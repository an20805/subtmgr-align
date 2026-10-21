#!/usr/bin/env python3
"""
run_iterative_alignment_gpu.py
------------------------------
End-to-End Native GPU-Accelerated Iterative Subtomogram Alignment (AlignOT Barycentric 1-NN SGD).

Mathematically Equivariant & Verified:
  1. Exact Topology Representing Network (TRN) Neural Gas sampling from trn.py.
  2. Native GPU Batched Barycentric 1-NN SGD Alignment (Sinkhorn OT + Exact Analytical Polynomial Gradients on GPU Tensor Cores).
  3. Ultra-fast GPU 3D rigid volume transformation (~0.3 ms/volume on RTX 5060).
  4. In-memory GPU streaming accumulator for fast reference generation.
  5. Formats all timestamps in local timezone (Asia/Kolkata / IST).

Usage:
  python alignOT/src/run_iterative_alignment_gpu.py \\
      --cluster_dir Data/0.1/Cluster_0 \\
      --output_base_dir alignOT/outputs/gpu_runs \\
      --init_reference Data/0.mrc \\
      --selection_mode all \\
      --n_rounds 3 \\
      --num_workers 16 \\
      --reg 100 --sgd_lr 0.001 --sgd_iters 500
"""

import argparse
import contextlib
import gc
import io
import logging
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Set local timezone
os.environ["TZ"] = "Asia/Kolkata"
if hasattr(time, "tzset"):
    time.tzset()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ── Module Path Setup ─────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR.parent / "alignOT"))

import alignot_utils
import alignot_gpu_utils as gpu_utils
import coords as _coords
import utils as _utils


# ── Logger Setup ──────────────────────────────────────────────────────────────

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("AlignOT_Native_GPU_Barycentric")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── Worker Task for TRN Sampling ──────────────────────────────────────────────

def _sample_single_subtomogram(task_args: tuple) -> dict:
    """
    Parallel TRN sampling across CPU worker pool:
      1. Loads raw subtomogram.
      2. Pre-processes and extracts point cloud via Neural Gas TRN.
      3. Returns centered (X, Y, Z) point coordinates, centroid Bbar, and volume.
    """
    path, lowpass_sigma, threshold, num_points, seed = task_args
    volume = alignot_utils.load_mrc(path, permissive=True)

    src_prep = alignot_utils.prepare_sampling_volume(volume, sigma=lowpass_sigma)
    pts_src = alignot_utils.sample_volume_in_memory(
        src_prep,
        threshold=threshold,
        num_points=num_points,
        random_seed=seed,
    )

    Bbar = pts_src.mean(axis=0).astype(np.float64)
    pts_src_c = pts_src - Bbar
    src_xyz = pts_src_c[:, [2, 1, 0]].astype(np.float32)  # [z, y, x] -> (x, y, z)

    return {
        "subtomogram_id": int(path.stem),
        "file_name": path.name,
        "src_xyz": src_xyz,
        "Bbar": Bbar,
        "volume": volume,
    }


# ── CLI Arguments ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="High-Performance Native GPU Iterative Alignment (AlignOT Barycentric SGD)"
    )

    # Data
    p.add_argument("--cluster_dir", required=True,
                   help="Directory containing subtomogram .mrc files")
    p.add_argument("--output_base_dir", required=True,
                   help="Base directory for output runs")
    p.add_argument("--init_reference", default=None,
                   help="Path to initial reference .mrc. If not provided, computes streaming average.")

    # Particle selection
    p.add_argument("--selection_mode", default="all",
                   choices=["all", "first", "random", "even", "odd"],
                   help="Which particles to use: all / first N / random N / even / odd")
    p.add_argument("--n_sbtms", type=int, default=None,
                   help="Number of particles (for first / random mode)")

    # Rounds & Hardware
    p.add_argument("--n_rounds", type=int, default=3,
                   help="Number of alignment rounds (default: 3)")
    p.add_argument("--num_workers", type=int, default=16,
                   help="Number of parallel CPU worker processes for TRN sampling (default: 16)")
    p.add_argument("--device", type=str, default="cuda",
                   help="GPU device for Sinkhorn, SGD & 3D volume rotations (default: cuda)")

    # Sampling
    p.add_argument("--num_points", type=int, default=500,
                   help="TRN points per volume (default: 500)")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="Density threshold after inversion (default: 0.7)")
    p.add_argument("--lowpass_sigma", type=float, default=2.0,
                   help="Gaussian LP sigma (voxels) applied before sampling only (default: 2.0)")

    # AlignOT SGD
    p.add_argument("--reg", type=float, default=100.0,
                   help="Sinkhorn regularisation (default: 100.0)")
    p.add_argument("--sgd_iters", type=int, default=500,
                   help="SGD iterations per particle (default: 500)")
    p.add_argument("--sgd_lr", type=float, default=0.001,
                   help="SGD learning rate (default: 0.001)")
    p.add_argument("--sinkhorn_iters", type=int, default=15,
                   help="Sinkhorn scaling iterations per SGD step (default: 15)")
    p.add_argument("--cost_metric", type=str, default="l2_sq", choices=["l2_sq", "l1", "l2"],
                   help="Ground distance cost metric in OT and SGD: 'l2_sq' (default), 'l1', or 'l2'")
    p.add_argument("--target_mode", type=str, default="barycentric", choices=["barycentric", "base", "argmax"],
                   help="Target point selection mode: 'barycentric' (default, 1-NN barycentric projection) or 'base' / 'argmax' (hard argmax on transport plan)")

    # Misc
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    device = gpu_utils.get_device(args.device)

    # ── Output directory & logger ──────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metric_suffix = f"_{args.cost_metric}" if args.cost_metric != "l2_sq" else ""
    mode_label = "argmax" if args.target_mode in ("base", "argmax") else "barycentric"
    run_name = f"run_{timestamp}_{args.selection_mode}_gpu_{mode_label}{metric_suffix}"
    output_dir = Path(args.output_base_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "run.log")
    logger.info(f"Starting AlignOT Native GPU Iterative Subtomogram Alignment [Target Mode: {args.target_mode}, Cost Metric: {args.cost_metric}]")
    logger.info(f"Target GPU   : {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    logger.info(f"TRN Workers  : {args.num_workers} parallel sampling processes")
    logger.info(f"Run directory: {output_dir.resolve()}")
    logger.info(f"Arguments    : {args}")

    # ── Particle discovery & selection ─────────────────────────────────────────
    all_paths = alignot_utils.get_cluster_paths(args.cluster_dir)
    logger.info(f"Discovered {len(all_paths)} subtomograms in {args.cluster_dir}")

    if args.selection_mode == "all":
        cluster_paths = all_paths
    elif args.selection_mode == "even":
        cluster_paths = [p for p in all_paths if int(p.stem) % 2 == 0]
    elif args.selection_mode == "odd":
        cluster_paths = [p for p in all_paths if int(p.stem) % 2 == 1]
    elif args.selection_mode == "first":
        n = args.n_sbtms or len(all_paths)
        cluster_paths = all_paths[:n]
    else:  # random
        n = args.n_sbtms or len(all_paths)
        cluster_paths = random.sample(all_paths, n)

    n_actual = len(cluster_paths)
    logger.info(f"Selected {n_actual} subtomograms ({args.selection_mode} mode)")

    # ── Round 0: Initial Reference ─────────────────────────────────────────────
    if args.init_reference is not None:
        logger.info(f"Using external initial reference: {args.init_reference}")
        current_reference = alignot_utils.load_mrc(args.init_reference, permissive=True)
        title_r0 = "External Initial Ref (Round 0)"
    else:
        logger.info("Computing initial reference (Round 0) — GPU streaming average of raw particles...")
        t0_avg = time.perf_counter()
        current_reference_tensor = gpu_utils.compute_streaming_average_gpu(
            cluster_paths, device=device, progress_every=max(1, n_actual // 10), logger=logger
        )
        current_reference = current_reference_tensor.cpu().numpy()
        logger.info(f"Initial streaming average computed in {time.perf_counter() - t0_avg:.2f}s")
        title_r0 = "Initial average (Round 0)"

    alignot_utils.save_mrc(output_dir / "reference_round_00.mrc", current_reference)
    gpu_utils.show_volume_slices(
        current_reference, title_r0,
        save_path=output_dir / "slices_round_00.png",
    )
    logger.info("Saved initial reference and slice plot.")

    all_references = [current_reference.copy()]
    round_metrics = []

    box_size = current_reference.shape[0]
    gpu_rotator = gpu_utils.GPUVolumeRotator(box_size=box_size, device=device)

    # ── Iterative Alignment Loop ───────────────────────────────────────────────
    for round_idx in range(1, args.n_rounds + 1):
        logger.info(f"\n--- Starting Round {round_idx}/{args.n_rounds} ---")
        round_start = time.perf_counter()

        # 1. Sample Reference Point Cloud
        t0_ref_trn = time.perf_counter()
        ref_prep = alignot_utils.prepare_sampling_volume(
            current_reference, sigma=args.lowpass_sigma
        )
        pts_ref = alignot_utils.sample_volume_in_memory(
            ref_prep,
            threshold=args.threshold,
            num_points=args.num_points,
            random_seed=args.seed + 1000 * round_idx,
        )
        Abar = pts_ref.mean(axis=0).astype(np.float64)
        pts_ref_c = pts_ref - Abar
        ref_xyz = pts_ref_c[:, [2, 1, 0]].astype(np.float32)  # [z, y, x] -> (x, y, z)
        logger.info(f"Sampled reference point cloud: {pts_ref.shape} in {time.perf_counter() - t0_ref_trn:.2f}s")

        # 2. Parallel TRN Sampling of Particles
        logger.info(f"Sampling {n_actual} subtomogram point clouds across {args.num_workers} workers...")
        t0_sample = time.perf_counter()
        sampling_tasks = [
            (
                path,
                args.lowpass_sigma,
                args.threshold,
                args.num_points,
                args.seed + round_idx * 1000 + idx,
            )
            for idx, path in enumerate(cluster_paths)
        ]

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            sampled_items = list(executor.map(_sample_single_subtomogram, sampling_tasks))

        logger.info(f"Sampled all {n_actual} particles in {time.perf_counter() - t0_sample:.2f}s")

        # 3. Stack Batched Source Clouds for GPU Alignment
        src_b_np = np.stack([item["src_xyz"] for item in sampled_items], axis=0)  # (N_particles, 500, 3)

        # 4. Native GPU Batched SGD (Barycentric or Base Argmax)
        logger.info(f"Running Native GPU Batched SGD ({args.target_mode}) on {n_actual} particles...")
        t0_gpu_sgd = time.perf_counter()
        quats_gpu, costs_gpu = gpu_utils.SGD_barycentric_gpu_batched(
            src_b_np,
            ref_xyz,
            lr=args.sgd_lr,
            max_iter=args.sgd_iters,
            reg=args.reg,
            sinkhorn_iters=args.sinkhorn_iters,
            cost_metric=args.cost_metric,
            target_mode=args.target_mode,
            seed=args.seed + round_idx * 1000,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_gpu_sgd = time.perf_counter() - t0_gpu_sgd
        logger.info(f"Completed Native GPU Batched SGD in {t_gpu_sgd:.2f}s ({t_gpu_sgd / n_actual * 1000:.1f} ms/particle)!")

        # 5. GPU 3D Volume Rigid Transformation & Accumulation
        logger.info("Applying exact centroid-aware rigid transformations on GPU...")
        t0_gpu_rot = time.perf_counter()
        sum_rotated_gpu = torch.zeros((box_size, box_size, box_size), dtype=torch.float64, device=device)

        alignment_results = []
        progress_every = max(1, n_actual // 10)

        for item_idx, item in enumerate(sampled_items, start=1):
            subtomogram_start = time.perf_counter()
            vol_np = item["volume"]
            Bbar = item["Bbar"]
            q = quats_gpu[item_idx - 1]
            cost = costs_gpu[item_idx - 1]

            rot_vol_gpu = gpu_rotator.rotate_volume(vol_np, q, Bbar, Abar)
            sum_rotated_gpu += rot_vol_gpu.double()

            runtime_s = float(t_gpu_sgd / n_actual) + (time.perf_counter() - subtomogram_start)

            alignment_results.append({
                "subtomogram_id": item["subtomogram_id"],
                "file_name": item["file_name"],
                "wass_dist": float(cost),
                "runtime_s": float(runtime_s),
                "q": [float(v) for v in q],
            })

            if item_idx % progress_every == 0 or item_idx == n_actual:
                logger.info(
                    f"  Round {round_idx}: aligned {item_idx}/{n_actual} "
                    f"({runtime_s:.2f}s) — WassDist={float(cost):.2f}"
                )

            del vol_np, rot_vol_gpu

        if device.type == "cuda":
            torch.cuda.synchronize()
        logger.info(f"Transformed & accumulated {n_actual} volumes on GPU in {time.perf_counter() - t0_gpu_rot:.2f}s")

        # 6. Build new reference
        next_reference = (sum_rotated_gpu / float(n_actual)).float().cpu().numpy()
        del sum_rotated_gpu, sampled_items, src_b_np
        if device.type == "cuda":
            torch.cuda.empty_cache()

        alignot_utils.save_mrc(
            output_dir / f"reference_round_{round_idx:02d}.mrc", next_reference
        )
        gpu_utils.show_volume_slices(
            next_reference, f"Round {round_idx} average (Native GPU Barycentric 1-NN)",
            save_path=output_dir / f"slices_round_{round_idx:02d}.png",
        )

        # 7. Save per-particle CSV metrics
        particle_rows = [
            {
                "round": round_idx,
                "subtomogram_id": r["subtomogram_id"],
                "file_name": r["file_name"],
                "wass_dist": r["wass_dist"],
                "runtime_s": r["runtime_s"],
                "q_w": r["q"][0],
                "q_x": r["q"][1],
                "q_y": r["q"][2],
                "q_z": r["q"][3],
            }
            for r in alignment_results
        ]
        round_df = pd.DataFrame(particle_rows)
        round_df.to_csv(
            output_dir / f"metrics_particles_round_{round_idx:02d}.csv", index=False
        )
        round_df.to_csv(
            output_dir / "quaternions.csv", index=False
        )

        # 8. Round-level metrics
        change = alignot_utils.reference_change_stats(current_reference, next_reference, round_idx)
        change.update({
            "n_subtomograms": int(n_actual),
            "mean_wass_dist": float(round_df["wass_dist"].mean()),
            "mean_runtime_s": float(round_df["runtime_s"].mean()),
            "round_runtime_s": float(time.perf_counter() - round_start),
            "reference_path": str(output_dir / f"reference_round_{round_idx:02d}.mrc"),
        })
        round_metrics.append(change)

        logger.info(
            f"Round {round_idx} complete in {change['round_runtime_s']:.1f}s  "
            f"| mean WassDist={change['mean_wass_dist']:.2f}  "
            f"| ref L2 change={change['reference_l2_change']:.4f}"
        )

        current_reference = next_reference
        all_references.append(current_reference.copy())
        del alignment_results, next_reference, round_df
        gc.collect()

    # ── Summary & Metrics Export ───────────────────────────────────────────────
    summary_df = pd.DataFrame(round_metrics)
    summary_path = output_dir / "round_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved round metrics to {summary_path}")

    # Summary plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(summary_df["round"], summary_df["reference_l2_change"], marker="o")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Reference L2 change")
    axes[0].set_title("Reference change across rounds")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(summary_df["round"], summary_df["mean_wass_dist"], marker="o", color="orange")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Mean Wasserstein dist")
    axes[1].set_title("Mean alignment cost across rounds")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "summary_metrics_plot.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved summary plot to {plot_path}")

    # Combined slice grid across all rounds
    params_label = (
        f"M={args.num_points}, σ_LP={args.lowpass_sigma}, thr={args.threshold}, "
        f"reg={args.reg}, cost={args.cost_metric}, sink_iters={args.sinkhorn_iters}, iters={args.sgd_iters}, lr={args.sgd_lr} (Native GPU Barycentric 1-NN)"
    )
    combined = alignot_utils.make_combined_plot(all_references, params_label, output_dir)
    logger.info(f"Saved combined slices to {combined}")
    logger.info("Iterative Native GPU Barycentric 1-NN alignment complete!")


if __name__ == "__main__":
    main()

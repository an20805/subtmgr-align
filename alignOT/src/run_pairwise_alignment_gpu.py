#!/usr/bin/env python3
"""
run_pairwise_alignment_gpu.py
------------------------------
Reference-Free Joint Pairwise Wasserstein SGD Alignment.

Objective:
  {q_i*} = argmin_{q_i in S^3} sum_{i<j} W_eps(R(q_i)*X_i, R(q_j)*X_j)

Key properties:
  - No fixed reference: all particles serve as each other's alignment targets.
  - All C(N,2) pairs batched into a single GPU Sinkhorn call per step.
  - Symmetric gradient update: both q_i and q_j updated per pair.
  - Particle 0 (or --anchor_idx) is fixed at identity to remove gauge freedom.
  - Identity initialization: no warm-start, pure from-scratch optimization.

Usage:
  python alignOT/src/run_pairwise_alignment_gpu.py \
      --cluster_dir Data/0.1/Cluster_0 \
      --output_base_dir alignOT/outputs/pairwise_runs \
      --selection_mode first \
      --n_sbtms 20 \
      --sgd_iters 500 \
      --device cuda
"""

import argparse
import gc
import logging
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Set local timezone
os.environ["TZ"] = "Asia/Kolkata"
if hasattr(time, "tzset"):
    time.tzset()

# ── Module Path Setup ─────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR.parent / "alignOT"))

import alignot_utils
import alignot_gpu_utils as gpu_utils
import coords as _coords


# ── Logger ────────────────────────────────────────────────────────────────────

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("AlignOT_Pairwise_Wass")
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
    """Parallel TRN point-cloud sampling worker."""
    path, lowpass_sigma, threshold, num_points, seed = task_args
    volume = alignot_utils.load_mrc(path, permissive=True)
    src_prep = alignot_utils.prepare_sampling_volume(volume, sigma=lowpass_sigma)
    pts_src = alignot_utils.sample_volume_in_memory(
        src_prep, threshold=threshold, num_points=num_points, random_seed=seed,
    )
    Bbar = pts_src.mean(axis=0).astype(np.float64)
    pts_src_c = pts_src - Bbar
    src_xyz = pts_src_c[:, [2, 1, 0]].astype(np.float32)  # [z,y,x] -> (x,y,z)
    return {
        "subtomogram_id": int(path.stem),
        "file_name": path.name,
        "src_xyz": src_xyz,
        "Bbar": Bbar,
        "volume": volume,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Joint Pairwise Wasserstein SGD — Reference-Free Subtomogram Alignment"
    )
    p.add_argument("--cluster_dir", required=True)
    p.add_argument("--output_base_dir", required=True)
    p.add_argument("--selection_mode", default="first",
                   choices=["first", "random", "all", "even", "odd"])
    p.add_argument("--n_sbtms", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_points", type=int, default=500)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--lowpass_sigma", type=float, default=2.0)
    p.add_argument("--reg", type=float, default=100.0)
    p.add_argument("--sgd_iters", type=int, default=500)
    p.add_argument("--sgd_lr", type=float, default=0.001)
    p.add_argument("--sinkhorn_iters", type=int, default=15)
    p.add_argument("--anchor_idx", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = gpu_utils.get_device(args.device)

    # ── Output directory & logger ─────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}_{args.selection_mode}{args.n_sbtms}_pairwise_wass"
    output_dir = Path(args.output_base_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "run.log")
    logger.info("=" * 70)
    logger.info("Joint Pairwise Wasserstein SGD — Reference-Free Alignment")
    logger.info("=" * 70)
    logger.info(f"Device       : {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    logger.info(f"Run directory: {output_dir.resolve()}")
    logger.info(f"Arguments    : {args}")

    # ── Particle discovery & selection ────────────────────────────────────────
    all_paths = alignot_utils.get_cluster_paths(args.cluster_dir)
    logger.info(f"Discovered {len(all_paths)} subtomograms in {args.cluster_dir}")

    if args.selection_mode == "all":
        cluster_paths = all_paths
    elif args.selection_mode == "even":
        cluster_paths = [p for p in all_paths if int(p.stem) % 2 == 0]
    elif args.selection_mode == "odd":
        cluster_paths = [p for p in all_paths if int(p.stem) % 2 == 1]
    elif args.selection_mode == "first":
        cluster_paths = all_paths[:args.n_sbtms]
    else:  # random
        cluster_paths = random.sample(all_paths, args.n_sbtms)

    N = len(cluster_paths)
    n_pairs = N * (N - 1) // 2
    logger.info(f"Selected {N} particles ({args.selection_mode} mode) → {n_pairs} pairs [C({N},2)]")

    # ── Parallel TRN sampling ─────────────────────────────────────────────────
    logger.info(f"Sampling {N} point clouds across {args.num_workers} CPU workers...")
    t0 = time.perf_counter()
    sampling_tasks = [
        (path, args.lowpass_sigma, args.threshold, args.num_points, args.seed + idx)
        for idx, path in enumerate(cluster_paths)
    ]
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        sampled_items = list(executor.map(_sample_single_subtomogram, sampling_tasks))
    logger.info(f"TRN sampling complete in {time.perf_counter() - t0:.2f}s")

    src_all_np = np.stack([item["src_xyz"] for item in sampled_items], axis=0)  # (N, N_pts, 3)
    Bbars = [item["Bbar"] for item in sampled_items]
    box_size = sampled_items[0]["volume"].shape[0]

    # ── Save raw average (no alignment) ──────────────────────────────────────
    logger.info("Computing initial raw average (no alignment)...")
    raw_avg = np.mean(
        np.stack([item["volume"] for item in sampled_items], axis=0), axis=0
    ).astype(np.float32)
    alignot_utils.save_mrc(output_dir / "reference_raw_avg.mrc", raw_avg)
    gpu_utils.show_volume_slices(
        raw_avg, "Raw Average (no alignment)",
        save_path=output_dir / "slices_raw_avg.png"
    )
    logger.info("Saved raw average and slices.")
    del raw_avg

    # ── Joint Pairwise Wasserstein SGD ────────────────────────────────────────
    logger.info("Starting Joint Pairwise Wasserstein SGD:")
    logger.info(f"  Particles   : {N}  |  Pairs: {n_pairs}")
    logger.info(f"  SGD iters   : {args.sgd_iters}  |  lr: {args.sgd_lr}")
    logger.info(f"  Sinkhorn ε  : {args.reg}  |  Sinkhorn iters: {args.sinkhorn_iters}")
    logger.info(f"  Anchor idx  : {args.anchor_idx} (identity, fixed throughout)")
    logger.info(f"  Init        : all q_i = identity (no warm-start)")

    t0_sgd = time.perf_counter()
    quats, loss_curve = gpu_utils.SGD_pairwise_gpu(
        src_all_np=src_all_np,
        lr=args.sgd_lr,
        max_iter=args.sgd_iters,
        reg=args.reg,
        seed=args.seed,
        anchor_idx=args.anchor_idx,
        sinkhorn_iters=args.sinkhorn_iters,
        log_every=args.log_every,
        device=device,
    )
    t_sgd = time.perf_counter() - t0_sgd
    logger.info(f"Pairwise SGD complete in {t_sgd:.2f}s  ({t_sgd / args.sgd_iters * 1000:.1f} ms/iter)")
    logger.info(f"Final mean pairwise W2: {loss_curve[-1]:.4f}  (initial: {loss_curve[0]:.4f})")

    # ── Save loss curve ───────────────────────────────────────────────────────
    loss_df = pd.DataFrame({"iter": list(range(1, args.sgd_iters + 1)), "mean_pairwise_W2": loss_curve})
    loss_df.to_csv(output_dir / "loss_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(loss_df["iter"], loss_df["mean_pairwise_W2"], linewidth=1.5, color="steelblue")
    ax.set_xlabel("SGD Iteration")
    ax.set_ylabel("Mean Pairwise W₂")
    ax.set_title(f"Joint Pairwise Wasserstein Loss — N={N}, ε={args.reg}, lr={args.sgd_lr}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved loss curve CSV and plot.")

    # ── Save per-particle quaternion CSV ──────────────────────────────────────
    quat_rows = [
        {
            "particle_idx": i,
            "subtomogram_id": sampled_items[i]["subtomogram_id"],
            "file_name": sampled_items[i]["file_name"],
            "q_w": quats[i][0], "q_x": quats[i][1],
            "q_y": quats[i][2], "q_z": quats[i][3],
            "anchored": (i == args.anchor_idx),
        }
        for i in range(N)
    ]
    pd.DataFrame(quat_rows).to_csv(output_dir / "quaternions.csv", index=False)
    logger.info("Saved per-particle quaternion CSV.")

    # ── Apply rotations & build final reference ───────────────────────────────
    logger.info("Applying recovered rotations to original volumes on GPU...")
    gpu_rotator = gpu_utils.GPUVolumeRotator(box_size=box_size, device=device)

    # Global centroid target: mean of all particle centroids (Abar equivalent)
    Abar_global = np.mean(np.stack(Bbars, axis=0), axis=0)

    t0_rot = time.perf_counter()
    sum_vol = torch.zeros((box_size, box_size, box_size), dtype=torch.float64, device=device)
    for i, item in enumerate(sampled_items):
        rot_vol = gpu_rotator.rotate_volume(item["volume"], quats[i], item["Bbar"], Abar_global)
        sum_vol += rot_vol.double()
        del rot_vol
    torch.cuda.synchronize()
    logger.info(f"Volume rotation & accumulation in {time.perf_counter() - t0_rot:.2f}s")

    final_ref = (sum_vol / float(N)).float().cpu().numpy()
    del sum_vol
    torch.cuda.empty_cache()
    gc.collect()

    # ── Save final reference & slices ─────────────────────────────────────────
    ref_path = output_dir / "reference_pairwise_final.mrc"
    alignot_utils.save_mrc(ref_path, final_ref)
    logger.info(f"Saved final reference MRC: {ref_path}")

    gpu_utils.show_volume_slices(
        final_ref,
        f"Pairwise Wass Final (N={N}, iters={args.sgd_iters}, lr={args.sgd_lr}, ε={args.reg})",
        save_path=output_dir / "slices_pairwise_final.png",
    )
    logger.info("Saved final reference slice plot.")

    # ── Side-by-side raw vs aligned comparison ────────────────────────────────
    raw_avg_reload = alignot_utils.load_mrc(output_dir / "reference_raw_avg.mrc")
    cz, cy, cx = [s // 2 for s in final_ref.shape]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="white")
    slices_raw = [raw_avg_reload[cz,:,:], raw_avg_reload[:,cy,:], raw_avg_reload[:,:,cx]]
    slices_aln = [final_ref[cz,:,:],      final_ref[:,cy,:],      final_ref[:,:,cx]]
    labels = ["XY (Z-mid)", "XZ (Y-mid)", "YZ (X-mid)"]
    for col, (sl_r, sl_a, lbl) in enumerate(zip(slices_raw, slices_aln, labels)):
        axes[0, col].imshow(sl_r, cmap="gray", origin="lower")
        axes[0, col].set_title(f"Raw Avg — {lbl}", fontsize=10)
        axes[0, col].axis("off")
        axes[1, col].imshow(sl_a, cmap="gray", origin="lower")
        axes[1, col].set_title(f"Pairwise Aligned — {lbl}", fontsize=10)
        axes[1, col].axis("off")
    fig.suptitle(
        f"Joint Pairwise Wasserstein | N={N}, iters={args.sgd_iters}, "
        f"lr={args.sgd_lr}, ε={args.reg}, anchor=particle {args.anchor_idx}",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_raw_vs_aligned.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved raw vs aligned comparison plot.")

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Run complete.")
    logger.info(f"  Particles    : {N}")
    logger.info(f"  Pairs        : {n_pairs}")
    logger.info(f"  SGD iters    : {args.sgd_iters}")
    logger.info(f"  Total time   : {t_sgd:.1f}s")
    logger.info(f"  Final W2     : {loss_curve[-1]:.4f}")
    logger.info(f"  Output dir   : {output_dir.resolve()}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_spectral_sync_alignment_gpu.py
------------------------------------
Reference-Free Two-Stage Rotation Synchronization + Iterative Refinement.

Pipeline:
  Stage 1 — Decoupled Pairwise OT (GPU):
    For each of C(N,2) pairs (i,j) independently:
        q_{ij}* = argmin_{q in S^3} W_eps(R(q) X_i, X_j)
    No gradient tug-of-war. Each pair converges cleanly.

  Stage 2 — Spectral Rotation Synchronization (~0.05 s):
    Build (3N x 3N) block matrix H from pairwise rotations.
    Top-3 eigenvectors → project 3x3 blocks to SO(3) → absolute rotations.

  Stage 3 — Iterative Refinement (optional, K rounds):
    Use spectral sync rotations as warm start for reference-guided GPU alignment.
    Same SGD_barycentric_gpu_batched loop as run_iterative_alignment_gpu.py.

Validation:
  Synthetic data: evaluates geodesic angular error vs ground truth CSV.
  Real data: report pairwise residual and reference L2 change per round.

Usage:
  python alignOT/src/run_spectral_sync_alignment_gpu.py \\
      --cluster_dir Data/0.1/Cluster_0 \\
      --output_base_dir alignOT/outputs/spectral_sync_runs \\
      --n_sbtms 40 --selection_mode first \\
      --n_refine_rounds 3 \\
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
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Timezone
os.environ["TZ"] = "Asia/Kolkata"
if hasattr(time, "tzset"):
    time.tzset()

# ── Module path setup ─────────────────────────────────────────────────────────
SRC_DIR  = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR.parent / "alignOT"))

import alignot_utils
import alignot_gpu_utils as gpu_utils
import alignot_sync_utils as sync_utils
import coords as _coords


# ── Logger ────────────────────────────────────────────────────────────────────

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("AlignOT_SpectralSync")
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


# ── TRN sampling worker (same as run_iterative_alignment_gpu.py) ──────────────

def _sample_single_subtomogram(task_args: tuple) -> dict:
    path, lowpass_sigma, threshold, num_points, seed = task_args
    volume  = alignot_utils.load_mrc(path, permissive=True)
    src_prep = alignot_utils.prepare_sampling_volume(volume, sigma=lowpass_sigma)
    pts_src  = alignot_utils.sample_volume_in_memory(
        src_prep, threshold=threshold, num_points=num_points, random_seed=seed
    )
    Bbar     = pts_src.mean(axis=0).astype(np.float64)
    pts_src_c = pts_src - Bbar
    src_xyz  = pts_src_c[:, [2, 1, 0]].astype(np.float32)  # [z,y,x] → (x,y,z)
    return {
        "subtomogram_id": int(path.stem),
        "file_name":      path.name,
        "src_xyz":        src_xyz,
        "Bbar":           Bbar,
        "volume":         volume,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Reference-Free Spectral Rotation Sync + Iterative Refinement"
    )
    # Data
    p.add_argument("--cluster_dir",      required=True)
    p.add_argument("--output_base_dir",  required=True)

    # Particle selection
    p.add_argument("--selection_mode", default="first",
                   choices=["first", "random", "all", "even", "odd"])
    p.add_argument("--n_sbtms", type=int, default=40)

    # Hardware
    p.add_argument("--device",       default="cuda")
    p.add_argument("--num_workers",  type=int, default=16)

    # TRN sampling
    p.add_argument("--num_points",    type=int,   default=500)
    p.add_argument("--threshold",     type=float, default=0.7)
    p.add_argument("--lowpass_sigma", type=float, default=2.0)

    # Stage 1 — Decoupled Pairwise OT
    p.add_argument("--reg",              type=float, default=100.0)
    p.add_argument("--sgd_iters",        type=int,   default=500)
    p.add_argument("--sgd_lr",           type=float, default=0.001)
    p.add_argument("--sinkhorn_iters",   type=int,   default=15)
    p.add_argument("--pair_batch_size",  type=int,   default=200,
                   help="Pairs per GPU mini-batch (memory knob)")

    # Stage 3 — Iterative refinement
    p.add_argument("--n_refine_rounds", type=int, default=3,
                   help="Iterative refinement rounds after spectral sync (0 = skip)")

    # Validation
    p.add_argument("--gt_csv", default=None,
                   help="Ground-truth CSV for synthetic benchmark evaluation")

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = gpu_utils.get_device(args.device)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Output dir & logger ───────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = (f"run_{timestamp}_{args.selection_mode}{args.n_sbtms}"
                 f"_spectral_sync")
    output_dir = Path(args.output_base_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "run.log")
    logger.info("=" * 70)
    logger.info("Two-Stage Rotation Synchronization: Decoupled Pairwise OT + Spectral Sync")
    logger.info("=" * 70)
    logger.info(f"Device       : {device}"
                f"  ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    logger.info(f"Run dir      : {output_dir.resolve()}")
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
    else:
        cluster_paths = random.sample(all_paths, args.n_sbtms)

    N = len(cluster_paths)
    n_pairs = N * (N - 1) // 2
    logger.info(f"Selected {N} particles ({args.selection_mode} mode)  →  {n_pairs} pairs")

    # ── TRN sampling (parallel) ───────────────────────────────────────────────
    logger.info(f"TRN sampling {N} point clouds across {args.num_workers} workers...")
    t0 = time.perf_counter()
    tasks = [
        (path, args.lowpass_sigma, args.threshold, args.num_points, args.seed + idx)
        for idx, path in enumerate(cluster_paths)
    ]
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        sampled_items = list(ex.map(_sample_single_subtomogram, tasks))
    logger.info(f"TRN sampling done in {time.perf_counter()-t0:.2f}s")

    src_all_np = np.stack([item["src_xyz"] for item in sampled_items], axis=0)  # (N, M, 3)
    Bbars      = [item["Bbar"] for item in sampled_items]
    box_size   = sampled_items[0]["volume"].shape[0]
    M          = src_all_np.shape[1]

    # Centroid target: global average of all particle centroids
    Abar_global = np.mean(np.stack(Bbars, axis=0), axis=0)

    # ── Build all C(N,2) pair indices ─────────────────────────────────────────
    pairs      = [(i, j) for i in range(N) for j in range(i + 1, N)]
    idx_i_list = [p[0] for p in pairs]
    idx_j_list = [p[1] for p in pairs]
    logger.info(f"Built {len(pairs)} pairs [C({N},2)]")

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Decoupled Pairwise OT
    # ════════════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("─" * 60)
    logger.info("STAGE 1: Decoupled Pairwise OT")
    logger.info(f"  {n_pairs} pairs  |  iters={args.sgd_iters}  "
                f"|  lr={args.sgd_lr}  |  eps={args.reg}  "
                f"|  pair_batch={args.pair_batch_size}")
    logger.info("─" * 60)

    t0_s1 = time.perf_counter()
    q_pairs, s1_loss_curve = gpu_utils.SGD_decoupled_pairwise_gpu(
        src_all_np      = src_all_np,
        pairs           = pairs,
        lr              = args.sgd_lr,
        max_iter        = args.sgd_iters,
        reg             = args.reg,
        seed            = args.seed,
        sinkhorn_iters  = args.sinkhorn_iters,
        pair_batch_size = args.pair_batch_size,
        log_every       = max(1, args.sgd_iters // 10),
        device          = device,
    )
    torch.cuda.synchronize() if device.type == "cuda" else None
    t_s1 = time.perf_counter() - t0_s1
    logger.info(f"Stage 1 done in {t_s1:.2f}s  "
                f"| final mean W2 = {s1_loss_curve[-1]:.4f}  "
                f"| initial mean W2 = {s1_loss_curve[0]:.4f}")

    # Save pairwise quaternions CSV
    df_pairs = pd.DataFrame([
        {"pair_i": pairs[p][0], "pair_j": pairs[p][1],
         "q_w": q_pairs[p][0], "q_x": q_pairs[p][1],
         "q_y": q_pairs[p][2], "q_z": q_pairs[p][3]}
        for p in range(len(pairs))
    ])
    df_pairs.to_csv(output_dir / "stage1_pairwise_quaternions.csv", index=False)

    # Save Stage 1 loss curve
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, len(s1_loss_curve) + 1), s1_loss_curve, linewidth=1.5, color="steelblue")
    ax.set_xlabel("SGD Iteration")
    ax.set_ylabel("Mean Pairwise W2")
    ax.set_title(f"Stage 1: Decoupled Pairwise OT  (N={N}, eps={args.reg}, lr={args.sgd_lr})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "stage1_loss_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Spectral Rotation Synchronization
    # ════════════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("─" * 60)
    logger.info("STAGE 2: Spectral Rotation Synchronization")
    logger.info("─" * 60)

    R_abs, q_align, eigenvalues, eigen_gap = sync_utils.spectral_rotation_sync(
        q_pairs      = q_pairs,
        idx_i_list   = idx_i_list,
        idx_j_list   = idx_j_list,
        N            = N,
        device       = "cpu",      # eigh is fast on CPU for N <= 200
        logger       = logger,
    )

    # Pairwise residual diagnostic
    residual = sync_utils.compute_pairwise_residual(
        R_abs, q_pairs, idx_i_list, idx_j_list
    )
    logger.info(f"Pairwise consistency residual: {residual:.4f}  "
                f"(Frobenius^2 per pair; <1.0 = good)")

    # Save eigenvalue spectrum plot
    fig, ax = plt.subplots(figsize=(9, 4))
    top_k = min(20, len(eigenvalues))
    ax.bar(range(1, top_k + 1), sorted(eigenvalues, reverse=True)[:top_k],
           color=["red"] * 3 + ["steelblue"] * (top_k - 3))
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"H eigenspectrum (N={N})  |  gap = {eigen_gap:.2f}  "
                 f"(ideal ≈ {N})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "stage2_eigenspectrum.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Save per-particle absolute rotations & alignment quaternions
    quat_rows = [
        {
            "particle_idx":    i,
            "subtomogram_id":  sampled_items[i]["subtomogram_id"],
            "file_name":       sampled_items[i]["file_name"],
            "q_w":             q_align[i][0],
            "q_x":             q_align[i][1],
            "q_y":             q_align[i][2],
            "q_z":             q_align[i][3],
        }
        for i in range(N)
    ]
    df_quat = pd.DataFrame(quat_rows)
    df_quat.to_csv(output_dir / "stage2_quaternions.csv", index=False)

    # ── Apply spectral-sync rotations → initial reference volume ─────────────
    logger.info("Applying spectral-sync rotations to build initial reference...")
    gpu_rotator = gpu_utils.GPUVolumeRotator(box_size=box_size, device=device)
    t0_rot = time.perf_counter()
    sum_vol = torch.zeros((box_size, box_size, box_size), dtype=torch.float64, device=device)
    for i, item in enumerate(sampled_items):
        rot_vol = gpu_rotator.rotate_volume(item["volume"], q_align[i], item["Bbar"], Abar_global)
        sum_vol += rot_vol.double()
        del rot_vol
    if device.type == "cuda":
        torch.cuda.synchronize()
    logger.info(f"Rotation + accumulation in {time.perf_counter()-t0_rot:.2f}s")

    current_reference = (sum_vol / float(N)).float().cpu().numpy()
    del sum_vol
    torch.cuda.empty_cache() if device.type == "cuda" else None

    alignot_utils.save_mrc(output_dir / "reference_spectral_sync.mrc", current_reference)
    gpu_utils.show_volume_slices(
        current_reference, "Spectral Sync Initial Reference",
        save_path=output_dir / "slices_spectral_sync.png"
    )
    logger.info("Saved spectral-sync initial reference MRC and slices.")

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 3 — Iterative Refinement (optional)
    # ════════════════════════════════════════════════════════════════════════
    all_references = [current_reference.copy()]

    if args.n_refine_rounds > 0:
        logger.info("")
        logger.info("─" * 60)
        logger.info(f"STAGE 3: Iterative Refinement ({args.n_refine_rounds} rounds)")
        logger.info("─" * 60)

        round_metrics = []
        current_quats = list(q_align)  # warm-start from spectral sync result

        for round_idx in range(1, args.n_refine_rounds + 1):
            logger.info(f"\n--- Refinement Round {round_idx}/{args.n_refine_rounds} ---")
            round_start = time.perf_counter()

            # Sample reference point cloud
            ref_prep = alignot_utils.prepare_sampling_volume(
                current_reference, sigma=args.lowpass_sigma
            )
            pts_ref = alignot_utils.sample_volume_in_memory(
                ref_prep, threshold=args.threshold,
                num_points=args.num_points,
                random_seed=args.seed + 1000 * round_idx,
            )
            Abar_ref = pts_ref.mean(axis=0).astype(np.float64)
            ref_xyz  = (pts_ref - Abar_ref)[:, [2, 1, 0]].astype(np.float32)  # (x,y,z)

            # Re-sample particle point clouds for this round
            logger.info(f"Re-sampling {N} particle clouds...")
            t0_s = time.perf_counter()
            tasks_r = [
                (path, args.lowpass_sigma, args.threshold, args.num_points,
                 args.seed + round_idx * 1000 + idx)
                for idx, path in enumerate(cluster_paths)
            ]
            with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
                sampled_items_r = list(ex.map(_sample_single_subtomogram, tasks_r))
            src_b_np_r = np.stack([it["src_xyz"] for it in sampled_items_r], axis=0)
            logger.info(f"  Sampling done in {time.perf_counter()-t0_s:.2f}s")

            # GPU Barycentric 1-NN SGD (reference-guided)
            logger.info("  Running GPU Barycentric SGD...")
            t0_sgd = time.perf_counter()
            quats_r, costs_r = gpu_utils.SGD_barycentric_gpu_batched(
                src_b_np_r, ref_xyz,
                lr       = args.sgd_lr,
                max_iter = args.sgd_iters,
                reg      = args.reg,
                seed     = args.seed + round_idx * 1000,
                device   = device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_sgd = time.perf_counter() - t0_sgd
            logger.info(f"  SGD done in {t_sgd:.2f}s  |  "
                        f"mean W2 = {float(np.mean(costs_r)):.4f}")
            current_quats = quats_r

            # Rotate volumes and accumulate new reference
            sum_vol_r = torch.zeros((box_size, box_size, box_size),
                                    dtype=torch.float64, device=device)
            for i, item in enumerate(sampled_items_r):
                rv = gpu_rotator.rotate_volume(item["volume"], quats_r[i],
                                               item["Bbar"], Abar_ref)
                sum_vol_r += rv.double()
                del rv
            if device.type == "cuda":
                torch.cuda.synchronize()

            next_ref = (sum_vol_r / float(N)).float().cpu().numpy()
            del sum_vol_r
            torch.cuda.empty_cache() if device.type == "cuda" else None

            # Save round outputs
            alignot_utils.save_mrc(
                output_dir / f"reference_round_{round_idx:02d}.mrc", next_ref
            )
            gpu_utils.show_volume_slices(
                next_ref, f"Refinement Round {round_idx}",
                save_path=output_dir / f"slices_round_{round_idx:02d}.png"
            )

            # Per-particle CSV
            rows_r = [
                {
                    "round":          round_idx,
                    "subtomogram_id": sampled_items_r[i]["subtomogram_id"],
                    "file_name":      sampled_items_r[i]["file_name"],
                    "wass_dist":      float(costs_r[i]),
                    "q_w":            quats_r[i][0], "q_x": quats_r[i][1],
                    "q_y":            quats_r[i][2], "q_z": quats_r[i][3],
                }
                for i in range(N)
            ]
            pd.DataFrame(rows_r).to_csv(
                output_dir / f"metrics_round_{round_idx:02d}.csv", index=False
            )

            change = alignot_utils.reference_change_stats(current_reference, next_ref, round_idx)
            change.update({
                "n_subtomograms":  N,
                "mean_wass_dist":  float(np.mean(costs_r)),
                "round_runtime_s": float(time.perf_counter() - round_start),
            })
            round_metrics.append(change)
            logger.info(f"  Round {round_idx}: L2 change = {change['reference_l2_change']:.4f}  "
                        f"|  mean W2 = {change['mean_wass_dist']:.4f}  "
                        f"|  runtime = {change['round_runtime_s']:.1f}s")

            current_reference = next_ref
            all_references.append(current_reference.copy())
            del sampled_items_r, src_b_np_r
            gc.collect()

        # Save round metrics summary
        pd.DataFrame(round_metrics).to_csv(output_dir / "round_metrics.csv", index=False)

        # Save final quaternions (compatible with evaluate_rotation_recovery.py)
        final_quat_rows = [
            {
                "particle_idx":   i,
                "subtomogram_id": sampled_items[i]["subtomogram_id"],
                "file_name":      sampled_items[i]["file_name"],
                "q_w":            current_quats[i][0],
                "q_x":            current_quats[i][1],
                "q_y":            current_quats[i][2],
                "q_z":            current_quats[i][3],
            }
            for i in range(N)
        ]
        df_final = pd.DataFrame(final_quat_rows)
        df_final.to_csv(output_dir / "quaternions.csv", index=False)
        logger.info("Saved final quaternions CSV (compatible with evaluate_rotation_recovery.py)")
    else:
        # No refinement: use spectral sync quaternions as final
        df_quat.rename(columns={}).to_csv(output_dir / "quaternions.csv", index=False)

    # ── Combined slice grid across all rounds ─────────────────────────────────
    params_label = (
        f"N={N}, eps={args.reg}, iters={args.sgd_iters}, lr={args.sgd_lr}, "
        f"pair_batch={args.pair_batch_size}, refine_rounds={args.n_refine_rounds}"
    )
    combined = alignot_utils.make_combined_plot(all_references, params_label, output_dir)
    logger.info(f"Saved combined slices plot: {combined}")

    # ── Optional: evaluate against ground truth ───────────────────────────────
    if args.gt_csv is not None:
        gt_csv = Path(args.gt_csv)
        if gt_csv.exists():
            logger.info(f"\nEvaluating against ground truth: {gt_csv}")
            _evaluate_vs_gt(
                output_dir / "quaternions.csv",
                gt_csv,
                sampled_items,
                logger,
            )
        else:
            logger.warning(f"Ground truth CSV not found: {gt_csv}")

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Run complete.")
    logger.info(f"  N particles      : {N}")
    logger.info(f"  Pairs            : {n_pairs}")
    logger.info(f"  Eigenvalue gap   : {eigen_gap:.2f}  (ideal ≈ {N})")
    logger.info(f"  Pairwise residual: {residual:.4f}")
    logger.info(f"  Refinement rounds: {args.n_refine_rounds}")
    logger.info(f"  Output directory : {output_dir.resolve()}")
    logger.info("=" * 70)


def _evaluate_vs_gt(est_csv: Path, gt_csv: Path, sampled_items, logger):
    """Compare estimated quaternions to ground truth; log geodesic errors."""
    from scipy.spatial.transform import Rotation

    df_gt  = pd.read_csv(gt_csv)
    df_est = pd.read_csv(est_csv)

    # Build subtomogram_id from sampled_items if not present in est_csv
    if "subtomogram_id" not in df_est.columns:
        df_est["subtomogram_id"] = [item["subtomogram_id"] for item in sampled_items]

    merged = pd.merge(df_gt, df_est, on="subtomogram_id", suffixes=("_gt", "_est"))

    errors = []
    for _, row in merged.iterrows():
        q_gt  = [row["q_w_gt"],  row["q_x_gt"],  row["q_y_gt"],  row["q_z_gt"]]
        q_est = [row["q_w_est"], row["q_x_est"], row["q_y_est"], row["q_z_est"]]
        r_gt  = Rotation.from_quat([q_gt[1],  q_gt[2],  q_gt[3],  q_gt[0]])
        r_est = Rotation.from_quat([q_est[1], q_est[2], q_est[3], q_est[0]])
        ang1  = float(np.degrees((r_est * r_gt.inv()).magnitude()))
        ang2  = float(np.degrees((r_est.inv() * r_gt.inv()).magnitude()))
        errors.append(min(ang1, ang2))

    logger.info("─" * 50)
    logger.info(" ROTATION RECOVERY vs GROUND TRUTH")
    logger.info("─" * 50)
    logger.info(f"  Mean Error   : {np.mean(errors):.2f}°")
    logger.info(f"  Median Error : {np.median(errors):.2f}°")
    logger.info(f"  Max Error    : {np.max(errors):.2f}°")
    logger.info(f"  Success <5°  : {np.sum(np.array(errors)<5)}/{len(errors)}")
    logger.info(f"  Success <10° : {np.sum(np.array(errors)<10)}/{len(errors)}")
    logger.info("─" * 50)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_coarse_to_fine_gpu.py
--------------------------
Reference-Guided Coarse-to-Fine Wasserstein Search + Iterative Refinement.

Mirrors classical STA (RELION/EMAN2) but uses W_ε on TRN point clouds instead
of cross-correlation on Fourier-space volumes:

    R_i* = argmin_{R ∈ G_Δ}  W_ε(R · X_i,  X_ref)

where G_Δ is a discrete uniform SO(3) grid at angular step Δ.

Pipeline per round:
  1. Sample reference and particle TRN point clouds (CPU, parallel).
  2. Stage A: evaluate W_ε for ALL K_coarse rotations × ALL N particles (GPU).
  3. Stage B: refine top-k candidates per particle with finer local grid (GPU).
  4. Apply best rotations to 3D volumes → accumulate streaming average (GPU).
  5. Save new reference MRC, per-particle metrics CSV, and diagnostic plots.

Usage:
    python alignOT/src/run_coarse_to_fine_gpu.py \\
        --cluster_dir Data/synthetic_clean_40 \\
        --output_base_dir alignOT/outputs/ctf_runs \\
        --n_rounds 3 --coarse_step 15.0 --fine_step 5.0 \\
        --gt_csv Data/synthetic_clean_40/ground_truth_rotations.csv \\
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

os.environ["TZ"] = "Asia/Kolkata"
if hasattr(time, "tzset"):
    time.tzset()

SRC_DIR   = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR.parent / "alignOT"))

import alignot_utils
import alignot_gpu_utils as gpu_utils
from alignot_w2_search import coarse_to_fine_w2_search, DEFAULT_CASCADE, DEFAULT_SINKHORN
from alignot_cc_search import coarse_to_fine_cc_search
from alignot_so3_grid import count_so3_grid, count_local_so3_grid


# ── Logger ────────────────────────────────────────────────────────────────────

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("AlignOT_CTF")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for h in [logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]:
        h.setLevel(logging.INFO)
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# ── TRN sampling worker ────────────────────────────────────────────────────────

def _sample_single_subtomogram(task_args: tuple) -> dict:
    path, lowpass_sigma, threshold, num_points, seed = task_args
    volume   = alignot_utils.load_mrc(path, permissive=True)
    src_prep = alignot_utils.prepare_sampling_volume(volume, sigma=lowpass_sigma)
    pts      = alignot_utils.sample_volume_in_memory(
        src_prep, threshold=threshold, num_points=num_points, random_seed=seed
    )
    Bbar     = pts.mean(axis=0).astype(np.float64)
    pts_c    = pts - Bbar
    src_xyz  = pts_c[:, [2, 1, 0]].astype(np.float32)   # [z,y,x] → (x,y,z)
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
        description="Reference-Guided Coarse-to-Fine W2 Search + Iterative Alignment"
    )
    # Data
    p.add_argument("--cluster_dir",      required=True)
    p.add_argument("--output_base_dir",  required=True)
    p.add_argument("--init_reference",   default="",
                   help="Initial reference MRC (leave empty to use streaming average)")

    # Particle selection
    p.add_argument("--selection_mode", default="all",
                   choices=["first", "random", "all", "even", "odd"])
    p.add_argument("--n_sbtms", type=int, default=40)

    # Hardware
    p.add_argument("--device",      default="cuda")
    p.add_argument("--num_workers", type=int, default=16)

    # TRN sampling
    p.add_argument("--num_points",    type=int,   default=500)
    p.add_argument("--threshold",     type=float, default=0.7)
    p.add_argument("--lowpass_sigma", type=float, default=2.0)

    # Coarse-to-Fine cascade
    p.add_argument("--cascade_steps",    default="60,20,7,2.5,1",
                   help="Comma-separated angular steps (degrees) for each level. "
                        "Level 0 = global grid, levels 1+ = local cone grids. "
                        "Default: 60→20→7→2.5→1 (5 levels, ≤1° accuracy)")
    p.add_argument("--cascade_sinkhorn", default="5,8,10,12,15",
                   help="Comma-separated Sinkhorn iters per level. "
                        "More iterations at finer levels for accurate W2 ranking.")
    p.add_argument("--top_k",           type=int, default=3,
                   help="Top-k candidates kept at each intermediate cascade level")
    p.add_argument("--fine_rot_batch",  type=int, default=200,
                   help="Rotation batch size for local levels (memory knob, ~200 MB)")

    # Objective Metric
    p.add_argument("--metric", default="w2", choices=["w2", "l1", "w1", "l2", "cc"],
                   help="Search objective metric: 'w2' (Wasserstein W2), 'l1' (Manhattan W1), "
                        "'l2'/'w1' (Euclidean W1) on TRN points, or 'cc' (Cross-Correlation on 3D volumes)")
    p.add_argument("--cost_metric", default="l2_sq", choices=["l2_sq", "l1", "l2"],
                   help="Ground distance cost metric for OT search: 'l2_sq' (default), 'l1', or 'l2'")
    p.add_argument("--reg", type=float, default=100.0,
                   help="Sinkhorn entropic regularization (used when OT metric is selected)")

    # Iterative rounds
    p.add_argument("--n_rounds", type=int, default=3)

    # Validation
    p.add_argument("--gt_csv", default="",
                   help="Ground-truth CSV for synthetic benchmark evaluation")

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Ground-truth evaluation ───────────────────────────────────────────────────

def _evaluate_vs_gt(est_csv: Path, gt_csv: Path, logger):
    from scipy.spatial.transform import Rotation

    df_gt  = pd.read_csv(gt_csv)
    df_est = pd.read_csv(est_csv)
    merged = pd.merge(df_gt, df_est, on="subtomogram_id", suffixes=("_gt", "_est"))

    errors = []
    for _, row in merged.iterrows():
        r_gt  = Rotation.from_quat([row["q_x_gt"],  row["q_y_gt"],
                                     row["q_z_gt"],  row["q_w_gt"]])
        r_est = Rotation.from_quat([row["q_x_est"], row["q_y_est"],
                                     row["q_z_est"], row["q_w_est"]])
        ang1  = float(np.degrees((r_est * r_gt.inv()).magnitude()))
        ang2  = float(np.degrees((r_est.inv() * r_gt.inv()).magnitude()))
        errors.append(min(ang1, ang2))

    logger.info("─" * 52)
    logger.info("  ROTATION RECOVERY vs GROUND TRUTH")
    logger.info("─" * 52)
    logger.info(f"  Matched particles  : {len(errors)}")
    logger.info(f"  Mean Error         : {np.mean(errors):.2f}°")
    logger.info(f"  Median Error       : {np.median(errors):.2f}°")
    logger.info(f"  Min / Max          : {np.min(errors):.2f}° / {np.max(errors):.2f}°")
    logger.info(f"  Success < 5°       : {np.sum(np.array(errors)<5)}/{len(errors)}")
    logger.info(f"  Success < 10°      : {np.sum(np.array(errors)<10)}/{len(errors)}")
    logger.info(f"  Success < 15°      : {np.sum(np.array(errors)<15)}/{len(errors)}")
    logger.info("─" * 52)
    return errors


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = gpu_utils.get_device(args.device)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Output dir ────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve metric configuration
    if args.metric == "l1" or (args.metric == "w2" and args.cost_metric == "l1"):
        cost_metric = "l1"
        is_ot = True
        metric_tag = "l1"
        metric_desc = "Optimal Transport (L1 Manhattan on TRN Point Clouds)"
        score_name = "L1_cost"
    elif args.metric in ["l2", "w1"] or (args.metric == "w2" and args.cost_metric == "l2"):
        cost_metric = "l2"
        is_ot = True
        metric_tag = "l2"
        metric_desc = "Optimal Transport (L2 Euclidean on TRN Point Clouds)"
        score_name = "L2_cost"
    elif args.metric == "w2":
        cost_metric = "l2_sq"
        is_ot = True
        metric_tag = "w2"
        metric_desc = "Optimal Transport (W2 Squared Euclidean on TRN Point Clouds)"
        score_name = "W2"
    else: # "cc"
        cost_metric = None
        is_ot = False
        metric_tag = "cc"
        metric_desc = "Normalized Cross-Correlation (CC on 3D Volumes)"
        score_name = "NCC"

    run_name  = (f"run_{timestamp}_{args.selection_mode}"
                 f"{args.n_sbtms}_ctf_{metric_tag}_L{len([float(s) for s in args.cascade_steps.split(',')])}")
    output_dir = Path(args.output_base_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "run.log")

    cascade_steps    = [float(s) for s in args.cascade_steps.split(",")]
    cascade_sinkhorn = [int(s)   for s in args.cascade_sinkhorn.split(",")]
    assert len(cascade_steps) == len(cascade_sinkhorn), \
        "cascade_steps and cascade_sinkhorn must have same number of entries"
    n_levels = len(cascade_steps)
    K_L0     = count_so3_grid(cascade_steps[0])

    logger.info("=" * 70)
    logger.info("Coarse-to-Fine Search: Reference-Guided STA")
    logger.info("=" * 70)
    logger.info(f"Device        : {device}"
                f"  ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    logger.info(f"Run dir       : {output_dir.resolve()}")
    logger.info(f"Metric        : {metric_desc}")
    cascade_str = "  →  ".join([f"{s}°" for s in cascade_steps])
    logger.info(f"Cascade       : {cascade_str}  ({n_levels} levels)")
    if is_ot:
        logger.info(f"Sinkhorn      : {cascade_sinkhorn}  |  reg={args.reg}  |  cost={cost_metric}")
    logger.info(f"Top-k kept    : {args.top_k}  at each intermediate level")
    logger.info(f"L0 global K   : {K_L0}")
    logger.info(f"Rounds        : {args.n_rounds}")
    logger.info(f"Args          : {args}")

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
    logger.info(f"Selected {N} particles ({args.selection_mode} mode)")

    # ── Initial TRN sampling ──────────────────────────────────────────────────
    logger.info(f"TRN sampling {N} particle clouds ({args.num_workers} workers)...")
    t0 = time.perf_counter()
    tasks = [
        (path, args.lowpass_sigma, args.threshold, args.num_points, args.seed + idx)
        for idx, path in enumerate(cluster_paths)
    ]
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        sampled_items = list(ex.map(_sample_single_subtomogram, tasks))
    logger.info(f"TRN sampling done in {time.perf_counter()-t0:.2f}s")

    box_size   = sampled_items[0]["volume"].shape[0]
    Bbars      = [item["Bbar"] for item in sampled_items]
    Abar_global = np.mean(np.stack(Bbars), axis=0)

    # ── Initial reference ─────────────────────────────────────────────────────
    if args.init_reference and Path(args.init_reference).is_file():
        logger.info(f"Using provided initial reference: {args.init_reference}")
        current_reference = alignot_utils.load_mrc(args.init_reference, permissive=True)
    else:
        logger.info("Computing streaming average as initial reference (no alignment)...")
        t0 = time.perf_counter()
        current_reference = gpu_utils.compute_streaming_average_gpu(
            cluster_paths, device=device
        )
        if hasattr(current_reference, 'numpy'):
            current_reference = current_reference.cpu().numpy()
        logger.info(f"Streaming average done in {time.perf_counter()-t0:.2f}s")
        alignot_utils.save_mrc(output_dir / "reference_round_00.mrc", current_reference)

    gpu_rotator = gpu_utils.GPUVolumeRotator(box_size=box_size, device=device)
    all_references = [current_reference.copy()]

    # ── Iterative coarse-to-fine rounds ──────────────────────────────────────
    round_metrics = []
    current_quats: Optional[List[List[float]]] = None
    raw_vols = [item["volume"] for item in sampled_items]

    for round_idx in range(1, args.n_rounds + 1):
        logger.info("")
        logger.info("─" * 70)
        logger.info(f"ROUND {round_idx}/{args.n_rounds}  (Metric: {metric_tag.upper()})")
        logger.info("─" * 70)
        round_start = time.perf_counter()

        if is_ot:
            # ── Sample reference TRN ──────────────────────────────────────────
            ref_prep = alignot_utils.prepare_sampling_volume(
                current_reference, sigma=args.lowpass_sigma
            )
            pts_ref  = alignot_utils.sample_volume_in_memory(
                ref_prep, threshold=args.threshold,
                num_points=args.num_points,
                random_seed=args.seed + 1000 * round_idx,
            )
            Abar_ref = pts_ref.mean(axis=0).astype(np.float64)
            ref_xyz  = (pts_ref - Abar_ref)[:, [2, 1, 0]].astype(np.float32)  # (x,y,z)

            # ── Re-sample particle TRN clouds ─────────────────────────────────
            logger.info(f"Re-sampling {N} particle clouds...")
            t0 = time.perf_counter()
            tasks_r = [
                (path, args.lowpass_sigma, args.threshold, args.num_points,
                 args.seed + round_idx * 1000 + idx)
                for idx, path in enumerate(cluster_paths)
            ]
            with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
                sampled_r = list(ex.map(_sample_single_subtomogram, tasks_r))
            src_b_np = np.stack([it["src_xyz"] for it in sampled_r], axis=0)  # (N, M, 3)
            logger.info(f"  Sampling done in {time.perf_counter()-t0:.2f}s")

            # ── Coarse-to-fine OT search (W2, L1, or L2) ─────────────────────
            logger.info(f"  Coarse-to-fine OT search (cost: {cost_metric}, cascade: {cascade_str})...")
            t0 = time.perf_counter()
            q_best, R_best, scores = coarse_to_fine_w2_search(
                src_b_np, ref_xyz,
                cascade_steps       = cascade_steps,
                cascade_sinkhorn    = cascade_sinkhorn,
                top_k               = args.top_k,
                reg                 = args.reg,
                cost_metric         = cost_metric,
                fine_rot_batch_size = args.fine_rot_batch,
                device              = device,
                logger              = logger,
            )
            rotate_Bbars = [it["Bbar"] for it in sampled_r]
            rotate_Abar  = Abar_ref
            del sampled_r, src_b_np
        else:
            # ── Coarse-to-fine CC search on raw 3D volumes ────────────────────
            logger.info(f"  Coarse-to-fine Cross-Correlation search (cascade: {cascade_str})...")
            t0 = time.perf_counter()
            q_best, R_best, scores = coarse_to_fine_cc_search(
                raw_vols, current_reference,
                cascade_steps       = cascade_steps,
                top_k               = args.top_k,
                rot_batch_size      = 20,
                device              = device,
                logger              = logger,
            )
            rotate_Bbars = [np.zeros(3, dtype=np.float64) for _ in range(N)]
            rotate_Abar  = np.zeros(3, dtype=np.float64)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_search = time.perf_counter() - t0
        logger.info(f"  Search done in {t_search:.2f}s  "
                    f"mean {score_name}={scores.mean():.4f}  "
                    f"median {score_name}={np.median(scores):.4f}")
        current_quats = q_best

        # ── Rotate volumes and accumulate new reference ───────────────────────
        logger.info(f"  Rotating and averaging {N} volumes...")
        t0 = time.perf_counter()
        sum_vol = torch.zeros((box_size, box_size, box_size),
                              dtype=torch.float64, device=device)
        for i in range(N):
            rv = gpu_rotator.rotate_volume(raw_vols[i], q_best[i],
                                           rotate_Bbars[i], rotate_Abar)
            sum_vol += rv.double()
            del rv
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_avg = time.perf_counter() - t0

        next_ref = (sum_vol / float(N)).float().cpu().numpy()
        del sum_vol
        torch.cuda.empty_cache() if device.type == "cuda" else None

        # ── Save round outputs ─────────────────────────────────────────────────
        mrc_path = output_dir / f"reference_round_{round_idx:02d}.mrc"
        alignot_utils.save_mrc(mrc_path, next_ref)
        gpu_utils.show_volume_slices(
            next_ref, f"Round {round_idx} ({args.metric.upper()}: {cascade_str})",
            save_path=output_dir / f"slices_round_{round_idx:02d}.png"
        )

        # Per-particle CSV
        rows = [
            {
                "round":          round_idx,
                "subtomogram_id": sampled_items[i]["subtomogram_id"],
                "file_name":      sampled_items[i]["file_name"],
                "metric":         metric_tag,
                "score":          float(scores[i]),
                "wass_dist":      float(scores[i]) if is_ot else np.nan,
                "ncc_score":      float(scores[i]) if not is_ot else np.nan,
                "q_w": q_best[i][0], "q_x": q_best[i][1],
                "q_y": q_best[i][2], "q_z": q_best[i][3],
            }
            for i in range(N)
        ]
        pd.DataFrame(rows).to_csv(
            output_dir / f"metrics_round_{round_idx:02d}.csv", index=False
        )

        change = alignot_utils.reference_change_stats(current_reference, next_ref, round_idx)
        change.update({
            "n_subtomograms":  N,
            "metric":          metric_tag,
            "mean_score":      float(scores.mean()),
            "mean_wass_dist":  float(scores.mean()) if is_ot else np.nan,
            "mean_ncc":        float(scores.mean()) if not is_ot else np.nan,
            "search_time_s":   float(t_search),
            "avg_time_s":      float(t_avg),
            "round_runtime_s": float(time.perf_counter() - round_start),
        })
        round_metrics.append(change)
        logger.info(f"  Round {round_idx}: L2 change={change['reference_l2_change']:.4f}  "
                    f"mean {score_name}={scores.mean():.4f}  "
                    f"search={t_search:.1f}s  avg={t_avg:.2f}s  "
                    f"total={change['round_runtime_s']:.1f}s")

        current_reference = next_ref
        all_references.append(current_reference.copy())
        gc.collect()

    # ── Save final quaternions CSV ────────────────────────────────────────────
    if current_quats is not None:
        final_rows = [
            {
                "subtomogram_id": sampled_items[i]["subtomogram_id"],
                "file_name":      sampled_items[i]["file_name"],
                "q_w": current_quats[i][0], "q_x": current_quats[i][1],
                "q_y": current_quats[i][2], "q_z": current_quats[i][3],
            }
            for i in range(N)
        ]
        df_final = pd.DataFrame(final_rows)
        df_final.to_csv(output_dir / "quaternions.csv", index=False)
        logger.info("Saved final quaternions.csv (compatible with evaluate_rotation_recovery.py)")

    # ── Round metrics summary ─────────────────────────────────────────────────
    pd.DataFrame(round_metrics).to_csv(output_dir / "round_metrics.csv", index=False)

    # ── Combined slice grid ───────────────────────────────────────────────────
    params_label = (
        f"N={N}  metric={args.metric.upper()}  Cascade={cascade_str}  "
        f"top_k={args.top_k}  rounds={args.n_rounds}"
    )
    alignot_utils.make_combined_plot(all_references, params_label, output_dir)

    # ── Loss curve (Score vs round) ──────────────────────────────────────────
    if round_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        rounds = [m["round"] for m in round_metrics]
        scores_arr = [m["mean_score"] for m in round_metrics]
        l2s    = [m["reference_l2_change"] for m in round_metrics]
        axes[0].plot(rounds, scores_arr, marker='o', linewidth=2, color='steelblue')
        axes[0].set_xlabel("Round")
        if is_ot:
            axes[0].set_ylabel(f"Mean {score_name} (Lower is better)")
            axes[0].set_title(f"Mean {score_name} per Round")
        else:
            axes[0].set_ylabel("Mean NCC (Higher is better)")
            axes[0].set_title("Mean Normalized Cross-Correlation per Round")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(rounds, l2s, marker='o', linewidth=2, color='firebrick')
        axes[1].set_xlabel("Round"); axes[1].set_ylabel("L2 Reference Change")
        axes[1].set_title("Reference Volume Change per Round")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "convergence_plots.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ── Ground-truth evaluation ───────────────────────────────────────────────
    if args.gt_csv and Path(args.gt_csv).is_file():
        logger.info(f"\nEvaluating against ground truth: {args.gt_csv}")
        _evaluate_vs_gt(output_dir / "quaternions.csv", Path(args.gt_csv), logger)

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Run complete.")
    logger.info(f"  N particles      : {N}")
    logger.info(f"  Metric           : {args.metric.upper()} ({metric_desc})")
    logger.info(f"  Cascade          : {cascade_str}")
    if args.metric == "w2":
        logger.info(f"  Sinkhorn/level   : {cascade_sinkhorn}")
    logger.info(f"  Rounds           : {args.n_rounds}")
    logger.info(f"  Output directory : {output_dir.resolve()}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

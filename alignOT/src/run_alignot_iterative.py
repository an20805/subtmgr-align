#!/usr/bin/env python3
"""
run_alignot_iterative.py

Iterative subtomogram alignment using AlignOT's quaternion-SGD rotation
optimizer.  Point cloud sampling reuses the same threshold=0.7 pipeline
as the EMPOT iterative script (prepare_sampling_volume +
sample_volume_in_memory) so that both methods are compared on equal footing.

AlignOT reference: https://github.com/bio-oi/alignOT
Difference from EMPOT: rotation is optimised by SGD on SO(3) quaternion
space instead of UGW; translation is recovered from point-cloud centroids.
"""

import os
import sys
import time
import argparse
import logging
import gc
import math
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── import shared utilities ────────────────────────────────────────────────────
EMPOT_SRC = Path(__file__).parent.parent.parent / "EMPOT" / "src"
sys.path.insert(0, str(EMPOT_SRC))
from iterative_utils import (
    get_cluster_paths,
    load_mrc,
    save_mrc,
    compute_streaming_average,
    show_volume_slices,
    reference_change_stats,
    prepare_sampling_volume,
    sample_volume_in_memory,
    apply_rigid_transform_to_volume,
    gaussian_lowpass_filter,
)

# ── import AlignOT SGD components ──────────────────────────────────────────────
ALIGNOT_SRC = Path(__file__).parent.parent / "alignOT"
sys.path.insert(0, str(ALIGNOT_SRC))
# suppress the polynomial print-out that executes at import time
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from utils import SGD, perform, get_quaternion_vals  # noqa: E402
    from coords import quaternion_to_R as _quat_to_R    # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────────

def quaternion_to_rotation_matrix(q):
    """Convert AlignOT quaternion [w, x, y, z] to 3×3 rotation matrix.
    Uses coords.quaternion_to_R from AlignOT to stay consistent."""
    return _quat_to_R(q).astype(np.float64)


from scipy.spatial.transform import Rotation

def run_alignot_alignment(pts_ref, pts_tgt,
                          lr=1e-5, max_iter=200, reg=30.0,
                          random_seed=None):
    """
    Align pts_tgt to pts_ref using AlignOT quaternion-SGD.

    Parameters
    ----------
    pts_ref : (N, 3) ndarray  –  reference point cloud (already sampled)
    pts_tgt : (N, 3) ndarray  –  target point cloud to align
    lr      : float            –  SGD learning rate
    max_iter: int              –  SGD iterations
    reg     : float            –  Sinkhorn regularisation
    random_seed : int or None

    Returns
    -------
    center_ref : (3,) ndarray  – centroid of reference cloud (Abar)
    center_tgt : (3,) ndarray  – centroid of target cloud   (Bbar)
    R_rec      : scipy Rotation – rotation object
    final_cost : float         – final OT cost
    """
    # Compute centroids and mean-centre both clouds
    center_ref = pts_ref.mean(axis=0)
    center_tgt = pts_tgt.mean(axis=0)
    pts_ref_c = pts_ref - center_ref
    pts_tgt_c = pts_tgt - center_tgt

    # AlignOT SGD expects separate coordinate lists
    xr = pts_ref_c[:, 0].tolist()
    yr = pts_ref_c[:, 1].tolist()
    zr = pts_ref_c[:, 2].tolist()
    x  = pts_tgt_c[:, 0].tolist()
    y  = pts_tgt_c[:, 1].tolist()
    z  = pts_tgt_c[:, 2].tolist()

    quaternions, costs = SGD(
        x, y, z,          # target to rotate
        xr, yr, zr,       # reference (fixed)
        lr=lr,
        max_iter=max_iter,
        reg=reg,
        num_samples=1,
        random_seed=random_seed,
    )

    R = quaternion_to_rotation_matrix(quaternions[-1])
    R_rec = Rotation.from_matrix(R)
    return center_ref, center_tgt, R_rec, costs[-1]


def setup_logger(log_file):
    logger = logging.getLogger("AlignOTIterative")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt); logger.addHandler(ch)
    return logger


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Iterative subtomogram alignment with AlignOT quaternion-SGD"
    )
    parser.add_argument("--cluster_dir",      type=str, required=True)
    parser.add_argument("--output_base_dir",  type=str,
                        default="EMPOT/outputs/alignot_runs")
    parser.add_argument("--selection_mode",   type=str,
                        choices=["first", "random", "even", "odd"],
                        default="even")
    parser.add_argument("--n_sbtms",          type=int, default=None)
    parser.add_argument("--n_rounds",         type=int, default=3)
    parser.add_argument("--threshold",        type=float, default=0.7,
                        help="Density threshold for point-cloud sampling "
                             "(fraction of 1-norm; 0.7 = darkest ~1%% of voxels)")
    parser.add_argument("--num_points",       type=int, default=500)
    parser.add_argument("--lr",               type=float, default=1e-5,
                        help="SGD learning rate for quaternion optimisation "
                             "(default 1e-5 matches AlignOT notebook values)")
    parser.add_argument("--max_iter",         type=int, default=200,
                        help="SGD iterations per subtomogram per round")
    parser.add_argument("--reg",              type=float, default=30.0,
                        help="Sinkhorn regularisation in AlignOT OT step "
                             "(default 30 matches AlignOT notebook values)")
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--init_reference",   type=str, default=None,
                        help="Optional MRC to use as round-0 reference "
                             "(auto-cropped to subtomogram size if larger)")
    parser.add_argument("--lowpass_sigma",    type=float, default=None)
    args = parser.parse_args()

    # ── run folder ─────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.selection_mode in ("even", "odd"):
        run_name = f"run_{timestamp}_{args.selection_mode}"
    else:
        n_str = str(args.n_sbtms) if args.n_sbtms else "All"
        run_name = f"run_{timestamp}_N{n_str}_{args.selection_mode}"
    output_dir = Path(args.output_base_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "run.log")
    logger.info("Starting AlignOT Iterative Alignment")
    logger.info(f"Run directory: {output_dir.resolve()}")
    logger.info(f"Arguments: {args}")

    np.random.seed(args.seed)
    random.seed(args.seed)

    # ── discover subtomograms ──────────────────────────────────────────────────
    all_paths = get_cluster_paths(args.cluster_dir)
    logger.info(f"Discovered {len(all_paths)} subtomograms in {args.cluster_dir}")

    if args.selection_mode == "even":
        cluster_paths = [p for p in all_paths if int(p.stem) % 2 == 0]
        logger.info(f"Selected {len(cluster_paths)} even-numbered subtomograms")
    elif args.selection_mode == "odd":
        cluster_paths = [p for p in all_paths if int(p.stem) % 2 == 1]
        logger.info(f"Selected {len(cluster_paths)} odd-numbered subtomograms")
    elif args.n_sbtms and args.n_sbtms < len(all_paths):
        if args.selection_mode == "first":
            cluster_paths = all_paths[:args.n_sbtms]
        else:
            cluster_paths = random.sample(all_paths, args.n_sbtms)
        logger.info(f"Selected {len(cluster_paths)} subtomograms ({args.selection_mode})")
    else:
        cluster_paths = all_paths
        logger.info(f"Using all {len(cluster_paths)} subtomograms")

    n_actual = len(cluster_paths)

    # ── initial reference (round 0) ────────────────────────────────────────────
    if args.init_reference is not None:
        logger.info(f"Loading external reference: {args.init_reference}")
        ext_ref = load_mrc(Path(args.init_reference), permissive=True)
        sbtm_shape = np.array(load_mrc(cluster_paths[0], permissive=True).shape)
        ref_shape  = np.array(ext_ref.shape)
        if not np.array_equal(ref_shape, sbtm_shape):
            if np.any(ref_shape < sbtm_shape):
                raise ValueError(
                    f"init_reference {tuple(ref_shape)} smaller than "
                    f"subtomogram {tuple(sbtm_shape)}"
                )
            slices = tuple(
                slice((r - s) // 2, (r - s) // 2 + s)
                for r, s in zip(ref_shape, sbtm_shape)
            )
            ext_ref = ext_ref[slices].copy()
            logger.info(f"Cropped external reference: {tuple(ref_shape)} → {ext_ref.shape}")
        reference_round_00 = ext_ref.astype(np.float32)
    else:
        logger.info("Computing round-0 reference from raw average...")
        reference_round_00 = compute_streaming_average(
            cluster_paths, logger=logger,
            progress_every=max(1, n_actual // 10)
        )

    save_mrc(output_dir / "reference_round_00.mrc", reference_round_00)
    show_volume_slices(
        reference_round_00, "Initial average (Round 0)",
        save_path=output_dir / "slices_round_00.png"
    )
    logger.info("Saved round-0 reference")

    # ── iterative alignment ────────────────────────────────────────────────────
    current_reference = reference_round_00.copy()
    round_metrics = []

    for round_idx in range(1, args.n_rounds + 1):
        logger.info(f"--- Round {round_idx}/{args.n_rounds} ---")
        logger.info(f"  SGD: lr={args.lr}  max_iter={args.max_iter}  reg={args.reg}")
        round_start = time.perf_counter()

        # Sample reference point cloud
        ref_for_sampling = gaussian_lowpass_filter(current_reference, args.lowpass_sigma)
        ref_sampling_vol = prepare_sampling_volume(ref_for_sampling)
        pts_ref = sample_volume_in_memory(
            ref_sampling_vol,
            threshold=args.threshold,
            num_points=args.num_points,
            random_seed=args.seed + 1000 * round_idx,
        )

        sum_transformed = np.zeros_like(current_reference, dtype=np.float64)
        count = 0
        subtomogram_rows = []
        progress_every = max(1, n_actual // 10)

        for item_idx, path in enumerate(cluster_paths, start=1):
            sbtm_start = time.perf_counter()

            # Load + sample subtomogram
            volume = load_mrc(path, permissive=True)
            vol_for_sampling = gaussian_lowpass_filter(volume, args.lowpass_sigma)
            samp_vol = prepare_sampling_volume(vol_for_sampling)
            pts_tgt = sample_volume_in_memory(
                samp_vol,
                threshold=args.threshold,
                num_points=args.num_points,
                random_seed=args.seed + round_idx + count,
            )

            # AlignOT alignment
            try:
                center_ref, center_tgt, R, ot_cost = run_alignot_alignment(
                    pts_ref, pts_tgt,
                    lr=args.lr,
                    max_iter=args.max_iter,
                    reg=args.reg,
                    random_seed=args.seed + round_idx * 10000 + item_idx,
                )
            except Exception as e:
                logger.error(f"Alignment failed for {path.name}: {e}")
                del volume, samp_vol, pts_tgt
                continue

            # Apply rotation + translation to volume
            transformed = apply_rigid_transform_to_volume(
                volume, R, center_ref, center_tgt
            )

            sum_transformed += transformed
            count += 1
            runtime_s = time.perf_counter() - sbtm_start

            subtomogram_rows.append({
                "round":          round_idx,
                "subtomogram_id": int(path.stem),
                "file_name":      path.name,
                "runtime_s":      float(runtime_s),
                "ot_cost":        float(ot_cost),
            })

            if item_idx % progress_every == 0 or item_idx == n_actual:
                avg_cost = np.mean([r["ot_cost"] for r in subtomogram_rows])
                logger.info(
                    f"  Round {round_idx}: aligned {item_idx}/{n_actual}  "
                    f"mean_OT={avg_cost:.4e}"
                )

            del volume, samp_vol, pts_tgt, transformed, R
            if item_idx % 10 == 0:
                gc.collect()

        if count == 0:
            logger.error("No subtomograms aligned this round. Aborting.")
            break

        # New reference
        next_reference = (sum_transformed / count).astype(np.float32)
        ref_path = save_mrc(
            output_dir / f"reference_round_{round_idx:02d}.mrc", next_reference
        )
        show_volume_slices(
            next_reference, f"Average (Round {round_idx})",
            save_path=output_dir / f"slices_round_{round_idx:02d}.png"
        )

        round_df = pd.DataFrame(subtomogram_rows)
        round_df.to_csv(
            output_dir / f"metrics_details_round_{round_idx:02d}.csv", index=False
        )

        change_row = reference_change_stats(current_reference, next_reference, round_idx)
        change_row.update({
            "n_subtomograms":  int(count),
            "mean_runtime_s":  float(round_df["runtime_s"].mean()),
            "mean_ot_cost":    float(round_df["ot_cost"].mean()),
            "reference_path":  str(ref_path),
            "round_runtime_s": float(time.perf_counter() - round_start),
        })
        round_metrics.append(change_row)

        current_reference = next_reference
        del ref_sampling_vol, pts_ref, sum_transformed, round_df, subtomogram_rows
        gc.collect()

        logger.info(
            f"Round {round_idx} done in {change_row['round_runtime_s']:.1f}s  |  "
            f"mean_OT={change_row['mean_ot_cost']:.4e}  |  "
            f"L2_change={change_row['reference_l2_change']:.4f}"
        )

    # ── summaries ──────────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(round_metrics)
    metrics_df.to_csv(output_dir / "round_metrics_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(metrics_df["round"], metrics_df["reference_l2_change"], marker="o")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Reference L2 change")
    axes[0].set_title("Reference change across rounds"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(metrics_df["round"], metrics_df["mean_ot_cost"], marker="o", color="orange")
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Mean OT cost")
    axes[1].set_title("Mean AlignOT cost across rounds"); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "summary_metrics_plot.png")
    plt.close(fig)

    logger.info(f"AlignOT iterative alignment complete. Output: {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import sys
import time
import argparse
import logging
import gc
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the utility functions
from iterative_utils import (
    get_cluster_paths,
    load_mrc,
    save_mrc,
    compute_streaming_average,
    show_volume_slices,
    reference_change_stats,
    prepare_sampling_volume,
    sample_volume_in_memory,
    run_empot_alignment,
    apply_rigid_transform_to_volume,
    gaussian_lowpass_filter,
)


def rotate_pts_around_z(pts: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a (N,3) point cloud by angle_deg degrees around the Z axis.

    Used for multi-hypothesis symmetry alignment: we rotate the reference
    point cloud by each symmetry angle (0°, 60°, 120°, …) and pick the
    rotation that gives the lowest UGW cost against the target.
    """
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s, 0.0],
                   [s,  c, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    return (pts.astype(np.float64) @ Rz.T).astype(pts.dtype)


def setup_logger(log_file):
    logger = logging.getLogger("IterativeAlignment")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def main():
    parser = argparse.ArgumentParser(description="Run EMPOT Iterative Subtomogram Alignment")
    parser.add_argument("--cluster_dir", type=str, default="/home/anshu/subtmgr-align/Data/0.1/Cluster_0", help="Directory containing subtomograms")
    parser.add_argument("--output_base_dir", type=str, default="/home/anshu/subtmgr-align/EMPOT/outputs/iterative_runs", help="Base directory for output runs")
    parser.add_argument("--n_sbtms", type=int, default=5, help="Number of subtomograms to process (default: all)")
    parser.add_argument("--selection_mode", type=str,
                        choices=["first", "random", "even", "odd", "all"],
                        default="first",
                        help="How to select subtomograms: "
                             "'all'    = every particle in cluster_dir (ignores --n_sbtms), "
                             "'first'  = first N by filename order, "
                             "'random' = random N, "
                             "'even'   = all even-numbered files (0,2,4,...), "
                             "'odd'    = all odd-numbered files (1,3,5,...). "
                             "For even/odd/all, --n_sbtms is ignored.")
    parser.add_argument("--n_rounds", type=int, default=1, help="Number of alignment rounds")
    parser.add_argument("--threshold", type=float, default=0.1, help="Density threshold for sampling")
    parser.add_argument("--num_points", type=int, default=2000, help="Number of points to sample per volume")
    parser.add_argument("--eps", type=float, default=10000, help="Entropy regularization parameter for EMPOT UGW")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--init_reference", type=str, default=None,
        help="Optional path to an external MRC file to use as the round-0 reference "
             "instead of computing a streaming average from the input particles. "
             "The volume is center-cropped to match the subtomogram shape if sizes differ."
    )
    parser.add_argument(
        "--lowpass_sigma", type=float, default=None,
        help="Sigma (in voxels) of the Gaussian low-pass filter applied to volumes "
             "BEFORE point-cloud sampling. Does NOT affect the averaged output. "
             "Cutoff resolution ≈ 2.35 * sigma * apix Å  "
             "(e.g. sigma=2 at apix=4 Å ≈ 19 Å cutoff). "
             "Default: None (no filtering)."
    )
    parser.add_argument(
        "--rejection_fraction", type=float, default=0.0,
        help="Fraction of worst-scoring particles (highest UGW distance) to exclude "
             "from averaging each round. Applied per-round independently. "
             "0.0 = keep all (default, backward-compatible). "
             "0.2 = drop worst 20%%, keep best 80%% (EMAN2 light). "
             "0.33 = drop worst 33%% (EMAN2 aggressive). "
             "Rejected particles are still aligned and logged, just not averaged."
    )
    parser.add_argument(
        "--symmetry", type=str, default="C1",
        help="Symmetry of the target molecule for multi-hypothesis alignment. "
             "Accepted values: C1 (no symmetry, default), C2, C3, C4, C5, C6, C7, C8 … "
             "For Cn symmetry the reference point cloud is rotated by k×(360/n)° "
             "around Z for k=0…n-1; the rotation giving the lowest UGW distance is kept. "
             "This resolves the n-fold alignment ambiguity (symmetry trap) at the cost "
             "of n× more UGW evaluations per particle. "
             "Example: --symmetry C6 for a hexameric ring."
    )

    args = parser.parse_args()

    # Parse symmetry order from e.g. 'C6' → 6
    sym_str = args.symmetry.upper().strip()
    if not sym_str.startswith("C") or not sym_str[1:].isdigit():
        raise ValueError(f"--symmetry must be Cn (e.g. C1, C6). Got: {args.symmetry!r}")
    symmetry_order = int(sym_str[1:])
    if symmetry_order < 1:
        raise ValueError(f"Symmetry order must be >= 1. Got: {symmetry_order}")

    # 1. Setup run folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.selection_mode in ("even", "odd", "all"):
        run_name = f"run_{timestamp}_{args.selection_mode}"
    else:
        run_name = f"run_{timestamp}_N{args.n_sbtms if args.n_sbtms else 'All'}_{args.selection_mode}"
    output_dir = Path(args.output_base_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Setup logger
    logger = setup_logger(output_dir / "run.log")
    logger.info(f"Starting EMPOT Iterative Alignment")
    logger.info(f"Run directory: {output_dir.resolve()}")
    logger.info(f"Arguments: {args}")

    # Set global seed for subset selection
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 3. Discover and select subtomograms
    all_paths = get_cluster_paths(args.cluster_dir)
    logger.info(f"Discovered {len(all_paths)} total subtomograms in {args.cluster_dir}")

    if args.selection_mode == "all":
        cluster_paths = all_paths
        logger.info(f"Using ALL {len(cluster_paths)} available subtomograms.")
    elif args.selection_mode == "even":
        # Select files whose numeric stem is even: 0.mrc, 2.mrc, 4.mrc, ...
        cluster_paths = [p for p in all_paths if int(p.stem) % 2 == 0]
        logger.info(f"Selected {len(cluster_paths)} even-numbered subtomograms (half-set FSC split).")
    elif args.selection_mode == "odd":
        # Select files whose numeric stem is odd: 1.mrc, 3.mrc, 5.mrc, ...
        cluster_paths = [p for p in all_paths if int(p.stem) % 2 == 1]
        logger.info(f"Selected {len(cluster_paths)} odd-numbered subtomograms (half-set FSC split).")
    elif args.n_sbtms is not None and args.n_sbtms < len(all_paths):
        if args.selection_mode == "first":
            cluster_paths = all_paths[:args.n_sbtms]
            logger.info(f"Selected first {args.n_sbtms} subtomograms.")
        else:  # random
            cluster_paths = random.sample(all_paths, args.n_sbtms)
            logger.info(f"Randomly selected {args.n_sbtms} subtomograms.")
    else:
        cluster_paths = all_paths
        logger.info(f"Using all {len(cluster_paths)} available subtomograms.")

    n_actual = len(cluster_paths)
    
    # 4. Compute or load the initial reference (Round 0)
    if args.init_reference is not None:
        # ── External seed reference (e.g., EMAN2 final average) ──────────────
        logger.info(f"Loading external initial reference from: {args.init_reference}")
        external_ref = load_mrc(Path(args.init_reference), permissive=True)

        # Auto center-crop to match the subtomogram shape if needed
        sbtm_shape = np.array(load_mrc(cluster_paths[0], permissive=True).shape)
        ref_shape   = np.array(external_ref.shape)
        if not np.array_equal(ref_shape, sbtm_shape):
            if np.any(ref_shape < sbtm_shape):
                raise ValueError(
                    f"init_reference shape {tuple(ref_shape)} is smaller than "
                    f"subtomogram shape {tuple(sbtm_shape)}. Cannot crop."
                )
            slices = tuple(
                slice((r - s) // 2, (r - s) // 2 + s)
                for r, s in zip(ref_shape, sbtm_shape)
            )
            external_ref = external_ref[slices].copy()
            logger.info(
                f"Center-cropped external reference: {tuple(ref_shape)} → {external_ref.shape}"
            )
        else:
            logger.info(f"External reference shape matches subtomograms: {tuple(ref_shape)}")

        reference_round_00 = external_ref.astype(np.float32)
        reference_round_00_path = save_mrc(
            output_dir / "reference_round_00.mrc", reference_round_00
        )
        logger.info(f"Saved (external) initial reference to {reference_round_00_path}")
    else:
        # ── Default: compute streaming average of the input particles ─────────
        logger.info("Computing initial reference (Round 0) from input particles...")
        reference_round_00 = compute_streaming_average(
            cluster_paths, logger=logger, progress_every=max(1, n_actual // 10)
        )
        reference_round_00_path = save_mrc(
            output_dir / "reference_round_00.mrc", reference_round_00
        )
        logger.info(f"Saved initial reference to {reference_round_00_path}")

    show_volume_slices(reference_round_00, "Initial average (Round 0)", save_path=output_dir / "slices_round_00.png")

    # 5. Iterative loop
    current_reference = reference_round_00.copy()
    round_metrics = []

    for round_idx in range(1, args.n_rounds + 1):
        logger.info(f"--- Starting Round {round_idx}/{args.n_rounds} ---")
        round_start = time.perf_counter()
        


        # Prepare and sample the reference
        # Apply low-pass filter before sampling (not before averaging)
        ref_for_sampling = gaussian_lowpass_filter(current_reference, args.lowpass_sigma)
        reference_sampling_volume = prepare_sampling_volume(ref_for_sampling)
        pts_ref = sample_volume_in_memory(
            reference_sampling_volume,
            threshold=args.threshold,
            num_points=args.num_points,
            random_seed=args.seed + 1000 * round_idx,
        )
        

        # ── Single-pass: align, transform, and store all volumes in memory ───────
        # Each particle is loaded once: aligned, transformed, kept in memory.
        # Transformed volumes are ~575 MB total (200 × 91³ × 4 bytes) — acceptable.
        # Rejection decision and accumulation happen after the loop.
        alignment_results = []   # list of dicts: subtomogram_id, ugw_dist, transformed_vol, runtime_s
        n_aligned = 0

        progress_every = max(1, n_actual // 10)
        cleanup_every  = 10

        for item_idx, path in enumerate(cluster_paths, start=1):
            subtomogram_start = time.perf_counter()

            # Load raw volume (kept for transformation below)
            volume           = load_mrc(path, permissive=True)
            vol_for_sampling = gaussian_lowpass_filter(volume, args.lowpass_sigma)
            sampling_volume  = prepare_sampling_volume(vol_for_sampling)
            pts_tgt = sample_volume_in_memory(
                sampling_volume,
                threshold=args.threshold,
                num_points=args.num_points,
                random_seed=args.seed + round_idx + n_aligned,
            )
            del vol_for_sampling, sampling_volume

            # ── Multi-hypothesis alignment ────────────────────────────────
            # For Cn symmetry: try all n rotations of pts_ref around Z
            # (0°, 360/n°, 2×360/n°, …) and keep the best (lowest UGW dist).
            # For C1 (no symmetry): a single alignment is run as before.
            best_dist  = np.inf
            best_Abar  = None
            best_Bbar  = None
            best_R     = None
            best_hyp   = 0   # which hypothesis index won
            any_succeeded = False

            for hyp_k in range(symmetry_order):
                angle_deg = hyp_k * (360.0 / symmetry_order)
                pts_ref_hyp = rotate_pts_around_z(pts_ref, angle_deg) if hyp_k > 0 else pts_ref
                try:
                    Abar_k, Bbar_k, R_k, dist_k = run_empot_alignment(
                        pts_ref_hyp, pts_tgt,
                        eps=args.eps,
                        nits_plan=100, nits_sinkhorn=100
                    )
                except Exception as e:
                    logger.warning(f"  Hypothesis {hyp_k} failed for {path.name}: {e}")
                    continue

                any_succeeded = True
                if dist_k < best_dist:
                    best_dist = dist_k
                    best_Abar = Abar_k
                    best_Bbar = Bbar_k
                    best_R    = R_k
                    best_hyp  = hyp_k

            if not any_succeeded:
                logger.error(f"All {symmetry_order} hypotheses failed for {path.name}. Skipping.")
                del volume, pts_tgt
                continue

            del pts_tgt

            # Transform using the best hypothesis and store in memory
            transformed_volume = apply_rigid_transform_to_volume(volume, best_R, best_Abar, best_Bbar)
            del volume

            n_aligned += 1
            runtime_s = time.perf_counter() - subtomogram_start

            alignment_results.append({
                "subtomogram_id":   int(path.stem),
                "file_name":        path.name,
                "ugw_dist":         float(best_dist),
                "best_hypothesis":  int(best_hyp),   # which C6 rotation won (0-based)
                "transformed_vol":  transformed_volume,
                "runtime_s":        float(runtime_s),
            })

            if item_idx % progress_every == 0 or item_idx == n_actual:
                logger.info(f"  Round {round_idx}: aligned {item_idx}/{n_actual} subtomograms")

            if item_idx % cleanup_every == 0:
                gc.collect()

        if n_aligned == 0:
            logger.error("No subtomograms successfully aligned in this round. Aborting.")
            break

        # ── Rejection decision ────────────────────────────────────────────────
        if args.rejection_fraction > 0.0:
            alignment_results.sort(key=lambda r: r["ugw_dist"])
            n_keep        = max(1, int(round(n_aligned * (1.0 - args.rejection_fraction))))
            accepted      = alignment_results[:n_keep]
            rejected      = alignment_results[n_keep:]
            ugw_threshold = float(accepted[-1]["ugw_dist"])
            logger.info(
                f"  Particle rejection (fraction={args.rejection_fraction}): "
                f"keeping {n_keep}/{n_aligned}, "
                f"dropped {len(rejected)}, "
                f"UGW acceptance threshold = {ugw_threshold:.2f}"
            )
        else:
            accepted      = alignment_results
            rejected      = []
            ugw_threshold = float("nan")

        accepted_ids = {id(r) for r in accepted}

        # ── Accumulate accepted transformed volumes (already in memory) ───────
        sum_transformed = np.zeros_like(current_reference, dtype=np.float64)
        for res in accepted:
            sum_transformed += res["transformed_vol"]
        count = len(accepted)

        # Free all stored volumes now that accumulation is done
        for res in alignment_results:
            del res["transformed_vol"]
        gc.collect()

        # ── Build per-round per-particle CSV rows ─────────────────────────────
        subtomogram_rows = [
            {
                "round":           round_idx,
                "subtomogram_id":  res["subtomogram_id"],
                "file_name":       res["file_name"],
                "runtime_s":       res["runtime_s"],
                "ugw_dist":        res["ugw_dist"],
                "best_hypothesis": res["best_hypothesis"],
                "accepted":        id(res) in accepted_ids,
            }
            for res in alignment_results
        ]

        # Form next reference
        next_reference = (sum_transformed / count).astype(np.float32)
        reference_path = save_mrc(output_dir / f"reference_round_{round_idx:02d}.mrc", next_reference)
        show_volume_slices(next_reference, f"Average (Round {round_idx})", save_path=output_dir / f"slices_round_{round_idx:02d}.png")

        # Metrics
        round_df = pd.DataFrame(subtomogram_rows)
        round_df.to_csv(output_dir / f"metrics_details_round_{round_idx:02d}.csv", index=False)

        change_row = reference_change_stats(current_reference, next_reference, round_idx)
        change_row.update({
            "n_subtomograms":  int(n_aligned),
            "n_accepted":      int(count),
            "n_rejected":      int(len(rejected)),
            "ugw_threshold":   ugw_threshold,
            "mean_runtime_s":  float(round_df["runtime_s"].mean()),
            "mean_ugw_dist":   float(round_df["ugw_dist"].mean()),
            "mean_ugw_dist_accepted": float(
                round_df.loc[round_df["accepted"], "ugw_dist"].mean()
            ) if count > 0 else float("nan"),
            "reference_path":  str(reference_path),
            "round_runtime_s": float(time.perf_counter() - round_start),
        })
        round_metrics.append(change_row)

        current_reference = next_reference
        del reference_sampling_volume, pts_ref, sum_transformed, round_df, subtomogram_rows
        del alignment_results, accepted, rejected, next_reference
        gc.collect()
        
        logger.info(f"Completed Round {round_idx} in {change_row['round_runtime_s']:.1f}s")
        logger.info(f"  Mean UGW dist: {change_row['mean_ugw_dist']:.4e}")
        logger.info(f"  Ref L2 change: {change_row['reference_l2_change']:.4f}")

    # 6. Save final summaries
    round_metrics_df = pd.DataFrame(round_metrics)
    round_metrics_path = output_dir / "round_metrics_summary.csv"
    round_metrics_df.to_csv(round_metrics_path, index=False)
    logger.info(f"Saved summary metrics to {round_metrics_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(round_metrics_df["round"], round_metrics_df["reference_l2_change"], marker="o")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Reference L2 change")
    axes[0].set_title("Reference change across rounds")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(round_metrics_df["round"], round_metrics_df["mean_ugw_dist"], marker="o", color="orange")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Mean OT Cost (UGW)")
    axes[1].set_title("Mean Alignment Cost across rounds")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "summary_metrics_plot.png"
    plt.savefig(plot_path)
    plt.close(fig)
    logger.info(f"Saved summary plot to {plot_path}")

    logger.info("Iterative alignment complete!")

if __name__ == "__main__":
    main()

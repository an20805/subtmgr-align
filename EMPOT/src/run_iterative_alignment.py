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
                        choices=["first", "random", "even", "odd"],
                        default="first",
                        help="How to select subtomograms: "
                             "'first' = first N by filename order, "
                             "'random' = random N, "
                             "'even' = all even-numbered files (0,2,4,...), "
                             "'odd'  = all odd-numbered files (1,3,5,...). "
                             "For even/odd, --n_sbtms is ignored.")
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
    
    args = parser.parse_args()

    # 1. Setup run folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.selection_mode in ("even", "odd"):
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

    if args.selection_mode == "even":
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
        

        sum_transformed = np.zeros_like(current_reference, dtype=np.float64)
        count = 0
        subtomogram_rows = []

        progress_every = max(1, n_actual // 10)
        cleanup_every = 10

        for item_idx, path in enumerate(cluster_paths, start=1):
            subtomogram_start = time.perf_counter()
            
            # Load and sample (apply low-pass only for sampling, not for accumulation)
            volume = load_mrc(path, permissive=True)
            vol_for_sampling = gaussian_lowpass_filter(volume, args.lowpass_sigma)
            sampling_volume = prepare_sampling_volume(vol_for_sampling)
            pts_tgt = sample_volume_in_memory(
                sampling_volume,
                threshold=args.threshold,
                num_points=args.num_points,
                random_seed=args.seed + round_idx + count,
            )

            # Align using EMPOT
            try:
                # start = time.perf_counter()
                Abar, Bbar, R_recovered, dist = run_empot_alignment(
                    pts_ref, pts_tgt,
                    eps=args.eps,
                    nits_plan=100, nits_sinkhorn=100
                )
                # end = time.perf_counter()
                # print(f"Time taken: {end - start:.6f} seconds")
                # exit()
            except Exception as e:
                logger.error(f"Alignment failed for {path.name}: {e}")
                del volume, sampling_volume, pts_tgt
                continue
            
            # Transform volume
            transformed_volume = apply_rigid_transform_to_volume(volume, R_recovered, Abar, Bbar)

            # Accumulate
            sum_transformed += transformed_volume
            count += 1
            runtime_s = time.perf_counter() - subtomogram_start
            
            subtomogram_rows.append({
                "round": round_idx,
                "subtomogram_id": int(path.stem),
                "file_name": path.name,
                "runtime_s": float(runtime_s),
                "ugw_dist": float(dist),
            })

            if item_idx % progress_every == 0 or item_idx == n_actual:
                logger.info(f"  Round {round_idx}: aligned {item_idx}/{n_actual} subtomograms")

            # Cleanup
            del volume, sampling_volume, pts_tgt, transformed_volume, R_recovered
            if item_idx % cleanup_every == 0:
                gc.collect()

        if count == 0:
            logger.error("No subtomograms successfully aligned in this round. Aborting.")
            break

        # Form next reference
        next_reference = (sum_transformed / count).astype(np.float32)
        reference_path = save_mrc(output_dir / f"reference_round_{round_idx:02d}.mrc", next_reference)
        show_volume_slices(next_reference, f"Average (Round {round_idx})", save_path=output_dir / f"slices_round_{round_idx:02d}.png")

        # Metrics
        round_df = pd.DataFrame(subtomogram_rows)
        # Save per-round detailed metrics
        round_df.to_csv(output_dir / f"metrics_details_round_{round_idx:02d}.csv", index=False)
        
        change_row = reference_change_stats(current_reference, next_reference, round_idx)
        change_row.update({
            "n_subtomograms": int(count),
            "mean_runtime_s": float(round_df["runtime_s"].mean()),
            "mean_ugw_dist": float(round_df["ugw_dist"].mean()),
            "reference_path": str(reference_path),
            "round_runtime_s": float(time.perf_counter() - round_start),
        })
        round_metrics.append(change_row)

        current_reference = next_reference
        del reference_sampling_volume, pts_ref, sum_transformed, round_df, subtomogram_rows, next_reference
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

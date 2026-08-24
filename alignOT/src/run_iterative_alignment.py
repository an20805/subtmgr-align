#!/usr/bin/env python3

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
from datetime import datetime
from pathlib import Path

# Set local timezone
os.environ["TZ"] = "Asia/Kolkata"
if hasattr(time, "tzset"):
    time.tzset()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import alignot_utils


# ── Logger setup ───────────────────────────────────────────────────────────────

def setup_logger(log_file):
    logger = logging.getLogger("alignot_iterative")
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


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Iterative Subtomogram Alignment using AlignOT"
    )

    # Data
    p.add_argument("--cluster_dir",      required=True,
                   help="Directory containing subtomogram .mrc files")
    p.add_argument("--output_base_dir",  required=True,
                   help="Base directory for output runs")
    p.add_argument("--init_reference",   default=None,
                   help="Path to initial reference .mrc. If not provided, computes streaming average.")

    # Particle selection
    p.add_argument("--selection_mode", default="all",
                   choices=["all", "first", "random", "even", "odd"],
                   help="Which particles to use: all / first N / random N / even / odd")
    p.add_argument("--n_sbtms", type=int, default=None,
                   help="Number of particles (for first / random mode)")

    # Rounds
    p.add_argument("--n_rounds", type=int, default=3,
                   help="Number of alignment rounds (default: 3)")

    # Sampling
    p.add_argument("--num_points",    type=int,   default=500,
                   help="TRN points per volume (default: 500)")
    p.add_argument("--threshold",     type=float, default=0.7,
                   help="Density threshold after inversion (default: 0.7)")
    p.add_argument("--lowpass_sigma", type=float, default=None,
                   help="Gaussian LP sigma (voxels) applied BEFORE sampling only. "
                        "None = no filtering. Example: 2.0")

    # AlignOT SGD
    p.add_argument("--reg",       type=float, default=30.0,
                   help="Sinkhorn regularisation (default: 30.0)")
    p.add_argument("--sgd_iters", type=int,   default=500,
                   help="SGD iterations per particle (default: 500)")
    p.add_argument("--sgd_lr",    type=float, default=1e-5,
                   help="SGD learning rate (default: 1e-5)")

    # Misc
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ── Output directory & logger ──────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"run_{timestamp}_{args.selection_mode}"
    output_dir = Path(args.output_base_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "run.log")
    logger.info("Starting AlignOT Iterative Subtomogram Alignment")
    logger.info(f"Run directory: {output_dir.resolve()}")
    logger.info(f"Arguments: {args}")

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

    # ── Round 0: initial reference ─────────────────────────────────────────────
    if args.init_reference is not None:
        logger.info(f"Using external initial reference: {args.init_reference}")
        current_reference = alignot_utils.load_mrc(args.init_reference, permissive=True)
        title_r0 = "External Initial Ref (Round 0)"
    else:
        logger.info("Computing initial reference (Round 0) — streaming average of raw particles...")
        current_reference = alignot_utils.compute_streaming_average(
            cluster_paths,
            logger=logger,
            progress_every=max(1, n_actual // 10),
        )
        title_r0 = "Initial average (Round 0)"

    alignot_utils.save_mrc(output_dir / "reference_round_00.mrc", current_reference)
    alignot_utils.show_volume_slices(
        current_reference, title_r0,
        save_path=output_dir / "slices_round_00.png",
    )
    logger.info("Saved initial reference and slice plot.")

    all_references = [current_reference.copy()]
    round_metrics  = []

    # ── Iterative loop ─────────────────────────────────────────────────────────
    for round_idx in range(1, args.n_rounds + 1):
        logger.info(f"\n--- Starting Round {round_idx}/{args.n_rounds} ---")
        round_start = time.perf_counter()

        # Sample reference point cloud
        ref_prep = alignot_utils.prepare_sampling_volume(
            current_reference, sigma=args.lowpass_sigma
        )
        pts_ref = alignot_utils.sample_volume_in_memory(
            ref_prep,
            threshold=args.threshold,
            num_points=args.num_points,
            random_seed=args.seed + 1000 * round_idx,
        )
        logger.info(f"Sampled reference point cloud: {pts_ref.shape}")

        # Per-particle alignment
        alignment_results = []
        n_aligned = 0
        progress_every = max(1, n_actual // 10)
        cleanup_every  = 10

        for item_idx, path in enumerate(cluster_paths, start=1):
            subtomogram_start = time.perf_counter()

            # Load original raw volume
            volume = alignot_utils.load_mrc(path, permissive=True)

            # Sample source point cloud
            src_prep = alignot_utils.prepare_sampling_volume(
                volume, sigma=args.lowpass_sigma
            )
            pts_src = alignot_utils.sample_volume_in_memory(
                src_prep,
                threshold=args.threshold,
                num_points=args.num_points,
                random_seed=args.seed + round_idx + n_aligned,
            )

            # Align: returns q, costs, wass_dist, Bbar (source centroid), Abar (ref centroid)
            q, costs, wass_dist, Bbar, Abar = alignot_utils.run_alignot_alignment(
                pts_src, pts_ref,
                reg=args.reg,
                sgd_iters=args.sgd_iters,
                lr=args.sgd_lr,
            )

            # Rotate full original volume using centroid-aware EMPOT-style offset
            vol_rot = alignot_utils.apply_rotation_to_volume(volume, q, Bbar, Abar)
            del volume, src_prep, pts_src

            n_aligned += 1
            runtime_s = time.perf_counter() - subtomogram_start

            alignment_results.append({
                "subtomogram_id": int(path.stem),
                "file_name":      path.name,
                "wass_dist":      float(wass_dist),
                "q":              list(q),
                "vol_rot":        vol_rot,
                "runtime_s":      float(runtime_s),
            })

            if item_idx % progress_every == 0 or item_idx == n_actual:
                logger.info(
                    f"  Round {round_idx}: aligned {item_idx}/{n_actual} "
                    f"({runtime_s:.2f}s) — WassDist={wass_dist:.2f}"
                )

            if item_idx % cleanup_every == 0:
                gc.collect()

        # ── Accumulate rotated volumes ─────────────────────────────────────────
        sum_rotated = np.zeros_like(current_reference, dtype=np.float64)
        for r in alignment_results:
            sum_rotated += r["vol_rot"].astype(np.float64)

        # Free rotated volumes
        for r in alignment_results:
            del r["vol_rot"]
        gc.collect()

        # ── Build new reference ────────────────────────────────────────────────
        next_reference = (sum_rotated / len(alignment_results)).astype(np.float32)
        del sum_rotated

        alignot_utils.save_mrc(
            output_dir / f"reference_round_{round_idx:02d}.mrc", next_reference
        )
        alignot_utils.show_volume_slices(
            next_reference, f"Round {round_idx} average",
            save_path=output_dir / f"slices_round_{round_idx:02d}.png",
        )

        # ── Per-particle CSV ───────────────────────────────────────────────────
        particle_rows = [
            {
                "round":          round_idx,
                "subtomogram_id": r["subtomogram_id"],
                "file_name":      r["file_name"],
                "wass_dist":      r["wass_dist"],
                "runtime_s":      r["runtime_s"],
                "q_w":            r["q"][0],
                "q_x":            r["q"][1],
                "q_y":            r["q"][2],
                "q_z":            r["q"][3],
            }
            for r in alignment_results
        ]
        round_df = pd.DataFrame(particle_rows)
        round_df.to_csv(
            output_dir / f"metrics_particles_round_{round_idx:02d}.csv", index=False
        )

        # ── Round-level metrics ────────────────────────────────────────────────
        change = alignot_utils.reference_change_stats(current_reference, next_reference, round_idx)
        change.update({
            "n_subtomograms":    int(n_aligned),
            "mean_wass_dist":    float(round_df["wass_dist"].mean()),
            "mean_runtime_s":    float(round_df["runtime_s"].mean()),
            "round_runtime_s":   float(time.perf_counter() - round_start),
            "reference_path":    str(output_dir / f"reference_round_{round_idx:02d}.mrc"),
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

    # ── Save round-level summary ───────────────────────────────────────────────
    summary_df = pd.DataFrame(round_metrics)
    summary_path = output_dir / "round_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved round metrics to {summary_path}")

    # ── Summary plots ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(summary_df["round"], summary_df["reference_l2_change"], marker="o")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Reference L2 change")
    axes[0].set_title("Reference change across rounds")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(summary_df["round"], summary_df["mean_wass_dist"], marker="o", color="orange")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Mean Wasserstein distance")
    axes[1].set_title("Mean alignment cost across rounds")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    summary_plot_path = output_dir / "summary_metrics_plot.png"
    plt.savefig(summary_plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved summary metrics plot to {summary_plot_path}")

    # ── Combined slice grid ────────────────────────────────────────────────────
    params_label = (
        f"M={args.num_points}, σ_LP={args.lowpass_sigma}, threshold={args.threshold}, "
        f"reg={args.reg}, lr={args.sgd_lr}, iters={args.sgd_iters}"
    )
    combined_path = alignot_utils.make_combined_plot(all_references, params_label, output_dir)
    logger.info(f"Saved combined slice plot to {combined_path}")

    logger.info("Iterative alignment complete!")


if __name__ == "__main__":
    main()

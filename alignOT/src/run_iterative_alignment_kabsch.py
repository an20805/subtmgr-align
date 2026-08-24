#!/usr/bin/env python3
"""
run_iterative_alignment_kabsch.py
----------------------------------
Iterative subtomogram alignment using AlignOT TRN sampling + Kabsch SVD.

Replaces the SGD rotation optimiser (500 iterations, ~2.5 s/particle) with:
  1. One-shot balanced Sinkhorn transport plan  (~17 ms for 500×500)
  2. Hard correspondences  (argmax over plan rows)
  3. Kabsch SVD            (closed-form optimal R, Abar, Bbar)
  4. EMPOT-style affine transform: offset = Bbar + center - R^T @ (Abar + center)

This is ~150× faster per particle while recovering larger rotations more
reliably than SGD with a fixed learning rate.

Usage
-----
See scripts/run_alignot_kabsch_all_particles.sh for an example invocation.
"""

import argparse
import gc
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import alignot_kabsch_utils as ku


# ── Logger ────────────────────────────────────────────────────────────────────

def setup_logger(log_file):
    logger = logging.getLogger("alignot_kabsch")
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Iterative Subtomogram Alignment — AlignOT TRN + Kabsch SVD"
    )

    # Data
    p.add_argument("--cluster_dir",     required=True,
                   help="Directory containing subtomogram .mrc files")
    p.add_argument("--output_base_dir", required=True,
                   help="Base directory for output runs")
    p.add_argument("--init_reference", default=None,
                   help="Path to initial reference .mrc. If not provided, computes streaming average.")

    # Particle selection
    p.add_argument("--selection_mode", default="all",
                   choices=["all", "first", "random", "even", "odd"])
    p.add_argument("--n_sbtms", type=int, default=None,
                   help="Number of particles (for first / random mode)")

    # Rounds
    p.add_argument("--n_rounds", type=int, default=3,
                   help="Number of alignment rounds (default: 3)")

    # TRN sampling
    p.add_argument("--num_points",    type=int,   default=500,
                   help="TRN points per volume (default: 500)")
    p.add_argument("--threshold",     type=float, default=0.7,
                   help="Density threshold after inversion (default: 0.7)")
    p.add_argument("--lowpass_sigma", type=float, default=2.0,
                   help="Gaussian LP sigma in voxels applied before TRN sampling only "
                        "(default: 2.0; set to 0 to disable)")

    # Sinkhorn + Kabsch
    p.add_argument("--reg",            type=float, default=5.0,
                   help="Sinkhorn entropy regularisation (default: 5.0; "
                        "lower = sharper correspondences)")
    p.add_argument("--sinkhorn_iters", type=int,   default=5000,
                   help="Max Sinkhorn iterations (default: 5000)")

    # Misc
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # ── Output directory & logger ──────────────────────────────────────────────
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name   = f"run_{timestamp}_{args.selection_mode}_kabsch"
    output_dir = Path(args.output_base_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "run.log")
    logger.info("AlignOT Kabsch Iterative Subtomogram Alignment")
    logger.info(f"Run directory : {output_dir.resolve()}")
    logger.info(f"Arguments     : {args}")

    # ── Particle discovery & selection ─────────────────────────────────────────
    all_paths = ku.get_cluster_paths(args.cluster_dir)
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
    else:   # random
        n = args.n_sbtms or len(all_paths)
        cluster_paths = random.sample(all_paths, n)

    n_actual = len(cluster_paths)
    logger.info(f"Selected {n_actual} subtomograms ({args.selection_mode} mode)")

    # ── Round 0: initial reference ────────────────────────────────────────────
    if args.init_reference is not None:
        logger.info(f"Using external initial reference: {args.init_reference}")
        current_reference = ku.load_mrc(args.init_reference)
        title_r0 = "External Initial Ref (R0)"
    else:
        logger.info("Computing initial reference (Round 0) — streaming average of raw particles…")
        current_reference = ku.compute_streaming_average(
            cluster_paths, logger=logger, progress_every=max(1, n_actual // 10)
        )
        title_r0 = "Initial average (R0)"

    ku.save_mrc(output_dir / "reference_round_00.mrc", current_reference)
    ku.show_volume_slices(
        current_reference, title_r0,
        save_path=output_dir / "slices_round_00.png",
    )
    logger.info("Saved initial reference and slice plot.")

    all_references = [current_reference.copy()]
    round_metrics  = []

    # ── Iterative loop ─────────────────────────────────────────────────────────
    for round_idx in range(1, args.n_rounds + 1):
        logger.info(f"\n--- Round {round_idx}/{args.n_rounds} ---")
        round_start = time.perf_counter()

        # Sample reference point cloud once per round
        ref_prep = ku.prepare_sampling_volume(current_reference, sigma=args.lowpass_sigma)
        pts_ref  = ku.sample_volume_in_memory(
            ref_prep,
            threshold=args.threshold,
            num_points=args.num_points,
            random_seed=args.seed + 1000 * round_idx,
        )
        logger.info(f"Sampled reference point cloud: {pts_ref.shape}")

        # ── Per-particle alignment ─────────────────────────────────────────────
        alignment_results = []
        n_failed          = 0
        progress_every    = max(1, n_actual // 10)

        for item_idx, path in enumerate(cluster_paths, start=1):
            t_start = time.perf_counter()

            # Load ORIGINAL raw volume
            volume = ku.load_mrc(path)

            # Sample source point cloud
            src_prep = ku.prepare_sampling_volume(volume, sigma=args.lowpass_sigma)
            pts_src  = ku.sample_volume_in_memory(
                src_prep,
                threshold=args.threshold,
                num_points=args.num_points,
                random_seed=args.seed + round_idx * 10000 + item_idx,
            )

            # ── Sinkhorn + Kabsch ──────────────────────────────────────────────
            try:
                R, Abar, Bbar, ot_dist = ku.run_kabsch_alignment(
                    pts_src, pts_ref,
                    reg=args.reg,
                    sinkhorn_iters=args.sinkhorn_iters,
                )
                vol_rot = ku.apply_rigid_transform_to_volume(volume, R, Abar, Bbar)
                failed  = False
            except Exception as exc:
                logger.warning(f"  Particle {path.name} alignment failed: {exc}. Using original.")
                vol_rot  = volume.copy()
                R        = np.eye(3)
                Abar     = np.zeros(3)
                Bbar     = np.zeros(3)
                ot_dist  = float("nan")
                failed   = True
                n_failed += 1

            del volume, src_prep, pts_src

            runtime_s = time.perf_counter() - t_start
            alignment_results.append({
                "subtomogram_id": int(path.stem),
                "file_name":      path.name,
                "ot_dist":        float(ot_dist),
                "vol_rot":        vol_rot,
                "R":              R.tolist(),
                "Abar":           Abar.tolist(),
                "Bbar":           Bbar.tolist(),
                "runtime_s":      float(runtime_s),
                "failed":         failed,
            })

            if item_idx % progress_every == 0 or item_idx == n_actual:
                logger.info(
                    f"  Round {round_idx}: aligned {item_idx}/{n_actual} "
                    f"({runtime_s*1000:.0f} ms) — OT_dist={ot_dist:.2f}"
                )

        if n_failed:
            logger.warning(f"  {n_failed} particles failed alignment and used identity transform.")

        # ── Build new reference: mean of all rotated volumes ──────────────
        sum_rotated = np.zeros_like(current_reference, dtype=np.float64)
        for r in alignment_results:
            sum_rotated += r["vol_rot"].astype(np.float64)

        # Free rotated volumes
        for r in alignment_results:
            del r["vol_rot"]
        gc.collect()

        next_reference = (sum_rotated / len(alignment_results)).astype(np.float32)
        del sum_rotated

        ku.save_mrc(output_dir / f"reference_round_{round_idx:02d}.mrc", next_reference)
        ku.show_volume_slices(
            next_reference, f"Round {round_idx}",
            save_path=output_dir / f"slices_round_{round_idx:02d}.png",
        )

        # ── Per-particle CSV ───────────────────────────────────────────────────
        particle_rows = []
        for r in alignment_results:
            R_flat = np.array(r["R"]).flatten()
            particle_rows.append({
                "round":          round_idx,
                "subtomogram_id": r["subtomogram_id"],
                "file_name":      r["file_name"],
                "ot_dist":        r["ot_dist"],
                "runtime_s":      r["runtime_s"],
                "failed":         r["failed"],
                "Abar_z": r["Abar"][0], "Abar_y": r["Abar"][1], "Abar_x": r["Abar"][2],
                "Bbar_z": r["Bbar"][0], "Bbar_y": r["Bbar"][1], "Bbar_x": r["Bbar"][2],
                **{f"R_{i}{j}": R_flat[3*i+j] for i in range(3) for j in range(3)},
            })
        round_df = pd.DataFrame(particle_rows)
        round_df.to_csv(
            output_dir / f"metrics_particles_round_{round_idx:02d}.csv", index=False
        )

        # ── Round-level metrics ────────────────────────────────────────────────
        change = ku.reference_change_stats(current_reference, next_reference, round_idx)
        valid_dists = round_df.loc[~round_df["failed"], "ot_dist"]
        change.update({
            "n_subtomograms":    int(n_actual),
            "n_failed":          int(n_failed),
            "mean_ot_dist":      float(valid_dists.mean()) if len(valid_dists) else float("nan"),
            "mean_runtime_ms":   float(round_df["runtime_s"].mean() * 1000),
            "round_runtime_s":   float(time.perf_counter() - round_start),
            "reference_path":    str(output_dir / f"reference_round_{round_idx:02d}.mrc"),
        })
        round_metrics.append(change)

        logger.info(
            f"Round {round_idx} done in {change['round_runtime_s']:.1f}s  "
            f"| mean OT_dist={change['mean_ot_dist']:.2f}  "
            f"| ref L2 change={change['reference_l2_change']:.4f}  "
            f"| mean time/particle={change['mean_runtime_ms']:.0f} ms"
        )

        current_reference = next_reference
        all_references.append(current_reference.copy())
        del alignment_results, next_reference, round_df
        gc.collect()

    # ── Round-level summary CSV ────────────────────────────────────────────────
    summary_df   = pd.DataFrame(round_metrics)
    summary_path = output_dir / "round_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved round metrics → {summary_path}")

    # ── Summary plots ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(summary_df["round"], summary_df["reference_l2_change"], marker="o")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Reference L2 change")
    axes[0].set_title("Reference change across rounds"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(summary_df["round"], summary_df["mean_ot_dist"], marker="o", color="orange")
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Mean OT distance")
    axes[1].set_title("Mean alignment cost across rounds"); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "summary_metrics_plot.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved summary plot → {plot_path}")

    # ── Combined slice grid ────────────────────────────────────────────────────
    params_label = (
        f"M={args.num_points}, σ_LP={args.lowpass_sigma}, thr={args.threshold}, "
        f"reg={args.reg}, iters={args.sinkhorn_iters}"
    )
    combined = ku.make_combined_plot(all_references, params_label, output_dir)
    logger.info(f"Saved combined slices → {combined}")

    logger.info("Iterative Kabsch alignment complete!")


if __name__ == "__main__":
    main()

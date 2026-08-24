#!/usr/bin/env python3
"""
run_iterative_alignment_barycentric.py
---------------------------------------
Iterative Subtomogram Alignment using AlignOT SGD with Barycentric Projection + 1-NN.

Theory:
  Instead of hard argmax on the Sinkhorn transport plan rows:
    1. Normalize row i of transport plan T so sum(weights) = 1.
    2. Compute continuous barycentric target point: p_bary = sum_j (w_j * q_j).
    3. Find the 1-nearest neighbor (1-NN) in the target point cloud:
         dest = argmin_j || p_bary - q_j ||_2
    4. Compute quaternion gradient with respect to this matched point.
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
from datetime import datetime
from pathlib import Path

# Set local timezone
os.environ["TZ"] = "Asia/Kolkata"
if hasattr(time, "tzset"):
    time.tzset()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR.parent / "alignOT"))

import alignot_utils
import utils as _utils


# ── Logger setup ───────────────────────────────────────────────────────────────

def setup_logger(log_file):
    logger = logging.getLogger("alignot_barycentric")
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


# ── Barycentric 1-NN SGD Optimizer ─────────────────────────────────────────────

def SGD_barycentric_1nn(x, y, z, xr, yr, zr, lr=0.005, max_iter=100, reg=0.1, num_samples=1, verbose=False, random_seed=None):
    """
    SGD rotation optimizer with Barycentric projection + 1-NN matching.
    """
    px = _utils.Quaternion(
        _utils.Polynomial([_utils.Term(coef=0, exps=[0, 0, 0, 0, 1, 0, 0, 0])]),
        _utils.Polynomial([_utils.Term(coef=1, exps=[0, 0, 0, 0, 0, 1, 0, 0])]),
        _utils.Polynomial([_utils.Term(coef=1, exps=[0, 0, 0, 0, 0, 0, 1, 0])]),
        _utils.Polynomial([_utils.Term(coef=1, exps=[0, 0, 0, 0, 0, 0, 0, 1])]),
    )

    bx = _utils.q * px * _utils.qs
    i_der = [bx.i_pol.derivative(i) for i in range(4)]
    j_der = [bx.j_pol.derivative(i) for i in range(4)]
    k_der = [bx.k_pol.derivative(i) for i in range(4)]

    if random_seed is not None:
        random.seed(random_seed)

    vals = _utils.get_quaternion_vals(0, 0, 0, 1)
    quaternions = []
    costs = []
    prev = None
    u, v = None, None

    ref_coords = np.column_stack((xr, yr, zr)).astype(np.float64)  # (M, 3)

    for i in range(max_iter):
        quaternions.append(list(vals))

        xx, yy, zz = _utils.perform(x, y, z, vals)

        if u is not None:
            prev = (u, v)
        T, cost, u, v = _utils.OT(xr, yr, zr, xx, yy, zz, reg=reg, prev=prev)
        costs.append(cost)

        grad = [0.0, 0.0, 0.0, 0.0]

        for s in range(num_samples):
            sample_point = int(random.random() * len(x))
            x1, y1, z1 = x[sample_point], y[sample_point], z[sample_point]

            # ── Barycentric weighted average + 1-NN ───────────────────────────
            row = np.array([T[sample_point][j] for j in range(len(xr))], dtype=np.float64)
            row_sum = float(row.sum())
            weights = row / row_sum if row_sum > 0 else np.ones(len(xr)) / len(xr)

            # Expected target coordinates (Barycentric projection)
            bary_pt = np.dot(weights, ref_coords)  # (3,)

            # 1-Nearest Neighbor in target point cloud
            sq_dists = np.sum((ref_coords - bary_pt) ** 2, axis=1)
            dest = int(np.argmin(sq_dists))

            x2, y2, z2 = xr[dest], yr[dest], zr[dest]

            # ── Gradient evaluation ───────────────────────────────────────────
            new_vals = list(vals) + [0, x1, y1, z1]
            i_vals = bx.i_pol.evaluate(new_vals)
            j_vals = bx.j_pol.evaluate(new_vals)
            k_vals = bx.k_pol.evaluate(new_vals)

            i_der_vals = [i_der[k].evaluate(new_vals) for k in range(4)]
            j_der_vals = [j_der[k].evaluate(new_vals) for k in range(4)]
            k_der_vals = [k_der[k].evaluate(new_vals) for k in range(4)]

            for k in range(4):
                temp_k = 2 * (
                    i_der_vals[k] * (i_vals - x2)
                    + j_der_vals[k] * (j_vals - y2)
                    + k_der_vals[k] * (k_vals - z2)
                )
                grad[k] += temp_k * T[sample_point][dest] * len(x) / num_samples

        d_prod = _utils.dot(grad, vals)
        for j in range(4):
            grad[j] -= d_prod * vals[j]

        for j in range(4):
            vals[j] -= lr * grad[j]

        norm = math.sqrt(sum(v ** 2 for v in vals))
        if norm > 0:
            vals = [v / norm for v in vals]

    quaternions.append(list(vals))
    return quaternions, costs


def run_alignot_barycentric_alignment(pts_src, pts_ref, reg=30, sgd_iters=500, lr=0.00001, random_seed=None):
    """
    Align a subtomogram point cloud to reference using Barycentric 1-NN SGD.
    """
    Bbar = pts_src.mean(axis=0).astype(np.float64)   # source centroid [z, y, x]
    Abar = pts_ref.mean(axis=0).astype(np.float64)   # reference centroid [z, y, x]

    pts_src_c = pts_src - Bbar
    pts_ref_c = pts_ref - Abar

    # TRN gives cols [z, y, x]; SGD expects separate (x, y, z) lists
    src_x = list(pts_src_c[:, 2].astype(float))
    src_y = list(pts_src_c[:, 1].astype(float))
    src_z = list(pts_src_c[:, 0].astype(float))
    ref_x = list(pts_ref_c[:, 2].astype(float))
    ref_y = list(pts_ref_c[:, 1].astype(float))
    ref_z = list(pts_ref_c[:, 0].astype(float))

    with contextlib.redirect_stdout(io.StringIO()):
        quaternions, costs = SGD_barycentric_1nn(
            src_x, src_y, src_z,
            ref_x, ref_y, ref_z,
            lr=lr,
            max_iter=sgd_iters,
            reg=reg,
            num_samples=1,
            verbose=False,
            random_seed=random_seed,
        )

    final_q = quaternions[-1]
    wass_dist = float(costs[-1])
    return final_q, costs, wass_dist, Bbar, Abar


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Iterative Subtomogram Alignment using AlignOT (Barycentric 1-NN SGD)"
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

    # Rounds
    p.add_argument("--n_rounds", type=int, default=3,
                   help="Number of alignment rounds (default: 3)")

    # Sampling
    p.add_argument("--num_points", type=int, default=500,
                   help="TRN points per volume (default: 500)")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="Density threshold after inversion (default: 0.7)")
    p.add_argument("--lowpass_sigma", type=float, default=2.0,
                   help="Gaussian LP sigma (voxels) applied before sampling only (default: 2.0)")

    # AlignOT SGD
    p.add_argument("--reg", type=float, default=30.0,
                   help="Sinkhorn regularisation (default: 30.0)")
    p.add_argument("--sgd_iters", type=int, default=500,
                   help="SGD iterations per particle (default: 500)")
    p.add_argument("--sgd_lr", type=float, default=1e-5,
                   help="SGD learning rate (default: 1e-5)")

    # Misc
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # ── Output directory & logger ──────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}_{args.selection_mode}_barycentric"
    output_dir = Path(args.output_base_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "run.log")
    logger.info("Starting AlignOT Barycentric 1-NN Iterative Subtomogram Alignment")
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
    round_metrics = []

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
        cleanup_every = 10

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

            # Align with Barycentric 1-NN SGD
            q, costs, wass_dist, Bbar, Abar = run_alignot_barycentric_alignment(
                pts_src, pts_ref,
                reg=args.reg,
                sgd_iters=args.sgd_iters,
                lr=args.sgd_lr,
                random_seed=args.seed + round_idx * 1000 + item_idx,
            )

            # Rotate full original volume using centroid-aware rigid transform
            vol_rot = alignot_utils.apply_rotation_to_volume(volume, q, Bbar, Abar)
            del volume, src_prep, pts_src

            n_aligned += 1
            runtime_s = time.perf_counter() - subtomogram_start

            alignment_results.append({
                "subtomogram_id": int(path.stem),
                "file_name": path.name,
                "wass_dist": float(wass_dist),
                "q": [float(val) for val in q],
                "vol_rot": vol_rot,
                "runtime_s": float(runtime_s),
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
            next_reference, f"Round {round_idx} average (Barycentric 1-NN)",
            save_path=output_dir / f"slices_round_{round_idx:02d}.png",
        )

        # ── Per-particle CSV ───────────────────────────────────────────────────
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

        # ── Round-level metrics ────────────────────────────────────────────────
        change = alignot_utils.reference_change_stats(current_reference, next_reference, round_idx)
        change.update({
            "n_subtomograms": int(n_aligned),
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
    axes[1].set_ylabel("Mean Wasserstein dist")
    axes[1].set_title("Mean alignment cost across rounds")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "summary_metrics_plot.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved summary plot to {plot_path}")

    # ── Combined slice grid ────────────────────────────────────────────────────
    params_label = (
        f"M={args.num_points}, σ_LP={args.lowpass_sigma}, thr={args.threshold}, "
        f"reg={args.reg}, iters={args.sgd_iters}, lr={args.sgd_lr} (Barycentric 1-NN)"
    )
    combined = alignot_utils.make_combined_plot(all_references, params_label, output_dir)
    logger.info(f"Saved combined slices to {combined}")
    logger.info("Iterative Barycentric 1-NN alignment complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
calculate_cost_matrix_stats.py
------------------------------
Calculate and report the statistics of the cost matrix used in AlignOT.

Distance Metric:
  Squared Euclidean Distance (L2^2):
    M_ij = || p_i - q_j ||_2^2 = (x_i - x_j')^2 + (y_i - y_j')^2 + (z_i - z_j')^2

Dimensions:
  (N, M), where N is the number of points in point cloud 1 (e.g. 500)
  and M is the number of points in point cloud 2 (e.g. 500).
  Default dimension: 500 x 500.

This script randomly selects two subtomograms from the cluster directory,
samples their TRN point clouds using standard AlignOT parameters, computes the
cost matrix, and calculates:
  - Minimum cost
  - Maximum cost
  - Mean cost (Average)
  - Row sum statistics (mean, min, max, std)
  - Column sum statistics (mean, min, max, std)
Repeated for 10 trials.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add alignOT paths
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR.parent / "alignOT"))

import alignot_kabsch_utils as ku


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute AlignOT cost matrix statistics for random subtomogram pairs"
    )
    parser.add_argument(
        "--cluster_dir",
        type=str,
        default=str(REPO_ROOT / "Data" / "0.1" / "Cluster_0"),
        help="Directory containing subtomogram .mrc files",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=500,
        help="Number of TRN points per volume (default: 500)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Density threshold for TRN sampling (default: 0.7)",
    )
    parser.add_argument(
        "--lowpass_sigma",
        type=float,
        default=2.0,
        help="Gaussian lowpass filter sigma in voxels (default: 2.0)",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=10,
        help="Number of random pairs to test (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def compute_cost_matrix(pts1, pts2):
    """
    Compute pairwise squared Euclidean distance matrix between two 3D point sets.
    pts1: (N, 3)
    pts2: (M, 3)
    Returns M: (N, M)
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    # M_ij = sum_d (pts1[i, d] - pts2[j, d])^2
    diff = pts1[:, None, :] - pts2[None, :, :]
    return np.sum(diff ** 2, axis=-1)


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    all_paths = ku.get_cluster_paths(args.cluster_dir)
    print(f"=" * 80)
    print(f"ALIGNOT COST MATRIX STATISTICS EVALUATION")
    print(f"=" * 80)
    print(f"Cluster directory : {args.cluster_dir}")
    print(f"Total subtomograms: {len(all_paths)}")
    print(f"TRN points (N, M) : ({args.num_points}, {args.num_points})")
    print(f"Density threshold : {args.threshold}")
    print(f"Low-pass sigma    : {args.lowpass_sigma}")
    print(f"Number of trials  : {args.num_trials}")
    print(f"Distance metric   : Squared Euclidean Distance ||p_i - q_j||_2^2")
    print(f"Matrix dimension  : {args.num_points} x {args.num_points}")
    print(f"=" * 80)

    if len(all_paths) < 2:
        raise ValueError("Need at least 2 subtomograms in cluster directory.")

    trial_records = []

    for trial_idx in range(1, args.num_trials + 1):
        # Randomly select 2 distinct subtomograms
        pair = random.sample(all_paths, 2)
        path1, path2 = pair[0], pair[1]

        # Load volumes
        vol1 = ku.load_mrc(path1)
        vol2 = ku.load_mrc(path2)

        # Prepare volumes for sampling
        prep1 = ku.prepare_sampling_volume(vol1, sigma=args.lowpass_sigma)
        prep2 = ku.prepare_sampling_volume(vol2, sigma=args.lowpass_sigma)

        # Sample TRN point clouds
        pts1 = ku.sample_volume_in_memory(
            prep1,
            threshold=args.threshold,
            num_points=args.num_points,
            random_seed=args.seed + trial_idx * 100 + 1,
        )
        pts2 = ku.sample_volume_in_memory(
            prep2,
            threshold=args.threshold,
            num_points=args.num_points,
            random_seed=args.seed + trial_idx * 100 + 2,
        )

        # In AlignOT, clouds are centered before alignment:
        pts1_c = pts1 - pts1.mean(axis=0)
        pts2_c = pts2 - pts2.mean(axis=0)

        # Compute cost matrix (N x M)
        cost_mat = compute_cost_matrix(pts1_c, pts2_c)

        # Compute stats
        cost_min = float(np.min(cost_mat))
        cost_max = float(np.max(cost_mat))
        cost_mean = float(np.mean(cost_mat))
        cost_std = float(np.std(cost_mat))

        # Row sums (sum across columns for each row -> length N)
        row_sums = np.sum(cost_mat, axis=1)
        row_sum_mean = float(np.mean(row_sums))
        row_sum_min = float(np.min(row_sums))
        row_sum_max = float(np.max(row_sums))
        row_sum_std = float(np.std(row_sums))

        # Column sums (sum across rows for each column -> length M)
        col_sums = np.sum(cost_mat, axis=0)
        col_sum_mean = float(np.mean(col_sums))
        col_sum_min = float(np.min(col_sums))
        col_sum_max = float(np.max(col_sums))
        col_sum_std = float(np.std(col_sums))

        record = {
            "trial": trial_idx,
            "subtomogram_1": path1.name,
            "subtomogram_2": path2.name,
            "dim_rows": cost_mat.shape[0],
            "dim_cols": cost_mat.shape[1],
            "min_cost": cost_min,
            "max_cost": cost_max,
            "mean_cost": cost_mean,
            "std_cost": cost_std,
            "row_sum_mean": row_sum_mean,
            "row_sum_min": row_sum_min,
            "row_sum_max": row_sum_max,
            "row_sum_std": row_sum_std,
            "col_sum_mean": col_sum_mean,
            "col_sum_min": col_sum_min,
            "col_sum_max": col_sum_max,
            "col_sum_std": col_sum_std,
        }
        trial_records.append(record)

        print(
            f"Trial {trial_idx:2d}: Pair ({path1.name:>7s}, {path2.name:>7s}) | "
            f"Cost: min={cost_min:7.2f}, max={cost_max:8.2f}, avg={cost_mean:7.2f} | "
            f"RowSum: avg={row_sum_mean:9.2f} (min={row_sum_min:8.2f}, max={row_sum_max:8.2f}) | "
            f"ColSum: avg={col_sum_mean:9.2f} (min={col_sum_min:8.2f}, max={col_sum_max:8.2f})"
        )

    df = pd.DataFrame(trial_records)

    print(f"\n" + "=" * 80)
    print("SUMMARY OF 10 TRIALS:")
    print("=" * 80)
    print(
        df[
            [
                "trial",
                "subtomogram_1",
                "subtomogram_2",
                "min_cost",
                "max_cost",
                "mean_cost",
                "row_sum_mean",
                "col_sum_mean",
            ]
        ].to_string(index=False)
    )

    print(f"\n" + "=" * 80)
    print("AGGREGATE STATISTICS (Across 10 Trials, Mean ± Std):")
    print("=" * 80)
    print(f"  Dimensions (Rows x Cols) : {df['dim_rows'].iloc[0]} x {df['dim_cols'].iloc[0]}")
    print(f"  Min Cost                 : {df['min_cost'].mean():.3f} ± {df['min_cost'].std():.3f}")
    print(f"  Max Cost                 : {df['max_cost'].mean():.3f} ± {df['max_cost'].std():.3f}")
    print(f"  Average Cost (Mean)      : {df['mean_cost'].mean():.3f} ± {df['mean_cost'].std():.3f}")
    print(f"  Row Sum Mean             : {df['row_sum_mean'].mean():.3f} ± {df['row_sum_mean'].std():.3f}")
    print(f"  Row Sum Range (Min ~ Max): [{df['row_sum_min'].min():.2f}, {df['row_sum_max'].max():.2f}]")
    print(f"  Col Sum Mean             : {df['col_sum_mean'].mean():.3f} ± {df['col_sum_mean'].std():.3f}")
    print(f"  Col Sum Range (Min ~ Max): [{df['col_sum_min'].min():.2f}, {df['col_sum_max'].max():.2f}]")
    print(f"=" * 80)


if __name__ == "__main__":
    main()

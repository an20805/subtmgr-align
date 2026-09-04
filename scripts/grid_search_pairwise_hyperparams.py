#!/usr/bin/env python3
"""
grid_search_pairwise_hyperparams.py
-----------------------------------
Fast in-memory hyperparameter grid search for Joint Pairwise Wasserstein SGD on GPU.
Samples the synthetic point clouds once, then rapidly evaluates multiple combinations
of (reg, sgd_lr, sgd_iters) on GPU, scoring each by:
  1. Mean angular recovery error (vs ground-truth).
  2. Success rate (< 5° and < 10° error).
  3. Total Wasserstein cost reduction (ΔW2).
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "alignOT" / "src"))
sys.path.insert(0, str(REPO_ROOT / "alignOT" / "alignOT"))

import alignot_utils
import alignot_gpu_utils as gpu_utils
import coords as _coords


def compute_angular_error(q_est: list, q_gt: list) -> float:
    r_est = Rotation.from_quat([q_est[1], q_est[2], q_est[3], q_est[0]])
    r_gt = Rotation.from_quat([q_gt[1], q_gt[2], q_gt[3], q_gt[0]])
    r_diff1 = r_est * r_gt.inv()
    ang1 = float(np.degrees(r_diff1.magnitude()))
    r_diff2 = r_est.inv() * r_gt.inv()
    ang2 = float(np.degrees(r_diff2.magnitude()))
    return min(ang1, ang2)


def _sample_single_subtomogram(task_args: tuple) -> dict:
    path, lowpass_sigma, threshold, num_points, seed = task_args
    volume = alignot_utils.load_mrc(path, permissive=True)
    src_prep = alignot_utils.prepare_sampling_volume(volume, sigma=lowpass_sigma)
    pts_src = alignot_utils.sample_volume_in_memory(
        src_prep, threshold=threshold, num_points=num_points, random_seed=seed,
    )
    Bbar = pts_src.mean(axis=0).astype(np.float64)
    pts_src_c = pts_src - Bbar
    src_xyz = pts_src_c[:, [2, 1, 0]].astype(np.float32)
    return {
        "subtomogram_id": int(path.stem),
        "src_xyz": src_xyz,
        "Bbar": Bbar,
    }


def main():
    parser = argparse.ArgumentParser(description="Fast GPU Grid Search for Pairwise Wasserstein Hyperparameters.")
    parser.add_argument("--cluster_dir", default=str(REPO_ROOT / "Data" / "synthetic_clean_40"))
    parser.add_argument("--gt_csv", default=str(REPO_ROOT / "Data" / "synthetic_clean_40" / "ground_truth_rotations.csv"))
    parser.add_argument("--output_csv", default=str(REPO_ROOT / "alignOT" / "outputs" / "pairwise_hyperparam_grid_results.csv"))
    parser.add_argument("--num_points", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--lowpass_sigma", type=float, default=2.0)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = gpu_utils.get_device(args.device)
    df_gt = pd.read_csv(args.gt_csv)

    # 1. Load & sample all point clouds ONCE in parallel
    cluster_paths = alignot_utils.get_cluster_paths(args.cluster_dir)
    N = len(cluster_paths)
    print(f"Sampling {N} point clouds from {args.cluster_dir} (done once in memory)...")
    t0 = time.perf_counter()
    sampling_tasks = [
        (path, args.lowpass_sigma, args.threshold, args.num_points, args.seed + idx)
        for idx, path in enumerate(cluster_paths)
    ]
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        sampled_items = list(executor.map(_sample_single_subtomogram, sampling_tasks))
    print(f"Point clouds sampled in {time.perf_counter() - t0:.2f}s\n")

    src_all_np = np.stack([item["src_xyz"] for item in sampled_items], axis=0)

    # Candidate hyperparameter grid
    reg_list = [50.0, 100.0, 200.0]
    lr_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    iters_list = [500]

    grid_results = []
    total_runs = len(reg_list) * len(lr_list) * len(iters_list)
    run_count = 0

    print("=" * 85)
    print(f" STARTING GPU HYPERPARAMETER GRID SEARCH ({total_runs} Configurations)")
    print("=" * 85)
    print(f" {'Run':<4} | {'Reg (ε)':<8} | {'LR (η)':<8} | {'Iters':<6} | {'W2 Init':<9} | {'W2 Final':<9} | {'ΔW2 (%)':<8} | {'Mean Err':<9} | {'< 10°':<6} | {'Time':<6}")
    print("-" * 85)

    for iters in iters_list:
        for reg in reg_list:
            for lr in lr_list:
                run_count += 1
                t0_run = time.perf_counter()

                # Run GPU Pairwise SGD
                quats, loss_curve = gpu_utils.SGD_pairwise_gpu(
                    src_all_np=src_all_np,
                    lr=lr,
                    max_iter=iters,
                    reg=reg,
                    seed=args.seed,
                    anchor_idx=0,
                    sinkhorn_iters=15,
                    log_every=0,  # silent
                    device=device,
                )
                t_run = time.perf_counter() - t0_run

                # Evaluate angular error vs ground truth
                errors = []
                for i, item in enumerate(sampled_items):
                    sid = item["subtomogram_id"]
                    gt_row = df_gt[df_gt["subtomogram_id"] == sid].iloc[0]
                    q_gt = [gt_row["q_w"], gt_row["q_x"], gt_row["q_y"], gt_row["q_z"]]
                    q_est = quats[i]
                    errors.append(compute_angular_error(q_est, q_gt))

                mean_err = float(np.mean(errors))
                median_err = float(np.median(errors))
                succ_10 = int(np.sum(np.array(errors) < 10.0))
                succ_5 = int(np.sum(np.array(errors) < 5.0))

                w2_init = loss_curve[0]
                w2_final = loss_curve[-1]
                delta_w2_pct = ((w2_final - w2_init) / w2_init) * 100.0

                print(f" {run_count:2d}/{total_runs} | {reg:7.1f} | {lr:8.4f} | {iters:5d} | {w2_init:8.2f} | {w2_final:8.2f} | {delta_w2_pct:7.2f}% | {mean_err:7.2f}° | {succ_10:2d}/{N}   | {t_run:5.1f}s")

                grid_results.append({
                    "reg": reg,
                    "lr": lr,
                    "iters": iters,
                    "w2_init": w2_init,
                    "w2_final": w2_final,
                    "delta_w2_pct": delta_w2_pct,
                    "mean_angular_error": mean_err,
                    "median_angular_error": median_err,
                    "success_lt_5deg": succ_5,
                    "success_lt_10deg": succ_10,
                    "runtime_s": t_run,
                })

    out_df = pd.DataFrame(grid_results)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print("=" * 85)
    print(f"\nGrid search completed! Results saved to: {args.output_csv}")

    # Display top 5 best configurations
    top_by_error = out_df.sort_values(by="mean_angular_error").head(5)
    print("\nTop 5 Configurations by Smallest Angular Error:")
    print(top_by_error[["reg", "lr", "iters", "w2_final", "mean_angular_error", "success_lt_10deg"]].to_string(index=False))


if __name__ == "__main__":
    main()

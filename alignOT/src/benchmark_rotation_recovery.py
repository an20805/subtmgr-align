#!/usr/bin/env python3
"""
benchmark_rotation_recovery.py
------------------------------
Standalone diagnostic and benchmark tool for testing rotation recovery of
subtomogram alignment methods in AlignOT (SGD vs. Kabsch).

Pre-processing Pipeline:
  1. Load volume: vol = mrc.data.astype(np.float32)
  2. Shift to positive: vol = vol - vol.min()
  3. Scale to [0, 1]: vol = vol / (vol.max() + 1e-8)
  4. Invert contrast: vol_dark = 1.0 - vol
  5. Threshold and sample via Topology Representing Networks (TRN).

Usage:
  # Run standalone with all default parameters defined inside:
  python alignOT/src/benchmark_rotation_recovery.py

  # Or optionally override via CLI:
  python alignOT/src/benchmark_rotation_recovery.py --method kabsch
  python alignOT/src/benchmark_rotation_recovery.py --method sgd_barycentric
  python alignOT/src/benchmark_rotation_recovery.py --method all
"""

import argparse
import contextlib
import io
import math
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# ── Module Path Setup ─────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR.parent / "alignOT"))

import utils as _utils
import trn
import run_iterative_alignment_barycentric as bary_sgd


# ==============================================================================
# DEFAULT PARAMETERS (Configurable directly in-script)
# ==============================================================================

# Data
DEFAULT_MRC_PATH = str(REPO_ROOT / "Data" / "0.1" / "Cluster_0" / "0.mrc")
if not Path(DEFAULT_MRC_PATH).exists():
    DEFAULT_MRC_PATH = str(REPO_ROOT / "Data" / "0.mrc")

# Alignment Method: "all", "sgd_barycentric", "sgd_standard", "kabsch"
DEFAULT_METHOD = "kabsch"

# Injected Rotation Geometry
DEFAULT_ROTATION_AXIS = "y"  # 'x', 'y', 'z'
DEFAULT_TEST_ANGLES = [15, 30, 45, 60, 90, 120, 180]  # Degrees

# TRN Sampling Hyperparameters
DEFAULT_NUM_POINTS = 500      # Number of representative TRN points
DEFAULT_THRESHOLD = 0.7       # Density cutoff after normalization & inversion

# AlignOT SGD Hyperparameters
DEFAULT_SGD_REG = 100.0        # Sinkhorn entropic regularization
DEFAULT_SGD_LR = 0.001      # Quaternion gradient descent learning rate
DEFAULT_SGD_ITERS = 1000      # Number of SGD iterations for convergence
DEFAULT_SGD_NUM_SAMPLES = 1   # Stochastic sample points per iteration

# AlignOT Kabsch Hyperparameters
DEFAULT_KABSCH_REG = 30.0     # Sinkhorn entropic regularization for one-shot plan
DEFAULT_KABSCH_ITERS = 5000   # Max Sinkhorn iterations

# General
DEFAULT_SEED = 42


# ==============================================================================
# ALIGNMENT IMPLEMENTATIONS
# ==============================================================================

def align_sgd_barycentric(x, y, z, xr, yr, zr, q_ideal, reg, lr, iters, num_samples, seed):
    """AlignOT SGD using Barycentric weighted sum + 1-NN projection."""
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        quats, costs = bary_sgd.SGD_barycentric_1nn(
            x, y, z,
            xr, yr, zr,
            lr=lr,
            max_iter=iters,
            reg=reg,
            num_samples=num_samples,
            random_seed=seed,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    q_final = quats[-1]  # [w, x, y, z]
    rot = Rotation.from_quat([q_final[1], q_final[2], q_final[3], q_final[0]])
    rec_angle = float(rot.magnitude() * 180.0 / math.pi)
    ang_error = float(_utils.diff_quaternions(q_final, q_ideal))

    return {
        "method": "SGD (Barycentric 1-NN)",
        "recovered_angle_deg": rec_angle,
        "angular_error_deg": ang_error,
        "initial_cost": float(costs[0]),
        "final_cost": float(costs[-1]),
        "cost_reduction_pct": float(100.0 * (costs[0] - costs[-1]) / max(costs[0], 1e-8)),
        "quaternion": [float(val) for val in q_final],
        "runtime_ms": elapsed_ms,
    }


def align_sgd_standard(x, y, z, xr, yr, zr, q_ideal, reg, lr, iters, num_samples, seed):
    """AlignOT SGD using Standard Argmax correspondence."""
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        quats, costs = _utils.SGD(
            x, y, z,
            xr, yr, zr,
            lr=lr,
            max_iter=iters,
            reg=reg,
            num_samples=num_samples,
            random_seed=seed,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    q_final = quats[-1]  # [w, x, y, z]
    rot = Rotation.from_quat([q_final[1], q_final[2], q_final[3], q_final[0]])
    rec_angle = float(rot.magnitude() * 180.0 / math.pi)
    ang_error = float(_utils.diff_quaternions(q_final, q_ideal))

    return {
        "method": "SGD (Standard Argmax)",
        "recovered_angle_deg": rec_angle,
        "angular_error_deg": ang_error,
        "initial_cost": float(costs[0]),
        "final_cost": float(costs[-1]),
        "cost_reduction_pct": float(100.0 * (costs[0] - costs[-1]) / max(costs[0], 1e-8)),
        "quaternion": [float(val) for val in q_final],
        "runtime_ms": elapsed_ms,
    }


def align_kabsch(x, y, z, xr, yr, zr, q_ideal, reg, iters):
    """AlignOT One-Shot Euclidean Sinkhorn + Kabsch SVD."""
    t0 = time.perf_counter()
    pts_src = np.column_stack((x, y, z)).astype(np.float64)
    pts_ref = np.column_stack((xr, yr, zr)).astype(np.float64)

    M = np.sum((pts_src[:, None, :] - pts_ref[None, :, :]) ** 2, axis=-1)
    N = len(pts_src)
    a = np.ones(N) / N
    b = np.ones(N) / N

    with contextlib.redirect_stdout(io.StringIO()):
        plan, _, _ = _utils.my_sinkhorn(a, b, M, reg=reg, numItermax=iters)

    init_cost = float(np.sum(M) / (N * N))
    final_cost = float(np.sum(plan * M))

    best_j = np.argmax(plan, axis=1)
    A = pts_ref[best_j]
    B = pts_src

    Abar = A.mean(axis=0)
    Bbar = B.mean(axis=0)
    H = (A - Abar).T @ (B - Bbar)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(U @ Vt)
    R_mat = U @ np.diag([1.0, 1.0, d]) @ Vt

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    rot = Rotation.from_matrix(R_mat)
    rec_angle = float(rot.magnitude() * 180.0 / math.pi)
    q_scipy = rot.as_quat()  # [x, y, z, w]
    q_kabsch = [q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]]  # [w, x, y, z]
    ang_error = float(_utils.diff_quaternions(q_kabsch, q_ideal))

    return {
        "method": "Kabsch (Euclidean OT)",
        "recovered_angle_deg": rec_angle,
        "angular_error_deg": ang_error,
        "initial_cost": init_cost,
        "final_cost": final_cost,
        "cost_reduction_pct": float(100.0 * (init_cost - final_cost) / max(init_cost, 1e-8)),
        "quaternion": q_kabsch,
        "runtime_ms": elapsed_ms,
    }


# ==============================================================================
# MAIN BENCHMARK PIPELINE
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="AlignOT Rotation Recovery Benchmark (SGD vs. Kabsch)"
    )
    parser.add_argument("--mrc_path", type=str, default=DEFAULT_MRC_PATH,
                        help="Path to input reference MRC file")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD,
                        choices=["all", "sgd_barycentric", "sgd_standard", "kabsch"],
                        help="Alignment method to evaluate")
    parser.add_argument("--axis", type=str, default=DEFAULT_ROTATION_AXIS,
                        help="Rotation axis ('x', 'y', 'z')")
    parser.add_argument("--num_points", type=int, default=DEFAULT_NUM_POINTS,
                        help="Number of TRN points to sample")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Density threshold for TRN sampling")
    parser.add_argument("--sgd_reg", type=float, default=DEFAULT_SGD_REG,
                        help="Sinkhorn regularization for SGD")
    parser.add_argument("--sgd_lr", type=float, default=DEFAULT_SGD_LR,
                        help="Learning rate for SGD")
    parser.add_argument("--sgd_iters", type=int, default=DEFAULT_SGD_ITERS,
                        help="Iterations for SGD")
    parser.add_argument("--kabsch_reg", type=float, default=DEFAULT_KABSCH_REG,
                        help="Sinkhorn regularization for Kabsch")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def run_benchmark(args=None):
    if args is None:
        args = parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 90)
    print("ALIGNOT ROTATION RECOVERY BENCHMARK & DIAGNOSTIC")
    print("=" * 90)
    print(f"Input MRC file     : {args.mrc_path}")
    print(f"Selected Method    : {args.method.upper()}")
    print(f"Rotation Axis      : {args.axis.upper()}")
    print(f"Test Angles (deg)  : {DEFAULT_TEST_ANGLES}")
    print(f"TRN Points         : {args.num_points} (threshold={args.threshold})")
    if args.method in ["all", "sgd_barycentric", "sgd_standard"]:
        print(f"SGD Parameters     : reg={args.sgd_reg}, lr={args.sgd_lr}, iters={args.sgd_iters}")
    if args.method in ["all", "kabsch"]:
        print(f"Kabsch Parameters  : reg={args.kabsch_reg}")
    print("=" * 90)

    # 1. Load Volume & Apply Pre-processing Pipeline
    print("\n[1/3] Loading MRC and applying pre-processing pipeline...")
    with mrcfile.open(args.mrc_path, permissive=True) as mrc:
        vol = mrc.data.astype(np.float32)

    # Step 1: shift to positive
    vol = vol - vol.min()

    # Step 2: scale to [0, 1]
    vol = vol / (vol.max() + 1e-8)
    vol_dark = 1.0 - vol

    # Threshold and sample TRN point cloud
    map_th = vol_dark.copy()
    map_th[map_th < args.threshold] = 0.0

    rm0, arr_flat, arr_idx, xyz, coords_1d = trn.trn_rm0(
        map_th, M=args.num_points, random_seed=args.seed
    )
    l0 = 0.005 * args.num_points
    lf = 0.5
    tf = args.num_points * 8
    e0 = 0.3
    ef = 0.05
    rms, _, _ = trn.trn_iterate(
        rm0, arr_flat, arr_idx, xyz,
        n_save=10, e0=e0, ef=ef, l0=l0, lf=lf, tf=tf,
        do_log=True, log_n=10
    )

    x = [p[0] for p in rms[10]]
    y = [p[1] for p in rms[10]]
    z = [p[2] for p in rms[10]]

    print(f"  Reference point cloud sampled: {len(x)} points")

    # 2. Run Benchmarks Across Angles
    print("\n[2/3] Running rotation recovery benchmarks...")
    methods_to_run = []
    if args.method in ["all", "sgd_barycentric"]:
        methods_to_run.append("sgd_barycentric")
    if args.method in ["all", "sgd_standard"]:
        methods_to_run.append("sgd_standard")
    if args.method in ["all", "kabsch"]:
        methods_to_run.append("kabsch")

    records = []

    # Axis vector mapping
    axis_vec = {
        "x": (1, 0, 0),
        "y": (0, 1, 0),
        "z": (0, 0, 1),
    }.get(args.axis.lower(), (0, 1, 0))

    for angle in DEFAULT_TEST_ANGLES:
        q_ideal = _utils.get_quaternion_vals(angle * math.pi / 180.0, axis_vec[0], axis_vec[1], axis_vec[2])
        xr, yr, zr = _utils.perform(x, y, z, q_ideal)

        for m in methods_to_run:
            if m == "sgd_barycentric":
                res = align_sgd_barycentric(
                    x, y, z, xr, yr, zr, q_ideal,
                    reg=args.sgd_reg,
                    lr=args.sgd_lr,
                    iters=args.sgd_iters,
                    num_samples=DEFAULT_SGD_NUM_SAMPLES,
                    seed=args.seed,
                )
            elif m == "sgd_standard":
                res = align_sgd_standard(
                    x, y, z, xr, yr, zr, q_ideal,
                    reg=args.sgd_reg,
                    lr=args.sgd_lr,
                    iters=args.sgd_iters,
                    num_samples=DEFAULT_SGD_NUM_SAMPLES,
                    seed=args.seed,
                )
            elif m == "kabsch":
                res = align_kabsch(
                    x, y, z, xr, yr, zr, q_ideal,
                    reg=args.kabsch_reg,
                    iters=DEFAULT_KABSCH_ITERS,
                )

            res["injected_angle_deg"] = angle
            records.append(res)
            print(f"  Injected: {angle:3d}° | {res['method']:<25s} -> Error: {res['angular_error_deg']:6.2f}° | Cost: {res['initial_cost']:.1f} -> {res['final_cost']:.1f} ({res['runtime_ms']:.0f} ms)")

    # 3. Format and Display Summary Table
    print("\n[3/3] Benchmark Results Summary:")
    print("=" * 90)
    df = pd.DataFrame(records)

    if len(methods_to_run) > 1:
        pivot_err = df.pivot(index="injected_angle_deg", columns="method", values="angular_error_deg")
        pivot_cost = df.pivot(index="injected_angle_deg", columns="method", values="final_cost")
        print("\nANGULAR ERROR (Degrees, lower is better):")
        print(pivot_err.to_string())
        print("\nFINAL WASSERSTEIN COSTS:")
        print(pivot_cost.to_string())
    else:
        display_cols = ["injected_angle_deg", "method", "recovered_angle_deg", "angular_error_deg", "initial_cost", "final_cost", "cost_reduction_pct", "runtime_ms"]
        print(df[display_cols].to_string(index=False))

    print("=" * 90)

    return df


if __name__ == "__main__":
    run_benchmark()

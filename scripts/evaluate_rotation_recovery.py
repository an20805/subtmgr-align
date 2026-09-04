#!/usr/bin/env python3
"""
evaluate_rotation_recovery.py
-----------------------------
Compares estimated quaternions from an alignment run against ground-truth quaternions.
Calculates geodesic angular error in degrees on SO(3).
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation

# Add repo root and src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "alignOT" / "src"))
sys.path.insert(0, str(REPO_ROOT / "alignOT" / "alignOT"))
import coords as _coords


def compute_angular_error(q_est: list, q_gt: list) -> float:
    """
    Computes angular distance between two rotations in degrees.
    Note: q_est is the inverse rotation applied to align to reference,
    so R_est @ R_gt should equal Identity if alignment is perfect.
    Or if q_est recovers the orientation directly: R_est ≈ R_gt.
    """
    # Convert [w, x, y, z] to scipy Rotation format [x, y, z, w]
    r_est = Rotation.from_quat([q_est[1], q_est[2], q_est[3], q_est[0]])
    r_gt = Rotation.from_quat([q_gt[1], q_gt[2], q_gt[3], q_gt[0]])

    # Error rotation: R_diff = R_est * R_gt^-1
    # Check both direct alignment and inverse alignment
    r_diff1 = r_est * r_gt.inv()
    ang1 = float(np.degrees(r_diff1.magnitude()))

    r_diff2 = r_est.inv() * r_gt.inv()
    ang2 = float(np.degrees(r_diff2.magnitude()))

    return min(ang1, ang2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate rotation recovery against ground truth.")
    parser.add_argument("--gt_csv", default=str(REPO_ROOT / "Data" / "synthetic_clean_40" / "ground_truth_rotations.csv"),
                        help="Path to ground truth rotations CSV")
    parser.add_argument("--est_csv", required=True,
                        help="Path to estimated quaternions CSV")
    args = parser.parse_args()

    df_gt = pd.read_csv(args.gt_csv)
    df_est = pd.read_csv(args.est_csv)

    merged = pd.merge(df_gt, df_est, on="subtomogram_id", suffixes=("_gt", "_est"))

    errors = []
    for _, row in merged.iterrows():
        q_gt = [row["q_w_gt"], row["q_x_gt"], row["q_y_gt"], row["q_z_gt"]]
        q_est = [row["q_w_est"], row["q_x_est"], row["q_y_est"], row["q_z_est"]]
        err = compute_angular_error(q_est, q_gt)
        errors.append(err)

    merged["angular_error_deg"] = errors

    print("=" * 70)
    print(" ROTATION RECOVERY EVALUATION AGAINST GROUND TRUTH")
    print("=" * 70)
    print(f"  Ground Truth : {args.gt_csv}")
    print(f"  Estimated    : {args.est_csv}")
    print(f"  Particles    : {len(merged)}")
    print("-" * 70)
    print(f"  Mean Angular Error   : {np.mean(errors):.2f}°")
    print(f"  Median Angular Error : {np.median(errors):.2f}°")
    print(f"  Min Angular Error    : {np.min(errors):.2f}°")
    print(f"  Max Angular Error    : {np.max(errors):.2f}°")
    print(f"  Success (Error < 5°) : {np.sum(np.array(errors) < 5.0)} / {len(errors)} ({np.mean(np.array(errors) < 5.0)*100:.1f}%)")
    print(f"  Success (Error < 10°): {np.sum(np.array(errors) < 10.0)} / {len(errors)} ({np.mean(np.array(errors) < 10.0)*100:.1f}%)")
    print("=" * 70)

    # Print first 10 particles
    print("\nSample Particles Breakdown:")
    for _, r in merged.head(10).iterrows():
        print(f"  Particle {int(r['subtomogram_id']):02d}: Injected = {r['injected_angle_deg']:5.1f}°, Error = {r['angular_error_deg']:5.2f}°")


if __name__ == "__main__":
    main()

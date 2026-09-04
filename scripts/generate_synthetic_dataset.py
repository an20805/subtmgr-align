#!/usr/bin/env python3
"""
generate_synthetic_dataset.py
-----------------------------
Generates a clean synthetic subtomogram dataset by applying known random 3D rotations
to a ground-truth reference map (e.g. Data/0.mrc).

Output:
  - N clean .mrc files (0.mrc, 1.mrc, ..., (N-1).mrc)
  - ground_truth_rotations.csv containing true quaternions & angles (degrees).
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation
import mrcfile

# Add repo root and src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "alignOT" / "src"))
sys.path.insert(0, str(REPO_ROOT / "alignOT" / "alignOT"))

import alignot_utils
import coords as _coords


def rotate_volume_scipy(volume: np.ndarray, q: list) -> np.ndarray:
    """
    Rotate 3D volume around box center using scipy ndimage affine transform
    with the exact coordinate convention of AlignOT.
    """
    from scipy.ndimage import affine_transform

    N = volume.shape[0]
    center = (N - 1) / 2.0

    # R in (x, y, z) Euclidean frame
    R_xyz = _coords.quaternion_to_R(list(q))

    # Permutation matrix P: [z, y, x] <-> [x, y, z]
    P = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]], dtype=np.float64)

    # R in [z, y, x] array index frame
    R_zyx = P @ R_xyz @ P.T
    R_inv = R_zyx.T

    # Offset to rotate strictly around box center
    c = np.array([center, center, center], dtype=np.float64)
    offset = c - R_inv @ c

    rot_vol = affine_transform(
        volume.astype(np.float32),
        matrix=R_inv,
        offset=offset,
        order=2,
        mode="constant",
        cval=0.0,
        prefilter=True,
    )
    return rot_vol.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate clean synthetic subtomogram dataset.")
    parser.add_argument("--ref_mrc", default=str(REPO_ROOT / "Data" / "0.mrc"),
                        help="Path to ground truth reference MRC")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "Data" / "synthetic_clean_40"),
                        help="Directory to save synthetic .mrc files")
    parser.add_argument("--n_particles", type=int, default=40,
                        help="Number of synthetic particles to generate")
    parser.add_argument("--max_angle", type=float, default=60.0,
                        help="Maximum random rotation angle in degrees (default: 60.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading reference volume: {args.ref_mrc}")
    with mrcfile.open(args.ref_mrc, permissive=True) as mrc:
        base_vol = mrc.data.astype(np.float32)

    rows = []

    print(f"Generating {args.n_particles} synthetic subtomograms (max angle: {args.max_angle}°)...")
    for i in range(args.n_particles):
        if i == 0:
            # Anchor particle is unrotated
            q = [1.0, 0.0, 0.0, 0.0]
            rot_vol = base_vol.copy()
            angle_deg = 0.0
        else:
            # Sample random rotation axis on unit sphere
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)

            # Sample random angle up to max_angle
            angle_deg = float(np.random.uniform(5.0, args.max_angle))
            angle_rad = np.radians(angle_deg)

            # Convert axis-angle to unit quaternion [w, x, y, z]
            w = float(np.cos(angle_rad / 2.0))
            xyz = axis * float(np.sin(angle_rad / 2.0))
            q = [w, float(xyz[0]), float(xyz[1]), float(xyz[2])]

            # Apply rotation to clean volume
            rot_vol = rotate_volume_scipy(base_vol, q)

        out_path = out_dir / f"{i}.mrc"
        with mrcfile.new(str(out_path), overwrite=True) as mrc:
            mrc.set_data(rot_vol)

        rows.append({
            "subtomogram_id": i,
            "file_name": f"{i}.mrc",
            "q_w": q[0],
            "q_x": q[1],
            "q_y": q[2],
            "q_z": q[3],
            "injected_angle_deg": angle_deg,
        })
        print(f"  Particle {i:02d}: injected rotation = {angle_deg:5.1f}° (axis = {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f})")

    csv_path = out_dir / "ground_truth_rotations.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSynthetic dataset generated successfully in: {out_dir}")
    print(f"Ground-truth rotations saved to: {csv_path}")


if __name__ == "__main__":
    main()

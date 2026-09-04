#!/usr/bin/env python3
"""
alignot_so3_grid.py
-------------------
SO(3) uniform rotation grid generation for exhaustive coarse-to-fine
Wasserstein search.

Two grid types:
  1. generate_so3_grid(step)         – global uniform grid via ZYZ Euler angles
  2. generate_local_so3_grid(R, r, s) – local cone grid via Lie algebra exp map

Convention: all rotation matrices R act on XYZ Euclidean column vectors:
    x_rotated = R @ x    ←→    x_row @ R.T  (row-vector convention used by torch)
This matches the AlignOT coordinate frame (ZYX → XYZ permutation already applied).
"""

import numpy as np
from scipy.spatial.transform import Rotation as ScipyR
from typing import List, Tuple


# ── 1. Global Uniform SO(3) Grid ─────────────────────────────────────────────

def generate_so3_grid(angular_step_deg: float) -> np.ndarray:
    """
    Approximately uniform SO(3) grid via ZYZ Euler angles.

    Sampling strategy:
      theta  – arccos of values uniform in [-1, 1]  (equal-area latitude rings)
      phi    – uniform in [0, 2pi), N_phi ∝ sin(theta) (equal-area arcs)
      psi    – uniform in [0, 2pi)  (equal-area in-plane rotation)

    Grid size: K ≈ π⁴ / Δ³   (e.g. Δ=15° → ~4848,  Δ=10° → ~16000)

    Args:
        angular_step_deg: Angular spacing in degrees.

    Returns:
        (K, 3, 3) float32 rotation matrices.
    """
    step = np.radians(angular_step_deg)

    n_theta = max(2, int(np.round(np.pi / step)))
    cos_thetas = np.linspace(-1.0, 1.0, n_theta)

    n_psi = max(1, int(np.round(2.0 * np.pi / step)))
    psis = np.linspace(0.0, 2.0 * np.pi, n_psi, endpoint=False)

    euler_angles: List[List[float]] = []

    for ct in cos_thetas:
        theta = float(np.arccos(np.clip(ct, -1.0, 1.0)))
        sin_t = float(np.sqrt(max(0.0, 1.0 - ct**2)))

        # sin-weighted phi count: fewer near poles, more near equator
        n_phi = max(1, int(np.round(2.0 * np.pi * sin_t / step)))
        phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

        for phi in phis:
            for psi in psis:
                euler_angles.append([phi, theta, psi])

    euler_arr = np.array(euler_angles, dtype=np.float64)   # (K, 3)
    R_grid = ScipyR.from_euler('ZYZ', euler_arr).as_matrix().astype(np.float32)
    return R_grid                                           # (K, 3, 3)


def count_so3_grid(angular_step_deg: float) -> int:
    """Estimate number of rotations in SO(3) grid without generating it."""
    step = np.radians(angular_step_deg)
    n_theta = max(2, int(np.round(np.pi / step)))
    cos_thetas = np.linspace(-1.0, 1.0, n_theta)
    n_psi = max(1, int(np.round(2.0 * np.pi / step)))

    total = 0
    for ct in cos_thetas:
        sin_t = float(np.sqrt(max(0.0, 1.0 - ct**2)))
        n_phi = max(1, int(np.round(2.0 * np.pi * sin_t / step)))
        total += n_phi * n_psi
    return total


# ── 2. Local Cone Grid (Lie Algebra Exp Map) ──────────────────────────────────

def generate_local_so3_grid(
    R_center: np.ndarray,
    cone_radius_deg: float,
    angular_step_deg: float,
) -> np.ndarray:
    """
    Generate SO(3) grid within cone_radius_deg of R_center.

    Uses the rotation-vector (so(3) Lie algebra) parameterization:
        R_local[k] = R_center @ exp(v_k)
    where {v_k} are vectors in a ball of radius `cone_radius` in R^3,
    sampled on a regular grid with step `angular_step`.

    This gives an approximately uniform grid in the geodesic ball around R_center.

    Args:
        R_center:        (3, 3) center rotation matrix.
        cone_radius_deg: Cone half-angle in degrees.
        angular_step_deg: Grid step in degrees.

    Returns:
        (K_local, 3, 3) float32 rotation matrices (includes the center itself).
    """
    radius = np.radians(cone_radius_deg)
    step   = np.radians(angular_step_deg)
    n      = max(1, int(np.ceil(radius / step)))

    vals = np.linspace(-radius, radius, 2 * n + 1)  # (2n+1,) from -r to +r

    rotvecs: List[np.ndarray] = []
    tol = radius + step * 0.6   # slight tolerance so corner points are included
    for v1 in vals:
        for v2 in vals:
            for v3 in vals:
                v = np.array([v1, v2, v3], dtype=np.float64)
                if np.linalg.norm(v) <= tol:
                    rotvecs.append(v)

    if not rotvecs:
        rotvecs = [np.zeros(3)]  # degenerate: just the center

    rotvecs_arr = np.array(rotvecs, dtype=np.float64)          # (K_local, 3)
    dR = ScipyR.from_rotvec(rotvecs_arr).as_matrix()           # (K_local, 3, 3)
    R_local = (R_center[None] @ dR).astype(np.float32)         # (K_local, 3, 3)
    return R_local


def count_local_so3_grid(cone_radius_deg: float, angular_step_deg: float) -> int:
    """Estimate number of rotations in local cone grid without generating it."""
    radius = np.radians(cone_radius_deg)
    step   = np.radians(angular_step_deg)
    n      = max(1, int(np.ceil(radius / step)))
    vals   = np.linspace(-radius, radius, 2 * n + 1)
    tol    = radius + step * 0.6
    count  = 0
    for v1 in vals:
        for v2 in vals:
            for v3 in vals:
                v = np.array([v1, v2, v3])
                if np.linalg.norm(v) <= tol:
                    count += 1
    return count


# ── 3. Quaternion Conversion Utilities ───────────────────────────────────────

def rotation_matrix_to_quaternion_wxyz(R: np.ndarray) -> List[float]:
    """Convert (3, 3) SO(3) matrix → [w, x, y, z] AlignOT quaternion."""
    r = ScipyR.from_matrix(R)
    q = r.as_quat()          # scipy convention: [x, y, z, w]
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def rotations_to_quaternions_wxyz(R_batch: np.ndarray) -> np.ndarray:
    """Convert (K, 3, 3) rotation matrices → (K, 4) [w, x, y, z] quaternions."""
    r = ScipyR.from_matrix(R_batch.astype(np.float64))
    q = r.as_quat()          # (K, 4) in [x, y, z, w]
    return np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])  # (K, 4) [w,x,y,z]

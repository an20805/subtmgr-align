#!/usr/bin/env python3
"""
alignot_sync_utils.py
---------------------
Stage 2 of the Two-Stage Rotation Synchronization pipeline:
Spectral Recovery of Absolute Rotations from Pairwise Relative Rotation Estimates.

Mathematical background (Singer & Shkolnisky 2011, SIAM J. Imaging Sci.):
  Given pairwise relative rotations R_{ij} ≈ R_{j,abs} R_{i,abs}^T obtained
  from Stage 1 (decoupled pairwise OT), assemble the (3N × 3N) block matrix H:

      H[3i:3i+3, 3j:3j+3] = R_{i,abs} R_{j,abs}^T  (Singer "connection block")

  In the noiseless case H = R R^T where R = [R_1; ...; R_N] in R^{3N x 3}.
  H is rank-3 with top-3 eigenvalues = N and spectral gap = N → 0.
  The top-3 eigenvectors recover [R_1,...,R_N]^T up to a global SO(3) gauge.

Coordinate Convention (matches AlignOT throughout):
  All rotations are in (X, Y, Z) Euclidean space.
  AlignOT Stage 1 convention:  R(q_{ij}) * X_i ≈ X_j
  which means:                  R(q_{ij}) = R_{j,abs} R_{i,abs}^T
  So for Singer's H:            H[3i:3i+3, 3j:3j+3] = R(q_{ij})^T
                                H[3j:3j+3, 3i:3i+3] = R(q_{ij})
"""

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR.parent / "alignOT"))


# ── 1. Quaternion / Matrix Utilities ─────────────────────────────────────────

def quaternion_to_matrix_batch(q: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of unit quaternions to rotation matrices.

    Args:
        q: (P, 4) tensor of [w, x, y, z] quaternions.

    Returns:
        R: (P, 3, 3) rotation matrices in SO(3).
    """
    q = q.to(dtype=torch.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        x**2 + w**2 - z**2 - y**2,  2*(x*y - w*z),              2*(x*z + w*y),
        2*(x*y + w*z),               y**2 + w**2 - x**2 - z**2,  2*(y*z - w*x),
        2*(x*z - w*y),               2*(y*z + w*x),              z**2 + w**2 - y**2 - x**2,
    ], dim=1).reshape(-1, 3, 3)
    return R


def matrix_to_quaternion_wxyz(R: np.ndarray) -> List[float]:
    """
    Convert a single 3x3 SO(3) rotation matrix to quaternion [w, x, y, z].

    Args:
        R: (3, 3) numpy rotation matrix.

    Returns:
        [w, x, y, z] as list of floats.
    """
    from scipy.spatial.transform import Rotation as ScipyR
    r = ScipyR.from_matrix(R)
    q_xyzw = r.as_quat()  # scipy: [x, y, z, w]
    return [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]


# ── 2. Spectral Rotation Synchronization ─────────────────────────────────────

def spectral_rotation_sync(
    q_pairs: List[List[float]],
    idx_i_list: List[int],
    idx_j_list: List[int],
    N: int,
    device: str = "cpu",
    logger=None,
) -> Tuple[np.ndarray, List[List[float]], np.ndarray, float]:
    """
    Stage 2: Spectral Rotation Synchronization.

    Recovers N absolute rotation matrices R_abs[0..N-1] in SO(3) from the
    pairwise relative rotations {q_{ij}} produced by Stage 1 (decoupled OT).

    Algorithm:
      1. Convert q_{ij} → R_{ij} = R(q_{ij}) in SO(3).
      2. Assemble (3N × 3N) symmetric block connection matrix H:
             H[3i:3i+3, 3j:3j+3] = R_ij^T   [= R_{i,abs} R_{j,abs}^T]
             H[3j:3j+3, 3i:3i+3] = R_ij      [= R_{j,abs} R_{i,abs}^T]
             H[3i:3i+3, 3i:3i+3] = I_3
      3. Eigendecompose H via torch.linalg.eigh (ascending order).
      4. Top-3 eigenvectors V = eigvecs[:, -3:] in R^{3N x 3}.
             V = [R_{1,abs}; ...; R_{N,abs}] @ O  (gauge O in SO(3))
      5. Project each 3x3 block Ř_i = V[3i:3i+3, :] to SO(3) via SVD:
             U_i S_i Wh_i = Ř_i,   R_abs[i] = U_i diag(1,1,det(U_i Wh_i)) Wh_i

    Alignment quaternion convention:
      To rotate particle i to the canonical frame: apply R_abs[i]^T.
      q_align[i] = quaternion for R_abs[i]^T.
      Pass q_align[i] to GPUVolumeRotator.rotate_volume().

    Args:
        q_pairs:     P quaternions [w,x,y,z], one per pair (ordered as pairs list).
        idx_i_list:  P source indices i (i < j).
        idx_j_list:  P target indices j.
        N:           Number of particles.
        device:      'cpu' or 'cuda' for eigendecomposition.
        logger:      Optional Python logger.

    Returns:
        R_abs:        (N, 3, 3) numpy float64 absolute rotation matrices in SO(3).
        q_align:      List of N [w,x,y,z] quaternions for R_abs[i]^T.
        eigenvalues:  (3N,) numpy array of eigenvalues (ascending).
        eigen_gap:    Ratio λ_{3rd-largest} / |λ_{4th-largest}|.
                      Should be >> 1 for clean pairwise estimates (ideal ≈ N).
    """
    P = len(q_pairs)
    assert len(idx_i_list) == P and len(idx_j_list) == P

    def _log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    _log(f"[spectral_sync] N={N}, P={P} pairs  →  H shape {3*N}x{3*N}")
    t0 = time.perf_counter()

    # Step 1: quaternions → rotation matrices (P, 3, 3)
    q_tensor = torch.tensor(q_pairs, dtype=torch.float64)  # (P, 4)
    R_pairs  = quaternion_to_matrix_batch(q_tensor)         # (P, 3, 3)

    # Step 2: build 3N × 3N block matrix H
    H = torch.zeros(3 * N, 3 * N, dtype=torch.float64)

    for p in range(P):
        i = idx_i_list[p]
        j = idx_j_list[p]
        R_ij = R_pairs[p]          # R_{j,abs} R_{i,abs}^T
        H[3*i:3*i+3, 3*j:3*j+3] = R_ij.T   # R_{i,abs} R_{j,abs}^T
        H[3*j:3*j+3, 3*i:3*i+3] = R_ij     # R_{j,abs} R_{i,abs}^T

    for i in range(N):
        H[3*i:3*i+3, 3*i:3*i+3] = torch.eye(3, dtype=torch.float64)

    _log(f"[spectral_sync] H assembled in {time.perf_counter()-t0:.3f}s")

    # Step 3: eigendecompose H (symmetric — use eigh for numerical stability)
    t1 = time.perf_counter()
    H_dev = H.to(device)
    eigenvalues, eigenvectors = torch.linalg.eigh(H_dev)  # ascending order
    eigenvalues  = eigenvalues.cpu()
    eigenvectors = eigenvectors.cpu()
    _log(f"[spectral_sync] eigh in {time.perf_counter()-t1:.3f}s  "
         f"top-5 eigenvalues: {[round(float(v), 3) for v in eigenvalues[-5:].tolist()]}")

    # Eigenvalue gap: ratio of 3rd-largest to 4th-largest eigenvalue
    lambda_3rd = float(eigenvalues[-3])
    lambda_4th = float(eigenvalues[-4])
    eigen_gap  = lambda_3rd / (abs(lambda_4th) + 1e-8)
    _log(f"[spectral_sync] Eigenvalue gap  lambda_3 / |lambda_4| = {eigen_gap:.2f}  "
         f"(ideal ≈ {N};  >>1 is good,  ~1 means noisy pairwise estimates)")

    # Step 4: top-3 eigenvectors → V in R^{3N x 3}
    V    = eigenvectors[:, -3:]          # (3N, 3)  last 3 columns = 3 largest
    V_np = V.numpy().astype(np.float64)

    # Step 5: project each 3x3 block to SO(3) via SVD
    t2 = time.perf_counter()
    R_abs   = np.zeros((N, 3, 3), dtype=np.float64)
    q_align: List[List[float]] = []

    for i in range(N):
        R_tilde = V_np[3*i:3*i+3, :]           # (3, 3)
        U, _, Wh = np.linalg.svd(R_tilde)
        d = np.linalg.det(U @ Wh)
        D = np.diag([1.0, 1.0, float(d)])
        R_abs[i] = U @ D @ Wh                  # in SO(3)

        # Alignment quaternion: R_abs[i]^T rotates particle i → canonical frame
        q_align.append(matrix_to_quaternion_wxyz(R_abs[i].T))

    _log(f"[spectral_sync] SO(3) projection in {time.perf_counter()-t2:.4f}s")
    _log(f"[spectral_sync] Total time: {time.perf_counter()-t0:.3f}s")

    return R_abs, q_align, eigenvalues.numpy(), eigen_gap


# ── 3. Pairwise Residual Diagnostic ──────────────────────────────────────────

def compute_pairwise_residual(
    R_abs: np.ndarray,
    q_pairs: List[List[float]],
    idx_i_list: List[int],
    idx_j_list: List[int],
) -> float:
    """
    Mean pairwise Frobenius consistency residual after spectral sync:

        residual = (1/P) sum_{i<j} ||R_{ij,measured} - R_{j,abs} R_{i,abs}^T||_F^2

    A small residual (<1.0 per pair) confirms self-consistency.

    Returns:
        mean_residual: float (Frobenius norm squared, averaged over pairs).
    """
    q_tensor = torch.tensor(q_pairs, dtype=torch.float64)
    R_meas   = quaternion_to_matrix_batch(q_tensor).numpy()  # (P, 3, 3)

    residuals = []
    for p, (i, j) in enumerate(zip(idx_i_list, idx_j_list)):
        R_pred = R_abs[j] @ R_abs[i].T          # R_{j,abs} R_{i,abs}^T
        diff   = R_meas[p] - R_pred
        residuals.append(float(np.sum(diff ** 2)))

    return float(np.mean(residuals))

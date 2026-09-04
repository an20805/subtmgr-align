#!/usr/bin/env python3
"""
alignot_w2_search.py
---------------------
Multi-Level Coarse-to-Fine Wasserstein Distance Search on SO(3).

HOW ROTATIONS ARE SAMPLED
==========================
Two fundamentally different parameterizations are used, one per stage:

Level 0 — Global coarse grid (ZYZ Euler angles, Haar-measure sampling):
  Every rotation R ∈ SO(3) is written as R = Rz(φ) Ry(θ) Rz(ψ).
  The uniform (Haar) measure on SO(3) in ZYZ coordinates is:
      dR = sin(θ) dφ dθ dψ  →  uniform in (φ, cos θ, ψ)
  So we do NOT sample θ on a regular grid — we sample cos(θ) uniformly in [-1,1]
  (equal-area latitude rings). The number of φ samples is proportional to sin(θ)
  (denser near the equator where there is more volume, sparser near the poles).
  Result: ≈48 rotations at 60° step, ≈4896 at 15° — approximately uniform on SO(3).

Levels 1-4 — Local cone grids (Rotation vectors = so(3) Lie algebra):
  We do NOT use Euler angles for local refinement. Euler angles have gimbal-lock
  singularities at θ=0 and θ=π. Near any given rotation, ZYZ behaves poorly.
  Instead we use the exponential map: for a rotation vector v ∈ R³,
      R = R_center · exp(v̂)
  where exp(v̂) is the Rodrigues rotation by angle |v| around axis v/|v|.
  We sample v on a regular 3D cubic lattice inside the ball |v| ≤ cone_radius:
      v ∈ {v₁e₁+v₂e₂+v₃e₃ : vᵢ ∈ linspace(-r, r, 2n+1), |v| ≤ r+Δ/2}
  This gives approximately uniform geodesic coverage of the cone neighborhood.

Default 5-level cascade: 60° → 20° → 7° → 2.5° → 1°
  L0:  48 rotations, global  (ZYZ Euler)
  L1: 179 rotations, cone 60°  (rotation vector)
  L2: 203 rotations, cone 20°  (rotation vector)
  L3: 203 rotations, cone 7°   (rotation vector)
  L4: 203 rotations, cone 2.5° (rotation vector)
  Total per particle: 2412 W2 evaluations → ≤ 1° final accuracy.

Sinkhorn convention (matches alignot_gpu_utils.py exactly):
  C[n, m, j] = ||X_ref[m] - X_rot[n][j]||²
  K = exp(-C / reg),   Kp = M * K   (uniform marginal a = b = 1/M)
  u ← 1 / (Kp v + ε),   v ← (1/M) / (Kᵀ u + ε)
  T = u[:,None] * K * v[None,:],   W2 ≈ sum(T * C)
"""

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from alignot_so3_grid import (
    generate_so3_grid,
    generate_local_so3_grid,
    count_so3_grid,
    count_local_so3_grid,
    rotation_matrix_to_quaternion_wxyz,
)

# Default 5-level cascade (angular steps in degrees)
DEFAULT_CASCADE   = [60.0, 20.0, 7.0, 2.5, 1.0]
# Sinkhorn iterations per level (more iterations for finer differences)
DEFAULT_SINKHORN  = [5,    8,    10,  12,  15]


# ── 1. Level-0: Exhaustive W2, All Rotations × All Particles ─────────────────

def evaluate_w2_over_rotations_gpu(
    src_all_np: np.ndarray,   # (N, M, 3) — particle clouds in XYZ
    ref_np:     np.ndarray,   # (M, 3)    — reference cloud in XYZ
    R_grid:     np.ndarray,   # (K, 3, 3) — candidate rotations
    reg:            float = 100.0,
    sinkhorn_iters: int   = 5,
    cost_metric:    str   = "l2_sq",
    log_every:      int   = 10,
    device: Optional[torch.device] = None,
) -> np.ndarray:              # (N, K) W2 values
    """
    Evaluate W_ε(R_k · X_i, X_ref) for ALL N particles and K rotations.

    Loop over K rotations; each iteration processes all N particles in parallel
    with one batched (N, M, M) Sinkhorn.

    Memory: (N, M, M) × 4 bytes — constant, independent of K.
    Intended for Level 0 (global grid) where all particles share the same grid.
    """
    N, M, _ = src_all_np.shape
    K = len(R_grid)

    X     = torch.from_numpy(src_all_np.astype(np.float32)).to(device)  # (N, M, 3)
    ref   = torch.from_numpy(ref_np.astype(np.float32)).to(device)      # (M, 3)
    ref_e = ref.unsqueeze(0).expand(N, M, 3)                            # (N, M, 3)
    R_all = torch.from_numpy(R_grid.astype(np.float32)).to(device)      # (K, 3, 3)

    W2 = np.zeros((N, K), dtype=np.float32)

    for k in range(K):
        # Apply rotation k to all N particles: x_row @ R.T = R @ x_col
        X_rot_k = X @ R_all[k].T                               # (N, M, 3)

        # Cost: C[n,m,j] based on cost_metric
        if cost_metric == "l1":
            C_k = torch.cdist(ref_e, X_rot_k, p=1.0)           # (N, M, M) Manhattan L1
        elif cost_metric == "l2":
            C_k = torch.cdist(ref_e, X_rot_k, p=2.0)           # (N, M, M) Euclidean L2
        else: # "l2_sq" default
            C_k = torch.cdist(ref_e, X_rot_k, p=2.0) ** 2     # (N, M, M) Squared Euclidean L2^2

        K_m = torch.exp(-C_k / reg)                             # (N, M, M)
        Kp  = float(M) * K_m                                    # uniform marginal

        u = torch.ones(N, M, device=device) / float(M)
        v = torch.ones(N, M, device=device) / float(M)
        for _ in range(sinkhorn_iters):
            KtU = torch.bmm(K_m.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)
            v   = (1.0 / float(M)) / (KtU + 1e-16)
            KpV = torch.bmm(Kp, v.unsqueeze(-1)).squeeze(-1)
            u   = 1.0 / (KpV + 1e-16)

        T_k      = u.unsqueeze(2) * K_m * v.unsqueeze(1)       # (N, M, M)
        W2[:, k] = (T_k * C_k).sum(dim=(1, 2)).cpu().numpy()

        del X_rot_k, C_k, K_m, Kp, T_k, u, v

        if log_every and (k + 1) % log_every == 0:
            best_mean = float(W2[:, :k+1].min(axis=1).mean())
            metric_label = "W1" if cost_metric in ["l1", "l2"] else "W2"
            print(f"  [L0] {k+1:4d}/{K}  best mean {metric_label}: {best_mean:.3f}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return W2   # (N, K)


# ── 2. Levels 1-4: W2 for One Particle, Batched Rotations ────────────────────

def _evaluate_w2_one_particle_batched(
    src_np: np.ndarray,   # (M, 3) — one particle cloud
    ref_np: np.ndarray,   # (M, 3) — reference cloud
    R_grid: np.ndarray,   # (K_loc, 3, 3) — candidate rotations
    reg:            float = 100.0,
    sinkhorn_iters: int   = 10,
    cost_metric:    str   = "l2_sq",
    rot_batch_size: int   = 200,
    device: Optional[torch.device] = None,
) -> np.ndarray:          # (K_loc,) W2 values
    """
    Evaluate W_ε for a SINGLE particle across K_loc rotations.

    Batches rot_batch_size rotations simultaneously through a (B, M, M) Sinkhorn.
    Memory per batch: rot_batch_size × M² × 4 bytes.
    """
    M       = src_np.shape[0]
    K_loc   = R_grid.shape[0]

    src   = torch.from_numpy(src_np.astype(np.float32)).to(device)    # (M, 3)
    ref   = torch.from_numpy(ref_np.astype(np.float32)).to(device)    # (M, 3)
    R_all = torch.from_numpy(R_grid.astype(np.float32)).to(device)    # (K_loc, 3, 3)

    W2 = np.zeros(K_loc, dtype=np.float32)

    for b_s in range(0, K_loc, rot_batch_size):
        b_e     = min(b_s + rot_batch_size, K_loc)
        B       = b_e - b_s
        R_batch = R_all[b_s:b_e]                                       # (B, 3, 3)

        # Batch-rotate: src_rot[b, m, :] = R_batch[b] @ src[m]
        # = (R_batch @ src.T.unsqueeze(0)).permute(0, 2, 1)
        src_rot = (R_batch @ src.T.unsqueeze(0)).permute(0, 2, 1)     # (B, M, 3)
        ref_e   = ref.unsqueeze(0).expand(B, M, 3)                     # (B, M, 3)

        # Cost matrix based on cost_metric
        if cost_metric == "l1":
            C = torch.cdist(ref_e, src_rot, p=1.0)                     # (B, M, M) Manhattan L1
        elif cost_metric == "l2":
            C = torch.cdist(ref_e, src_rot, p=2.0)                     # (B, M, M) Euclidean L2
        else: # "l2_sq" default
            C = torch.cdist(ref_e, src_rot, p=2.0) ** 2                 # (B, M, M) Squared Euclidean L2^2

        K_m = torch.exp(-C / reg)
        Kp  = float(M) * K_m

        u = torch.ones(B, M, device=device) / float(M)
        v = torch.ones(B, M, device=device) / float(M)
        for _ in range(sinkhorn_iters):
            KtU = torch.bmm(K_m.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)
            v   = (1.0 / float(M)) / (KtU + 1e-16)
            KpV = torch.bmm(Kp, v.unsqueeze(-1)).squeeze(-1)
            u   = 1.0 / (KpV + 1e-16)

        T          = u.unsqueeze(2) * K_m * v.unsqueeze(1)
        W2[b_s:b_e] = (T * C).sum(dim=(1, 2)).cpu().numpy()

        del src_rot, ref_e, C, K_m, Kp, T, u, v

    return W2   # (K_loc,)


# ── 3. Multi-Level Coarse-to-Fine Search (Main Entry Point) ──────────────────

def coarse_to_fine_w2_search(
    src_all_np: np.ndarray,                   # (N, M, 3) particle clouds
    ref_np:     np.ndarray,                   # (M, 3)    reference cloud
    cascade_steps:    List[float] = None,     # angular steps per level (degrees)
    cascade_sinkhorn: List[int]   = None,     # sinkhorn iters per level
    top_k:            int         = 3,        # candidates kept at each level
    reg:              float       = 100.0,
    cost_metric:      str         = "l2_sq",
    fine_rot_batch_size: int      = 200,
    device: Optional[torch.device] = None,
    logger=None,
) -> Tuple[List[List[float]], np.ndarray, np.ndarray]:
    """
    Multi-level coarse-to-fine Wasserstein rotation search.
    Supports ground metrics: 'l2_sq' (default W2), 'l1' (Manhattan W1), and 'l2' (Euclidean W1).
    """
    if cascade_steps    is None: cascade_steps    = DEFAULT_CASCADE
    if cascade_sinkhorn is None: cascade_sinkhorn = DEFAULT_SINKHORN

    n_levels = len(cascade_steps)
    assert len(cascade_sinkhorn) == n_levels, \
        "cascade_sinkhorn must have same length as cascade_steps"

    N, M, _ = src_all_np.shape
    metric_tag = "L1" if cost_metric == "l1" else ("L2" if cost_metric == "l2" else "W2")

    def _log(msg):
        if logger: logger.info(msg)
        else:      print(msg)

    t_total = time.perf_counter()

    # ── Level 0: Global exhaustive grid (ZYZ Euler, Haar measure) ────────────
    step0    = cascade_steps[0]
    K0       = count_so3_grid(step0)
    sin0     = cascade_sinkhorn[0]
    _log(f"[CTF-{metric_tag}] L0: global ZYZ grid  Δ={step0}°  K={K0}  "
         f"sinkhorn={sin0}  cost={cost_metric}  sampling all {N} particles simultaneously")

    t0    = time.perf_counter()
    R_L0  = generate_so3_grid(step0)                       # (K0, 3, 3)
    W2_L0 = evaluate_w2_over_rotations_gpu(
        src_all_np, ref_np, R_L0,
        reg=reg, sinkhorn_iters=sin0,
        cost_metric=cost_metric,
        log_every=max(1, K0 // 4),
        device=device,
    )                                                       # (N, K0)
    t_L0 = time.perf_counter() - t0
    _log(f"[CTF-{metric_tag}] L0 done in {t_L0:.2f}s  "
         f"mean best-{metric_tag}={W2_L0.min(axis=1).mean():.3f}")

    # top_k best rotation indices per particle → (N, top_k, 3, 3)
    top_k_idx      = np.argsort(W2_L0, axis=1)[:, :top_k]   # (N, top_k)
    R_candidates   = R_L0[top_k_idx]                          # (N, top_k, 3, 3)
    W2_candidates  = np.take_along_axis(
        W2_L0, top_k_idx, axis=1
    )                                                          # (N, top_k)

    _log(f"[CTF-{metric_tag}] L0 top-{top_k} per particle: "
         f"mean best {metric_tag}={W2_candidates[:, 0].mean():.3f}  "
         f"mean {top_k}th {metric_tag}={W2_candidates[:, -1].mean():.3f}")

    # ── Levels 1 .. L-2: local cone, keep top_k ──────────────────────────────
    for l in range(1, n_levels - 1):
        step = cascade_steps[l]
        cone = cascade_steps[l - 1]
        sin_l = cascade_sinkhorn[l]
        K_loc = count_local_so3_grid(cone, step)

        _log(f"[CTF-{metric_tag}] L{l}: local rotvec  Δ={step}°  cone={cone}°  "
             f"K_local={K_loc}  top_k×K={top_k*K_loc}  sinkhorn={sin_l}")

        t0 = time.perf_counter()
        R_new   = np.zeros((N, top_k, 3, 3), dtype=np.float32)
        W2_new  = np.full((N, top_k), np.inf, dtype=np.float32)

        for i in range(N):
            # Build combined local grid for all top_k candidates of particle i
            local_grids = [
                generate_local_so3_grid(R_candidates[i, k], cone, step)
                for k in range(top_k)
            ]                                                  # list of (K_loc, 3, 3)
            R_all_loc  = np.concatenate(local_grids, axis=0)  # (top_k×K_loc, 3, 3)

            W2_loc = _evaluate_w2_one_particle_batched(
                src_all_np[i], ref_np, R_all_loc,
                reg=reg,
                sinkhorn_iters=sin_l,
                cost_metric=cost_metric,
                rot_batch_size=fine_rot_batch_size,
                device=device,
            )                                                  # (top_k×K_loc,)

            # Pick top_k from ALL (top_k×K_loc) evaluated rotations
            best_idx    = np.argsort(W2_loc)[:top_k]
            R_new[i]    = R_all_loc[best_idx]
            W2_new[i]   = W2_loc[best_idx]

            if device.type == "cuda" and i % 20 == 19:
                torch.cuda.empty_cache()

        R_candidates  = R_new
        W2_candidates = W2_new
        t_l = time.perf_counter() - t0
        _log(f"[CTF-{metric_tag}] L{l} done in {t_l:.2f}s  "
             f"mean best-{metric_tag}={W2_candidates[:, 0].mean():.3f}")

    # ── Final level: local cone, keep top-1 ──────────────────────────────────
    l_final = n_levels - 1
    step_f  = cascade_steps[l_final]
    cone_f  = cascade_steps[l_final - 1]
    sin_f   = cascade_sinkhorn[l_final]
    K_loc_f = count_local_so3_grid(cone_f, step_f)

    _log(f"[CTF-{metric_tag}] L{l_final} (final): local rotvec  "
         f"Δ={step_f}°  cone={cone_f}°  K_local={K_loc_f}  "
         f"top_k×K={top_k*K_loc_f}  sinkhorn={sin_f}  → top-1")

    t0      = time.perf_counter()
    R_best  = np.zeros((N, 3, 3), dtype=np.float32)
    W2_min  = np.full(N, np.inf, dtype=np.float32)

    for i in range(N):
        local_grids = [
            generate_local_so3_grid(R_candidates[i, k], cone_f, step_f)
            for k in range(top_k)
        ]
        R_all_loc = np.concatenate(local_grids, axis=0)

        W2_loc    = _evaluate_w2_one_particle_batched(
            src_all_np[i], ref_np, R_all_loc,
            reg=reg,
            sinkhorn_iters=sin_f,
            cost_metric=cost_metric,
            rot_batch_size=fine_rot_batch_size,
            device=device,
        )

        best_i    = int(np.argmin(W2_loc))
        R_best[i] = R_all_loc[best_i]
        W2_min[i] = float(W2_loc[best_i])

        if device.type == "cuda" and i % 20 == 19:
            torch.cuda.empty_cache()

    t_f = time.perf_counter() - t0
    _log(f"[CTF-{metric_tag}] L{l_final} done in {t_f:.2f}s  "
         f"mean best-{metric_tag}={W2_min.mean():.3f}  "
         f"median={float(np.median(W2_min)):.3f}")
    _log(f"[CTF-{metric_tag}] Total search time: {time.perf_counter()-t_total:.2f}s  "
         f"Accuracy: ≤ {step_f}° geodesic error")

    # Convert R_best → [w, x, y, z] quaternions for GPUVolumeRotator
    q_best: List[List[float]] = [
        rotation_matrix_to_quaternion_wxyz(R_best[i]) for i in range(N)
    ]

    return q_best, R_best, W2_min

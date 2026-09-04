#!/usr/bin/env python3
"""
alignot_cc_search.py
---------------------
Multi-Level Coarse-to-Fine Cross-Correlation (CC) Search on SO(3).

This is the classical STA metric: for each candidate rotation R,
rotate the particle volume and measure normalized cross-correlation with
the reference volume.

    R_i* = argmax_{R ∈ G_Δ}  NCC(R · V_i,  V_ref)

where NCC is the normalized cross-correlation coefficient:

    NCC(A, B) = sum(A * B) / (||A||_F * ||B||_F)

Works on raw 3D volumes (no TRN point cloud needed). Minimization convention:
returns -NCC so that the same cascade code can uniformly minimize the score
(consistent with the W2 engine which minimizes W2 distance).

Rotation Convention (matches GPUVolumeRotator exactly):
  - R_xyz is the rotation matrix in (x, y, z) Euclidean space.
  - P = [[0,0,1],[0,1,0],[1,0,0]] permutes [z,y,x] ↔ (x,y,z).
  - R_zyx = P @ R_xyz @ P.T  —  rotation in [z,y,x] voxel array frame.
  - R_inv = R_zyx.T          —  inverse (used in grid_sample lookup).
  - Grid lookup: out_vox = R_inv @ in_vox + offset
    where offset = center - R_inv @ center  (rotate around box center).
  - Grid coordinates normalized to [-1, 1] as required by F.grid_sample,
    with (x_norm, y_norm, z_norm) = (W, H, D) axes respectively.

Stage A (global grid):
  - For each rotation k (loop over K_coarse):
    Build one (D,H,W,3) grid and apply it to ALL N particles simultaneously
    via a single batched F.grid_sample call (input: (N,1,D,H,W)).
    Memory = N × D³ × 4 bytes  — constant per rotation, independent of K.

Stages 1-N (local cone grids):
  - Per particle: batch B_rot rotations through one (B_rot,D,H,W,3) grid set.
    Memory = B_rot × D³ × (1 + 3) × 4 bytes per batch.
"""

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR.parent / "alignOT"))

import coords as _coords
from alignot_so3_grid import (
    generate_so3_grid,
    generate_local_so3_grid,
    count_so3_grid,
    count_local_so3_grid,
    rotation_matrix_to_quaternion_wxyz,
)

# Permutation matrix: [z,y,x] ↔ (x,y,z)
_P = np.array([[0, 0, 1],
               [0, 1, 0],
               [1, 0, 0]], dtype=np.float64)


# ── Grid Builder ──────────────────────────────────────────────────────────────

def _build_grid_from_R_xyz(
    R_xyz: np.ndarray,   # (3,3) rotation matrix in Euclidean (x,y,z) frame
    box_size: int,
    device: torch.device,
) -> torch.Tensor:       # (D, H, W, 3)  normalized grid for F.grid_sample
    """
    Build a F.grid_sample coordinate grid for rotation R_xyz, rotating
    around the box center (no centroid correction).

    The grid encodes: for each output voxel position p_out (in [z,y,x]),
    the corresponding input voxel position p_in = R_inv @ (p_out - c) + c,
    normalized to [-1, 1].

    grid[..., 0] = x_norm (W axis)
    grid[..., 1] = y_norm (H axis)
    grid[..., 2] = z_norm (D axis)
    """
    N = box_size
    c = N / 2.0   # box center matching GPUVolumeRotator (N / 2.0)

    # Convert to [z,y,x] voxel frame and invert
    R_zyx = _P @ R_xyz @ _P.T
    R_inv = R_zyx.T   # (3,3), float64

    # offset so that rotation is around the box center
    c_vec = np.array([c, c, c], dtype=np.float64)
    offset = c_vec - R_inv @ c_vec   # (3,) in [z,y,x]

    R_inv_t = torch.from_numpy(R_inv.astype(np.float32)).to(device)    # (3,3)
    offset_t = torch.from_numpy(offset.astype(np.float32)).to(device)  # (3,)

    # Output voxel coordinates [z,y,x]: (D*H*W, 3)
    z = torch.arange(N, dtype=torch.float32, device=device)
    y = torch.arange(N, dtype=torch.float32, device=device)
    x = torch.arange(N, dtype=torch.float32, device=device)
    gz, gy, gx = torch.meshgrid(z, y, x, indexing="ij")
    coords_zyx = torch.stack([gz, gy, gx], dim=-1).reshape(-1, 3)  # (D*H*W, 3)

    # Map output coords → input coords: in_zyx = R_inv @ out_zyx + offset
    in_zyx = coords_zyx @ R_inv_t.T + offset_t                     # (D*H*W, 3)

    # Normalize to [-1, 1]  (align_corners=True convention)
    norm = (2.0 / (N - 1)) * in_zyx - 1.0                          # (D*H*W, 3)

    # grid_sample expects (x, y, z) = (W, H, D) ordering
    # in_zyx[:,0]=z → D axis → norm index 2
    # in_zyx[:,1]=y → H axis → norm index 1
    # in_zyx[:,2]=x → W axis → norm index 0
    grid_xyz = torch.stack([norm[:, 2], norm[:, 1], norm[:, 0]], dim=-1)  # (D*H*W, 3)
    return grid_xyz.reshape(N, N, N, 3)                             # (D, H, W, 3)


# ── Stage A: All N particles × K rotations ────────────────────────────────────

def evaluate_cc_over_rotations_gpu(
    vols_all_np: List[np.ndarray],      # list of N (D,H,W) volumes
    vol_ref_np:  np.ndarray,            # (D,H,W) reference volume
    R_grid:      np.ndarray,            # (K, 3, 3) candidate rotations
    part_batch_size: int = 10,          # N-particles batch size (memory knob)
    log_every:   int  = 10,
    device: Optional[torch.device] = None,
) -> np.ndarray:                        # (N, K)  negative NCC (lower = better)
    """
    Evaluate -NCC(R_k · V_i, V_ref) for ALL N particles and K rotations.

    Loop over K rotations; each iteration applies one rotation grid to all N
    particles via batched F.grid_sample and computes dot-product with reference.

    Memory: N × D³ × 4 bytes per rotation (constant, independent of K).

    Returns negative NCC so downstream code can uniformly minimize.
    """
    N   = len(vols_all_np)
    K   = R_grid.shape[0]
    D, H, W = vol_ref_np.shape

    # Pre-load all particle volumes to GPU: (N, 1, D, H, W)
    vols_gpu = torch.stack([
        torch.from_numpy(v.astype(np.float32)) for v in vols_all_np
    ]).unsqueeze(1).to(device)   # (N, 1, D, H, W)

    # Reference flat + norm
    ref_gpu  = torch.from_numpy(vol_ref_np.astype(np.float32)).to(device)  # (D,H,W)
    ref_flat = ref_gpu.reshape(1, -1)   # (1, D*H*W)
    ref_norm = ref_flat.norm()          # scalar

    neg_ncc = np.zeros((N, K), dtype=np.float32)

    for k, R_k in enumerate(R_grid):
        grid_k = _build_grid_from_R_xyz(R_k, D, device)           # (D, H, W, 3)

        # Expand grid to all N particles: (N, D, H, W, 3)
        grid_batch = grid_k.unsqueeze(0).expand(N, -1, -1, -1, -1)

        rot_vols = F.grid_sample(
            vols_gpu, grid_batch,
            mode="bilinear", padding_mode="zeros", align_corners=True
        )                                                           # (N, 1, D, H, W)
        rot_flat = rot_vols.reshape(N, -1)                         # (N, D*H*W)

        # NCC per particle
        numer       = (rot_flat * ref_flat).sum(dim=1)             # (N,)
        rot_norms   = rot_flat.norm(dim=1).clamp(min=1e-10)        # (N,)
        ncc         = numer / (ref_norm * rot_norms)               # (N,)

        neg_ncc[:, k] = (-ncc).cpu().numpy()

        del grid_batch, rot_vols, rot_flat, numer, rot_norms, ncc
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if log_every and (k + 1) % log_every == 0:
            best_mean = float(-neg_ncc[:, :k+1].min(axis=1).mean())
            print(f"  [L0] {k+1:4d}/{K}  best mean NCC: {best_mean:.4f}")

    del vols_gpu, ref_gpu, ref_flat
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return neg_ncc   # (N, K),  negative NCC


# ── Stages 1-L: One Particle × K_local Rotations ─────────────────────────────

def _evaluate_cc_one_particle_batched(
    vol_np:     np.ndarray,   # (D, H, W) one particle volume
    vol_ref_np: np.ndarray,   # (D, H, W) reference volume
    R_grid:     np.ndarray,   # (K_loc, 3, 3) candidate rotations
    rot_batch_size: int = 20,
    device: Optional[torch.device] = None,
) -> np.ndarray:              # (K_loc,)  negative NCC (lower = better)
    """
    Evaluate -NCC for a SINGLE particle across K_loc rotations.

    Batches rot_batch_size rotations simultaneously through one
    F.grid_sample call with input (B_rot, 1, D, H, W).

    Memory per batch: B_rot × D³ × 4 bytes (input) + B_rot × D³ × 3 × 4 bytes (grid).
    For B_rot=20, D=91: ~23 MB input + ~68 MB grid = ~91 MB total.
    """
    K_loc   = R_grid.shape[0]
    D, H, W = vol_np.shape

    vol_gpu  = torch.from_numpy(vol_np.astype(np.float32)).to(device)      # (D,H,W)
    ref_gpu  = torch.from_numpy(vol_ref_np.astype(np.float32)).to(device)  # (D,H,W)
    ref_flat = ref_gpu.reshape(1, -1)                                        # (1, D*H*W)
    ref_norm = ref_flat.norm()

    neg_ncc = np.zeros(K_loc, dtype=np.float32)

    for b_s in range(0, K_loc, rot_batch_size):
        b_e = min(b_s + rot_batch_size, K_loc)
        B   = b_e - b_s

        # Build grid for each rotation in this batch: (B, D, H, W, 3)
        grids = torch.stack([
            _build_grid_from_R_xyz(R_grid[b], D, device)
            for b in range(b_s, b_e)
        ])                                                           # (B, D, H, W, 3)

        # Repeat particle volume B times: (B, 1, D, H, W)
        vol_batch = vol_gpu.unsqueeze(0).unsqueeze(0).expand(B, 1, D, H, W)

        rot_vols = F.grid_sample(
            vol_batch, grids,
            mode="bilinear", padding_mode="zeros", align_corners=True
        )                                                            # (B, 1, D, H, W)
        rot_flat = rot_vols.reshape(B, -1)                          # (B, D*H*W)

        numer     = (rot_flat * ref_flat).sum(dim=1)                # (B,)
        rot_norms = rot_flat.norm(dim=1).clamp(min=1e-10)           # (B,)
        ncc       = numer / (ref_norm * rot_norms)                  # (B,)

        neg_ncc[b_s:b_e] = (-ncc).cpu().numpy()

        del grids, vol_batch, rot_vols, rot_flat, numer, rot_norms, ncc

    del vol_gpu, ref_gpu, ref_flat
    return neg_ncc   # (K_loc,),  negative NCC


# ── Main Entry Point: Multi-Level Cascade ────────────────────────────────────

def coarse_to_fine_cc_search(
    vols_all_np:   List[np.ndarray],          # list of N (D,H,W) raw volumes
    vol_ref_np:    np.ndarray,                # (D,H,W) reference volume
    cascade_steps: List[float],               # angular steps per level (degrees)
    top_k:         int          = 3,          # candidates kept per intermediate level
    part_batch_size: int        = 10,         # Stage A: N-particle batch (memory knob)
    rot_batch_size:  int        = 20,         # Stages 1+: rotation batch per particle
    device: Optional[torch.device] = None,
    logger = None,
) -> Tuple[List[List[float]], np.ndarray, np.ndarray]:
    """
    Multi-level coarse-to-fine cross-correlation rotation search.

    Level 0 — Global uniform SO(3) grid at cascade_steps[0]:
      ZYZ Euler angles, Haar measure (cos θ uniform). All K₀ rotations
      evaluated for ALL N particles simultaneously (batched grid_sample).
      Keeps top_k candidates per particle.

    Levels 1 .. L-2 — Local cone grids (rotation vector / Lie algebra):
      For each of top_k candidates per particle, evaluates all (top_k × K_local)
      rotations using batched CC. Keeps top_k overall best candidates.

    Level L-1 (final) — Local cone grid → top-1 per particle.

    Args:
        vols_all_np:     List of N (D,H,W) float32 raw particle volumes.
        vol_ref_np:      (D,H,W) float32 reference volume.
        cascade_steps:   Angular step per level in degrees.
                         e.g. [60, 20, 7, 2.5, 1].
        top_k:           Number of candidates kept at each intermediate level.
        part_batch_size: Number of particles processed per batch in Stage A.
        rot_batch_size:  Rotation batch size for local levels (Stages 1+).
        device:          torch.device.
        logger:          Python logger or None.

    Returns:
        q_best:  List of N [w, x, y, z] quaternions for GPUVolumeRotator.
        R_best:  (N, 3, 3) float32 rotation matrices.
        ncc_best: (N,) best NCC score per particle (positive, higher = better).
    """
    n_levels = len(cascade_steps)
    N        = len(vols_all_np)
    D        = vols_all_np[0].shape[0]

    def _log(msg):
        if logger: logger.info(msg)
        else:      print(msg)

    t_total = time.perf_counter()

    # ── Level 0: Global exhaustive grid (ZYZ Euler, Haar measure) ────────────
    step0 = cascade_steps[0]
    K0    = count_so3_grid(step0)
    _log(f"[CTF-CC] L0: global ZYZ grid  Δ={step0}°  K={K0}  "
         f"all {N} particles simultaneously")

    t0     = time.perf_counter()
    R_L0   = generate_so3_grid(step0)                              # (K0, 3, 3)
    negNCC = evaluate_cc_over_rotations_gpu(
        vols_all_np, vol_ref_np, R_L0,
        part_batch_size=part_batch_size,
        log_every=max(1, K0 // 4),
        device=device,
    )                                                               # (N, K0)
    t_L0 = time.perf_counter() - t0
    best_ncc_L0 = -negNCC.min(axis=1).mean()
    _log(f"[CTF-CC] L0 done in {t_L0:.2f}s  mean best NCC={best_ncc_L0:.4f}")

    # top_k best rotation indices per particle → (N, top_k, 3, 3)
    top_k_idx    = np.argsort(negNCC, axis=1)[:, :top_k]          # (N, top_k) ascending = best
    R_candidates = R_L0[top_k_idx]                                 # (N, top_k, 3, 3)
    negNCC_cands = np.take_along_axis(negNCC, top_k_idx, axis=1)   # (N, top_k)
    _log(f"[CTF-CC] L0 top-{top_k}: mean best NCC={-negNCC_cands[:,0].mean():.4f}  "
         f"mean {top_k}th NCC={-negNCC_cands[:,-1].mean():.4f}")

    # ── Levels 1 .. L-2: local cone, keep top_k ──────────────────────────────
    for l in range(1, n_levels - 1):
        step  = cascade_steps[l]
        cone  = cascade_steps[l - 1]
        K_loc = count_local_so3_grid(cone, step)
        _log(f"[CTF-CC] L{l}: local rotvec  Δ={step}°  cone={cone}°  "
             f"K_local={K_loc}  top_k×K={top_k * K_loc}")

        t0     = time.perf_counter()
        R_new  = np.zeros((N, top_k, 3, 3), dtype=np.float32)
        nN_new = np.full((N, top_k), np.inf, dtype=np.float32)

        for i in range(N):
            local_grids = [
                generate_local_so3_grid(R_candidates[i, k], cone, step)
                for k in range(top_k)
            ]
            R_all_loc = np.concatenate(local_grids, axis=0)       # (top_k×K_loc, 3, 3)

            neg_scores = _evaluate_cc_one_particle_batched(
                vols_all_np[i], vol_ref_np, R_all_loc,
                rot_batch_size=rot_batch_size,
                device=device,
            )                                                       # (top_k×K_loc,)

            best_idx     = np.argsort(neg_scores)[:top_k]
            R_new[i]     = R_all_loc[best_idx]
            nN_new[i]    = neg_scores[best_idx]

            if device.type == "cuda" and i % 20 == 19:
                torch.cuda.empty_cache()

        R_candidates = R_new
        negNCC_cands = nN_new
        t_l = time.perf_counter() - t0
        _log(f"[CTF-CC] L{l} done in {t_l:.2f}s  "
             f"mean best NCC={-negNCC_cands[:,0].mean():.4f}")

    # ── Final level: local cone → top-1 ──────────────────────────────────────
    l_f    = n_levels - 1
    step_f = cascade_steps[l_f]
    cone_f = cascade_steps[l_f - 1]
    K_loc_f = count_local_so3_grid(cone_f, step_f)
    _log(f"[CTF-CC] L{l_f} (final): local rotvec  Δ={step_f}°  cone={cone_f}°  "
         f"K_local={K_loc_f}  top_k×K={top_k * K_loc_f}  → top-1")

    t0      = time.perf_counter()
    R_best  = np.zeros((N, 3, 3), dtype=np.float32)
    ncc_best = np.full(N, -np.inf, dtype=np.float32)  # positive NCC (higher = better)

    for i in range(N):
        local_grids = [
            generate_local_so3_grid(R_candidates[i, k], cone_f, step_f)
            for k in range(top_k)
        ]
        R_all_loc = np.concatenate(local_grids, axis=0)

        neg_scores = _evaluate_cc_one_particle_batched(
            vols_all_np[i], vol_ref_np, R_all_loc,
            rot_batch_size=rot_batch_size,
            device=device,
        )

        best_i      = int(np.argmin(neg_scores))
        R_best[i]   = R_all_loc[best_i]
        ncc_best[i] = float(-neg_scores[best_i])   # store positive NCC

        if device.type == "cuda" and i % 20 == 19:
            torch.cuda.empty_cache()

    t_f = time.perf_counter() - t0
    _log(f"[CTF-CC] L{l_f} done in {t_f:.2f}s  "
         f"mean NCC={ncc_best.mean():.4f}  median={float(np.median(ncc_best)):.4f}")
    _log(f"[CTF-CC] Total search time: {time.perf_counter()-t_total:.2f}s  "
         f"Accuracy: ≤ {step_f}° geodesic error")

    # Convert R_best → [w, x, y, z] quaternions for GPUVolumeRotator
    q_best: List[List[float]] = [
        rotation_matrix_to_quaternion_wxyz(R_best[i]) for i in range(N)
    ]

    return q_best, R_best, ncc_best

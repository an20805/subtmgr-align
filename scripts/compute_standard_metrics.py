#!/usr/bin/env python3
"""
compute_standard_metrics.py
---------------------------
Official standardized evaluation tool for subtomogram alignment runs.

Evaluates an aligned consensus reference map and its corresponding quaternions against
the raw subtomograms using two standard metrics:

1. Cross-Correlation (CC / NCC):
   - Evaluated on continuous 3D density volumes rotated in their native bounding box.
   - Computes normalized dot-product: <V_rot, V_ref> / (||V_rot|| * ||V_ref||).

2. 2-Wasserstein Distance (W2):
   - Evaluated on centered TRN point clouds (M=500, sigma=2.0, threshold=0.70).
   - Computes optimal transport cost: <T*, C> using Sinkhorn OT with regularization --reg.

Usage:
  python scripts/compute_standard_metrics.py \\
      --ref path/to/reference_round_03.mrc \\
      --quats path/to/quaternions.csv \\
      [--cluster_dir Data/0.1/Cluster_0] \\
      [--reg 100.0] \\
      [--device cuda]
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "alignOT" / "src"))
sys.path.insert(0, str(REPO_ROOT / "alignOT" / "alignOT"))

import alignot_utils
import coords as _coords

_P = np.array([[0, 0, 1],
               [0, 1, 0],
               [1, 0, 0]], dtype=np.float64)


def build_native_box_grid(R_xyz: np.ndarray, box_size: int, device: torch.device) -> torch.Tensor:
    """Build coordinate grid for 3D rotation around native box center."""
    c = box_size / 2.0
    R_zyx = _P @ R_xyz @ _P.T
    R_inv = R_zyx.T
    c_vec = np.array([c, c, c], dtype=np.float64)
    offset = c_vec - R_inv @ c_vec

    lin = np.arange(box_size, dtype=np.float32)
    Z, Y, X = np.meshgrid(lin, lin, lin, indexing="ij")
    p_tgt = np.stack([Z, Y, X], axis=-1).reshape(-1, 3)

    p_src = (p_tgt @ R_inv.T) + offset
    p_src_norm = (p_src / (c - 0.5)) - 1.0

    x_norm = p_src_norm[:, 2]
    y_norm = p_src_norm[:, 1]
    z_norm = p_src_norm[:, 0]
    grid = np.stack([x_norm, y_norm, z_norm], axis=-1).reshape(box_size, box_size, box_size, 3)
    return torch.from_numpy(grid.astype(np.float32)).to(device)


def _sample_particle_worker(args):
    idx, path = args
    v = alignot_utils.load_mrc(path, permissive=True)
    prep = alignot_utils.prepare_sampling_volume(v, sigma=2.0)
    pts = alignot_utils.sample_volume_in_memory(prep, threshold=0.70, num_points=500, random_seed=42 + idx)
    pts_c = pts - pts.mean(axis=0)
    pts_xyz = pts_c[:, [2, 1, 0]].astype(np.float32)
    return idx, pts_xyz


def parse_args():
    p = argparse.ArgumentParser(description="Standardized CC and W2 Metric Evaluator")
    p.add_argument("--ref", required=True, help="Path to reference .mrc volume")
    p.add_argument("--quats", required=True, help="Path to quaternions.csv (or metrics_particles.csv)")
    p.add_argument("--cluster_dir", default="Data/0.1/Cluster_0", help="Directory with raw subtomograms")
    p.add_argument("--reg", type=float, default=100.0, help="Sinkhorn regularization for W2 (default: 100.0)")
    p.add_argument("--sinkhorn_iters", type=int, default=15, help="Number of Sinkhorn iterations for W2 (default: 15)")
    p.add_argument("--device", default="cuda", help="Computation device: 'cuda' or 'cpu'")
    p.add_argument("--output_csv", default=None, help="Optional path to save per-particle metrics CSV")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ref_path = Path(args.ref)
    quats_path = Path(args.quats)
    cluster_dir = REPO_ROOT / args.cluster_dir

    if not ref_path.is_file():
        sys.exit(f"[ERROR] Reference file not found: {ref_path}")

    if quats_path.is_dir():
        for candidate in ["quaternions.csv", "metrics_particles_round_03.csv", "metrics_particles.csv"]:
            if (quats_path / candidate).is_file():
                quats_path = quats_path / candidate
                break

    if not quats_path.is_file():
        # Check if parent dir has metrics_particles_round_03.csv
        parent = quats_path.parent
        for candidate in ["quaternions.csv", "metrics_particles_round_03.csv", "metrics_particles.csv"]:
            if (parent / candidate).is_file():
                quats_path = parent / candidate
                break

    if not quats_path.is_file():
        sys.exit(f"[ERROR] Quaternions file not found: {quats_path}")

    # Load Quaternions
    df_q = pd.read_csv(quats_path)
    if "q_w" not in df_q.columns:
        sys.exit(f"[ERROR] Quaternions CSV must contain ['q_w', 'q_x', 'q_y', 'q_z'] columns.")
    quats = df_q[["q_w", "q_x", "q_y", "q_z"]].values
    N = len(quats)

    print("=" * 76, flush=True)
    print(" STANDARDIZED SUBTOMOGRAM ALIGNMENT EVALUATION", flush=True)
    print("=" * 76, flush=True)
    print(f"Reference Map    : {ref_path}", flush=True)
    print(f"Quaternions File : {quats_path} ({N} particles)", flush=True)
    print(f"Subtomograms Dir : {cluster_dir}", flush=True)
    print(f"Sinkhorn Reg (ε) : {args.reg}", flush=True)
    print(f"Device           : {device}", flush=True)
    print("-" * 76, flush=True)

    # 1. Load Raw Volumes
    paths = sorted([p for p in cluster_dir.iterdir() if p.suffix == ".mrc" and "Zone" not in p.name],
                   key=lambda p: int(p.stem))[:N]
    if len(paths) < N:
        sys.exit(f"[ERROR] Found only {len(paths)} volumes in {cluster_dir}, but quaternions has {N} rows.")

    raw_vols = [alignot_utils.load_mrc(p, permissive=True) for p in paths]
    ref_vol = alignot_utils.load_mrc(ref_path, permissive=True)
    box_size = raw_vols[0].shape[0]

    # 2. Compute Native 3D Continuous Box CC
    print("1. Computing Native 3D Continuous Box Cross-Correlation (CC)...", flush=True)
    t0_cc = time.perf_counter()
    vols_gpu = torch.stack([torch.from_numpy(v.astype(np.float32)) for v in raw_vols]).unsqueeze(1).to(device)
    ref_gpu = torch.from_numpy(ref_vol.astype(np.float32)).to(device)
    ref_flat = ref_gpu.reshape(1, -1)
    ref_norm = ref_flat.norm().clamp(min=1e-10)

    R_list = [_coords.quaternion_to_R(list(q)) for q in quats]
    grids = torch.stack([build_native_box_grid(R, box_size, device) for R in R_list])

    rot_vols = F.grid_sample(vols_gpu, grids, mode="bilinear", padding_mode="zeros", align_corners=True)
    rot_flat = rot_vols.reshape(N, -1)
    numer = (rot_flat * ref_flat).sum(dim=1)
    denom = ref_norm * rot_flat.norm(dim=1).clamp(min=1e-10)
    cc_scores = (numer / denom).cpu().numpy()
    dt_cc = time.perf_counter() - t0_cc

    # 3. Compute Standard W2 (Centered TRN Point Transport)
    print("2. Sampling TRN point clouds and computing 2-Wasserstein Distance (W2)...", flush=True)
    t0_w2 = time.perf_counter()
    tasks = [(idx, p) for idx, p in enumerate(paths)]
    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as ex:
        sampled_results = list(ex.map(_sample_particle_worker, tasks))
    sampled_results = sorted(sampled_results, key=lambda x: x[0])
    particle_trn_xyz = [x[1] for x in sampled_results]

    # Reference TRN
    ref_prep = alignot_utils.prepare_sampling_volume(ref_vol, sigma=2.0)
    pts_ref = alignot_utils.sample_volume_in_memory(ref_prep, threshold=0.70, num_points=500, random_seed=42)
    pts_ref_c = pts_ref - pts_ref.mean(axis=0)
    ref_xyz = pts_ref_c[:, [2, 1, 0]].astype(np.float32)
    ref_xyz_t = torch.from_numpy(ref_xyz).to(device)

    M = 500
    u_init = torch.ones(M, device=device) / float(M)
    v_init = torch.ones(M, device=device) / float(M)
    w2_scores = []

    for i in range(N):
        R_t = torch.from_numpy(R_list[i].astype(np.float32)).to(device)
        src_xyz_t = torch.from_numpy(particle_trn_xyz[i]).to(device)
        src_rot = src_xyz_t @ R_t.T

        C = torch.cdist(ref_xyz_t, src_rot) ** 2
        K = torch.exp(-C / args.reg)
        Kp = float(M) * K

        u = u_init.clone()
        v = v_init.clone()
        for _ in range(args.sinkhorn_iters):
            KtU = torch.mv(K.t(), u)
            v = (1.0 / float(M)) / (KtU + 1e-16)
            KpV = torch.mv(Kp, v)
            u = 1.0 / (KpV + 1e-16)

        T = torch.outer(u, v) * K
        w2_val = float((T * C).sum().cpu().item())
        w2_scores.append(w2_val)

    w2_scores = np.array(w2_scores)
    dt_w2 = time.perf_counter() - t0_w2

    # 4. Print Summary Report
    print("\n" + "=" * 76, flush=True)
    print(" STANDARDIZED EVALUATION REPORT", flush=True)
    print("=" * 76, flush=True)
    print(f"Number of Particles (N) : {N}", flush=True)
    print(f"Reference Volume        : {ref_path.name}", flush=True)
    print("-" * 76, flush=True)
    print(f"1. Cross-Correlation (CC) [Native 3D Continuous Box]:")
    print(f"   - Mean CC            : {cc_scores.mean():.4f}")
    print(f"   - CC Range           : [{cc_scores.min():.4f}, {cc_scores.max():.4f}]")
    print()
    print(f"2. 2-Wasserstein Distance (W2) [Centered TRN Points, reg={args.reg:g}, iters={args.sinkhorn_iters}]:")
    print(f"   - Mean W2            : {w2_scores.mean():.2f}")
    print(f"   - W2 Range           : [{w2_scores.min():.2f}, {w2_scores.max():.2f}]")
    print("=" * 76, flush=True)

    # 5. Save Output CSV if requested
    if args.output_csv:
        out_csv_path = Path(args.output_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_out = pd.DataFrame({
            "subtomogram_id": range(N),
            "file_name": [p.name for p in paths],
            "cc_score": cc_scores,
            "w2_distance": w2_scores,
        })
        df_out.to_csv(out_csv_path, index=False)
        print(f"Saved per-particle metrics to: {out_csv_path}", flush=True)


if __name__ == "__main__":
    main()


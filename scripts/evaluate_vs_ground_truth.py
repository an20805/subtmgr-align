#!/usr/bin/env python3
"""
evaluate_vs_ground_truth.py
----------------------------
Standardized evaluation tool to benchmark any subtomogram average against
the ground truth macromolecular reference (Data/0.mrc).

Calculates:
  1. Raw Pearson CC with Ground Truth (unaligned)
  2. Optimal 3-DOF SO(3) rigid registration against Ground Truth
  3. Aligned Pearson CC (masked with smooth spherical cosine envelope)
  4. Full Fourier Shell Correlation (FSC) curve against Ground Truth (before & after alignment)
  5. Resolution determination at FSC = 0.500 and FSC = 0.143 thresholds (in Å)

Usage:
  # Run with defaults:
  python scripts/evaluate_vs_ground_truth.py

  # Evaluate a specific volume:
  python scripts/evaluate_vs_ground_truth.py \\
      --ref alignOT/outputs/gpu_runs/run_.../round_05/ref_combined_round_05.mrc \\
      --ground_truth Data/0.mrc \\
      --apix 4.0 \\
      --output_dir experiments/my_evaluation
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as ScipyR
import torch
import torch.nn.functional as F


# ── Spherical Cosine Envelope Mask ───────────────────────────────────────────

def soft_sphere(D: int, frac_in: float = 0.40, frac_out: float = 0.48) -> np.ndarray:
    """
    Smooth spherical envelope to eliminate box-corner noise and boundary interpolation artifacts.
    Smoothly tapers from 1.0 (inside frac_in) to 0.0 (outside frac_out) via a raised cosine.
    """
    c = D // 2
    ax = np.arange(D) - c
    r = np.sqrt(ax[:, None, None]**2 + ax[None, :, None]**2 + ax[None, None, :]**2)
    m = np.clip((frac_out * D - r) / (frac_out * D - frac_in * D), 0, 1)
    return (0.5 - 0.5 * np.cos(np.pi * m)).astype(np.float64)


# ── MRC Load / Save ───────────────────────────────────────────────────────────

def load_mrc(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"MRC file not found: {path}")
    with mrcfile.open(str(path), permissive=True) as m:
        return np.array(m.data, dtype=np.float32)


def save_mrc(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(np.asarray(arr, dtype=np.float32))


# ── Correlation Metrics ───────────────────────────────────────────────────────

def pearson_cc(A: np.ndarray, B: np.ndarray, mask: np.ndarray = None) -> float:
    """Computes real-space Pearson cross-correlation."""
    if mask is not None:
        a = (A * mask).ravel()
        b = (B * mask).ravel()
    else:
        a = A.ravel()
        b = B.ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def masked_corr_torch(A_t: torch.Tensor, B_batch_t: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
    """Batched masked Pearson correlation on GPU."""
    B_sz = B_batch_t.shape[0]
    A_m = (A_t * mask_t).reshape(1, -1)
    A_m = A_m - A_m.mean(dim=-1, keepdim=True)
    A_norm = torch.norm(A_m, dim=-1, keepdim=True)

    B_m = (B_batch_t * mask_t).reshape(B_sz, -1)
    B_m = B_m - B_m.mean(dim=-1, keepdim=True)
    B_norm = torch.norm(B_m, dim=-1, keepdim=True)

    denom = (A_norm * B_norm + 1e-16).squeeze(-1)
    dots = (A_m * B_m).sum(dim=-1)
    return dots / denom


# ── GPU SO(3) Rigid Registration ──────────────────────────────────────────────

def rotate_volume_batch_gpu(vol_t: torch.Tensor, R_mats_t: torch.Tensor, grid_xyz: torch.Tensor) -> torch.Tensor:
    """Rotates volume on GPU using 3D trilinear interpolation (F.grid_sample)."""
    B = R_mats_t.shape[0]
    D = vol_t.shape[-1]
    grid_rot = torch.matmul(grid_xyz, R_mats_t).reshape(B, D, D, D, 3)
    vol_batch = vol_t.expand(B, 1, D, D, D)
    rot_vols = F.grid_sample(vol_batch, grid_rot, mode="bilinear", padding_mode="zeros", align_corners=True)
    return rot_vols.squeeze(1)


def register_so3_to_ground_truth(vol: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                                 n_coarse: int = 300, device: torch.device = torch.device("cuda")):
    """
    Finds the optimal rotation matrix R in SO(3) that superimposes vol onto gt.
    Two-phase search: Coarse global sweep on SO(3) + Multiscale Lie-algebra refinement (8°, 3°, 1°).
    """
    D = gt.shape[-1]
    rng = np.random.default_rng(0)

    # 1. Coarse Candidate Rotations
    cands_rotvec = [np.zeros(3)]  # Identity (0°)
    for ang in (5, 10, 15, 20, 30, 45):
        for _ in range(20):
            ax = rng.normal(size=3)
            ax /= np.linalg.norm(ax)
            cands_rotvec.append(np.deg2rad(ang) * ax)
    for _ in range(n_coarse):
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        cands_rotvec.append(rng.uniform(0, np.pi) * ax)

    Rs_np = ScipyR.from_rotvec(np.array(cands_rotvec)).as_matrix()

    coords = torch.linspace(-1, 1, D, device=device)
    gz, gy, gx = torch.meshgrid(coords, coords, coords, indexing="ij")
    grid_xyz = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    gt_t = torch.tensor(gt, dtype=torch.float32, device=device).unsqueeze(0)
    vol_t = torch.tensor(vol, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)

    CH = 64
    best_c = -2.0
    best_R = np.eye(3)

    for s in range(0, len(Rs_np), CH):
        chunk_np = Rs_np[s:s + CH]
        chunk_t = torch.tensor(chunk_np, dtype=torch.float32, device=device)
        rot_b = rotate_volume_batch_gpu(vol_t, chunk_t, grid_xyz)
        corrs = masked_corr_torch(gt_t, rot_b, mask_t).cpu().numpy()
        max_idx = np.argmax(corrs)
        if corrs[max_idx] > best_c:
            best_c = float(corrs[max_idx])
            best_R = chunk_np[max_idx]

    # 2. Hierarchical Multiscale Refinement (8°, 3°, 1°)
    for scale in (8, 3, 1):
        improved = True
        while improved:
            improved = False
            base_rotvecs = []
            for k in range(3):
                for sgn in (1, -1):
                    v = np.zeros(3)
                    v[k] = sgn * np.deg2rad(scale)
                    base_rotvecs.append(v)
            base_Rs = ScipyR.from_rotvec(np.array(base_rotvecs)).as_matrix()
            trial_Rs = np.array([best_R @ b for b in base_Rs])
            trial_t = torch.tensor(trial_Rs, dtype=torch.float32, device=device)
            rot_b = rotate_volume_batch_gpu(vol_t, trial_t, grid_xyz)
            corrs = masked_corr_torch(gt_t, rot_b, mask_t).cpu().numpy()
            max_idx = np.argmax(corrs)
            if corrs[max_idx] > best_c + 1e-5:
                best_c = float(corrs[max_idx])
                best_R = trial_Rs[max_idx]
                improved = True

    best_R_t = torch.tensor(best_R, dtype=torch.float32, device=device).unsqueeze(0)
    vol_aligned = rotate_volume_batch_gpu(vol_t, best_R_t, grid_xyz).cpu().numpy()[0]
    rot_angle_deg = float(np.linalg.norm(ScipyR.from_matrix(best_R).as_rotvec(degrees=True)))

    return vol_aligned, best_c, best_R, rot_angle_deg


# ── Fourier Shell Correlation (FSC) ───────────────────────────────────────────

def compute_fsc(vol1: np.ndarray, vol2: np.ndarray, apix: float = 4.0, mask: np.ndarray = None):
    """
    Computes Fourier Shell Correlation between two volumes.
    Optionally applies real-space masking before FFT.
    """
    assert vol1.shape == vol2.shape
    D = vol1.shape[0]
    n_shells = D // 2

    v1 = vol1.astype(np.float64) * (mask if mask is not None else 1.0)
    v2 = vol2.astype(np.float64) * (mask if mask is not None else 1.0)

    F1 = np.fft.fftn(v1)
    F2 = np.fft.fftn(v2)

    freqs = [np.fft.fftfreq(s) for s in vol1.shape]
    fz, fy, fx = np.meshgrid(freqs[0], freqs[1], freqs[2], indexing="ij")
    r = np.sqrt(fz**2 + fy**2 + fx**2)

    shell_width = 0.5 / n_shells
    shell_centers = np.arange(n_shells) * shell_width + shell_width / 2

    fsc_vals = np.zeros(n_shells)
    n_voxels = np.zeros(n_shells, dtype=int)

    for i in range(n_shells):
        lo = i * shell_width
        hi = lo + shell_width
        shell_m = (r >= lo) & (r < hi) & (r > 0)
        n_vox = int(shell_m.sum())
        n_voxels[i] = n_vox
        if n_vox == 0:
            continue
        f1_sh = F1[shell_m]
        f2_sh = F2[shell_m]
        num = np.real(np.sum(f1_sh * np.conj(f2_sh)))
        den = np.sqrt(np.sum(np.abs(f1_sh)**2) * np.sum(np.abs(f2_sh)**2))
        fsc_vals[i] = num / den if den > 0 else 0.0

    def get_res(thresh):
        for i in range(len(fsc_vals) - 1):
            if n_voxels[i] <= 1:
                continue
            v_a, v_b = fsc_vals[i], fsc_vals[i + 1]
            f_a, f_b = shell_centers[i], shell_centers[i + 1]
            if v_a >= thresh >= v_b:
                freq = f_a if v_a == v_b else f_a + (thresh - v_a) * (f_b - f_a) / (v_b - v_a)
                return float(apix / freq) if freq > 0 else None
            elif v_a < thresh and i == 0:
                return float(apix / f_a) if f_a > 0 else None
        if fsc_vals[-1] >= thresh:
            return float(apix / shell_centers[-1])
        return None

    res_143 = get_res(0.143)
    res_500 = get_res(0.500)
    freq_inv_A = shell_centers / apix

    return freq_inv_A, fsc_vals, res_143, res_500


# ── Main Entrypoint ───────────────────────────────────────────────────────────

def main():
    repo_root = Path(__file__).resolve().parent.parent

    default_gt = repo_root / "Data" / "0.mrc"
    default_ref = (
        repo_root
        / "alignOT"
        / "outputs"
        / "gpu_runs"
        / "run_20261009_090752_snr0.1_alignot_20rounds_gpu_argmax_wedge_lp23"
        / "round_05"
        / "ref_combined_round_05.mrc"
    )

    parser = argparse.ArgumentParser(
        description="Standardized Map-to-Ground-Truth Evaluation (Aligned CC & FSC)"
    )
    parser.add_argument(
        "--ref", "-r", type=str, default=str(default_ref),
        help=f"Path to candidate volume .mrc to evaluate (default: {default_ref.relative_to(repo_root)})"
    )
    parser.add_argument(
        "--ground_truth", "-g", type=str, default=str(default_gt),
        help=f"Path to ground truth volume .mrc (default: {default_gt.relative_to(repo_root)})"
    )
    parser.add_argument(
        "--apix", type=float, default=4.0,
        help="Voxel size in Angstroms (default: 4.0)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for GPU accelerated SO(3) rotation search ('cuda' or 'cpu')"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default=None,
        help="Optional directory to save comparison plot, CSV curve data, and aligned MRC"
    )
    parser.add_argument(
        "--save_aligned_mrc", action="store_true",
        help="Save the rigidly rotated reference volume (.mrc) in output_dir"
    )

    args = parser.parse_args()

    ref_path = Path(args.ref)
    gt_path = Path(args.ground_truth)
    device = torch.device(args.device)

    print("=" * 80)
    print("STANDARDIZED MAP-TO-GROUND-TRUTH BENCHMARK (ALIGNED CC & FSC)")
    print("=" * 80)
    print(f"Candidate Reference : {ref_path}")
    print(f"Ground Truth Reference: {gt_path}")
    print(f"Pixel Size (apix)   : {args.apix} Å")
    print(f"Compute Device      : {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print("-" * 80)

    # 1. Load volumes
    vol_ref = load_mrc(ref_path)
    vol_gt = load_mrc(gt_path)

    if vol_ref.shape != vol_gt.shape:
        raise ValueError(f"Shape mismatch: Reference {vol_ref.shape} vs Ground Truth {vol_gt.shape}")

    D = vol_gt.shape[0]
    mask = soft_sphere(D)

    # 2. Raw unaligned metrics
    raw_cc = pearson_cc(vol_gt, vol_ref)
    freq_raw, fsc_raw, res143_raw, res500_raw = compute_fsc(vol_gt, vol_ref, apix=args.apix)

    # 3. GPU SO(3) Rigid Registration
    t0_align = time.perf_counter()
    vol_aligned, aligned_cc, best_R, rot_deg = register_so3_to_ground_truth(
        vol_ref, vol_gt, mask, device=device
    )
    t_align = time.perf_counter() - t0_align

    # 4. Aligned metrics (Standard and Masked)
    freq_aln, fsc_aln, res143_aln, res500_aln = compute_fsc(vol_gt, vol_aligned, apix=args.apix)
    _, fsc_aln_m, res143_aln_m, res500_aln_m = compute_fsc(vol_gt, vol_aligned, apix=args.apix, mask=mask)

    # 5. Display Benchmark Report
    print(f"\n[1] REAL-SPACE CORRELATION WITH GROUND TRUTH:")
    print(f"    • Raw (Unaligned) Pearson CC  : {raw_cc:.4f}")
    print(f"    • Aligned Pearson CC (Masked) : {aligned_cc:.4f}")
    print(f"    • Registration Search Time    : {t_align:.2f}s")
    print(f"    • Detected 3D Rotation Offset : {rot_deg:.2f}°")

    print(f"\n[2] FOURIER SHELL CORRELATION (FSC) AGAINST GROUND TRUTH:")
    raw_500_str = f"{res500_raw:.2f} Å" if res500_raw else "None"
    raw_143_str = f"{res143_raw:.2f} Å" if res143_raw else "None"
    aln_500_str = f"{res500_aln:.2f} Å" if res500_aln else "None"
    aln_143_str = f"{res143_aln:.2f} Å" if res143_aln else "None"
    alnm_500_str = f"{res500_aln_m:.2f} Å" if res500_aln_m else "None"
    alnm_143_str = f"{res143_aln_m:.2f} Å" if res143_aln_m else "None"

    print(f"    • Raw FSC @ 0.500 (Unregistered)    : {raw_500_str}")
    print(f"    • Raw FSC @ 0.143 (Unregistered)    : {raw_143_str}")
    print(f"    • Aligned FSC @ 0.500 (SO(3) Rigid) : {aln_500_str}")
    print(f"    • Aligned FSC @ 0.143 (SO(3) Rigid) : {aln_143_str}")
    print(f"    • Masked Aligned FSC @ 0.143        : {alnm_143_str}")
    print("=" * 80)

    # 6. Optional Save
    if args.output_dir:
        out_p = Path(args.output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        # Save summary CSV
        summary_data = {
            "reference": str(ref_path),
            "ground_truth": str(gt_path),
            "raw_pearson_cc": raw_cc,
            "aligned_pearson_cc": aligned_cc,
            "rotation_offset_deg": rot_deg,
            "raw_fsc_0500_A": res500_raw,
            "raw_fsc_0143_A": res143_raw,
            "aligned_fsc_0500_A": res500_aln,
            "aligned_fsc_0143_A": res143_aln,
            "masked_aligned_fsc_0143_A": res143_aln_m,
        }
        df_summary = pd.DataFrame([summary_data])
        df_summary.to_csv(out_p / "benchmark_summary.csv", index=False)

        # Save FSC curve CSV
        df_curves = pd.DataFrame({
            "freq_inv_A": freq_raw,
            "fsc_raw": fsc_raw,
            "fsc_aligned": fsc_aln,
            "fsc_aligned_masked": fsc_aln_m,
        })
        df_curves.to_csv(out_p / "fsc_curves.csv", index=False)

        # Save FSC plot
        fig, ax = plt.subplots(figsize=(8, 5))
        nyquist = 1.0 / (2.0 * args.apix)
        ax.plot(freq_raw, fsc_raw, "--", color="#7f7f7f", lw=1.8, label=f"Raw Unaligned ({raw_143_str})")
        ax.plot(freq_aln, fsc_aln, "-o", color="#1f77b4", markersize=3.5, lw=2.2, label=f"SO(3) Aligned ({aln_143_str})")
        ax.plot(freq_aln, fsc_aln_m, "-s", color="#2ca02c", markersize=3.5, lw=2.0, label=f"SO(3) Aligned + Masked ({alnm_143_str})")
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.7, label="FSC = 0.500")
        ax.axhline(0.143, color="red", linestyle="--", alpha=0.8, label="FSC = 0.143")
        ax.set_xlim(0, nyquist)
        ax.set_ylim(-0.2, 1.05)
        ax.set_xlabel(r"Spatial Frequency (1/$\mathrm{\AA}$)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Fourier Shell Correlation", fontsize=11, fontweight="bold")
        ax.set_title("Fourier Shell Correlation vs Ground Truth (Data/0.mrc)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_p / "fsc_vs_ground_truth.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Save aligned volume
        if args.save_aligned_mrc:
            save_mrc(out_p / "reference_aligned_to_gt.mrc", vol_aligned)

        print(f"\n[Saved Outputs to {out_p.resolve()}]")
        print(f"  • Summary CSV : {out_p / 'benchmark_summary.csv'}")
        print(f"  • FSC Curves  : {out_p / 'fsc_curves.csv'}")
        print(f"  • FSC Plot    : {out_p / 'fsc_vs_ground_truth.png'}")
        if args.save_aligned_mrc:
            print(f"  • Aligned MRC : {out_p / 'reference_aligned_to_gt.mrc'}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
run_iterative_alignment_gpu_halfset.py
--------------------------------------
Gold-Standard Even/Odd Iterative Subtomogram Alignment & Averaging using Native PyTorch GPU AlignOT.

Features:
- Completely independent Even/Odd half-set alignment for rigorous gold-standard FSC evaluation.
- Native PyTorch GPU acceleration on Tensor Cores (Batched Sinkhorn OT + Analytical Polynomial Gradients on S³).
- Fast GPU 3D volume rigid transformations (torch.nn.functional.grid_sample).
- Supports both Base AlignOT (hard argmax on transport plan) and Barycentric 1-NN SGD.
- Exact matching output structure and benchmarking reports as the EMAN2 baseline for direct apples-to-apples comparison.

Outputs per run:
  alignOT/outputs/gpu_runs/run_<timestamp>_alignot_5rounds_gpu_<mode>/
  ├── round_00/
  │   ├── ref_even_round_00.mrc
  │   ├── ref_odd_round_00.mrc
  │   ├── ref_combined_round_00.mrc
  │   ├── fsc_round_00.csv / .png
  │   └── slices_round_00.png
  ├── round_01/ ... round_05/
  │   ├── ref_even_round_0X.mrc
  │   ├── ref_odd_round_0X.mrc
  │   ├── ref_combined_round_0X.mrc
  │   ├── metrics_particles_even_round_0X.csv
  │   ├── metrics_particles_odd_round_0X.csv
  │   ├── transforms_even.json / transforms_odd.json
  │   ├── fsc_round_0X.csv / .png
  │   └── slices_round_0X.png
  ├── fsc_all_rounds_comparison.png
  ├── slices_all_rounds_comparison.png
  ├── timing_report.txt
  ├── benchmark_summary.json
  └── run.log
"""

import argparse
import gc
import json
import logging
import os
import platform
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Set local timezone to Asia/Kolkata (IST)
os.environ["TZ"] = "Asia/Kolkata"
if hasattr(time, "tzset"):
    time.tzset()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ── Module Path Setup ─────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR.parent / "alignOT"))

import alignot_utils
import alignot_gpu_utils as gpu_utils
import coords as _coords


# ── Logger Setup ──────────────────────────────────────────────────────────────

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("AlignOT_GPU_Halfset")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── System Info ───────────────────────────────────────────────────────────────

def get_system_info(num_workers: int, device: torch.device) -> dict:
    cpu_model = platform.processor()
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "None (CPU)"
    return {
        "platform": platform.platform(),
        "processor": cpu_model,
        "machine": platform.machine(),
        "cpu_count_logical": os.cpu_count(),
        "num_workers_used": num_workers,
        "gpu_device": str(device),
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }


# ── Worker Task for Parallel TRN Sampling ─────────────────────────────────────

def _sample_single_subtomogram(task_args: tuple) -> dict:
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    path, lowpass_sigma, threshold, num_points, seed = task_args
    volume = alignot_utils.load_mrc(path, permissive=True)

    src_prep = alignot_utils.prepare_sampling_volume(volume, sigma=lowpass_sigma)
    pts_src = alignot_utils.sample_volume_in_memory(
        src_prep,
        threshold=threshold,
        num_points=num_points,
        random_seed=seed,
    )

    Bbar = pts_src.mean(axis=0).astype(np.float64)
    pts_src_c = pts_src - Bbar
    src_xyz = pts_src_c[:, [2, 1, 0]].astype(np.float32)  # [z, y, x] -> (x, y, z)

    return {
        "subtomogram_id": int(path.stem),
        "file_name": path.name,
        "src_xyz": src_xyz,
        "Bbar": Bbar,
        "volume": volume,
    }


# ── Quaternion Utilities ──────────────────────────────────────────────────────

def quaternion_multiply(q1, q2):
    """
    Hamilton product of two quaternions q1 * q2 in [w, x, y, z] convention.
    Works with lists, numpy arrays, or individual quaternions.
    Returns a numpy array [w, x, y, z].
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def align_volumes_ot(
    vol_source: np.ndarray,
    vol_target: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    gpu_rotator,
    logger: logging.Logger,
    label: str = "even→odd",
    seed_offset: int = 9999,
) -> Tuple[np.ndarray, list, np.ndarray, np.ndarray, float]:
    """
    Align vol_source to vol_target using AlignOT SGD (single-volume alignment).

    Steps:
      1. TRN-sample both volumes into point clouds.
      2. Run Sinkhorn OT + quaternion SGD to find the optimal rotation.
      3. Apply the rigid transform to vol_source using GPUVolumeRotator.

    Returns:
      aligned_vol   : The source volume rigidly transformed into the target frame.
      q_align       : The alignment quaternion [w, x, y, z].
      Bbar_align    : The source centroid (in [z, y, x] voxel space).
      Abar_align    : The target centroid (in [z, y, x] voxel space).
      cost_align    : The final Wasserstein distance.
    """
    t0 = time.perf_counter()
    logger.info(f"--- Inter-half alignment ({label}): TRN sampling & SGD ---")

    # Sample target (reference) point cloud
    ref_prep = alignot_utils.prepare_sampling_volume(vol_target, sigma=args.lowpass_sigma)
    pts_ref = alignot_utils.sample_volume_in_memory(
        ref_prep,
        threshold=args.threshold,
        num_points=args.num_points,
        random_seed=args.seed + seed_offset,
    )
    Abar_align = pts_ref.mean(axis=0).astype(np.float64)
    pts_ref_c = pts_ref - Abar_align
    ref_xyz = pts_ref_c[:, [2, 1, 0]].astype(np.float32)  # [z, y, x] -> (x, y, z)

    # Sample source point cloud
    src_prep = alignot_utils.prepare_sampling_volume(vol_source, sigma=args.lowpass_sigma)
    pts_src = alignot_utils.sample_volume_in_memory(
        src_prep,
        threshold=args.threshold,
        num_points=args.num_points,
        random_seed=args.seed + seed_offset + 1,
    )
    Bbar_align = pts_src.mean(axis=0).astype(np.float64)
    pts_src_c = pts_src - Bbar_align
    src_xyz = pts_src_c[:, [2, 1, 0]].astype(np.float32)  # [z, y, x] -> (x, y, z)

    # Run SGD: single volume, so batch dimension B=1
    src_b_np = src_xyz[np.newaxis, :, :]  # (1, N, 3)
    quats, costs = gpu_utils.SGD_barycentric_gpu_batched(
        src_b_np,
        ref_xyz,
        lr=args.sgd_lr,
        max_iter=args.sgd_iters,
        reg=args.reg,
        sinkhorn_iters=args.sinkhorn_iters,
        cost_metric=args.cost_metric,
        target_mode=args.target_mode,
        seed=args.seed + seed_offset + 2,
        device=device,
    )

    q_align = quats[0]  # [w, x, y, z]
    cost_align = costs[0]

    # Apply rigid transform to source volume
    aligned_vol = gpu_rotator.rotate_volume(vol_source, q_align, Bbar_align, Abar_align)
    aligned_vol = aligned_vol.float().cpu().numpy()

    dt = time.perf_counter() - t0
    logger.info(f"  Inter-half alignment ({label}) complete in {dt:.2f}s | wass_dist={cost_align:.3f}")
    logger.info(f"  q_align = [w={q_align[0]:.6f}, x={q_align[1]:.6f}, y={q_align[2]:.6f}, z={q_align[3]:.6f}]")

    return aligned_vol, q_align, Bbar_align, Abar_align, cost_align


# ── FSC & Plotting Utilities ──────────────────────────────────────────────────

def compute_fsc(vol1_np: np.ndarray, vol2_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 3D Fourier Shell Correlation between two volumes."""
    assert vol1_np.shape == vol2_np.shape, f"Volume shapes must match: {vol1_np.shape} vs {vol2_np.shape}"
    N = vol1_np.shape[0]
    n_shells = N // 2

    F1 = np.fft.fftn(vol1_np.astype(np.float64))
    F2 = np.fft.fftn(vol2_np.astype(np.float64))

    freqs = [np.fft.fftfreq(s) for s in vol1_np.shape]
    fz, fy, fx = np.meshgrid(freqs[0], freqs[1], freqs[2], indexing="ij")
    r = np.sqrt(fz**2 + fy**2 + fx**2)

    shell_width = 0.5 / n_shells
    shell_centers = np.arange(n_shells) * shell_width + shell_width / 2.0

    fsc_values = np.zeros(n_shells)
    n_voxels = np.zeros(n_shells, dtype=int)

    for i in range(n_shells):
        lo = i * shell_width
        hi = lo + shell_width
        mask = (r >= lo) & (r < hi) & (r > 0)
        n_vox = int(mask.sum())
        n_voxels[i] = n_vox
        if n_vox == 0:
            fsc_values[i] = 0.0
            continue
        f1_shell = F1[mask]
        f2_shell = F2[mask]
        numerator = np.real(np.sum(f1_shell * np.conj(f2_shell)))
        denominator = np.sqrt(np.sum(np.abs(f1_shell)**2) * np.sum(np.abs(f2_shell)**2))
        fsc_values[i] = float(numerator / denominator) if denominator > 0 else 0.0

    return shell_centers, fsc_values, n_voxels


def resolution_at_threshold(shell_centers: np.ndarray, fsc_values: np.ndarray,
                            threshold: float, apix: float, n_voxels: Optional[np.ndarray] = None) -> Optional[float]:
    """Interpolate spatial resolution (in Angstroms) where FSC drops below threshold."""
    for i in range(len(fsc_values) - 1):
        if n_voxels is not None and n_voxels[i] <= 1:
            continue
        v1, v2 = fsc_values[i], fsc_values[i + 1]
        f1, f2 = shell_centers[i], shell_centers[i + 1]
        if v1 >= threshold >= v2:
            if v1 == v2:
                freq = f1
            else:
                freq = f1 + (threshold - v1) * (f2 - f1) / (v2 - v1)
            return float(apix / freq) if freq > 0 else None

    if fsc_values[-1] >= threshold:
        return float(apix / shell_centers[-1])
    return None


def plot_fsc_curve(shell_centers: np.ndarray, fsc_values: np.ndarray, apix: float,
                    res_500: Optional[float], res_143: Optional[float], title: str, out_png: Path):
    """Plot and save Gold-Standard FSC curve."""
    freq_nyq = 1.0 / (2.0 * apix)
    freqs_inv_a = shell_centers / apix

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(freqs_inv_a, fsc_values, "b-o", markersize=3, lw=1.8, label="FSC (Even vs Odd)")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7,
               label=f"FSC=0.5 ({res_500:.1f}Å)" if res_500 else "FSC=0.5")
    ax.axhline(0.143, color="red", linestyle="--", alpha=0.7,
               label=f"FSC=0.143 ({res_143:.1f}Å)" if res_143 else "FSC=0.143")

    if res_143:
        ax.axvline(1.0 / res_143, color="red", linestyle=":", alpha=0.6)
    if res_500:
        ax.axvline(1.0 / res_500, color="gray", linestyle=":", alpha=0.6)

    ax.set_xlim(0, freq_nyq)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel(r"Spatial Frequency (1/$\mathrm{\AA}$)", fontsize=11)
    ax.set_ylabel("Fourier Shell Correlation", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    # Top axis for resolution in Angstroms
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    res_ticks = [50.0, 30.0, 20.0, 15.0, 12.0, 10.0, 8.0]
    res_ticks = [r for r in res_ticks if (1.0 / r) <= freq_nyq]
    ax2.set_xticks([1.0 / r for r in res_ticks])
    ax2.set_xticklabels([f"{r:.0f}Å" for r in res_ticks], fontsize=8)
    ax2.set_xlabel(r"Resolution ($\mathrm{\AA}$)", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_orthoslices(vol_np: np.ndarray, title: str, out_png: Path):
    """Plot central XY, XZ, YZ orthoslices."""
    cz, cy, cx = [s // 2 for s in vol_np.shape]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    slices = [
        ("Central XY", vol_np[cz, :, :]),
        ("Central XZ", vol_np[:, cy, :]),
        ("Central YZ", vol_np[:, :, cx])
    ]

    for ax, (name, sl) in zip(axes, slices):
        im = ax.imshow(sl, cmap="gray", origin="lower")
        ax.set_title(f"{name} (std={sl.std():.3f})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_all_rounds_fsc(fsc_history: list, apix: float, out_png: Path):
    """Plot overlay of FSC curves across all rounds."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(fsc_history)))
    freq_nyq = 1.0 / (2.0 * apix)

    for (round_num, shell_centers, fsc_vals, res_143), col in zip(fsc_history, colors):
        freqs_inv_a = shell_centers / apix
        lbl = f"Round {round_num:02d}"
        if res_143 is not None:
            lbl += f" ({res_143:.1f}Å)"
        ax.plot(freqs_inv_a, fsc_vals, "-o", markersize=3, color=col, lw=1.8, label=lbl)

    ax.axhline(0.143, color="red", linestyle="--", alpha=0.7, label="FSC=0.143 threshold")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="FSC=0.5 threshold")
    ax.set_xlim(0, freq_nyq)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel(r"Spatial Frequency (1/$\mathrm{\AA}$)", fontsize=11)
    ax.set_ylabel("Fourier Shell Correlation", fontsize=11)
    ax.set_title("AlignOT Native GPU — Gold-Standard Half-Set FSC", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    res_ticks = [50.0, 30.0, 20.0, 15.0, 12.0, 10.0, 8.0]
    res_ticks = [r for r in res_ticks if (1.0 / r) <= freq_nyq]
    ax2.set_xticks([1.0 / r for r in res_ticks])
    ax2.set_xticklabels([f"{r:.0f}Å" for r in res_ticks], fontsize=8)
    ax2.set_xlabel(r"Resolution ($\mathrm{\AA}$)", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_all_rounds_slices(slices_history: list, cluster_dir: Union[str, Path], out_png: Path):
    """Plot stacked comparison of central orthoslices across all rounds for combined average."""
    n_rounds = len(slices_history)
    if n_rounds == 0:
        return

    fig, axes = plt.subplots(n_rounds, 3, figsize=(11, 3.2 * n_rounds))
    if n_rounds == 1:
        axes = np.expand_dims(axes, axis=0)

    for r_idx, (r_num, vol, res_143) in enumerate(slices_history):
        if isinstance(vol, (str, Path)):
            vol = alignot_utils.load_mrc(vol, permissive=True)
        cz, cy, cx = [s // 2 for s in vol.shape]
        slices = [
            ("Central XY", vol[cz, :, :]),
            ("Central XZ", vol[:, cy, :]),
            ("Central YZ", vol[:, :, cx])
        ]
        for col_idx, (name, sl) in enumerate(slices):
            ax = axes[r_idx, col_idx]
            im = ax.imshow(sl, cmap="gray", origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{name} (std={sl.std():.3f})", fontsize=10)
            if col_idx == 0:
                label = f"Round {r_num:02d}\n({res_143:.1f}Å)" if res_143 is not None else f"Round {r_num:02d}"
                ax.set_ylabel(label, fontsize=11, fontweight="bold", labelpad=10)

    cluster_p = Path(cluster_dir)
    prot_label = cluster_p.name
    prot_str = f"Protein {prot_label}" if prot_label else ""
    snr_str = ""
    for part in reversed(cluster_p.parts):
        if part.startswith("0."):
            snr_str = f"SNR {part}"
            break

    meta_parts = [p for p in [prot_str, snr_str] if p]
    meta_combined = ", ".join(meta_parts)
    meta_header = f" ({meta_combined})" if meta_combined else ""
    first_r = slices_history[0][0]
    last_r = slices_history[-1][0]
    subtitle = f"Round {first_r:02d} (Raw Average) through Round {last_r:02d}" if last_r > first_r else f"Round {first_r:02d} (Raw Average)"

    fig.suptitle(
        f"AlignOT Native GPU Baseline — Iterative Alignment Orthoslices{meta_header}\n{subtitle}",
        fontsize=13,
        fontweight="bold",
        y=0.995
    )

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


# ── Alignment Routine for One Half-Set ────────────────────────────────────────

def run_half_set_round_gpu(
    half_name: str,
    cluster_paths: List[Path],
    current_reference: np.ndarray,
    round_dir: Path,
    round_idx: int,
    args: argparse.Namespace,
    device: torch.device,
    gpu_rotator: gpu_utils.GPUVolumeRotator,
    logger: logging.Logger,
    wedge_mask_gpu: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, dict, list]:
    """
    One round of AlignOT alignment for a single half-set.

    wedge_mask_gpu : (D, H, W) bool/float tensor on `device`, or None.
        When provided, volume accumulation is done in Fourier space with
        per-particle rotated wedge masks — the Fourier-space masked average.
        When None (default), the original real-space running sum is used,
        preserving exact backward compatibility.
    """
    t0_half = time.perf_counter()
    n_actual = len(cluster_paths)
    box_size = current_reference.shape[0]
    use_wedge = wedge_mask_gpu is not None

    avg_mode = "Fourier-space wedge-masked" if use_wedge else "real-space sum"
    logger.info(f"--- [{half_name.upper()} HALF] Round {round_idx:02d} Starting ({n_actual} particles, device: {device}, avg: {avg_mode}) ---")

    # 1. Sample Reference Point Cloud via TRN
    t0_ref_trn = time.perf_counter()
    ref_prep = alignot_utils.prepare_sampling_volume(current_reference, sigma=args.lowpass_sigma)
    pts_ref = alignot_utils.sample_volume_in_memory(
        ref_prep,
        threshold=args.threshold,
        num_points=args.num_points,
        random_seed=args.seed + 1000 * round_idx + (1 if half_name == "odd" else 0),
    )
    Abar = pts_ref.mean(axis=0).astype(np.float64)
    pts_ref_c = pts_ref - Abar
    ref_xyz = pts_ref_c[:, [2, 1, 0]].astype(np.float32)  # [z, y, x] -> (x, y, z)
    t_ref_trn = time.perf_counter() - t0_ref_trn

    # 2. Parallel TRN Sampling of Particles on CPU Workers
    logger.info(f"Sampling {n_actual} subtomogram point clouds across {args.num_workers} CPU workers...")
    t0_sample = time.perf_counter()
    sampling_tasks = [
        (
            path,
            args.lowpass_sigma,
            args.threshold,
            args.num_points,
            args.seed + round_idx * 1000 + (500 if half_name == "odd" else 0) + idx,
        )
        for idx, path in enumerate(cluster_paths)
    ]
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        sampled_items = list(executor.map(_sample_single_subtomogram, sampling_tasks))
    t_sample = time.perf_counter() - t0_sample

    # 3. Stack Batched Source Clouds for GPU Alignment
    src_b_np = np.stack([item["src_xyz"] for item in sampled_items], axis=0)  # (B, num_points, 3)

    # 4. Native GPU Batched SGD (Base Argmax or Barycentric)
    t0_gpu_sgd = time.perf_counter()
    quats_gpu, costs_gpu = gpu_utils.SGD_barycentric_gpu_batched(
        src_b_np,
        ref_xyz,
        lr=args.sgd_lr,
        max_iter=args.sgd_iters,
        reg=args.reg,
        sinkhorn_iters=args.sinkhorn_iters,
        cost_metric=args.cost_metric,
        target_mode=args.target_mode,
        seed=args.seed + round_idx * 1000 + (100 if half_name == "odd" else 0),
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_gpu_sgd = time.perf_counter() - t0_gpu_sgd

    # 5. GPU 3D Volume Rigid Transformation & Accumulation
    t0_gpu_rot = time.perf_counter()
    alignment_results = []

    if use_wedge:
        # ── Fourier-space masked average (wedge-aware) ───────────────────────
        # Accumulators in Fourier space (centered DC coordinates matching wedge_mask):
        #   F_sum[k]  = Σ_i  M_i(k) · F_i(k)   (weighted Fourier coefficients)
        #   W_sum[k]  = Σ_i  M_i(k)             (total weight at each frequency)
        #
        # In wedge_mask.mrc, DC (k=0) is at the volume center [box_size//2, ...].
        # torch.fft.fftn produces unshifted FFT with DC at [0, 0, 0].
        # We use torch.fft.fftshift so F_i matches the centered wedge_mask.
        #
        # To rotate the mask around its center, Bbar_mask must be [0, 0, 0]
        # (in centered TRN coords, offset = center - R_inv @ center).
        F_sum = torch.zeros(
            (box_size, box_size, box_size), dtype=torch.complex128, device=device
        )
        W_sum = torch.zeros(
            (box_size, box_size, box_size), dtype=torch.float64, device=device
        )
        Bbar_mask = np.zeros(3, dtype=np.float64)

        for item_idx, item in enumerate(sampled_items, start=1):
            vol_np = item["volume"]
            Bbar   = item["Bbar"]
            q      = quats_gpu[item_idx - 1]
            cost   = float(costs_gpu[item_idx - 1])

            # 1. Rotate volume in real space (same as original path)
            rot_vol = gpu_rotator.rotate_volume(vol_np, q, Bbar, Abar)  # (D, H, W), float32

            # 2. Rotate wedge mask around center on GPU
            rot_mask_f = gpu_rotator.rotate_volume(
                wedge_mask_gpu, q, Bbar_mask, Bbar_mask
            ).double()  # (D, H, W), float64
            rot_mask = (rot_mask_f >= 0.5).double()

            # 3. FFT of rotated volume, shifted so DC is at the center
            F_i = torch.fft.fftshift(torch.fft.fftn(rot_vol.double()))  # complex128

            # 4. Accumulate in Fourier space
            F_sum += rot_mask * F_i
            W_sum += rot_mask

            alignment_results.append({
                "index": item_idx - 1,
                "subtomogram_id": item["subtomogram_id"],
                "file_name": item["file_name"],
                "wass_dist": cost,
                "q_w": float(q[0]), "q_x": float(q[1]),
                "q_y": float(q[2]), "q_z": float(q[3]),
            })
            del vol_np, rot_vol, rot_mask_f, rot_mask, F_i

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_gpu_rot = time.perf_counter() - t0_gpu_rot

        # 6. Reconstruct reference from weighted Fourier average
        # Divide F_sum / W_sum where at least one particle measured that frequency
        mask_measured = W_sum > 0.5
        F_avg = torch.zeros_like(F_sum)
        F_avg[mask_measured] = F_sum[mask_measured] / W_sum[mask_measured]

        # Shift back from center to standard FFT layout and IFFT to real space
        new_reference_t = torch.fft.ifftn(torch.fft.ifftshift(F_avg)).real.float()
        new_reference = new_reference_t.cpu().numpy()
        del F_sum, W_sum, mask_measured, F_avg, new_reference_t

    else:
        # ── Original real-space running sum (backward-compatible) ────────────
        sum_rotated_gpu = torch.zeros(
            (box_size, box_size, box_size), dtype=torch.float64, device=device
        )

        for item_idx, item in enumerate(sampled_items, start=1):
            vol_np = item["volume"]
            Bbar   = item["Bbar"]
            q      = quats_gpu[item_idx - 1]
            cost   = float(costs_gpu[item_idx - 1])

            rot_vol_gpu = gpu_rotator.rotate_volume(vol_np, q, Bbar, Abar)
            sum_rotated_gpu += rot_vol_gpu.double()

            alignment_results.append({
                "index": item_idx - 1,
                "subtomogram_id": item["subtomogram_id"],
                "file_name": item["file_name"],
                "wass_dist": cost,
                "q_w": float(q[0]), "q_x": float(q[1]),
                "q_y": float(q[2]), "q_z": float(q[3]),
            })
            del vol_np, rot_vol_gpu

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_gpu_rot = time.perf_counter() - t0_gpu_rot

        # 6. Build new half reference
        new_reference = (sum_rotated_gpu / float(n_actual)).float().cpu().numpy()
        del sum_rotated_gpu

    del sampled_items, src_b_np
    if device.type == "cuda":
        torch.cuda.empty_cache()


    total_half_time = time.perf_counter() - t0_half
    mean_cost = float(np.mean([r["wass_dist"] for r in alignment_results]))

    # Save per-particle metrics CSV
    df_metrics = pd.DataFrame(alignment_results)
    df_metrics["round"] = round_idx
    csv_out = round_dir / f"metrics_particles_{half_name}_round_{round_idx:02d}.csv"
    df_metrics.to_csv(csv_out, index=False)

    # Save transforms JSON
    json_out = round_dir / f"transforms_{half_name}.json"
    with open(json_out, "w") as f:
        json.dump({
            "half": half_name,
            "round": round_idx,
            "mean_score": mean_cost,
            "align_time_s": t_gpu_sgd,
            "total_time_s": total_half_time,
            "transforms": alignment_results
        }, f, indent=2)

    logger.info(f"[{half_name.upper()} HALF] Round {round_idx:02d} Complete: mean_wass={mean_cost:.3f}, "
                f"sample={t_sample:.1f}s, sgd={t_gpu_sgd:.2f}s, rot={t_gpu_rot:.2f}s, total={total_half_time:.1f}s "
                f"({total_half_time / n_actual:.3f}s/ptcl)")

    metrics_dict = {
        "mean_score": mean_cost,
        "sample_time_s": t_sample,
        "align_time_s": t_gpu_sgd,
        "accum_time_s": t_gpu_rot,
        "total_time_s": total_half_time,
        "sec_per_particle": total_half_time / n_actual
    }
    return new_reference, metrics_dict, alignment_results


# ── Write Summary Files ───────────────────────────────────────────────────────

def write_summary_files(run_dir: Path, sys_info: dict, args: argparse.Namespace, rounds_metrics: list, t0: float):
    total_time = time.perf_counter() - t0

    # JSON Summary
    summary_data = {
        "benchmark_metadata": {
            "method": f"AlignOT_GPU_{args.target_mode.upper()}",
            "target_mode": args.target_mode,
            "cost_metric": args.cost_metric,
            "reg": args.reg,
            "sinkhorn_iters": args.sinkhorn_iters,
            "sgd_iters": args.sgd_iters,
            "sgd_lr": args.sgd_lr,
            "num_points": args.num_points,
            "threshold": args.threshold,
            "lowpass_sigma": args.lowpass_sigma,
            "data_dir": str(Path(args.cluster_dir).resolve()),
            "total_rounds": args.n_rounds,
            "apix": args.apix,
            "num_workers": args.num_workers,
            "device": str(args.device),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
            "total_elapsed_seconds": total_time,
            "system_info": sys_info
        },
        "rounds": rounds_metrics
    }

    json_path = run_dir / "benchmark_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    # Text Report
    txt_path = run_dir / "timing_report.txt"
    mode_str = f"{args.target_mode} ({'argmax' if args.target_mode in ('base', 'argmax') else '1-NN'})"
    with open(txt_path, "w") as f:
        f.write("=" * 82 + "\n")
        f.write(f"ALIGNOT GPU ({args.target_mode.upper()}) SUBTOMOGRAM ALIGNMENT BENCHMARK REPORT\n")
        f.write("=" * 82 + "\n")
        f.write(f"Date & Time        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}\n")
        f.write(f"GPU Model          : {sys_info.get('gpu_name', 'Unknown')} ({sys_info.get('gpu_device', '')})\n")
        f.write(f"CPU Model          : {sys_info.get('processor', 'Unknown')} ({sys_info.get('machine', '')})\n")
        f.write(f"TRN Workers Used   : {args.num_workers}\n")
        f.write(f"Dataset            : {str(Path(args.cluster_dir).resolve())}\n")
        f.write(f"Target Mode        : {mode_str}\n")
        f.write(f"Cost Metric        : {args.cost_metric}\n")
        f.write(f"Sinkhorn Iters     : {args.sinkhorn_iters} (reg={args.reg})\n")
        f.write(f"SGD Iters / LR     : {args.sgd_iters} / {args.sgd_lr}\n")
        f.write(f"TRN Points / LP    : {args.num_points} / σ={args.lowpass_sigma} vox\n")
        f.write(f"Pixel Size (apix)  : {args.apix} Å\n")
        f.write(f"Total Pipeline Time: {total_time:.1f}s ({total_time / 60.0:.2f} min)\n")
        f.write("-" * 82 + "\n\n")

        f.write(f"{'Round':<7} | {'Even Time':<11} | {'Odd Time':<11} | {'Total Round':<12} | {'FSC@0.500':<12} | {'FSC@0.143':<12}\n")
        f.write("-" * 82 + "\n")
        for m in rounds_metrics:
            r = m["round"]
            ev_t = f"{m['even_metrics']['total_time_s']:.1f}s" if r > 0 else "-"
            od_t = f"{m['odd_metrics']['total_time_s']:.1f}s" if r > 0 else "-"
            rnd_t = f"{m['round_time_seconds']:.1f}s"
            fsc5 = f"{m['fsc_0500_res_angstrom']:.2f} Å" if m['fsc_0500_res_angstrom'] else "None"
            fsc143 = f"{m['fsc_0143_res_angstrom']:.2f} Å" if m['fsc_0143_res_angstrom'] else "None"
            f.write(f"{r:<7} | {ev_t:<11} | {od_t:<11} | {rnd_t:<12} | {fsc5:<12} | {fsc143:<12}\n")
        f.write("=" * 82 + "\n")


# ── Main Entrypoint ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gold-Standard Iterative Subtomogram Alignment with Native GPU AlignOT"
    )
    # Data & Output
    parser.add_argument("--cluster_dir", type=str, default="Data/0.1/subtomograms/0",
                        help="Directory containing subtomogram .mrc files")
    parser.add_argument("--output_base_dir", type=str, default="alignOT/outputs/gpu_runs",
                        help="Root output directory for runs")
    parser.add_argument("--init_reference", type=str, default=None,
                        help="Optional external initial reference .mrc. If None, computes raw unaligned streaming average.")
    parser.add_argument("--n_rounds", type=int, default=5,
                        help="Number of iterative alignment rounds")
    parser.add_argument("--n_sbtms", type=int, default=-1,
                        help="Limit number of subtomograms (-1 = all)")
    parser.add_argument("--apix", type=float, default=4.0,
                        help="Pixel / voxel size in Angstroms")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Custom run directory name")

    # TRN Sampling
    parser.add_argument("--num_points", type=int, default=500,
                        help="TRN points per volume (default: 500)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Density threshold after contrast inversion (default: 0.7)")
    parser.add_argument("--lowpass_sigma", type=float, default=2.0,
                        help="Gaussian lowpass filter sigma in voxels before sampling (default: 2.0)")

    # AlignOT SGD & Sinkhorn
    parser.add_argument("--reg", type=float, default=100.0,
                        help="Sinkhorn entropic regularisation (default: 100.0)")
    parser.add_argument("--sinkhorn_iters", type=int, default=50,
                        help="Sinkhorn scaling iterations (default: 50)")
    parser.add_argument("--cost_metric", type=str, default="l2_sq", choices=["l2_sq", "l1", "l2"],
                        help="Ground distance cost metric (default: 'l2_sq')")
    parser.add_argument("--target_mode", type=str, default="base", choices=["base", "argmax", "barycentric"],
                        help="Target point selection: 'base' / 'argmax' (hard argmax) or 'barycentric' (1-NN)")
    parser.add_argument("--sgd_iters", type=int, default=500,
                        help="SGD iterations per particle (default: 500)")
    parser.add_argument("--sgd_lr", type=float, default=0.001,
                        help="SGD learning rate (default: 0.001)")

    # Hardware & Multiprocessing
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel CPU workers for TRN sampling (default: 16)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for Sinkhorn, SGD & 3D volume rigid transformations (default: 'cuda')")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    # Fourier-space wedge-masked averaging (optional — off by default)
    parser.add_argument(
        "--wedge_mask", type=str, default=None,
        help=(
            "Path to a binary Fourier-space wedge mask (.mrc, same box size as subtomograms). "
            "When supplied, volume accumulation switches from real-space sum to a Fourier-space "
            "weighted average: each Fourier coefficient is weighted by the rotated per-particle "
            "wedge mask, suppressing missing-wedge artifacts in the reference. "
            "When omitted (default), the original real-space averaging is used unchanged."
        ),
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = gpu_utils.get_device(args.device)

    # 1. Output directory setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_label = "argmax" if args.target_mode in ("base", "argmax") else "barycentric"
    metric_suffix = f"_{args.cost_metric}" if args.cost_metric != "l2_sq" else ""
    cluster_p = Path(args.cluster_dir)
    snr_tag = ""
    for part in reversed(cluster_p.parts):
        if part.startswith("0."):
            snr_tag = f"_snr{part}"
            break

    if args.run_name:
        run_dir_name = f"run_{timestamp}_{args.run_name}"
    else:
        wedge_suffix = "_wedge" if args.wedge_mask is not None else ""
        run_dir_name = f"run_{timestamp}{snr_tag}_alignot_{args.n_rounds}rounds_gpu_{mode_label}{metric_suffix}{wedge_suffix}"

    run_dir = Path(args.output_base_dir) / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 2. Setup Logging
    log_file = run_dir / "run.log"
    logger = setup_logger(log_file)

    logger.info("=" * 78)
    logger.info("ALIGNOT NATIVE GPU GOLD-STANDARD ITERATIVE SUBTOMOGRAM ALIGNMENT")
    logger.info("=" * 78)
    logger.info(f"Start Time      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    logger.info(f"Output Directory: {run_dir.resolve()}")
    logger.info(f"Subtomograms Dir: {args.cluster_dir}")
    logger.info(f"Rounds          : {args.n_rounds}")
    logger.info(f"Target Mode     : {args.target_mode} ({mode_label})")
    logger.info(f"Cost Metric     : {args.cost_metric}")
    logger.info(f"Sinkhorn Iters  : {args.sinkhorn_iters} (reg={args.reg})")
    logger.info(f"SGD Iters / LR  : {args.sgd_iters} / {args.sgd_lr}")
    logger.info(f"TRN Points / LP : {args.num_points} / σ={args.lowpass_sigma} vox")
    logger.info(f"Pixel Size      : {args.apix} Å")
    logger.info(f"Device          : {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    logger.info(f"TRN Workers     : {args.num_workers}")

    # Load wedge mask (optional). Must match the subtomogram box size.
    wedge_mask_gpu: Optional[torch.Tensor] = None
    if args.wedge_mask is not None:
        import mrcfile as _mrcfile
        logger.info(f"Wedge Mask Path : {args.wedge_mask}")
        with _mrcfile.open(args.wedge_mask, permissive=True) as _mf:
            _mask_np = _mf.data.astype(np.float32)
        # Validate shape — must be (D, H, W) with D = H = W = box_size (91)
        if _mask_np.ndim != 3:
            logger.error(f"Wedge mask must be 3D; got shape {_mask_np.shape}. Aborting.")
            sys.exit(1)
        wedge_mask_gpu = torch.from_numpy(_mask_np).to(device)
        n_ones  = int((wedge_mask_gpu >= 0.5).sum().item())
        n_total = wedge_mask_gpu.numel()
        logger.info(
            f"Wedge Mask Shape: {tuple(wedge_mask_gpu.shape)}  "
            f"Fill: {n_ones}/{n_total} = {100*n_ones/n_total:.1f}%  "
            f"(averaging: Fourier-space wedge-masked)"
        )
    else:
        logger.info("Wedge Mask      : None  (averaging: real-space sum — original mode)")

    # Hardware Info
    sys_info = get_system_info(args.num_workers, device)
    for k, v in sys_info.items():
        logger.info(f"  SysInfo {k}: {v}")

    # 3. Discover subtomograms
    all_paths = alignot_utils.get_cluster_paths(args.cluster_dir)
    if args.n_sbtms > 0:
        all_paths = all_paths[:args.n_sbtms]

    num_total = len(all_paths)
    logger.info(f"Total Subtomograms Loaded: {num_total}")
    if num_total < 2:
        logger.error("Insufficient subtomograms for even/odd half-set alignment.")
        sys.exit(1)

    even_paths = [p for p in all_paths if int(p.stem) % 2 == 0]
    odd_paths  = [p for p in all_paths if int(p.stem) % 2 == 1]
    logger.info(f"Even Half-Set: {len(even_paths)} particles (indices 0, 2, ...)")
    logger.info(f"Odd Half-Set : {len(odd_paths)} particles (indices 1, 3, ...)")

    # 4. Round 0: Initial Seed References & Baseline FSC
    round_00_dir = run_dir / "round_00"
    round_00_dir.mkdir(parents=True, exist_ok=True)
    logger.info("\n" + "=" * 50)
    logger.info("ROUND 00: INITIAL SEED REFERENCES & BASELINE FSC")
    logger.info("=" * 50)

    t_r0_start = time.perf_counter()
    if args.init_reference is not None:
        logger.info(f"Loading external initial reference: {args.init_reference}")
        init_ref = alignot_utils.load_mrc(args.init_reference, permissive=True)
        ref_even = init_ref.copy()
        ref_odd  = init_ref.copy()
    else:
        logger.info(f"Computing GPU streaming raw averages ({len(even_paths)} Even, {len(odd_paths)} Odd)...")
        ref_even = gpu_utils.compute_streaming_average_gpu(
            even_paths, device=device, progress_every=max(1, len(even_paths) // 5), logger=logger
        ).cpu().numpy()
        ref_odd = gpu_utils.compute_streaming_average_gpu(
            odd_paths, device=device, progress_every=max(1, len(odd_paths) // 5), logger=logger
        ).cpu().numpy()

    # Save Round 0 References
    alignot_utils.save_mrc(round_00_dir / "ref_even_round_00.mrc", ref_even)
    alignot_utils.save_mrc(round_00_dir / "ref_odd_round_00.mrc", ref_odd)

    ref_comb_r0 = 0.5 * (ref_even + ref_odd)
    alignot_utils.save_mrc(round_00_dir / "ref_combined_round_00.mrc", ref_comb_r0)

    # Compute FSC Round 0
    shell_centers, fsc_vals, n_voxels = compute_fsc(ref_even, ref_odd)
    res_500 = resolution_at_threshold(shell_centers, fsc_vals, 0.5, args.apix, n_voxels)
    res_143 = resolution_at_threshold(shell_centers, fsc_vals, 0.143, args.apix, n_voxels)

    fsc_csv = round_00_dir / "fsc_round_00.csv"
    np.savetxt(fsc_csv, np.column_stack([shell_centers, shell_centers / args.apix, fsc_vals, n_voxels]),
               delimiter=",", header="shell_center_1_vox,freq_inv_angstrom,fsc,num_voxels", comments="")

    fsc_png = round_00_dir / "fsc_round_00.png"
    plot_fsc_curve(shell_centers, fsc_vals, args.apix, res_500, res_143,
                   f"Round 00 FSC (Unaligned Seed) — FSC@0.143: {f'{res_143:.1f}Å' if res_143 else 'None'}", fsc_png)

    plot_orthoslices(ref_comb_r0, "Round 00 Combined Reference (Unaligned)",
                     round_00_dir / "slices_round_00.png")

    dt_r0 = time.perf_counter() - t_r0_start
    logger.info(f"Round 00 Complete in {dt_r0:.1f}s | Resolution: FSC@0.5={f'{res_500:.2f}Å' if res_500 else 'None'}, "
                f"FSC@0.143={f'{res_143:.2f}Å' if res_143 else 'None'}")

    # History tracking
    fsc_history = [(0, shell_centers, fsc_vals, res_143)]
    slices_history = [(0, ref_comb_r0.copy(), res_143)]
    plot_all_rounds_slices(slices_history, args.cluster_dir, run_dir / "slices_all_rounds_comparison.png")
    rounds_metrics = [{
        "round": 0,
        "fsc_0500_res_angstrom": res_500,
        "fsc_0143_res_angstrom": res_143,
        "round_time_seconds": dt_r0,
        "even_metrics": {"total_time_s": 0.0, "mean_score": 0.0},
        "odd_metrics": {"total_time_s": 0.0, "mean_score": 0.0}
    }]

    total_pipeline_t0 = time.perf_counter()
    box_size = ref_even.shape[0]
    gpu_rotator = gpu_utils.GPUVolumeRotator(box_size=box_size, device=device)

    current_ref_even = ref_even
    current_ref_odd  = ref_odd

    # 5. Iterative Alignment Rounds 1 to N
    for r in range(1, args.n_rounds + 1):
        round_t0 = time.perf_counter()
        round_str = f"round_{r:02d}"
        round_dir = run_dir / round_str
        round_dir.mkdir(parents=True, exist_ok=True)

        logger.info("\n" + "=" * 60)
        logger.info(f"STARTING ITERATION ROUND {r:02d} / {args.n_rounds:02d}")
        logger.info("=" * 60)

        # Align Even Half
        new_ref_even, even_metrics, even_align_results = run_half_set_round_gpu(
            half_name="even",
            cluster_paths=even_paths,
            current_reference=current_ref_even,
            round_dir=round_dir,
            round_idx=r,
            args=args,
            device=device,
            gpu_rotator=gpu_rotator,
            logger=logger,
            wedge_mask_gpu=wedge_mask_gpu,
        )

        # Align Odd Half
        new_ref_odd, odd_metrics, odd_align_results = run_half_set_round_gpu(
            half_name="odd",
            cluster_paths=odd_paths,
            current_reference=current_ref_odd,
            round_dir=round_dir,
            round_idx=r,
            args=args,
            device=device,
            gpu_rotator=gpu_rotator,
            logger=logger,
            wedge_mask_gpu=wedge_mask_gpu,
        )

        # ── Inter-Half-Set Alignment (even_ref → odd_ref) ────────────────
        # Align even reference into the odd reference frame for FSC & combined average.
        # This is purely for evaluation — does NOT feed back into the iterative loop.
        aligned_ref_even, q_align, Bbar_align, Abar_align, cost_align = align_volumes_ot(
            vol_source=new_ref_even,
            vol_target=new_ref_odd,
            args=args,
            device=device,
            gpu_rotator=gpu_rotator,
            logger=logger,
            label=f"Round {r:02d} even→odd",
            seed_offset=9999 + r * 100,
        )

        # Compose alignment quaternion with each even particle's quaternion
        # q_total_i = q_align ∘ q_i  (Hamilton product)
        R_align = _coords.quaternion_to_R(q_align)  # (3, 3) in (x, y, z) Euclidean
        composed_even_results = []
        for res in even_align_results:
            q_i = [res["q_w"], res["q_x"], res["q_y"], res["q_z"]]
            q_composed = quaternion_multiply(q_align, q_i)
            q_norm = np.linalg.norm(q_composed)
            if q_norm > 1e-12:
                q_composed = q_composed / q_norm
            composed_even_results.append({
                **res,
                "q_w": float(q_composed[0]),
                "q_x": float(q_composed[1]),
                "q_y": float(q_composed[2]),
                "q_z": float(q_composed[3]),
                "q_intra_w": float(q_i[0]),
                "q_intra_x": float(q_i[1]),
                "q_intra_y": float(q_i[2]),
                "q_intra_z": float(q_i[3]),
            })

        # Overwrite even transforms JSON with composed quaternions
        json_out = round_dir / "transforms_even.json"
        with open(json_out, "w") as f:
            json.dump({
                "half": "even",
                "round": r,
                "mean_score": even_metrics["mean_score"],
                "align_time_s": even_metrics["align_time_s"],
                "total_time_s": even_metrics["total_time_s"],
                "inter_half_alignment": {
                    "q_align_w": float(q_align[0]),
                    "q_align_x": float(q_align[1]),
                    "q_align_y": float(q_align[2]),
                    "q_align_z": float(q_align[3]),
                    "wass_dist": float(cost_align),
                    "note": "Even particle quaternions are composed: q_total = q_align * q_intra"
                },
                "transforms": composed_even_results
            }, f, indent=2)

        # Overwrite even metrics CSV with composed quaternions
        df_metrics_even = pd.DataFrame(composed_even_results)
        df_metrics_even["round"] = r
        csv_out_even = round_dir / f"metrics_particles_even_{round_str}.csv"
        df_metrics_even.to_csv(csv_out_even, index=False)

        # Save references (aligned even, original odd)
        alignot_utils.save_mrc(round_dir / f"ref_even_{round_str}.mrc", aligned_ref_even)
        alignot_utils.save_mrc(round_dir / f"ref_odd_{round_str}.mrc", new_ref_odd)

        # Combined reference (aligned even + odd) — for saving & plotting only
        ref_comb = 0.5 * (aligned_ref_even + new_ref_odd)
        alignot_utils.save_mrc(round_dir / f"ref_combined_{round_str}.mrc", ref_comb)

        # Compute FSC between aligned even and odd (now in same frame)
        shell_centers, fsc_vals, n_voxels = compute_fsc(aligned_ref_even, new_ref_odd)
        res_500 = resolution_at_threshold(shell_centers, fsc_vals, 0.5, args.apix, n_voxels)
        res_143 = resolution_at_threshold(shell_centers, fsc_vals, 0.143, args.apix, n_voxels)

        # Save FSC CSV & PNG
        fsc_csv = round_dir / f"fsc_{round_str}.csv"
        np.savetxt(fsc_csv, np.column_stack([shell_centers, shell_centers / args.apix, fsc_vals, n_voxels]),
                   delimiter=",", header="shell_center_1_vox,freq_inv_angstrom,fsc,num_voxels", comments="")

        fsc_png = round_dir / f"fsc_{round_str}.png"
        plot_fsc_curve(shell_centers, fsc_vals, args.apix, res_500, res_143,
                       f"Round {r:02d} Gold-Standard FSC — FSC@0.143: {f'{res_143:.1f}Å' if res_143 else 'None'}", fsc_png)

        # Save orthoslices
        plot_orthoslices(ref_comb, f"Round {r:02d} Combined Reference",
                         round_dir / f"slices_{round_str}.png")

        round_total_time = time.perf_counter() - round_t0
        fsc_history.append((r, shell_centers, fsc_vals, res_143))
        slices_history.append((r, ref_comb.copy(), res_143))

        round_metric = {
            "round": r,
            "round_time_seconds": round_total_time,
            "even_metrics": even_metrics,
            "odd_metrics": odd_metrics,
            "inter_half_alignment": {
                "q_align": [float(q_align[0]), float(q_align[1]), float(q_align[2]), float(q_align[3])],
                "wass_dist": float(cost_align)
            },
            "fsc_0500_res_angstrom": res_500,
            "fsc_0143_res_angstrom": res_143
        }
        rounds_metrics.append(round_metric)

        logger.info(f"=== ROUND {r:02d} SUMMARY ===")
        logger.info(f"  Round Duration   : {round_total_time:.1f}s ({round_total_time / 60.0:.2f} min)")
        logger.info(f"  Even Half Time   : {even_metrics['total_time_s']:.1f}s (wass={even_metrics['mean_score']:.3f})")
        logger.info(f"  Odd Half Time    : {odd_metrics['total_time_s']:.1f}s (wass={odd_metrics['mean_score']:.3f})")
        logger.info(f"  Inter-Half Wass  : {cost_align:.3f}")
        logger.info(f"  Gold-Std FSC@0.5 : {f'{res_500:.2f}Å' if res_500 else 'None'}")
        logger.info(f"  Gold-Std FSC@.143: {f'{res_143:.2f}Å' if res_143 else 'None'}")

        # Update references for next iteration (ORIGINAL unaligned — no information leak)
        current_ref_even = new_ref_even
        current_ref_odd  = new_ref_odd

        # Periodically update summary files
        plot_all_rounds_fsc(fsc_history, args.apix, run_dir / "fsc_all_rounds_comparison.png")
        plot_all_rounds_slices(slices_history, args.cluster_dir, run_dir / "slices_all_rounds_comparison.png")
        write_summary_files(run_dir, sys_info, args, rounds_metrics, total_pipeline_t0)

    total_elapsed = time.perf_counter() - total_pipeline_t0
    logger.info("\n" + "=" * 78)
    logger.info("ALIGNOT NATIVE GPU ITERATIVE ALIGNMENT PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"Total Pipeline Runtime: {total_elapsed:.1f}s ({total_elapsed / 60.0:.2f} min)")
    logger.info(f"All outputs saved to  : {run_dir.resolve()}")
    logger.info("=" * 78)


if __name__ == "__main__":
    main()

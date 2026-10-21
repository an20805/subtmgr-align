#!/usr/bin/env python3
"""
run_iterative_alignment_eman2.py — Gold-Standard Iterative Subtomogram Alignment & Averaging (EMAN2 Baseline)

Pipeline:
  - Splits particles into independent Even (indices 0, 2, ...) and Odd (indices 1, 3, ...) half-sets.
  - Per round, each half independently aligns to its own reference using pure EMAN2:
      Aligner   : rotate_translate_3d_tree  (missing-wedge aware 3D tree search)
      Comparator: ccc.tomo.thresh           (tomography cross-correlation with threshold)
      Averager  : mean.tomo                 (wedge-weighted Fourier average)
  - Contrast is INVERTED before alignment (cryo-ET particles are dark; EMAN2 expects positive/bright protein).
  - Reference-free: Round 00 initial seeds are unaligned raw averages of each half-set.
  - Evaluation protocol matches AlignOT exactly:
      Independent Even/Odd gold-standard FSC at FSC = 0.143 and 0.5.
  - Generates outputs organized identically to the AlignOT GPU benchmark format:
      round_NN/
        ref_even_round_NN.mrc
        ref_odd_round_NN.mrc
        ref_combined_round_NN.mrc
        fsc_round_NN.csv / fsc_round_NN.png
        slices_round_NN.png
        transforms_even.json / transforms_odd.json
        metrics_particles_even_round_NN.csv / metrics_particles_odd_round_NN.csv
  - Run-level outputs:
        fsc_all_rounds_comparison.png
        slices_all_rounds_comparison.png
        benchmark_summary.json
        timing_report.txt
        halfsets.json
        run.log
"""

import gc
import glob
import json
import logging
import multiprocessing as mp
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# ── Timezone ─────────────────────────────────────────────────────────────────
os.environ["TZ"] = "Asia/Kolkata"
try:
    time.tzset()
except AttributeError:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from EMAN2 import EMData, Region, Averagers, Transform, from_numpy


# ── Global worker state ──────────────────────────────────────────────────────

_SHRINK: int = 3
_LP: float = 0.15
_REF_TMP: str = ""
_STACK_FILE: str = ""
_REFP: Optional[EMData] = None


def _init_worker(shrink: int, lp: float, reftmp: str, stack_file: str) -> None:
    global _SHRINK, _LP, _REF_TMP, _STACK_FILE, _REFP
    _SHRINK = shrink
    _LP = lp
    _REF_TMP = reftmp
    _STACK_FILE = stack_file
    _REFP = EMData(reftmp, 0)


def _prep_inplace(x: EMData) -> None:
    """Invert contrast, normalize, lowpass-filter, and shrink in-place."""
    x.mult(-1.0)  # dark cryo-ET -> positive/bright protein for EMAN2
    x.process_inplace("normalize")
    if _LP is not None and _LP > 0:
        x.process_inplace("filter.lowpass.gauss", {"cutoff_abs": _LP})
    if _SHRINK > 1:
        x.process_inplace("math.meanshrink", {"n": _SHRINK})


def _align_worker(idx: int) -> Tuple[int, dict, float]:
    """Per-particle alignment worker: loads, preps, aligns, returns transforms."""
    p = EMData(_STACK_FILE, idx)
    _prep_inplace(p)
    aln = p.align("rotate_translate_3d_tree", _REFP, {}, "ccc.tomo.thresh", {})
    xf = aln["xform.align3d"]
    score = float(aln["score"])
    return (idx, xf.get_params("eman"), score)


# ── Volume Utilities ─────────────────────────────────────────────────────────

def _emd_to_np(em: EMData) -> np.ndarray:
    """EMData -> numpy array (Z, Y, X)."""
    nz, ny, nx = em.get_zsize(), em.get_ysize(), em.get_xsize()
    return np.array(em.numpy(), copy=True).reshape(nz, ny, nx)


def _np_to_emd(arr: np.ndarray) -> EMData:
    return from_numpy(arr.astype(np.float32))


def load_mrc(path: Path, invert: bool = False) -> np.ndarray:
    """Load an MRC file to numpy, optionally inverting contrast."""
    em = EMData(str(path))
    arr = _emd_to_np(em)
    if invert:
        arr = -arr
    return arr


def save_mrc(path: Path, arr: np.ndarray) -> None:
    """Save numpy array to MRC."""
    path.parent.mkdir(parents=True, exist_ok=True)
    em = _np_to_emd(arr)
    em.write_image(str(path), 0)


# ── HDF Stack Builder ────────────────────────────────────────────────────────

def build_hdf_stack(mrc_files: List[Path], hdf_out: Path, logger: logging.Logger) -> None:
    """Pack MRCs into an HDF stack for rapid random-access from workers."""
    if hdf_out.exists():
        logger.info(f"Using existing HDF stack: {hdf_out}")
        return
    hdf_out.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Building HDF stack: {hdf_out} ({len(mrc_files)} particles)...")
    t0 = time.perf_counter()
    for idx, p in enumerate(mrc_files):
        em = EMData(str(p))
        em.write_image(str(hdf_out), idx)
        if (idx + 1) % 50 == 0 or (idx + 1) == len(mrc_files):
            logger.info(f"  Cached {idx + 1}/{len(mrc_files)} particles...")
    logger.info(f"HDF stack built in {time.perf_counter() - t0:.1f}s -> {hdf_out}")


# ── FSC & Plotting Utilities (Identical formula to AlignOT) ──────────────────

def compute_fsc(
    vol1_np: np.ndarray, vol2_np: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D Fourier Shell Correlation between two volumes (identical to AlignOT)."""
    assert vol1_np.shape == vol2_np.shape
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
        lo, hi = i * shell_width, (i + 1) * shell_width
        mask = (r >= lo) & (r < hi) & (r > 0)
        nv = int(mask.sum())
        n_voxels[i] = nv
        if nv == 0:
            continue
        f1s, f2s = F1[mask], F2[mask]
        num = np.real(np.sum(f1s * np.conj(f2s)))
        den = np.sqrt(np.sum(np.abs(f1s)**2) * np.sum(np.abs(f2s)**2))
        fsc_values[i] = float(num / den) if den > 0 else 0.0

    return shell_centers, fsc_values, n_voxels


def resolution_at_threshold(
    shell_centers: np.ndarray,
    fsc_values: np.ndarray,
    threshold: float,
    apix: float,
    n_voxels: Optional[np.ndarray] = None,
) -> Optional[float]:
    """Interpolated resolution in Å at a given FSC threshold."""
    for i in range(len(fsc_values) - 1):
        if n_voxels is not None and n_voxels[i] <= 1:
            continue
        v1, v2 = fsc_values[i], fsc_values[i + 1]
        f1, f2 = shell_centers[i], shell_centers[i + 1]
        if v1 >= threshold >= v2:
            freq = f1 if v1 == v2 else f1 + (threshold - v1) * (f2 - f1) / (v2 - v1)
            return float(apix / freq) if freq > 0 else None
        if v1 < threshold and i == 0:
            return float(apix / f1) if f1 > 0 else None
    if fsc_values[-1] >= threshold:
        return float(apix / shell_centers[-1])
    return None


def _fsc_ticks(freq_nyq: float) -> List[float]:
    return [1.0 / r for r in [50, 30, 20, 15, 12, 10, 8] if 1.0 / r <= freq_nyq]


def plot_fsc_curve(
    shell_centers: np.ndarray,
    fsc_values: np.ndarray,
    apix: float,
    res_500: Optional[float],
    res_143: Optional[float],
    title: str,
    out_png: Path,
) -> None:
    freq_nyq = 1.0 / (2.0 * apix)
    freqs = shell_centers / apix

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(freqs, fsc_values, "b-o", markersize=3, lw=1.8, label="FSC (Even vs Odd)")
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
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ticks = _fsc_ticks(freq_nyq)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{int(round(1.0/t))}Å" for t in ticks], fontsize=8)
    ax2.set_xlabel(r"Resolution ($\mathrm{\AA}$)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_orthoslices(vol_np: np.ndarray, title: str, out_png: Path) -> None:
    cz, cy, cx = [s // 2 for s in vol_np.shape]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    slices = [("Central XY", vol_np[cz, :, :]),
              ("Central XZ", vol_np[:, cy, :]),
              ("Central YZ", vol_np[:, :, cx])]
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


def plot_all_rounds_fsc(
    fsc_history: List[Tuple],
    apix: float,
    out_png: Path,
) -> None:
    """Overlay FSC curves for every round."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(fsc_history)))
    freq_nyq = 1.0 / (2.0 * apix)
    for (rnd, sc, fv, r143), col in zip(fsc_history, colors):
        lbl = f"Round {rnd:02d}" + (f" ({r143:.1f}Å)" if r143 else "")
        ax.plot(sc / apix, fv, "-o", markersize=3, color=col, lw=1.8, label=lbl)
    ax.axhline(0.143, color="red",  linestyle="--", alpha=0.7, label="FSC=0.143")
    ax.axhline(0.5,   color="gray", linestyle="--", alpha=0.7, label="FSC=0.5")
    ax.set_xlim(0, freq_nyq)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel(r"Spatial Frequency (1/$\mathrm{\AA}$)", fontsize=11)
    ax.set_ylabel("Fourier Shell Correlation", fontsize=11)
    ax.set_title("EMAN2 Iterative Alignment — Gold-Standard Half-Set FSC", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ticks = _fsc_ticks(freq_nyq)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{int(round(1.0/t))}Å" for t in ticks], fontsize=8)
    ax2.set_xlabel(r"Resolution ($\mathrm{\AA}$)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_all_rounds_slices(
    slices_history: List[Tuple],
    cluster_dir: str,
    out_png: Path,
) -> None:
    """Stack central XY/XZ/YZ slices of the combined reference for all rounds."""
    n_rounds = len(slices_history)
    fig, axes = plt.subplots(n_rounds, 3, figsize=(12, 3.5 * n_rounds), squeeze=False)

    snr_tag = ""
    for part in reversed(Path(cluster_dir).parts):
        if part.startswith("0."):
            snr_tag = f"SNR={part}"
            break

    for row, (rnd, vol_np, r143) in enumerate(slices_history):
        cz, cy, cx = [s // 2 for s in vol_np.shape]
        slices = [vol_np[cz, :, :], vol_np[:, cy, :], vol_np[:, :, cx]]
        labels = ["XY", "XZ", "YZ"]
        res_str = f"{r143:.1f}Å" if r143 else "N/A"
        rnd_label = f"Round {rnd:02d} (FSC@0.143: {res_str})"
        for col, (sl, lbl) in enumerate(zip(slices, labels)):
            ax = axes[row][col]
            ax.imshow(sl, cmap="gray", origin="lower", aspect="equal")
            ax.set_title(f"{rnd_label} [{lbl}]", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"EMAN2 All-Rounds Slices Comparison — Combined Reference  {snr_tag}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()


# ── System Info ──────────────────────────────────────────────────────────────

def get_system_info(num_workers: int) -> dict:
    cpu_model = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    return {
        "platform": platform.platform(),
        "processor": cpu_model,
        "machine": platform.machine(),
        "cpu_count_logical": mp.cpu_count(),
        "num_workers_used": num_workers,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }


# ── Logger ───────────────────────────────────────────────────────────────────

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("EMAN2_STA")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── Half-Set Alignment Round (Pure EMAN2) ────────────────────────────────────

def run_half_set_round(
    half_name: str,
    hdf_stack: Path,
    mrc_files: List[Path],
    current_ref_np: np.ndarray,
    round_dir: Path,
    round_idx: int,
    shrink: int,
    lowpass: float,
    num_workers: int,
    logger: logging.Logger,
) -> Tuple[np.ndarray, dict, list]:
    """
    Run one round of EMAN2 alignment for one half-set using pure EMAN2 primitives:
      Aligner   : rotate_translate_3d_tree
      Comparator: ccc.tomo.thresh
      Averager  : mean.tomo

    Returns:
        new_ref_np       : averaged reference (91^3, raw contrast convention)
        metrics_dict     : timing and score summary
        alignment_results: per-particle list of transforms and scores
    """
    t0 = time.perf_counter()
    n = len(mrc_files)
    logger.info(f"--- [{half_name.upper()} HALF] Round {round_idx:02d} Starting ({n} particles, {num_workers} workers) ---")

    # 1. Write shrunk+inverted reference temp file for workers
    reftmp = round_dir / f"_reftmp_{half_name}.hdf"
    ref_em = _np_to_emd(current_ref_np)
    ref_em.mult(-1.0)  # invert for EMAN2 positive density alignment
    ref_em.process_inplace("normalize")
    if lowpass > 0:
        ref_em.process_inplace("filter.lowpass.gauss", {"cutoff_abs": lowpass})
    if shrink > 1:
        ref_em.process_inplace("math.meanshrink", {"n": shrink})
    reftmp.parent.mkdir(parents=True, exist_ok=True)
    ref_em.write_image(str(reftmp), 0)

    # 2. Multiprocessing alignment (pure EMAN2 rotate_translate_3d_tree)
    t_align0 = time.perf_counter()
    indices = list(range(n))
    with mp.Pool(num_workers,
                 initializer=_init_worker,
                 initargs=(shrink, lowpass, str(reftmp), str(hdf_stack))) as pool:
        results = pool.map(_align_worker, indices,
                           chunksize=max(1, n // (num_workers * 2)))
    t_align = time.perf_counter() - t_align0

    # Clean temp reference
    try:
        reftmp.unlink()
    except OSError:
        pass

    # 3. Accumulate full-resolution average using EMAN2 mean.tomo averager
    t_accum0 = time.perf_counter()
    avg = Averagers.get("mean.tomo")
    scores: List[float] = []
    alignment_results: List[dict] = []

    for idx, params, score in results:
        scores.append(score)
        alignment_results.append({
            "index": idx,
            "subtomogram_id": int(mrc_files[idx].stem),
            "file_name": mrc_files[idx].name,
            "score": score,
            "alt": float(params.get("alt", 0.0)),
            "az":  float(params.get("az",  0.0)),
            "phi": float(params.get("phi", 0.0)),
            "tx":  float(params.get("tx",  0.0)) * shrink,
            "ty":  float(params.get("ty",  0.0)) * shrink,
            "tz":  float(params.get("tz",  0.0)) * shrink,
            "mirror": int(params.get("mirror", 0)),
        })
        # Unscale translation and apply to full-res particle
        full_params = dict(params)
        full_params["tx"] = float(params.get("tx", 0.0)) * shrink
        full_params["ty"] = float(params.get("ty", 0.0)) * shrink
        full_params["tz"] = float(params.get("tz", 0.0)) * shrink
        xf = Transform(full_params)
        pr = EMData(str(hdf_stack), idx)
        pr.transform(xf)
        avg.add_image(pr)
        del pr

    new_ref_em = avg.finish()
    new_ref_em.process_inplace("normalize")
    new_ref_np = _emd_to_np(new_ref_em)
    t_accum = time.perf_counter() - t_accum0
    t_total = time.perf_counter() - t0

    mean_score = float(np.mean(scores))
    logger.info(f"[{half_name.upper()} HALF] Round {round_idx:02d} Complete: "
                f"mean_score={mean_score:.5f}, align={t_align:.1f}s, accum={t_accum:.1f}s, "
                f"total={t_total:.1f}s ({t_total/n:.2f}s/ptcl)")

    # Save transforms JSON
    json_out = round_dir / f"transforms_{half_name}.json"
    with open(json_out, "w") as f:
        json.dump({
            "half": half_name,
            "round": round_idx,
            "mean_score": mean_score,
            "align_time_s": t_align,
            "total_time_s": t_total,
            "transforms": alignment_results,
        }, f, indent=2)

    # Save per-particle metrics CSV
    df = pd.DataFrame(alignment_results)
    df["round"] = round_idx
    df.to_csv(round_dir / f"metrics_particles_{half_name}_round_{round_idx:02d}.csv",
              index=False)

    metrics_dict = {
        "mean_score": mean_score,
        "align_time_s": t_align,
        "accum_time_s": t_accum,
        "total_time_s": t_total,
        "sec_per_particle": t_total / n,
    }
    return new_ref_np, metrics_dict, alignment_results


# ── Summary Writers ──────────────────────────────────────────────────────────

def write_summary_files(
    run_dir: Path,
    sys_info: dict,
    args: argparse.Namespace,
    rounds_metrics: list,
    t0: float,
) -> None:
    total_time = time.perf_counter() - t0

    summary_data = {
        "benchmark_metadata": {
            "method": "EMAN2_SPT_Baseline",
            "aligner": "rotate_translate_3d_tree",
            "comparator": "ccc.tomo.thresh",
            "averager": "mean.tomo",
            "contrast_inversion": True,
            "reference_free": True,
            "shrink": args.shrink,
            "lowpass": args.lowpass,
            "data_dir": str(Path(args.cluster_dir).resolve()),
            "total_rounds": args.n_rounds,
            "apix": args.apix,
            "num_workers": args.num_workers,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
            "total_elapsed_seconds": total_time,
            "system_info": sys_info,
        },
        "rounds": rounds_metrics,
    }
    with open(run_dir / "benchmark_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    txt_path = run_dir / "timing_report.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 82 + "\n")
        f.write("EMAN2 SUBTOMOGRAM ALIGNMENT BENCHMARK REPORT\n")
        f.write("=" * 82 + "\n")
        f.write(f"Date & Time        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}\n")
        f.write(f"CPU Model          : {sys_info.get('processor', 'Unknown')} ({sys_info.get('machine', '')})\n")
        f.write(f"Logical Cores      : {sys_info.get('cpu_count_logical')}\n")
        f.write(f"Workers Used       : {args.num_workers}\n")
        f.write(f"Dataset            : {args.cluster_dir}\n")
        f.write(f"Box Dimensions     : 91^3 (shrunk to {91//args.shrink}^3 for alignment)\n")
        f.write(f"Shrink Factor      : {args.shrink}\n")
        f.write(f"Lowpass Cutoff     : {args.lowpass}\n")
        f.write(f"Pixel Size (apix)  : {args.apix} Å\n")
        f.write(f"Contrast Inverted  : Yes (dark cryo-ET -> bright for EMAN2)\n")
        f.write(f"Reference Free     : Yes (raw streaming average seed per half-set)\n")
        f.write(f"Total Pipeline Time: {total_time:.1f}s ({total_time / 60.0:.2f} min)\n")
        f.write("-" * 82 + "\n\n")
        f.write(f"{'Round':<7} | {'Even Time':<11} | {'Odd Time':<11} | {'Total Round':<12} | {'FSC@0.500':<12} | {'FSC@0.143':<12}\n")
        f.write("-" * 82 + "\n")
        for m in rounds_metrics:
            r = m["round"]
            ev_t = f"{m['even_metrics']['total_time_s']:.1f}s" if r > 0 else "-"
            od_t = f"{m['odd_metrics']['total_time_s']:.1f}s" if r > 0 else "-"
            rnd_t = f"{m['round_time_seconds']:.1f}s"
            fsc5  = f"{m['fsc_0500_res_angstrom']:.2f} Å" if m['fsc_0500_res_angstrom'] else "None"
            fsc143 = f"{m['fsc_0143_res_angstrom']:.2f} Å" if m['fsc_0143_res_angstrom'] else "None"
            f.write(f"{r:<7} | {ev_t:<11} | {od_t:<11} | {rnd_t:<12} | {fsc5:<12} | {fsc143:<12}\n")
        f.write("=" * 82 + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gold-Standard Iterative Subtomogram Alignment with EMAN2"
    )
    parser.add_argument("--cluster_dir", type=str, default="Data/0.1/Cluster_0",
                        help="Directory containing subtomogram .mrc files")
    parser.add_argument("--output_base_dir", type=str, default="eman2_baseline/runs",
                        help="Root output directory")
    parser.add_argument("--n_rounds", type=int, default=5,
                        help="Number of iterative alignment rounds")
    parser.add_argument("--n_sbtms", type=int, default=-1,
                        help="Limit particle count (-1 = all)")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="CPU worker processes (physical cores recommended)")
    parser.add_argument("--shrink", type=int, default=3,
                        help="Downsampling factor for alignment (3 -> 30^3 box)")
    parser.add_argument("--lowpass", type=float, default=0.15,
                        help="Gaussian lowpass filter cutoff_abs (Nyquist units)")
    parser.add_argument("--apix", type=float, default=4.0,
                        help="Voxel size in Å")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Custom run folder name suffix")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (for reproducibility)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ── Output directory ────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cluster_p = Path(args.cluster_dir)
    snr_tag = ""
    for part in reversed(cluster_p.parts):
        if part.startswith("0."):
            snr_tag = f"_snr{part}"
            break

    if args.run_name:
        run_dir_name = f"run_{timestamp}_{args.run_name}"
    else:
        run_dir_name = f"run_{timestamp}{snr_tag}_eman2_{args.n_rounds}rounds_{args.num_workers}cpu"

    run_dir = Path(args.output_base_dir) / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Logger ──────────────────────────────────────────────────────────────
    logger = setup_logger(run_dir / "run.log")
    logger.info("=" * 78)
    logger.info("EMAN2 GOLD-STANDARD ITERATIVE SUBTOMOGRAM ALIGNMENT & AVERAGING")
    logger.info("=" * 78)
    logger.info(f"Start Time      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    logger.info(f"Output Directory: {run_dir.resolve()}")
    logger.info(f"Data Directory  : {args.cluster_dir}")
    logger.info(f"Rounds          : {args.n_rounds}")
    logger.info(f"CPU Workers     : {args.num_workers}")
    logger.info(f"Shrink Factor   : {args.shrink} (alignment box: {91 // args.shrink}^3)")
    logger.info(f"Lowpass Cutoff  : {args.lowpass}")
    logger.info(f"Pixel Size      : {args.apix} Å")
    logger.info(f"Contrast        : Inverted (dark cryo-ET -> bright for EMAN2)")
    logger.info(f"Reference Free  : True (raw streaming average seed per half-set)")

    sys_info = get_system_info(args.num_workers)
    for k, v in sys_info.items():
        logger.info(f"  SysInfo {k}: {v}")

    # ── Discover subtomograms ───────────────────────────────────────────────
    all_paths = sorted(
        [Path(f) for f in glob.glob(str(cluster_p / "*.mrc"))
         if not f.endswith("Zone.Identifier")],
        key=lambda x: int(x.stem),
    )
    if args.n_sbtms > 0:
        all_paths = all_paths[:args.n_sbtms]

    num_total = len(all_paths)
    logger.info(f"Total Subtomograms: {num_total}")
    if num_total < 2:
        logger.error("Insufficient subtomograms.")
        sys.exit(1)

    even_paths = [p for p in all_paths if int(p.stem) % 2 == 0]
    odd_paths  = [p for p in all_paths if int(p.stem) % 2 == 1]
    logger.info(f"Even Half-Set: {len(even_paths)} particles (indices 0, 2, ...)")
    logger.info(f"Odd Half-Set : {len(odd_paths)} particles (indices 1, 3, ...)")

    # Save halfsets.json
    halfsets_data = {
        "even_indices": [int(p.stem) for p in even_paths],
        "odd_indices": [int(p.stem) for p in odd_paths],
        "even_count": len(even_paths),
        "odd_count": len(odd_paths),
        "total_particles": num_total
    }
    with open(run_dir / "halfsets.json", "w") as f:
        json.dump(halfsets_data, f, indent=2)

    # ── Build cached HDF stacks ─────────────────────────────────────────────
    hdf_even = run_dir / "particles_even.hdf"
    hdf_odd  = run_dir / "particles_odd.hdf"
    build_hdf_stack(even_paths, hdf_even, logger)
    build_hdf_stack(odd_paths,  hdf_odd,  logger)

    # ── Round 00: Reference-Free Seed references & baseline FSC ─────────────
    round_00_dir = run_dir / "round_00"
    round_00_dir.mkdir(parents=True, exist_ok=True)
    logger.info("\n" + "=" * 50)
    logger.info("ROUND 00: INITIAL SEED REFERENCES & BASELINE FSC (REFERENCE-FREE)")
    logger.info("=" * 50)

    t_r0 = time.perf_counter()
    logger.info("Computing raw streaming averages for even/odd seeds...")
    def raw_avg(paths: List[Path]) -> np.ndarray:
        total = None
        for p in paths:
            v = load_mrc(p, invert=False)
            total = v if total is None else total + v
        return total / len(paths)

    ref_even_np = raw_avg(even_paths)
    ref_odd_np  = raw_avg(odd_paths)

    save_mrc(round_00_dir / "ref_even_round_00.mrc", ref_even_np)
    save_mrc(round_00_dir / "ref_odd_round_00.mrc",  ref_odd_np)

    comb_r0 = 0.5 * (ref_even_np + ref_odd_np)
    save_mrc(round_00_dir / "ref_combined_round_00.mrc", comb_r0)

    sc, fv, nv = compute_fsc(ref_even_np, ref_odd_np)
    res_500 = resolution_at_threshold(sc, fv, 0.5,   args.apix, nv)
    res_143 = resolution_at_threshold(sc, fv, 0.143, args.apix, nv)

    np.savetxt(round_00_dir / "fsc_round_00.csv",
               np.column_stack([sc, sc / args.apix, fv, nv]),
               delimiter=",",
               header="shell_center_1_vox,freq_inv_angstrom,fsc,num_voxels",
               comments="")
    plot_fsc_curve(sc, fv, args.apix, res_500, res_143,
                   f"Round 00 FSC (Unaligned Seed) — FSC@0.143: {f'{res_143:.1f}Å' if res_143 else 'None'}",
                   round_00_dir / "fsc_round_00.png")
    plot_orthoslices(comb_r0, "Round 00 Combined Reference (Unaligned)",
                     round_00_dir / "slices_round_00.png")

    dt_r0 = time.perf_counter() - t_r0
    logger.info(f"Round 00 Complete in {dt_r0:.1f}s | "
                f"FSC@0.5={f'{res_500:.2f}Å' if res_500 else 'None'}, "
                f"FSC@0.143={f'{res_143:.2f}Å' if res_143 else 'None'}")

    fsc_history    = [(0, sc, fv, res_143)]
    slices_history = [(0, comb_r0.copy(), res_143)]
    plot_all_rounds_slices(slices_history, args.cluster_dir,
                           run_dir / "slices_all_rounds_comparison.png")
    rounds_metrics = [{
        "round": 0,
        "round_time_seconds": dt_r0,
        "fsc_0500_res_angstrom": res_500,
        "fsc_0143_res_angstrom": res_143,
        "even_metrics": {"total_time_s": 0.0, "mean_score": 0.0},
        "odd_metrics":  {"total_time_s": 0.0, "mean_score": 0.0},
    }]

    total_t0 = time.perf_counter()
    current_ref_even = ref_even_np
    current_ref_odd  = ref_odd_np

    # ── Iterative rounds ────────────────────────────────────────────────────
    for r in range(1, args.n_rounds + 1):
        round_t0 = time.perf_counter()
        round_str = f"round_{r:02d}"
        round_dir = run_dir / round_str
        round_dir.mkdir(parents=True, exist_ok=True)

        logger.info("\n" + "=" * 60)
        logger.info(f"STARTING ITERATION ROUND {r:02d} / {args.n_rounds:02d}")
        logger.info("=" * 60)

        new_ref_even, even_metrics, _ = run_half_set_round(
            half_name="even",
            hdf_stack=hdf_even,
            mrc_files=even_paths,
            current_ref_np=current_ref_even,
            round_dir=round_dir,
            round_idx=r,
            shrink=args.shrink,
            lowpass=args.lowpass,
            num_workers=args.num_workers,
            logger=logger,
        )

        new_ref_odd, odd_metrics, _ = run_half_set_round(
            half_name="odd",
            hdf_stack=hdf_odd,
            mrc_files=odd_paths,
            current_ref_np=current_ref_odd,
            round_dir=round_dir,
            round_idx=r,
            shrink=args.shrink,
            lowpass=args.lowpass,
            num_workers=args.num_workers,
            logger=logger,
        )

        # Save half-map and combined MRCs
        save_mrc(round_dir / f"ref_even_{round_str}.mrc", new_ref_even)
        save_mrc(round_dir / f"ref_odd_{round_str}.mrc",  new_ref_odd)
        ref_comb = 0.5 * (new_ref_even + new_ref_odd)
        save_mrc(round_dir / f"ref_combined_{round_str}.mrc", ref_comb)

        # FSC (Identical formula to AlignOT)
        sc, fv, nv = compute_fsc(new_ref_even, new_ref_odd)
        res_500 = resolution_at_threshold(sc, fv, 0.5,   args.apix, nv)
        res_143 = resolution_at_threshold(sc, fv, 0.143, args.apix, nv)

        np.savetxt(round_dir / f"fsc_{round_str}.csv",
                   np.column_stack([sc, sc / args.apix, fv, nv]),
                   delimiter=",",
                   header="shell_center_1_vox,freq_inv_angstrom,fsc,num_voxels",
                   comments="")
        plot_fsc_curve(sc, fv, args.apix, res_500, res_143,
                       f"Round {r:02d} Gold-Standard FSC — FSC@0.143: {f'{res_143:.1f}Å' if res_143 else 'None'}",
                       round_dir / f"fsc_{round_str}.png")
        plot_orthoslices(ref_comb, f"Round {r:02d} Combined Reference",
                         round_dir / f"slices_{round_str}.png")

        round_total_time = time.perf_counter() - round_t0
        fsc_history.append((r, sc, fv, res_143))
        slices_history.append((r, ref_comb.copy(), res_143))

        rounds_metrics.append({
            "round": r,
            "round_time_seconds": round_total_time,
            "even_metrics": even_metrics,
            "odd_metrics":  odd_metrics,
            "fsc_0500_res_angstrom": res_500,
            "fsc_0143_res_angstrom": res_143,
        })

        logger.info(f"=== ROUND {r:02d} SUMMARY ===")
        logger.info(f"  Round Duration   : {round_total_time:.1f}s ({round_total_time / 60.0:.2f} min)")
        logger.info(f"  Even Half Time   : {even_metrics['total_time_s']:.1f}s (score={even_metrics['mean_score']:.5f})")
        logger.info(f"  Odd Half Time    : {odd_metrics['total_time_s']:.1f}s (score={odd_metrics['mean_score']:.5f})")
        logger.info(f"  Gold-Std FSC@0.5 : {f'{res_500:.2f}Å' if res_500 else 'None'}")
        logger.info(f"  Gold-Std FSC@.143: {f'{res_143:.2f}Å' if res_143 else 'None'}")

        current_ref_even = new_ref_even
        current_ref_odd  = new_ref_odd

        # Update summary files each round
        plot_all_rounds_fsc(fsc_history, args.apix, run_dir / "fsc_all_rounds_comparison.png")
        plot_all_rounds_slices(slices_history, args.cluster_dir,
                               run_dir / "slices_all_rounds_comparison.png")
        write_summary_files(run_dir, sys_info, args, rounds_metrics, total_t0)
        gc.collect()

    # ── Final ───────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - total_t0
    logger.info("\n" + "=" * 78)
    logger.info("EMAN2 ITERATIVE ALIGNMENT PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"Total Pipeline Runtime: {total_elapsed:.1f}s ({total_elapsed / 60.0:.2f} min)")
    logger.info(f"All outputs saved to  : {run_dir.resolve()}")
    logger.info("=" * 78)


if __name__ == "__main__":
    main()

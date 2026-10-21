#!/usr/bin/env python3
"""
compute_halfset_fsc_all_rounds.py
----------------------------------
Compute Gold-Standard Even/Odd FSC across all rounds (Round 00 to 10) for AlignOT GPU runs:
  Even: alignOT/outputs/gpu_runs/run_20260903_213100_even_gpu_argmax
  Odd : alignOT/outputs/gpu_runs/run_20260903_213156_odd_gpu_argmax
Saves:
  - fsc_all_rounds_comparison.csv
  - fsc_all_rounds_comparison.png
  - fsc_all_rounds_summary.txt
in both the even and odd run directories.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import mrcfile

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

APIX = 4.0  # Voxel size in Angstroms


def load_mrc(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as m:
        return np.array(m.data, dtype=np.float32)


def compute_fsc(vol1: np.ndarray, vol2: np.ndarray):
    assert vol1.shape == vol2.shape, f"Shape mismatch: {vol1.shape} vs {vol2.shape}"
    N = vol1.shape[0]
    n_shells = N // 2

    F1 = np.fft.fftn(vol1.astype(np.float64))
    F2 = np.fft.fftn(vol2.astype(np.float64))

    freqs = [np.fft.fftfreq(s) for s in vol1.shape]
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


def resolution_at_threshold(shell_centers: np.ndarray, fsc_values: np.ndarray, threshold: float, apix: float = APIX):
    for i in range(len(fsc_values) - 1):
        v1, v2 = fsc_values[i], fsc_values[i + 1]
        f1, f2 = shell_centers[i], shell_centers[i + 1]
        if v1 >= threshold >= v2:
            freq = f1 if v1 == v2 else f1 + (threshold - v1) * (f2 - f1) / (v2 - v1)
            return float(apix / freq) if freq > 0 else None
        elif v1 < threshold and i == 0:
            return float(apix / f1) if f1 > 0 else None
    if fsc_values[-1] >= threshold:
        return float(apix / shell_centers[-1])
    return None


def main():
    parser = argparse.ArgumentParser(description="Calculate FSC between even and odd run averages across all rounds.")
    parser.add_argument("--even_dir", type=str, default="alignOT/outputs/gpu_runs/run_20260903_213100_even_gpu_argmax",
                        help="Path to even run directory")
    parser.add_argument("--odd_dir", type=str, default="alignOT/outputs/gpu_runs/run_20260903_213156_odd_gpu_argmax",
                        help="Path to odd run directory")
    parser.add_argument("--apix", type=float, default=4.0, help="Voxel size in Angstroms")
    args = parser.parse_args()

    even_dir = Path(args.even_dir)
    odd_dir = Path(args.odd_dir)
    apix = args.apix

    # Discover all rounds common to both directories
    even_files = {int(p.stem.split("_")[-1]): p for p in even_dir.glob("reference_round_*.mrc") if "inverted" not in p.name}
    odd_files = {int(p.stem.split("_")[-1]): p for p in odd_dir.glob("reference_round_*.mrc") if "inverted" not in p.name}
    
    common_rounds = sorted(set(even_files.keys()) & set(odd_files.keys()))
    if not common_rounds:
        # Fallback to ref_even / ref_odd or other naming if needed
        raise FileNotFoundError(f"No matching reference_round_XX.mrc files found in {even_dir} and {odd_dir}")

    print("=" * 72)
    print("ALIGNOT GPU GOLD-STANDARD HALF-SET FSC (EVEN vs ODD)")
    print(f"Even Directory: {even_dir}")
    print(f"Odd Directory : {odd_dir}")
    print(f"Pixel Size    : {apix} Å")
    print(f"Rounds Found  : {common_rounds}")
    print("=" * 72)
    print(f"{'Round':<7} | {'FSC @ 0.500':<15} | {'FSC @ 0.143 (Gold-Std)':<24}")
    print("-" * 72)

    rounds_data = []
    shell_centers = None

    for r in common_rounds:
        v_even = load_mrc(even_files[r])
        v_odd = load_mrc(odd_files[r])
        sc, fsc, nv = compute_fsc(v_even, v_odd)
        if shell_centers is None:
            shell_centers = sc

        res500 = resolution_at_threshold(sc, fsc, 0.5, apix=apix)
        res143 = resolution_at_threshold(sc, fsc, 0.143, apix=apix)

        r5_str = f"{res500:.2f} Å" if res500 else "None"
        r143_str = f"{res143:.2f} Å" if res143 else "None"
        print(f"{r:<7} | {r5_str:<15} | {r143_str:<24}")

        rounds_data.append({
            "round": r,
            "fsc": fsc,
            "res500": res500,
            "res143": res143
        })

    print("=" * 72)

    # ── 1. Save CSV Table ─────────────────────────────────────────────────────
    freqs_inv_a = shell_centers / apix
    header_cols = ["shell_idx", "spatial_freq_inv_A"] + [f"fsc_round_{d['round']:02d}" for d in rounds_data]
    data_cols = [np.arange(len(shell_centers)), freqs_inv_a] + [d["fsc"] for d in rounds_data]
    data_mat = np.column_stack(data_cols)

    csv_paths = [
        even_dir / "fsc_all_rounds_comparison.csv",
        odd_dir / "fsc_all_rounds_comparison.csv"
    ]
    for p in csv_paths:
        np.savetxt(p, data_mat, delimiter=",", header=",".join(header_cols), comments="", fmt="%.6e")
        print(f"Saved CSV: {p}")

    # ── 2. Save Text Summary ──────────────────────────────────────────────────
    summary_txt = [
        "=" * 72,
        "ALIGNOT GPU (ARGMAX) ALL-ROUNDS GOLD-STANDARD FSC SUMMARY",
        "=" * 72,
        f"Even Directory: {even_dir}",
        f"Odd Directory : {odd_dir}",
        f"Voxel Size    : {apix} Å",
        "-" * 72,
        f"{'Round':<7} | {'FSC @ 0.500':<15} | {'FSC @ 0.143 (Gold-Std)':<24}",
        "-" * 72
    ]
    for d in rounds_data:
        r5_str = f"{d['res500']:.2f} Å" if d['res500'] else "None"
        r143_str = f"{d['res143']:.2f} Å" if d['res143'] else "None"
        summary_txt.append(f"{d['round']:<7} | {r5_str:<15} | {r143_str:<24}")
    summary_txt.append("=" * 72 + "\n")
    summary_content = "\n".join(summary_txt)

    for p in [even_dir / "fsc_all_rounds_summary.txt", odd_dir / "fsc_all_rounds_summary.txt"]:
        with open(p, "w") as f:
            f.write(summary_content)
        print(f"Saved Summary: {p}")

    # ── 3. Generate High-Quality Multi-Round Plot ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6.2))
    nyquist = 1.0 / (2.0 * apix)

    # Use colormap for distinct rounds
    cmap = plt.cm.tab10(np.linspace(0, 1, len(rounds_data)))

    for i, d in enumerate(rounds_data):
        r = d["round"]
        if r == 0:
            col = "#555555"
            sty = "--"
            lw = 2.0
            lbl = f"Round 00 (Seed) — {d['res143']:.1f}Å" if d['res143'] else "Round 00 (Seed)"
        else:
            col = cmap[i % len(cmap)]
            sty = "-"
            lw = 1.8
            lbl = f"Round {r:02d} — {d['res143']:.1f}Å" if d['res143'] else f"Round {r:02d}"

        ax.plot(freqs_inv_a, d["fsc"], color=col, linestyle=sty, lw=lw, label=lbl)

    # Standard Gold-Standard Thresholds
    ax.axhline(0.5, color="black", linestyle=":", alpha=0.6, label="FSC = 0.500")
    ax.axhline(0.143, color="red", linestyle="--", alpha=0.75, label="FSC = 0.143 (Gold-Standard)")

    ax.set_xlim(0, nyquist)
    ax.set_ylim(-0.15, 1.05)
    ax.set_xlabel(r"Spatial Frequency (1/$\mathrm{\AA}$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Fourier Shell Correlation", fontsize=11, fontweight="bold")
    ax.set_title("AlignOT GPU (Argmax) — Gold-Standard Half-Set FSC Progression (All Rounds 00–10)\n"
                 "Even vs. Odd Independent Alignment (200 particles, apix = 4.0 Å)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=9, framealpha=0.92, ncol=2)

    # Top axis for resolution in Angstroms
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    res_ticks = [50.0, 30.0, 25.0, 20.0, 16.0, 12.0, 10.0, 8.0]
    res_ticks = [r for r in res_ticks if (1.0 / r) <= nyquist]
    ax2.set_xticks([1.0 / r for r in res_ticks])
    ax2.set_xticklabels([f"{r:.0f}Å" for r in res_ticks], fontsize=8.5)
    ax2.set_xlabel(r"Resolution ($\mathrm{\AA}$)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_png_even = even_dir / "fsc_all_rounds_comparison.png"
    out_png_odd  = odd_dir  / "fsc_all_rounds_comparison.png"
    plt.savefig(out_png_even, dpi=200)
    plt.savefig(out_png_odd, dpi=200)
    plt.close()
    print(f"Saved Plots:\n  - {out_png_even}\n  - {out_png_odd}")


if __name__ == "__main__":
    main()


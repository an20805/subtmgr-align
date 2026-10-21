#!/usr/bin/env python3
"""
construct_even_odd_averages.py
------------------------------
Reconstructs the Even and Odd half-set averages from the all-particle AlignOT run:
  alignOT/outputs/gpu_runs/run_20260903_152736_all_gpu_argmax

Rotates the raw subtomograms using the saved quaternions and the exact
centroid offset math:
    offset = Bbar + center - R_inv @ (Abar + center)
Splits particles into even (0, 2, 4...) and odd (1, 3, 5...) sets,
averages them, computes Gold-Standard FSC, and generates comparison plots.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "alignOT" / "src"
sys.path.insert(0, str(SRC_DIR))
import alignot_utils

APIX = 4.0  # Voxel size in Angstroms


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


def _rotate_single_subtomogram(task_args):
    idx, path, q_vals, Abar, lowpass_sigma, threshold, num_points, seed_ptcl = task_args
    vol = alignot_utils.load_mrc(path, permissive=True)
    src_prep = alignot_utils.prepare_sampling_volume(vol, sigma=lowpass_sigma)
    pts_src = alignot_utils.sample_volume_in_memory(
        src_prep, threshold=threshold, num_points=num_points, random_seed=seed_ptcl
    )
    Bbar = pts_src.mean(axis=0).astype(np.float64)
    rot_vol = alignot_utils.apply_rotation_to_volume(vol, q_vals, Bbar, Abar)
    subtom_id = int(path.stem)
    return idx, subtom_id, rot_vol


def plot_single_fsc(shell_centers, fsc_vals, res_500, res_143, title, out_png, apix=APIX):
    freq_nyq = 1.0 / (2.0 * apix)
    freqs_inv_a = shell_centers / apix

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(freqs_inv_a, fsc_vals, "b-o", markersize=3.5, lw=2.0, label="FSC (Even vs Odd Average)")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7,
               label=f"FSC = 0.500 ({res_500:.1f}Å)" if res_500 else "FSC = 0.500")
    ax.axhline(0.143, color="red", linestyle="--", alpha=0.7,
               label=f"FSC = 0.143 ({res_143:.1f}Å)" if res_143 else "FSC = 0.143")

    if res_143:
        ax.axvline(1.0 / res_143, color="red", linestyle=":", alpha=0.6)
    if res_500:
        ax.axvline(1.0 / res_500, color="gray", linestyle=":", alpha=0.6)

    ax.set_xlim(0, freq_nyq)
    ax.set_ylim(-0.15, 1.05)
    ax.set_xlabel(r"Spatial Frequency (1/$\mathrm{\AA}$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Fourier Shell Correlation", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    # Secondary resolution axis in Angstroms
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    res_ticks = [50.0, 30.0, 25.0, 20.0, 16.0, 12.0, 10.0, 8.0]
    res_ticks = [r for r in res_ticks if (1.0 / r) <= freq_nyq]
    ax2.set_xticks([1.0 / r for r in res_ticks])
    ax2.set_xticklabels([f"{r:.0f}Å" for r in res_ticks], fontsize=8.5)
    ax2.set_xlabel(r"Resolution ($\mathrm{\AA}$)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_slices(vol, title, out_png):
    cz, cy, cx = [s // 2 for s in vol.shape]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    slices = [
        ("Central XY", vol[cz, :, :]),
        ("Central XZ", vol[:, cy, :]),
        ("Central YZ", vol[:, :, cx])
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


def main():
    parser = argparse.ArgumentParser(description="Construct Even/Odd averages and FSC for all-particle run.")
    parser.add_argument("--run_dir", type=str,
                        default="alignOT/outputs/gpu_runs/run_20260903_152736_all_gpu_argmax",
                        help="Path to all-particle run directory")
    parser.add_argument("--cluster_dir", type=str, default="Data/0.1/Cluster_0",
                        help="Directory with subtomogram MRC files")
    parser.add_argument("--num_workers", type=int, default=16, help="Parallel CPU workers")
    parser.add_argument("--apix", type=float, default=4.0, help="Voxel size in Angstroms")
    parser.add_argument("--target_round", type=int, default=5, help="Alignment round to reconstruct (default: 5)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cluster_dir = Path(args.cluster_dir)
    num_workers = args.num_workers
    apix = args.apix
    target_round = args.target_round

    seed = 42
    lowpass_sigma = 2.0
    threshold = 0.7
    num_points = 500

    cluster_paths = alignot_utils.get_cluster_paths(cluster_dir)
    print(f"Discovered {len(cluster_paths)} subtomograms in {cluster_dir}")

    # ── 1. Unaligned Round 00 Even/Odd Average ────────────────────────────────
    print("\n--- Constructing Round 00 (Raw Unaligned) Even/Odd Averages ---")
    even_paths_r0 = [p for p in cluster_paths if int(p.stem) % 2 == 0]
    odd_paths_r0  = [p for p in cluster_paths if int(p.stem) % 2 == 1]
    
    print(f"Even particles: {len(even_paths_r0)}, Odd particles: {len(odd_paths_r0)}")
    even_vols_r0 = [alignot_utils.load_mrc(p, permissive=True) for p in even_paths_r0]
    odd_vols_r0  = [alignot_utils.load_mrc(p, permissive=True) for p in odd_paths_r0]

    even_avg_r0 = np.mean(even_vols_r0, axis=0).astype(np.float32)
    odd_avg_r0  = np.mean(odd_vols_r0, axis=0).astype(np.float32)
    comb_avg_r0 = 0.5 * (even_avg_r0 + odd_avg_r0)

    alignot_utils.save_mrc(run_dir / "ref_even_round_00.mrc", even_avg_r0)
    alignot_utils.save_mrc(run_dir / "ref_odd_round_00.mrc", odd_avg_r0)
    alignot_utils.save_mrc(run_dir / "ref_combined_round_00.mrc", comb_avg_r0)

    sc_r0, fsc_r0, nv_r0 = compute_fsc(even_avg_r0, odd_avg_r0)
    res_500_r0 = resolution_at_threshold(sc_r0, fsc_r0, 0.5, apix=apix)
    res_143_r0 = resolution_at_threshold(sc_r0, fsc_r0, 0.143, apix=apix)
    print(f"Round 00 (Unaligned) FSC: FSC@0.5 = {res_500_r0:.2f}Å, FSC@0.143 = {res_143_r0:.2f}Å")

    # ── 2. Reconstruct Target Round (Round 05) ─────────────────────────────────
    print(f"\n--- Reconstructing Round {target_round:02d} (Aligned) Even/Odd Averages ---")
    ref_prev_path = run_dir / f"reference_round_{target_round - 1:02d}.mrc"
    ref_prev = alignot_utils.load_mrc(ref_prev_path, permissive=True)

    # Reference TRN sampling
    ref_prep = alignot_utils.prepare_sampling_volume(ref_prev, sigma=lowpass_sigma)
    pts_ref = alignot_utils.sample_volume_in_memory(
        ref_prep, threshold=threshold, num_points=num_points, random_seed=seed + 1000 * target_round
    )
    Abar = pts_ref.mean(axis=0).astype(np.float64)
    print(f"Reference Round {target_round - 1:02d} Centroid (Abar): {Abar}")

    # Load quaternions
    csv_q = run_dir / f"metrics_particles_round_{target_round:02d}.csv"
    if not csv_q.exists():
        csv_q = run_dir / "quaternions.csv"
    df_q = pd.read_csv(csv_q)
    print(f"Loaded {len(df_q)} quaternions from {csv_q.name}")

    # Build tasks for parallel rotation
    tasks = []
    for idx, path in enumerate(cluster_paths):
        subtom_id = int(path.stem)
        row = df_q[df_q["subtomogram_id"] == subtom_id].iloc[0]
        q_vals = [row["q_w"], row["q_x"], row["q_y"], row["q_z"]]
        seed_ptcl = seed + target_round * 1000 + idx
        tasks.append((idx, path, q_vals, Abar, lowpass_sigma, threshold, num_points, seed_ptcl))

    print(f"Applying rotations to {len(tasks)} subtomograms across {num_workers} workers...")
    t0_rot = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_rotate_single_subtomogram, tasks))
    print(f"Transformation completed in {time.perf_counter() - t0_rot:.2f}s")

    # Split into even and odd sets
    even_rotated = []
    odd_rotated = []

    for idx, subtom_id, rot_vol in results:
        if subtom_id % 2 == 0:
            even_rotated.append(rot_vol)
        else:
            odd_rotated.append(rot_vol)

    print(f"Accumulating {len(even_rotated)} Even volumes and {len(odd_rotated)} Odd volumes...")
    even_avg = np.mean(even_rotated, axis=0).astype(np.float32)
    odd_avg  = np.mean(odd_rotated, axis=0).astype(np.float32)
    comb_avg = 0.5 * (even_avg + odd_avg)

    # Save reconstructed MRC files
    ref_even_path = run_dir / f"ref_even_round_{target_round:02d}.mrc"
    ref_odd_path  = run_dir / f"ref_odd_round_{target_round:02d}.mrc"
    ref_comb_path = run_dir / f"ref_combined_round_{target_round:02d}.mrc"

    alignot_utils.save_mrc(ref_even_path, even_avg)
    alignot_utils.save_mrc(ref_odd_path, odd_avg)
    alignot_utils.save_mrc(ref_comb_path, comb_avg)
    print(f"Saved reconstructed MRC volumes:\n  - {ref_even_path}\n  - {ref_odd_path}\n  - {ref_comb_path}")

    # ── 3. Compute Gold-Standard FSC ──────────────────────────────────────────
    sc, fsc, nv = compute_fsc(even_avg, odd_avg)
    res_500 = resolution_at_threshold(sc, fsc, 0.5, apix=apix)
    res_143 = resolution_at_threshold(sc, fsc, 0.143, apix=apix)

    res_500_str = f"{res_500:.2f} Å" if res_500 else "None"
    res_143_str = f"{res_143:.2f} Å" if res_143 else "None"
    print("\n" + "=" * 60)
    print(f"ROUND {target_round:02d} EVEN vs ODD GOLD-STANDARD FSC:")
    print(f"  FSC @ 0.500: {res_500_str}")
    print(f"  FSC @ 0.143: {res_143_str}")
    print("=" * 60)

    # ── 4. Save FSC CSV Table ─────────────────────────────────────────────────
    fsc_csv_path = run_dir / f"fsc_round_{target_round:02d}.csv"
    np.savetxt(
        fsc_csv_path,
        np.column_stack([sc, sc / apix, fsc, nv]),
        delimiter=",",
        header="shell_center_1_vox,freq_inv_angstrom,fsc,num_voxels",
        comments="",
        fmt="%.6e"
    )
    print(f"Saved FSC table to: {fsc_csv_path}")

    # ── 5. Plot FSC Curves ────────────────────────────────────────────────────
    # A. Target Round FSC Plot
    fsc_png_path = run_dir / f"fsc_round_{target_round:02d}.png"
    plot_single_fsc(
        sc, fsc, res_500, res_143,
        title=f"AlignOT GPU (All Particles) — Round {target_round:02d} Gold-Standard FSC\n"
              f"Even (100) vs. Odd (100) Average (FSC@0.143 = {res_143_str})",
        out_png=fsc_png_path,
        apix=apix
    )
    print(f"Saved Round {target_round:02d} FSC plot to: {fsc_png_path}")

    # B. Round 00 vs Round 05 Comparison FSC Plot
    comp_png_path = run_dir / "fsc_comparison_r00_vs_r05.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    nyquist = 1.0 / (2.0 * apix)
    freqs_inv_a = sc / apix

    ax.plot(freqs_inv_a, fsc_r0, "--", color="#7f7f7f", lw=2.0,
            label=f"Round 00 (Raw Seed) — {res_143_r0:.1f}Å" if res_143_r0 else "Round 00 (Raw Seed)")
    ax.plot(freqs_inv_a, fsc, "-o", color="#1f77b4", markersize=3.5, lw=2.2,
            label=f"Round 05 (Aligned) — {res_143:.1f}Å" if res_143 else f"Round 05 (Aligned)")

    ax.axhline(0.5, color="black", linestyle=":", alpha=0.6, label="FSC = 0.500")
    ax.axhline(0.143, color="red", linestyle="--", alpha=0.75, label="FSC = 0.143 (Gold-Standard)")

    if res_143:
        ax.axvline(1.0 / res_143, color="#1f77b4", linestyle=":", alpha=0.5)

    ax.set_xlim(0, nyquist)
    ax.set_ylim(-0.15, 1.05)
    ax.set_xlabel(r"Spatial Frequency (1/$\mathrm{\AA}$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Fourier Shell Correlation", fontsize=11, fontweight="bold")
    ax.set_title(f"AlignOT GPU All-Particle Run — Gold-Standard Half-Set FSC\n"
                 f"Raw Seed vs. Round {target_round:02d} Aligned Average (200 particles)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    res_ticks = [50.0, 30.0, 25.0, 20.0, 16.0, 12.0, 10.0, 8.0]
    res_ticks = [r for r in res_ticks if (1.0 / r) <= nyquist]
    ax2.set_xticks([1.0 / r for r in res_ticks])
    ax2.set_xticklabels([f"{r:.0f}Å" for r in res_ticks], fontsize=8.5)
    ax2.set_xlabel(r"Resolution ($\mathrm{\AA}$)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(comp_png_path, dpi=200)
    plt.close()
    print(f"Saved Comparison FSC plot to: {comp_png_path}")

    # ── 6. Plot Slices for Even, Odd, and Combined Averages ────────────────────
    plot_slices(even_avg, f"Round {target_round:02d} Even Average (100 particles)", run_dir / f"slices_even_round_{target_round:02d}.png")
    plot_slices(odd_avg, f"Round {target_round:02d} Odd Average (100 particles)", run_dir / f"slices_odd_round_{target_round:02d}.png")
    plot_slices(comb_avg, f"Round {target_round:02d} Combined Average (200 particles)", run_dir / f"slices_combined_round_{target_round:02d}.png")
    print("Saved slice inspection plots for Even, Odd, and Combined averages.")

    # ── 7. Text Summary ───────────────────────────────────────────────────────
    summary_txt = [
        "=" * 70,
        f"ALIGNOT ALL-PARTICLE RUN (ROUND {target_round:02d}) HALF-SET FSC SUMMARY",
        "=" * 70,
        f"Run Directory: {run_dir}",
        f"Dataset      : {cluster_dir} (200 particles)",
        f"Pixel Size   : {apix} Å",
        "-" * 70,
        f"{'State':<25} | {'FSC @ 0.500':<18} | {'FSC @ 0.143':<18}",
        "-" * 70,
        f"{'Round 00 (Raw Unaligned)':<25} | {f'{res_500_r0:.2f} Å' if res_500_r0 else 'None':<18} | {f'{res_143_r0:.2f} Å' if res_143_r0 else 'None':<18}",
        f"{f'Round {target_round:02d} (Aligned)':<25} | {res_500_str:<18} | {res_143_str:<18}",
        "=" * 70,
        "",
        "Generated Files:",
        f"  - Even Average    : {ref_even_path.name}",
        f"  - Odd Average     : {ref_odd_path.name}",
        f"  - Combined Average: {ref_comb_path.name}",
        f"  - FSC Curve (CSV) : {fsc_csv_path.name}",
        f"  - FSC Plot (PNG)  : {fsc_png_path.name}",
        f"  - Comparison Plot : {comp_png_path.name}",
    ]
    summary_path = run_dir / f"halfset_fsc_summary_round_{target_round:02d}.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_txt) + "\n")
    print(f"Saved summary report to: {summary_path}")


if __name__ == "__main__":
    main()


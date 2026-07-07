#!/usr/bin/env python3
"""
compute_fsc.py — Fourier Shell Correlation evaluation tool.

Three modes:
  halfset   : compute FSC between the even-half average and odd-half average
              of a raw subtomogram cluster. Measures the data ceiling (no alignment).
  compare   : compute FSC between any two .mrc volumes.
  vs_baseline: compute FSC between the final reference in an experiment folder
              and the EMAN2 baseline reference.

Usage examples
--------------
# Mode 1 — raw half-set FSC (run once per dataset)
python scripts/compute_fsc.py halfset \\
    --cluster_dir /abs/path/Data/0.1/Cluster_0 \\
    --output experiments/halfset_fsc_snr01 \\
    --apix 4.0

# Mode 2 — compare two volumes (aligned halves, or any pair)
python scripts/compute_fsc.py compare \\
    --vol1 experiments/exp004_even/run_.../reference_round_01.mrc \\
    --vol2 experiments/exp004_odd/run_.../reference_round_01.mrc \\
    --label1 "even half" --label2 "odd half" \\
    --output experiments/exp004_halfset_fsc \\
    --apix 4.0

# Mode 3 — quick proxy: experiment final ref vs EMAN2 baseline
python scripts/compute_fsc.py vs_baseline \\
    --exp_dir experiments/exp004_empot_lowpass \\
    --baseline eman2_baseline/baseline_reference_91.mrc \\
    --apix 4.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import mrcfile
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_mrc(path):
    """Load an MRC file and return a float32 numpy array."""
    path = Path(path)
    if not path.exists():
        sys.exit(f"[ERROR] File not found: {path}")
    with mrcfile.open(str(path), permissive=True) as mrc:
        return np.array(mrc.data, dtype=np.float32)


def save_mrc(path, arr):
    arr = np.asarray(arr, dtype=np.float32)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(arr)


def get_cluster_paths(cluster_dir):
    """Return sorted list of .mrc paths (ignores Zone.Identifier sidecars)."""
    cluster_dir = Path(cluster_dir)
    paths = [p for p in cluster_dir.iterdir()
             if p.is_file() and p.suffix == ".mrc" and "Zone" not in p.name]
    return sorted(paths, key=lambda p: int(p.stem))


# ─────────────────────────────────────────────────────────────────────────────
# Volume utilities
# ─────────────────────────────────────────────────────────────────────────────

def match_sizes(vol1, vol2):
    """
    Crop both volumes to the smallest common box size (centred crop).
    Returns (vol1_cropped, vol2_cropped).
    """
    target = tuple(min(a, b) for a, b in zip(vol1.shape, vol2.shape))
    def centre_crop(v, tgt):
        slices = tuple(
            slice((s - t) // 2, (s - t) // 2 + t)
            for s, t in zip(v.shape, tgt)
        )
        return v[slices]
    return centre_crop(vol1, target), centre_crop(vol2, target)


def streaming_average(paths, progress_every=20):
    """Compute the average of a list of MRC files without loading all at once."""
    total = None
    count = 0
    for idx, p in enumerate(paths, 1):
        arr = load_mrc(p)
        if total is None:
            total = np.zeros(arr.shape, dtype=np.float64)
        total += arr
        count += 1
        if progress_every and (idx % progress_every == 0 or idx == len(paths)):
            print(f"  Averaged {idx}/{len(paths)} subtomograms...")
    if count == 0:
        sys.exit("[ERROR] No subtomograms found.")
    return (total / count).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Core FSC computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_fsc(vol1, vol2):
    """
    Compute the Fourier Shell Correlation between two real-space volumes.

    Both volumes must have the same shape (call match_sizes first if needed).

    Returns
    -------
    shell_centers : np.ndarray, shape (n_shells,)
        Spatial frequency of each shell in units of 1/voxel
        (0 = DC, 0.5 = Nyquist).
    fsc_values : np.ndarray, shape (n_shells,)
        FSC value in each shell, in [-1, 1].
    n_voxels : np.ndarray, shape (n_shells,)
        Number of Fourier voxels contributing to each shell.
    """
    assert vol1.shape == vol2.shape, \
        f"Volume shapes must match: {vol1.shape} vs {vol2.shape}"

    N = vol1.shape[0]  # assume cubic; works for non-cubic too (uses min dim)
    n_shells = N // 2  # shells from 0 to Nyquist

    # 3D FFT of both volumes
    F1 = np.fft.fftn(vol1.astype(np.float64))
    F2 = np.fft.fftn(vol2.astype(np.float64))

    # Build a grid of squared radii in frequency space (shifted so DC is centre)
    # freq coords run from -0.5 to +0.5
    freqs = [np.fft.fftfreq(s) for s in vol1.shape]
    fz, fy, fx = np.meshgrid(freqs[0], freqs[1], freqs[2], indexing="ij")
    r = np.sqrt(fz**2 + fy**2 + fx**2)   # frequency magnitude (0..~0.87)

    # Bin voxels into shells of width 1/N
    shell_width = 0.5 / n_shells          # each shell spans this range
    shell_centers = np.arange(n_shells) * shell_width + shell_width / 2

    fsc_values = np.zeros(n_shells)
    n_voxels   = np.zeros(n_shells, dtype=int)

    for i in range(n_shells):
        lo = i * shell_width
        hi = lo + shell_width
        # Exclude the DC component (r=0, the volume mean) — it is always
        # numerically unstable for normalized volumes and carries no
        # resolution information.  Apply r > 0 so the DC voxel is never
        # included in any shell.
        mask = (r >= lo) & (r < hi) & (r > 0)
        n_vox = int(mask.sum())
        n_voxels[i] = n_vox
        if n_vox == 0:
            fsc_values[i] = 0.0
            continue
        f1_shell = F1[mask]
        f2_shell = F2[mask]
        numerator   = np.real(np.sum(f1_shell * np.conj(f2_shell)))
        denominator = np.sqrt(
            np.sum(np.abs(f1_shell)**2) * np.sum(np.abs(f2_shell)**2)
        )
        fsc_values[i] = numerator / denominator if denominator > 0 else 0.0

    return shell_centers, fsc_values, n_voxels


def resolution_at_threshold(shell_centers, fsc_values, threshold, apix, n_voxels=None):
    """
    Return the spatial resolution (in Angstroms) where FSC first drops
    below `threshold`.  Returns None if FSC never drops below threshold.
    apix      : pixel size in Angstroms.
    n_voxels  : optional array; shells with <= 1 voxel (DC shell) are skipped.
    """
    for i, (freq, val) in enumerate(zip(shell_centers, fsc_values)):
        if freq <= 0:
            continue
        # Skip the DC/near-DC shell — it contains only the volume mean and
        # is numerically unstable for zero-mean normalized volumes.
        if n_voxels is not None and n_voxels[i] <= 1:
            continue
        if val < threshold:
            return apix / freq
    return None   # FSC stays above threshold up to Nyquist


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_fsc(shell_centers, fsc_curves, labels, apix, title, output_path,
             n_voxels_list=None):
    """
    Plot one or more FSC curves on the same axes.

    Parameters
    ----------
    fsc_curves      : list of np.ndarray  — one per curve
    labels          : list of str
    n_voxels_list   : optional list of np.ndarray — one per curve;
                      shells with n_voxels == 0 (empty DC shell) are excluded
                      from the plot so the curve starts at the first real shell.
    """
    # x-axis: resolution in Angstroms (apix / freq), skipping DC (freq=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        resolution_A = np.where(shell_centers > 0, apix / shell_centers, np.inf)

    # Nyquist resolution = 2 * apix
    nyquist_A = 2.0 * apix

    fig, ax = plt.subplots(figsize=(9, 5))

    x_max = nyquist_A  # will grow to fit all curve start points
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(fsc_curves)))

    for idx, (fsc, label, color) in enumerate(zip(fsc_curves, labels, colors)):
        # Build mask: only plot shells that have actual data (n_voxels > 0)
        if n_voxels_list is not None:
            n_vox = n_voxels_list[idx]
            valid = (n_vox > 0) & (resolution_A < np.inf)
        else:
            valid = resolution_A < np.inf

        res_plot = resolution_A[valid]
        fsc_plot = fsc[valid]

        if len(res_plot) == 0:
            continue

        ax.plot(res_plot, fsc_plot, label=label, color=color, linewidth=2)

        # Track the coarsest resolution plotted (for x-axis limit)
        x_max = max(x_max, res_plot.max())

    # Standard threshold lines
    ax.axhline(0.5,   color="grey",  linestyle="--", linewidth=1, label="FSC = 0.5")
    ax.axhline(0.143, color="black", linestyle=":",  linewidth=1, label="FSC = 0.143")

    ax.set_xlabel("Resolution (Å)", fontsize=12)
    ax.set_ylabel("FSC", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(-0.1, 1.05)

    # x-axis: high resolution (small Å) on right, low resolution (large Å) on left
    # Pad the left limit by 10% so the curve doesn't start at the plot edge
    ax.set_xlim(nyquist_A, x_max * 1.10)
    ax.invert_xaxis()

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [saved] {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────────

def mode_halfset(args):
    """Average even/odd halves of raw cluster, compute FSC between them."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_paths = get_cluster_paths(args.cluster_dir)
    print(f"Found {len(all_paths)} subtomograms in {args.cluster_dir}")

    even_paths = [p for p in all_paths if int(p.stem) % 2 == 0]
    odd_paths  = [p for p in all_paths if int(p.stem) % 2 == 1]
    print(f"Even half: {len(even_paths)} files  |  Odd half: {len(odd_paths)} files")

    print("Computing even-half average...")
    avg_even = streaming_average(even_paths)
    save_mrc(output_dir / "avg_even_raw.mrc", avg_even)

    print("Computing odd-half average...")
    avg_odd = streaming_average(odd_paths)
    save_mrc(output_dir / "avg_odd_raw.mrc", avg_odd)

    # Ensure same size before FSC
    avg_even, avg_odd = match_sizes(avg_even, avg_odd)

    print("Computing FSC...")
    shell_centers, fsc_values, n_voxels = compute_fsc(avg_even, avg_odd)

    # Report resolutions
    res_05  = resolution_at_threshold(shell_centers, fsc_values, 0.5,   args.apix, n_voxels)
    res_143 = resolution_at_threshold(shell_centers, fsc_values, 0.143, args.apix, n_voxels)
    print(f"\n{'─'*40}")
    print(f"  Half-set FSC (raw data, no alignment)")
    print(f"  Resolution @ FSC=0.5  : {f'{res_05:.1f} Å' if res_05 else 'not reached'}")
    print(f"  Resolution @ FSC=0.143: {f'{res_143:.1f} Å' if res_143 else 'not reached'}")
    print(f"{'─'*40}\n")

    # Save CSV
    import pandas as pd
    df = pd.DataFrame({
        "shell_freq":    shell_centers,
        "resolution_A":  np.where(shell_centers > 0, args.apix / shell_centers, np.inf),
        "fsc":           fsc_values,
        "n_voxels":      n_voxels,
    })
    csv_path = output_dir / "fsc_halfset_raw.csv"
    df.to_csv(csv_path, index=False)
    print(f"  [saved] {csv_path}")

    # Plot
    plot_fsc(
        shell_centers, [fsc_values],
        labels=[f"Half-set FSC (raw, no alignment)  |  apix={args.apix}Å"],
        apix=args.apix,
        title="Half-Set FSC — Raw Data (Floor / Ceiling Estimate)",
        output_path=output_dir / "fsc_halfset_raw.png",
        n_voxels_list=[n_voxels],
    )

    # Save summary txt
    summary = (
        f"Half-set FSC — raw data, no alignment\n"
        f"Cluster: {args.cluster_dir}\n"
        f"Even files: {len(even_paths)}  |  Odd files: {len(odd_paths)}\n"
        f"Pixel size: {args.apix} Å\n"
        f"Resolution @ FSC=0.5  : {f'{res_05:.1f} Å' if res_05 else 'not reached'}\n"
        f"Resolution @ FSC=0.143: {f'{res_143:.1f} Å' if res_143 else 'not reached'}\n"
    )
    (output_dir / "summary.txt").write_text(summary)
    print(summary)


def mode_compare(args):
    """Compute FSC between two explicit .mrc files."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading vol1: {args.vol1}")
    vol1 = load_mrc(args.vol1)
    print(f"Loading vol2: {args.vol2}")
    vol2 = load_mrc(args.vol2)

    if vol1.shape != vol2.shape:
        print(f"  Shapes differ ({vol1.shape} vs {vol2.shape}) — applying centred crop.")
        vol1, vol2 = match_sizes(vol1, vol2)
        print(f"  Cropped to {vol1.shape}")

    print("Computing FSC...")
    shell_centers, fsc_values, n_voxels = compute_fsc(vol1, vol2)

    label1 = args.label1 if args.label1 else Path(args.vol1).name
    label2 = args.label2 if args.label2 else Path(args.vol2).name

    res_05  = resolution_at_threshold(shell_centers, fsc_values, 0.5,   args.apix, n_voxels)
    res_143 = resolution_at_threshold(shell_centers, fsc_values, 0.143, args.apix, n_voxels)
    print(f"\n{'─'*40}")
    print(f"  FSC: {label1}  vs  {label2}")
    print(f"  Resolution @ FSC=0.5  : {f'{res_05:.1f} Å' if res_05 else 'not reached'}")
    print(f"  Resolution @ FSC=0.143: {f'{res_143:.1f} Å' if res_143 else 'not reached'}")
    print(f"{'─'*40}\n")

    import pandas as pd
    df = pd.DataFrame({
        "shell_freq":   shell_centers,
        "resolution_A": np.where(shell_centers > 0, args.apix / shell_centers, np.inf),
        "fsc":          fsc_values,
        "n_voxels":     n_voxels,
    })
    csv_path = output_dir / "fsc_compare.csv"
    df.to_csv(csv_path, index=False)
    print(f"  [saved] {csv_path}")

    plot_fsc(
        shell_centers, [fsc_values],
        labels=[f"{label1}  vs  {label2}"],
        apix=args.apix,
        title=f"FSC Comparison  |  apix={args.apix}Å",
        output_path=output_dir / "fsc_compare.png",
        n_voxels_list=[n_voxels],
    )

    summary = (
        f"FSC comparison\n"
        f"Vol1: {args.vol1}\n"
        f"Vol2: {args.vol2}\n"
        f"Pixel size: {args.apix} Å\n"
        f"Resolution @ FSC=0.5  : {f'{res_05:.1f} Å' if res_05 else 'not reached'}\n"
        f"Resolution @ FSC=0.143: {f'{res_143:.1f} Å' if res_143 else 'not reached'}\n"
    )
    (output_dir / "summary.txt").write_text(summary)
    print(summary)


def mode_vs_baseline(args):
    """
    Find the final reference MRC in an experiment folder and compute
    FSC against the EMAN2 baseline.
    """
    exp_dir  = Path(args.exp_dir)
    baseline = Path(args.baseline)

    if not exp_dir.exists():
        sys.exit(f"[ERROR] Experiment directory not found: {exp_dir}")
    if not baseline.exists():
        sys.exit(f"[ERROR] Baseline file not found: {baseline}")

    # Find the highest-numbered reference_round_XX.mrc inside the run subfolder
    run_dirs = sorted(exp_dir.glob("run_*"))
    if run_dirs:
        search_dir = run_dirs[-1]   # most recent run inside the experiment folder
    else:
        search_dir = exp_dir        # flat structure

    ref_files = sorted(search_dir.glob("reference_round_*.mrc"))
    if not ref_files:
        sys.exit(f"[ERROR] No reference_round_*.mrc files found under {search_dir}")

    final_ref = ref_files[-1]
    print(f"Using final reference: {final_ref}")
    print(f"Baseline:              {baseline}")

    vol_exp  = load_mrc(final_ref)
    vol_base = load_mrc(baseline)

    if vol_exp.shape != vol_base.shape:
        print(f"  Shapes differ ({vol_exp.shape} vs {vol_base.shape}) — applying centred crop.")
        vol_exp, vol_base = match_sizes(vol_exp, vol_base)
        print(f"  Cropped to {vol_exp.shape}")

    print("Computing FSC...")
    shell_centers, fsc_values, n_voxels = compute_fsc(vol_exp, vol_base)

    res_05  = resolution_at_threshold(shell_centers, fsc_values, 0.5,   args.apix, n_voxels)
    res_143 = resolution_at_threshold(shell_centers, fsc_values, 0.143, args.apix, n_voxels)
    print(f"\n{'─'*40}")
    print(f"  FSC: experiment final ref  vs  EMAN2 baseline")
    print(f"  Resolution @ FSC=0.5  : {f'{res_05:.1f} Å' if res_05 else 'not reached'}")
    print(f"  Resolution @ FSC=0.143: {f'{res_143:.1f} Å' if res_143 else 'not reached'}")
    print(f"{'─'*40}\n")

    # Save into experiment folder
    output_dir = exp_dir / "fsc_vs_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    df = pd.DataFrame({
        "shell_freq":   shell_centers,
        "resolution_A": np.where(shell_centers > 0, args.apix / shell_centers, np.inf),
        "fsc":          fsc_values,
        "n_voxels":     n_voxels,
    })
    csv_path = output_dir / "fsc_vs_baseline.csv"
    df.to_csv(csv_path, index=False)
    print(f"  [saved] {csv_path}")

    plot_fsc(
        shell_centers, [fsc_values],
        labels=[f"{exp_dir.name}  vs  EMAN2 baseline"],
        apix=args.apix,
        title=f"FSC vs EMAN2 Baseline  |  apix={args.apix}Å",
        output_path=output_dir / "fsc_vs_baseline.png",
        n_voxels_list=[n_voxels],
    )

    summary = (
        f"FSC vs EMAN2 baseline\n"
        f"Experiment:  {exp_dir}\n"
        f"Final ref:   {final_ref}\n"
        f"Baseline:    {baseline}\n"
        f"Pixel size:  {args.apix} Å\n"
        f"Resolution @ FSC=0.5  : {f'{res_05:.1f} Å' if res_05 else 'not reached'}\n"
        f"Resolution @ FSC=0.143: {f'{res_143:.1f} Å' if res_143 else 'not reached'}\n"
    )
    (output_dir / "summary.txt").write_text(summary)
    print(summary)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="Fourier Shell Correlation (FSC) evaluation tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── halfset ──────────────────────────────────────────────────────────────
    p_half = sub.add_parser(
        "halfset",
        help="FSC between even-half and odd-half averages of a raw cluster "
             "(measures data ceiling with no alignment).",
    )
    p_half.add_argument("--cluster_dir", required=True,
                        help="Directory containing subtomogram .mrc files.")
    p_half.add_argument("--output", required=True,
                        help="Directory to save results into.")
    p_half.add_argument("--apix", type=float, default=4.0,
                        help="Pixel size in Ångströms (default: 4.0).")

    # ── compare ──────────────────────────────────────────────────────────────
    p_cmp = sub.add_parser(
        "compare",
        help="FSC between any two .mrc volumes.",
    )
    p_cmp.add_argument("--vol1", required=True, help="Path to first volume.")
    p_cmp.add_argument("--vol2", required=True, help="Path to second volume.")
    p_cmp.add_argument("--label1", default=None, help="Legend label for vol1.")
    p_cmp.add_argument("--label2", default=None, help="Legend label for vol2.")
    p_cmp.add_argument("--output", required=True,
                       help="Directory to save results into.")
    p_cmp.add_argument("--apix", type=float, default=4.0,
                       help="Pixel size in Ångströms (default: 4.0).")

    # ── vs_baseline ──────────────────────────────────────────────────────────
    p_base = sub.add_parser(
        "vs_baseline",
        help="FSC between an experiment's final reference and the EMAN2 baseline.",
    )
    p_base.add_argument("--exp_dir", required=True,
                        help="Experiment folder (e.g. experiments/exp004_...).")
    p_base.add_argument("--baseline", required=True,
                        help="Path to baseline .mrc (e.g. eman2_baseline/baseline_reference_91.mrc).")
    p_base.add_argument("--apix", type=float, default=4.0,
                        help="Pixel size in Ångströms (default: 4.0).")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "halfset":
        mode_halfset(args)
    elif args.mode == "compare":
        mode_compare(args)
    elif args.mode == "vs_baseline":
        mode_vs_baseline(args)


if __name__ == "__main__":
    main()

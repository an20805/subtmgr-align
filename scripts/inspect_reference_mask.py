#!/usr/bin/env python3
"""
scripts/inspect_reference_mask.py
---------------------------------
Visualise and inspect the reference mask used for selective TRN sampling.

Uses the EXACT same implementation as the AlignOT iterative pipeline:
  1. Lowpass filter the reference volume (Gaussian filter, sigma=2.0 voxels).
  2. Binarise:
       - If threshold <= 0: voxels with ref_vol < threshold (dark protein signal).
       - If 0 < threshold <= 1: voxels on prepared volume > threshold.
  3. Fill internal cavities via binary_fill_holes.
  4. Dilate by dilation_vox via Euclidean Distance Transform (EDT).
  5. Apply soft Gaussian taper (sigma=3.0 voxels) to prevent sharp boundary artifacts.

Outputs:
  - If multiple thresholds are provided: a side-by-side grid comparing contours on Central XY, XZ, YZ.
  - If a single threshold is provided: a multi-panel view showing contours, masked density, and the mask slice.
  - Optional: saves the 3D soft mask as an .mrc volume (--save_mask).

Example Usage:
  # Scan multiple thresholds:
  python scripts/inspect_reference_mask.py \
      --ref_mrc alignOT/outputs/gpu_runs/run_20260924_155858_snr0.1_alignot_5rounds_gpu_argmax_wedge/round_03/ref_combined_round_03.mrc \
      --thresholds -0.10 -0.15 -0.20 -0.25 \
      --dilation 8

  # Inspect a single threshold in detail:
  python scripts/inspect_reference_mask.py \
      --ref_mrc alignOT/outputs/gpu_runs/run_20260924_155858_snr0.1_alignot_5rounds_gpu_argmax_wedge/round_03/ref_combined_round_03.mrc \
      --thresholds -0.15 \
      --dilation 8 \
      --save_mask
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter

# Add repo root to import path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "alignOT" / "src"))
sys.path.insert(0, str(REPO_ROOT / "alignOT" / "alignOT"))

import alignot_utils


def build_reference_mask(
    ref_vol: np.ndarray,
    mask_threshold: float = -0.15,
    dilation_vox: int = 8,
    taper_sigma: float = 3.0,
    lowpass_sigma: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact mirror of build_reference_mask in run_iterative_alignment_gpu_halfset.py.

    Returns:
        soft_mask: (D, H, W) float32 in [0, 1]
        binary_core: (D, H, W) bool, core protein detection before dilation
    """
    if 0.0 < mask_threshold <= 1.0:
        prep = alignot_utils.prepare_sampling_volume(ref_vol, sigma=lowpass_sigma)
        binary = (prep > mask_threshold)
    else:
        v_lp = alignot_utils.gaussian_lowpass_filter(ref_vol, sigma=lowpass_sigma)
        binary = (v_lp < mask_threshold)

    # Fill internal holes (e.g. cavity) so the entire protein barrel is solid
    binary_filled = binary_fill_holes(binary)

    # EDT of complement gives distance from protein boundary
    dist_outside = distance_transform_edt(~binary_filled)
    dilated = (binary_filled | (dist_outside <= dilation_vox)).astype(np.float32)
    soft = gaussian_filter(dilated.astype(np.float64), sigma=taper_sigma)
    soft_mask = np.clip(soft, 0.0, 1.0).astype(np.float32)
    return soft_mask, binary_filled


def compute_solvent_stats(vol: np.ndarray) -> Tuple[float, float]:
    """Estimate solvent background mean and standard deviation from outer corner voxels."""
    N = vol.shape[0]
    cz = cy = cx = N // 2
    z, y, x = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    r_dist = np.sqrt((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2)
    # Use voxels beyond 85% of half-box-width
    bg_voxels = vol[r_dist > (0.85 * (N / 2.0))]
    return float(bg_voxels.mean()), float(bg_voxels.std())


def plot_comparison_grid(
    vol: np.ndarray,
    thresholds: List[float],
    dilation_vox: int,
    taper_sigma: float,
    lowpass_sigma: float,
    out_png: Path,
    title_prefix: str = "Reference Mask Threshold Comparison",
):
    """Plot multi-column comparison across multiple thresholds."""
    N = vol.shape[0]
    cz = cy = cx = N // 2
    slices_raw = [
        ("Central XY", vol[cz, :, :]),
        ("Central XZ", vol[:, cy, :]),
        ("Central YZ", vol[:, :, cx]),
    ]

    n_th = len(thresholds)
    fig, axes = plt.subplots(3, n_th, figsize=(4.2 * n_th, 12))
    if n_th == 1:
        axes = np.expand_dims(axes, axis=1)

    print("\nThreshold Evaluation Summary:")
    print("-" * 75)
    print(f"{'Threshold':<11} | {'Core Voxels':<13} | {'Mask Fill %':<13} | {'Max Radius':<12}")
    print("-" * 75)

    for col_idx, th in enumerate(thresholds):
        soft_mask, core_bin = build_reference_mask(
            vol,
            mask_threshold=th,
            dilation_vox=dilation_vox,
            taper_sigma=taper_sigma,
            lowpass_sigma=lowpass_sigma,
        )

        fill_pct = 100.0 * float((soft_mask >= 0.5).sum()) / float(soft_mask.size)
        core_pct = 100.0 * float(core_bin.sum()) / float(core_bin.size)

        # Measure max radius from center
        z_idx, y_idx, x_idx = np.where(soft_mask >= 0.5)
        if len(z_idx) > 0:
            max_r = float(np.sqrt((z_idx - cz) ** 2 + (y_idx - cy) ** 2 + (x_idx - cx) ** 2).max())
        else:
            max_r = 0.0

        print(f"{th:<11.3f} | {core_pct:<12.2f}% | {fill_pct:<12.2f}% | {max_r:<10.1f} vox")

        slices_core = [core_bin[cz, :, :], core_bin[:, cy, :], core_bin[:, :, cx]]
        slices_mask = [soft_mask[cz, :, :], soft_mask[:, cy, :], soft_mask[:, :, cx]]

        for row_idx in range(3):
            ax = axes[row_idx, col_idx]
            sl_name, sl = slices_raw[row_idx]
            c_sl = slices_core[row_idx]
            m_sl = slices_mask[row_idx]

            ax.imshow(sl, cmap="gray", origin="lower")
            # Overlays:
            # 1. Red contour: core detected protein density
            if c_sl.any():
                ax.contour(c_sl, levels=[0.5], colors="red", linewidths=1.2)
            # 2. Cyan contour: dilated soft mask boundary (value = 0.5)
            if (m_sl >= 0.5).any():
                ax.contour(m_sl, levels=[0.5], colors="cyan", linewidths=1.8)
            # 3. Yellow dashed contour: soft taper outer boundary (value = 0.05)
            if (m_sl >= 0.05).any():
                ax.contour(m_sl, levels=[0.05], colors="yellow", linewidths=1.0, linestyles="dashed")

            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(
                    f"Threshold: {th:+.2f}\nCore: {core_pct:.1f}% | Mask: {fill_pct:.1f}%",
                    fontsize=11,
                    fontweight="bold",
                )
            if col_idx == 0:
                ax.set_ylabel(sl_name, fontsize=12, fontweight="bold")

    print("-" * 75)

    plt.suptitle(
        f"{title_prefix} (Dilation = {dilation_vox} vox, Taper σ = {taper_sigma} vox)\n"
        r"$\mathbf{Red}$: Core Protein Boundary | $\mathbf{Cyan}$: Dilated Mask (50%) | $\mathbf{Yellow\ Dashed}$: Soft Taper Edge (5%)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\nSaved threshold comparison figure to: {out_png}")


def plot_single_threshold_detail(
    vol: np.ndarray,
    th: float,
    dilation_vox: int,
    taper_sigma: float,
    lowpass_sigma: float,
    out_png: Path,
):
    """Detailed 3x3 plot for a single threshold: Raw + Contours, Masked Density, and Soft Mask."""
    N = vol.shape[0]
    cz = cy = cx = N // 2
    soft_mask, core_bin = build_reference_mask(
        vol,
        mask_threshold=th,
        dilation_vox=dilation_vox,
        taper_sigma=taper_sigma,
        lowpass_sigma=lowpass_sigma,
    )

    masked_vol = vol * soft_mask
    fill_pct = 100.0 * float((soft_mask >= 0.5).sum()) / float(soft_mask.size)
    core_pct = 100.0 * float(core_bin.sum()) / float(core_bin.size)

    fig, axes = plt.subplots(3, 3, figsize=(13, 12))

    slices_data = [
        ("Central XY", vol[cz, :, :], core_bin[cz, :, :], soft_mask[cz, :, :], masked_vol[cz, :, :]),
        ("Central XZ", vol[:, cy, :], core_bin[:, cy, :], soft_mask[:, cy, :], masked_vol[:, cy, :]),
        ("Central YZ", vol[:, :, cx], core_bin[:, :, cx], soft_mask[:, :, cx], masked_vol[:, :, cx]),
    ]

    for row_idx, (name, raw_sl, c_sl, m_sl, msk_sl) in enumerate(slices_data):
        # Column 0: Raw with contours
        ax0 = axes[row_idx, 0]
        ax0.imshow(raw_sl, cmap="gray", origin="lower")
        if c_sl.any():
            ax0.contour(c_sl, levels=[0.5], colors="red", linewidths=1.2)
        if (m_sl >= 0.5).any():
            ax0.contour(m_sl, levels=[0.5], colors="cyan", linewidths=1.8)
        if (m_sl >= 0.05).any():
            ax0.contour(m_sl, levels=[0.05], colors="yellow", linewidths=1.0, linestyles="dashed")
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_ylabel(name, fontsize=12, fontweight="bold")
        if row_idx == 0:
            ax0.set_title("Average with Mask Contours\n(Red=Core, Cyan=Mask, Yellow=Taper)", fontsize=11, fontweight="bold")

        # Column 1: Masked Average (what remains inside the mask)
        ax1 = axes[row_idx, 1]
        im1 = ax1.imshow(msk_sl, cmap="gray", origin="lower", vmin=raw_sl.min(), vmax=raw_sl.max())
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        if row_idx == 0:
            ax1.set_title("Masked Average Density\n(Outside Envelope Zeroed Out)", fontsize=11, fontweight="bold")

        # Column 2: 2D Soft Mask Slice
        ax2 = axes[row_idx, 2]
        im2 = ax2.imshow(m_sl, cmap="viridis", origin="lower", vmin=0.0, vmax=1.0)
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        if row_idx == 0:
            ax2.set_title("Soft Mask [0, 1]\n(Gaussian Taper Falloff)", fontsize=11, fontweight="bold")

    plt.suptitle(
        f"Detailed Reference Mask Inspection: Threshold = {th:+.2f} (Dilation = {dilation_vox} vox)\n"
        f"Core Protein: {core_pct:.1f}% | Dilated Mask Envelope: {fill_pct:.1f}% of box",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved detailed threshold figure to: {out_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and select reference mask thresholds for selective TRN sampling."
    )
    parser.add_argument(
        "--ref_mrc",
        type=str,
        required=True,
        help="Path to the reference/average .mrc file (e.g. ref_combined_round_03.mrc)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[-0.10, -0.15, -0.20, -0.25],
        help="One or more threshold values to evaluate (default: -0.10 -0.15 -0.20 -0.25)",
    )
    parser.add_argument(
        "--dilation",
        type=int,
        default=8,
        help="Dilation radius in voxels (default: 8, matching alignment pipeline)",
    )
    parser.add_argument(
        "--taper_sigma",
        type=float,
        default=3.0,
        help="Soft Gaussian taper sigma in voxels (default: 3.0)",
    )
    parser.add_argument(
        "--lowpass_sigma",
        type=float,
        default=2.0,
        help="Lowpass filter sigma before binarisation in voxels (default: 2.0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="temp/mask_inspections",
        help="Output directory for generated plots and masks (default: temp/mask_inspections)",
    )
    parser.add_argument(
        "--save_mask",
        action="store_true",
        help="Save 3D .mrc soft mask file for the first/selected threshold",
    )

    args = parser.parse_args()

    ref_path = Path(args.ref_mrc)
    if not ref_path.exists():
        print(f"Error: Reference file does not exist: {ref_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vol = alignot_utils.load_mrc(ref_path, permissive=True)
    print("=" * 75)
    print(f"REFERENCE VOLUME: {ref_path.name}")
    print(f"  Shape       : {vol.shape}")
    print(f"  Density min : {vol.min():.4f}, max: {vol.max():.4f}, mean: {vol.mean():.4f}, std: {vol.std():.4f}")

    bg_mean, bg_std = compute_solvent_stats(vol)
    print(f"  Solvent bg  : mean = {bg_mean:+.4f}, std = {bg_std:.4f} (3σ range: [{bg_mean - 3*bg_std:+.3f}, {bg_mean + 3*bg_std:+.3f}])")
    print("=" * 75)

    base_stem = ref_path.stem

    if len(args.thresholds) > 1:
        # Multi-threshold comparison grid
        comp_png = out_dir / f"mask_comparison_{base_stem}_dil{args.dilation}.png"
        plot_comparison_grid(
            vol=vol,
            thresholds=args.thresholds,
            dilation_vox=args.dilation,
            taper_sigma=args.taper_sigma,
            lowpass_sigma=args.lowpass_sigma,
            out_png=comp_png,
            title_prefix=f"Mask Comparison: {ref_path.name}",
        )
    else:
        # Detailed single-threshold plot
        th = args.thresholds[0]
        th_tag = f"{th:+.2f}".replace("+", "p").replace("-", "m").replace(".", "_")
        single_png = out_dir / f"mask_detail_{base_stem}_th{th_tag}_dil{args.dilation}.png"
        plot_single_threshold_detail(
            vol=vol,
            th=th,
            dilation_vox=args.dilation,
            taper_sigma=args.taper_sigma,
            lowpass_sigma=args.lowpass_sigma,
            out_png=single_png,
        )

    if args.save_mask:
        primary_th = args.thresholds[0]
        th_tag = f"{primary_th:+.2f}".replace("+", "p").replace("-", "m").replace(".", "_")
        soft_mask, _ = build_reference_mask(
            vol,
            mask_threshold=primary_th,
            dilation_vox=args.dilation,
            taper_sigma=args.taper_sigma,
            lowpass_sigma=args.lowpass_sigma,
        )
        mask_mrc = out_dir / f"ref_mask_{base_stem}_th{th_tag}_dil{args.dilation}.mrc"
        alignot_utils.save_mrc(mask_mrc, soft_mask)
        print(f"Saved 3D soft mask .mrc to: {mask_mrc}")


if __name__ == "__main__":
    main()

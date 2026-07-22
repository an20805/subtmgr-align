#!/usr/bin/env python3
"""
visualize_sampling.py  (AlignOT / alignOT folder)

Produces a 3×4 grid point-cloud sampling diagnostic matching the EMPOT
reference plot at experiments/empot_eman2ref_fsc/pointcloud_viz.png.

Layout (same as EMPOT reference):
  rows : XY (z=centre) | XZ (y=centre) | YZ (x=centre)
  cols : Volume slice | Point cloud (N pts) | Overlay (±15 vox slab) | Point density hexbin

Uses the exact same TRN sampling pipeline as EMPOT's iterative_utils.py:
  1. normalize to [0,1]
  2. invert  (1 – norm)  so high-density protein → high probability
  3. threshold
  4. trn_rm0 + trn_iterate (do_log=True!)  → rms[-1]

Usage:
  python visualize_sampling.py \\
      --mrc  /path/to/average.mrc \\
      --threshold 0.75 \\
      --num_points 500 \\
      --sigma 0 \\
      --out  /path/to/output.png
"""

import sys
import io
import contextlib
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mrcfile

# ── import EMPOT trn (proven pipeline) ────────────────────────────────────────
EMPOT_SRC = Path(__file__).parent.parent.parent / "EMPOT" / "src"
sys.path.insert(0, str(EMPOT_SRC))
import trn  # EMPOT's trn.py (do_log correctly handled)

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ── helpers ───────────────────────────────────────────────────────────────────

def normalize_volume(vol: np.ndarray) -> np.ndarray:
    v_min, v_max = vol.min(), vol.max()
    if v_max - v_min == 0:
        return vol.copy()
    return (vol - v_min) / (v_max - v_min)


def lowpass_filter(vol: np.ndarray, sigma: float) -> np.ndarray:
    if not HAS_SCIPY or sigma <= 0:
        return vol
    return gaussian_filter(vol.astype(np.float32), sigma=float(sigma))


def sample_inverted(volume: np.ndarray,
                    threshold: float,
                    num_points: int,
                    sigma: float = 0,
                    random_seed: int | None = None) -> np.ndarray:
    """
    The same pipeline used by EMPOT's iterative_utils.sample_volume_in_memory:
      lowpass → normalize → invert → threshold → TRN
    Returns rms[-1], shape (num_points, 3), centred coordinates.
    """
    vol = lowpass_filter(volume, sigma)
    norm = normalize_volume(vol)
    inv = 1.0 - norm

    map_th = inv.copy()
    map_th[map_th < threshold] = 0.0
    if map_th.sum() == 0:
        raise ValueError(
            f"threshold={threshold} zeros out the entire (inverted) volume. "
            "Try a lower threshold."
        )

    with contextlib.redirect_stdout(io.StringIO()):
        rm0, arr_flat, arr_idx, xyz, coords_1d = trn.trn_rm0(
            map_th, M=num_points, random_seed=random_seed
        )
        rms, rs, ts_save = trn.trn_iterate(
            rm0, arr_flat, arr_idx, xyz,
            n_save=10, e0=0.3, ef=0.05,
            l0=0.005 * num_points, lf=0.5,
            tf=num_points * 8,
            do_log=True,   # ← CRITICAL: must be True for rms to be filled
            log_n=10,
        )
    return rms[-1]   # shape (M, 3), centred units


# ── plotting ──────────────────────────────────────────────────────────────────

def _ax_off(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#333333")


def plot_sampling_grid(volume: np.ndarray,
                       pts: np.ndarray,
                       title: str,
                       out_path: str,
                       num_points: int,
                       threshold: float,
                       sigma: float,
                       slab_half: int = 15):
    """
    pts : (M, 3) centred coordinates (output of TRN).
          Axis convention from trn_rm0: pts[:,0]=x, pts[:,1]=y, pts[:,2]=z
    """
    N = volume.shape[0]
    centre = N // 2

    # ── volume slices (centred) ────────────────────────────────────────────
    sl_xy = volume[centre, :, :]   # z fixed
    sl_xz = volume[:, centre, :]   # y fixed
    sl_yz = volume[:, :, centre]   # x fixed

    # ── project points onto each plane ────────────────────────────────────
    #  TRN coords are centred at 0, shift by `centre` for image coords
    #  Axis mapping: dim0=x, dim1=y, dim2=z
    px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]

    # For imshow with origin='lower': horizontal = first dim, vertical = second dim
    # XY plane  (z slice): h=x, v=y; depth=z
    # XZ plane  (y slice): h=x, v=z; depth=y
    # YZ plane  (x slice): h=y, v=z; depth=x

    planes = [
        dict(h=px + centre, v=py + centre, depth=pz,
             sl=sl_xy,  lbl="XY (z=centre)"),
        dict(h=px + centre, v=pz + centre, depth=py,
             sl=sl_xz,  lbl="XZ (y=centre)"),
        dict(h=py + centre, v=pz + centre, depth=px,
             sl=sl_yz,  lbl="YZ (x=centre)"),
    ]

    col_titles = [
        "Volume slice",
        f"Point cloud ({num_points} pts)",
        "Overlay",
        "Point density (hexbin)",
    ]

    # ── figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 18), facecolor="#111111")
    outer = gridspec.GridSpec(
        3, 4, figure=fig,
        hspace=0.04, wspace=0.04,
        top=0.88, bottom=0.06, left=0.07, right=0.99,
    )

    # ── 3 rows ────────────────────────────────────────────────────────────
    for row, plane in enumerate(planes):
        sl     = plane["sl"]
        h_all  = plane["h"]
        v_all  = plane["v"]
        depth  = plane["depth"]
        lbl    = plane["lbl"]
        mask   = np.abs(depth) < slab_half

        # col 0 – volume slice
        ax = fig.add_subplot(outer[row, 0])
        ax.set_facecolor("#111111")
        ax.imshow(sl.T, cmap="gray", origin="lower",
                  extent=[0, N, 0, N], aspect="equal")
        ax.axhline(centre, color="red",  lw=0.8, alpha=0.5)
        ax.axvline(centre, color="red",  lw=0.8, alpha=0.5)
        ax.set_ylabel(lbl, color="white", fontsize=13, labelpad=4)
        if row == 0:
            ax.set_title(col_titles[0], color="white", fontsize=13, pad=6)
        _ax_off(ax)

        # col 1 – point cloud (all points, dark background)
        ax = fig.add_subplot(outer[row, 1])
        ax.set_facecolor("#111111")
        ax.scatter(h_all, v_all, c="cyan", s=8, alpha=0.7, linewidths=0)
        ax.axhline(centre, color="red",  lw=0.8, alpha=0.5)
        ax.axvline(centre, color="red",  lw=0.8, alpha=0.5)
        ax.set_xlim(0, N); ax.set_ylim(0, N)
        if row == 0:
            ax.set_title(col_titles[1], color="white", fontsize=13, pad=6)
        _ax_off(ax)

        # col 2 – overlay (slice + slab points)
        ax = fig.add_subplot(outer[row, 2])
        ax.set_facecolor("#111111")
        ax.imshow(sl.T, cmap="gray", origin="lower",
                  extent=[0, N, 0, N], aspect="equal")
        ax.scatter(h_all[mask], v_all[mask], c="cyan",
                   s=8, alpha=0.85, linewidths=0)
        if row == 0:
            ax.set_title(col_titles[2], color="white", fontsize=13, pad=6)
        _ax_off(ax)

        # col 3 – hexbin density
        ax = fig.add_subplot(outer[row, 3])
        ax.set_facecolor("#111111")
        if mask.sum() > 0:
            ax.hexbin(h_all[mask], v_all[mask],
                      gridsize=14, cmap="hot", mincnt=1,
                      extent=[0, N, 0, N])
        ax.axhline(centre, color="teal", lw=0.8, alpha=0.5)
        ax.axvline(centre, color="teal", lw=0.8, alpha=0.5)
        ax.set_xlim(0, N); ax.set_ylim(0, N)
        if row == 0:
            ax.set_title(col_titles[3], color="white", fontsize=13, pad=6)
        _ax_off(ax)

    # ── suptitle ──────────────────────────────────────────────────────────
    sigma_str = f"sigma={sigma}" if sigma > 0 else "no lowpass filter"
    fig.suptitle(
        f"{title}\n"
        f"{sigma_str}  |  threshold={threshold}  |  {num_points} points  |  "
        f"slab ±{slab_half} vox",
        color="white", fontsize=16, y=0.96,
    )

    # ── footer annotation (centroid + stats) ──────────────────────────────
    centroid = pts.mean(axis=0)          # centred coords
    centroid_vox = centroid + centre      # voxel indices
    kept_frac = (volume > 0).mean() * 100
    footer = (
        f"Point centroid (voxel): z={centroid_vox[2]:.1f}, "
        f"y={centroid_vox[1]:.1f}, x={centroid_vox[0]:.1f}  |  "
        f"Volume centre = ({centre}, {centre}, {centre})  |  "
        f"Threshold kept {kept_frac:.1f}% of voxels"
    )
    fig.text(0.5, 0.025, footer, ha="center", va="bottom",
             color="#aaaaaa", fontsize=9)

    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize AlignOT point-cloud sampling (EMPOT-style 3×4 grid)"
    )
    parser.add_argument("--mrc",        required=True,
                        help="Input MRC file (e.g. EMAN2 baseline_avg_iter02.mrc)")
    parser.add_argument("--threshold",  type=float, default=0.75,
                        help="Inversion threshold [0–1] (default 0.75)")
    parser.add_argument("--num_points", type=int,   default=500,
                        help="Number of TRN pseudo-atoms (default 500)")
    parser.add_argument("--sigma",      type=float, default=0.0,
                        help="Gaussian lowpass sigma in voxels (0 = off)")
    parser.add_argument("--slab",       type=int,   default=15,
                        help="Half-slab thickness in voxels for overlay (default 15)")
    parser.add_argument("--out",        type=str,
                        default="alignOT/sampling_viz.png",
                        help="Output PNG path")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    # Load
    with mrcfile.open(args.mrc, permissive=True) as mrc:
        volume = mrc.data.copy().astype(np.float32)

    mrc_name = Path(args.mrc).name
    print(f"Volume shape: {volume.shape}  min={volume.min():.4f}  max={volume.max():.4f}")

    # Sample
    print(f"Sampling {args.num_points} points (threshold={args.threshold}) …")
    pts = sample_inverted(volume,
                          threshold=args.threshold,
                          num_points=args.num_points,
                          sigma=args.sigma,
                          random_seed=args.seed)
    print(f"Points std: {pts.std(axis=0)}")
    print(f"Points centroid: {pts.mean(axis=0)}")

    # Plot
    title = f"{mrc_name} ({args.num_points} ptcl)  →  current sampling pipeline"
    plot_sampling_grid(
        volume=volume,
        pts=pts,
        title=title,
        out_path=args.out,
        num_points=args.num_points,
        threshold=args.threshold,
        sigma=args.sigma,
        slab_half=args.slab,
    )


if __name__ == "__main__":
    main()

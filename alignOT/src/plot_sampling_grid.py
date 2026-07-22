#!/usr/bin/env python3
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mrcfile

# ── import AlignOT TRN components ─────────────────────────────────────────────
ALIGNOT_SRC = Path(__file__).parent.parent / "alignOT"
sys.path.insert(0, str(ALIGNOT_SRC))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import trn

def normalize_volume(vol):
    v_min = np.min(vol)
    v_max = np.max(vol)
    if v_max - v_min == 0:
        return vol
    return (vol - v_min) / (v_max - v_min)

def sample_alignot_native(volume, threshold, num_points):
    norm = normalize_volume(volume)
    map_th = norm.copy()
    map_th[map_th < threshold] = 0
    with contextlib.redirect_stdout(io.StringIO()):
        rm0, arr_flat, arr_idx, xyz, coords_1d = trn.trn_rm0(map_th, num_points)
        rms, rs, ts_save = trn.trn_iterate(
            rm0, arr_flat, arr_idx, xyz,
            n_save=10, e0=0.3, ef=0.05,
            l0=0.005 * num_points, lf=0.5, tf=num_points * 8,
            do_log=False
        )
    return rms[-1]

def sample_inverted(volume, threshold, num_points):
    norm = normalize_volume(volume)
    inv = 1.0 - norm
    map_th = inv.copy()
    map_th[map_th < threshold] = 0
    with contextlib.redirect_stdout(io.StringIO()):
        rm0, arr_flat, arr_idx, xyz, coords_1d = trn.trn_rm0(map_th, num_points)
        rms, rs, ts_save = trn.trn_iterate(
            rm0, arr_flat, arr_idx, xyz,
            n_save=10, e0=0.3, ef=0.05,
            l0=0.005 * num_points, lf=0.5, tf=num_points * 8,
            do_log=False
        )
    return rms[-1]

def plot_grid(volume, pts, title, out_path):
    # Setup grid
    fig, axes = plt.subplots(3, 4, figsize=(24, 18), facecolor='#111111')
    fig.suptitle(title, color='white', fontsize=20, y=0.95)
    
    N = volume.shape[0]
    center = N // 2
    
    # Slices
    slices = [
        volume[center, :, :], # XY (z=center)
        volume[:, center, :], # XZ (y=center)
        volume[:, :, center]  # YZ (x=center)
    ]
    
    # Point coordinates and axes
    pt_coords = [
        (pts[:, 0], pts[:, 1]), # XY
        (pts[:, 0], pts[:, 2]), # XZ
        (pts[:, 1], pts[:, 2])  # YZ
    ]
    
    labels = ["XY (z=centre)", "XZ (y=centre)", "YZ (x=centre)"]
    col_titles = ["Volume slice", "Point cloud (500 pts)", "Overlay", "Point density (hexbin)"]
    
    for i in range(3):
        # Column 0: Volume slice
        ax = axes[i, 0]
        ax.imshow(slices[i].T, cmap='gray', origin='lower')
        ax.axhline(center, color='r', alpha=0.3)
        ax.axvline(center, color='r', alpha=0.3)
        ax.set_ylabel(labels[i], color='white', fontsize=14)
        if i == 0: ax.set_title(col_titles[0], color='white', fontsize=14)
        
        # Column 1: Point cloud
        ax = axes[i, 1]
        ax.set_facecolor('#111111')
        # Center points: pt_coords are in [-N/2, N/2] space (AlignOT native range)
        # Wait, TRN coords are usually centered. Let's shift to [0, N] for plotting
        px = pt_coords[i][0] + center
        py = pt_coords[i][1] + center
        ax.scatter(px, py, c='cyan', s=10, alpha=0.8)
        ax.set_xlim(0, N); ax.set_ylim(0, N)
        ax.axhline(center, color='r', alpha=0.3)
        ax.axvline(center, color='r', alpha=0.3)
        if i == 0: ax.set_title(col_titles[1], color='white', fontsize=14)
        
        # Column 2: Overlay
        ax = axes[i, 2]
        ax.imshow(slices[i].T, cmap='gray', origin='lower')
        # Filter points close to the slice
        z_coord = pts[:, 2] if i == 0 else (pts[:, 1] if i == 1 else pts[:, 0])
        mask = np.abs(z_coord) < 15  # +/- 15 voxels slab
        ax.scatter(px[mask], py[mask], c='cyan', s=10, alpha=0.8)
        if i == 0: ax.set_title(col_titles[2], color='white', fontsize=14)
        
        # Column 3: Hexbin
        ax = axes[i, 3]
        ax.set_facecolor('#111111')
        hb = ax.hexbin(px, py, gridsize=15, cmap='hot', mincnt=1, extent=(0, N, 0, N))
        ax.axhline(center, color='teal', alpha=0.5)
        ax.axvline(center, color='teal', alpha=0.5)
        if i == 0: ax.set_title(col_titles[3], color='white', fontsize=14)
        
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#333333')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrc", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--num_points", type=int, default=500)
    args = parser.parse_args()

    with mrcfile.open(args.mrc, permissive=True) as mrc:
        volume = mrc.data.copy()

    # AlignOT Native
    print("Sampling AlignOT native...")
    pts_native = sample_alignot_native(volume, args.threshold, args.num_points)
    plot_grid(volume, pts_native, 
              f"AlignOT Native Sampling\nthreshold={args.threshold} | {args.num_points} points | slab ±15 vox", 
              "/home/anshu/subtmgr-align/alignOT/sampling_grid_native.png")

    # Inverted (EMPOT-style)
    print("Sampling Inverted (EMPOT-style)...")
    pts_inv = sample_inverted(volume, args.threshold, args.num_points)
    plot_grid(volume, pts_inv, 
              f"Inverted (EMPOT-style) Sampling\nthreshold={args.threshold} | {args.num_points} points | slab ±15 vox", 
              "/home/anshu/subtmgr-align/alignOT/sampling_grid_inverted.png")

if __name__ == "__main__":
    main()

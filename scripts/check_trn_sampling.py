"""
check_trn_sampling.py
---------------------
Sample a point cloud from the EMAN2 baseline subtomogram average using
AlignOT's TRN algorithm.  Compares sampling WITHOUT vs WITH a Gaussian
low-pass filter applied before TRN.

Coordinate convention (from coords.py analysis):
  pts[:,0]=z  pts[:,1]=y  pts[:,2]=x
  (C-order flatten of vol[z,y,x] matched to coords after meshgrid+swap)

Usage (from repo root):
    /home/anshu/subtmgr-align/alignOT/alignot_env/bin/python \
        scripts/check_trn_sampling.py
"""

import sys, os
import numpy as np
import mrcfile
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── AlignOT source ────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
ALIGNOT_SRC = os.path.join(REPO_ROOT, "alignOT", "alignOT")
sys.path.insert(0, ALIGNOT_SRC)
import trn

# ── Config ────────────────────────────────────────────────────────────────────
# MRC_PATH   = os.path.join(REPO_ROOT, "eman2_baseline", "baseline_avg_iter02.mrc")   # EMAN2 average
MRC_PATH     = os.path.join(REPO_ROOT, "Data", "0.1", "Cluster_0", "100.mrc")           # raw subtomogram
OUT_COMPARE  = os.path.join(REPO_ROOT, "experiments", "alignot_trn_sampling_check.png")
N_POINTS     = 2000
THRESHOLD    = 0.7
RANDOM_SEED  = 42
THICKNESS    = 5     # voxels either side of slice plane
SIGMA        = 2.0   # Gaussian σ in voxels ≈ 19 Å resolution cutoff

# ── 1. Load ───────────────────────────────────────────────────────────────────
with mrcfile.open(MRC_PATH, permissive=True) as mrc:
    volume = mrc.data.copy().astype(np.float32)

D, H, W = volume.shape
print(f"Volume: {D}×{H}×{W}  |  min={volume.min():.3f}  max={volume.max():.3f}")

# ── 2. Prepare: normalize → invert ───────────────────────────────────────────
def prepare(vol, sigma=None):
    """Normalize, optionally lowpass, invert so signal→high values."""
    v = vol.astype(np.float32)
    if sigma is not None and sigma > 0:
        v = gaussian_filter(v, sigma=float(sigma))
    vmin, vmax = v.min(), v.max()
    norm = (v - vmin) / (vmax - vmin + 1e-8)
    return 1.0 - norm   # invert: signal → near 1

prep_no_lp = prepare(volume, sigma=None)
prep_lp    = prepare(volume, sigma=SIGMA)

# Show effect of filter on the probability field at slice z=D//2
cz, cy, cx = D//2, H//2, W//2
print(f"\nAt centre slice (z={cz}), threshold={THRESHOLD}:")
print(f"  No filter  — voxels above threshold: {(prep_no_lp[cz] >= THRESHOLD).sum()}")
print(f"  σ={SIGMA} filter — voxels above threshold: {(prep_lp[cz] >= THRESHOLD).sum()}")

# ── 3. TRN sampling (both variants) ──────────────────────────────────────────
def run_trn(prep_vol, label):
    """Call trn_rm0 + trn_iterate directly to avoid the reshape bug in
    trn_wrapper (it tries to reshape the padded flat array back to the
    original shape, which fails when the original dim is odd, e.g. 91→92)."""
    print(f"\nRunning TRN ({label}, M={N_POINTS}) ...")
    map_th = prep_vol.copy()
    map_th[map_th < THRESHOLD] = 0
    rm0, map_3d_flat, map_3d_idx, xyz, coords_1d = trn.trn_rm0(
        map_th, M=N_POINTS, random_seed=RANDOM_SEED,
    )
    rms, rs, ts_save = trn.trn_iterate(
        rm0, map_3d_flat, map_3d_idx, xyz,
        n_save=10, e0=0.3, ef=0.05,
        l0=0.005 * N_POINTS, lf=0.5,
        tf=N_POINTS * 8,
        do_log=True, log_n=10,
    )
    return rms[-1]   # shape (N_POINTS, 3)

pts_no_lp = run_trn(prep_no_lp, "no filter")
pts_lp    = run_trn(prep_lp,    f"σ={SIGMA}")

# ── 4. Centered → voxel coords ───────────────────────────────────────────────
N_cube = max(D, H, W)
N_cube += N_cube % 2

def to_vox(pts):
    pv = pts + N_cube // 2
    return pv[:, 0], pv[:, 1], pv[:, 2]   # z_v, y_v, x_v

z_no, y_no, x_no = to_vox(pts_no_lp)
z_lp,  y_lp,  x_lp  = to_vox(pts_lp)

# ── 5. Plot side-by-side comparison ──────────────────────────────────────────
# 2 rows (no-filter / with-filter), 3 cols (XY / XZ / YZ)
fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="white")
fig.suptitle(
    f"TRN Sampling: no filter  vs  Gaussian σ={SIGMA} vox (≈{2.35*SIGMA*4:.0f} Å)\n"
    f"M={N_POINTS}, threshold={THRESHOLD}, EMAN2 baseline avg iter02  ({D}×{H}×{W} vox)",
    fontsize=12, fontweight="bold", y=1.01,
)

PT_S = 3
CMAP = "gray"

def clim(sl, lo=2, hi=98):
    return float(np.percentile(sl, lo)), float(np.percentile(sl, hi))

def plot_row(axes_row, z_v, y_v, x_v, row_label, pt_color):
    # XY
    sl = volume[cz, :, :]
    vlo, vhi = clim(sl)
    axes_row[0].imshow(sl, cmap=CMAP, vmin=vlo, vmax=vhi, origin="lower",
                       extent=[0, W, 0, H])
    mask = np.abs(z_v - cz) <= THICKNESS
    axes_row[0].scatter(x_v[mask], y_v[mask], s=PT_S, c=pt_color, alpha=0.7, linewidths=0)
    axes_row[0].set_title(f"[{row_label}] XY  (z={cz}, {mask.sum()} pts)", fontsize=10)
    axes_row[0].set_xlabel("x (vox)"); axes_row[0].set_ylabel("y (vox)")

    # XZ
    sl = volume[:, cy, :]
    vlo, vhi = clim(sl)
    axes_row[1].imshow(sl, cmap=CMAP, vmin=vlo, vmax=vhi, origin="lower",
                       extent=[0, W, 0, D])
    mask = np.abs(y_v - cy) <= THICKNESS
    axes_row[1].scatter(x_v[mask], z_v[mask], s=PT_S, c=pt_color, alpha=0.7, linewidths=0)
    axes_row[1].set_title(f"[{row_label}] XZ  (y={cy}, {mask.sum()} pts)", fontsize=10)
    axes_row[1].set_xlabel("x (vox)"); axes_row[1].set_ylabel("z (vox)")

    # YZ
    sl = volume[:, :, cx]
    vlo, vhi = clim(sl)
    axes_row[2].imshow(sl, cmap=CMAP, vmin=vlo, vmax=vhi, origin="lower",
                       extent=[0, H, 0, D])
    mask = np.abs(x_v - cx) <= THICKNESS
    axes_row[2].scatter(y_v[mask], z_v[mask], s=PT_S, c=pt_color, alpha=0.7, linewidths=0)
    axes_row[2].set_title(f"[{row_label}] YZ  (x={cx}, {mask.sum()} pts)", fontsize=10)
    axes_row[2].set_xlabel("y (vox)"); axes_row[2].set_ylabel("z (vox)")

plot_row(axes[0], z_no, y_no, x_no, "No filter",          pt_color="red")
plot_row(axes[1], z_lp,  y_lp,  x_lp,  f"Gaussian σ={SIGMA}", pt_color="dodgerblue")

legend_handles = [
    Patch(facecolor="dimgray", label="Volume slice  (2–98 pct)"),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='red',
           markersize=6, label=f"No filter  (±{THICKNESS} vox slab)"),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='dodgerblue',
           markersize=6, label=f"Gaussian σ={SIGMA}  (±{THICKNESS} vox slab)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           fontsize=10, bbox_to_anchor=(0.5, -0.03))

plt.tight_layout()
plt.savefig(OUT_COMPARE, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nSaved comparison → {OUT_COMPARE}")

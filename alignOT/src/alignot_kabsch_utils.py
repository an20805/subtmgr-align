"""
alignot_kabsch_utils.py
-----------------------
Utility functions for the Kabsch-based AlignOT iterative subtomogram alignment.

Pipeline
--------
TRN sampling (AlignOT)  →  Balanced Sinkhorn OT plan  →  Kabsch SVD  →  Rigid transform

Key differences from alignot_utils.py (SGD-based)
--------------------------------------------------
- Alignment uses a one-shot Sinkhorn transport plan + Kabsch SVD instead of 500 SGD steps
- ~150x faster per particle (~17 ms vs ~2500 ms)
- Recovers both R and translation (Abar, Bbar) simultaneously as EMPOT does
- No learning rate, no quaternion parameterisation

Coordinate conventions
----------------------
TRN output : pts[:, 0] = z,  pts[:, 1] = y,  pts[:, 2] = x   (C-order)
Kabsch SVD : operates directly in this [z, y, x] space
affine_transform: R_inv in [z, y, x] space; offset = Bbar + center - R_inv @ (Abar + center)
"""

import contextlib
import gc
import io
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
from scipy.ndimage import affine_transform, gaussian_filter
from scipy.spatial.transform import Rotation

# ── Add AlignOT source to Python path ─────────────────────────────────────────
_ALIGNOT_SRC = Path(__file__).resolve().parent.parent / "alignOT"
if str(_ALIGNOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ALIGNOT_SRC))

# Suppress the module-level print() calls in utils.py
with contextlib.redirect_stdout(io.StringIO()):
    import trn
    import utils as _aligot_utils   # for my_sinkhorn only


# ── I/O helpers ───────────────────────────────────────────────────────────────

def get_cluster_paths(cluster_dir):
    """Return sorted list of .mrc paths (by numeric stem, Zone.Identifier excluded)."""
    d = Path(cluster_dir)
    paths = [
        p for p in d.iterdir()
        if p.is_file() and p.suffix == ".mrc" and "Zone" not in p.name
    ]
    return sorted(paths, key=lambda p: int(p.stem))


def load_mrc(path, permissive=True):
    with mrcfile.open(str(path), permissive=permissive) as mrc:
        return np.array(mrc.data, dtype=np.float32)


def save_mrc(path, arr):
    arr = np.asarray(arr, dtype=np.float32)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(arr)
    return Path(path)


def compute_streaming_average(paths, logger=None, progress_every=20):
    """Memory-efficient streaming voxel-wise average of .mrc files."""
    total = None
    count = 0
    n = len(paths)
    for i, path in enumerate(paths, start=1):
        arr = load_mrc(path)
        if total is None:
            total = np.zeros_like(arr, dtype=np.float64)
        total += arr.astype(np.float64)
        count += 1
        if progress_every and (i % progress_every == 0 or i == n):
            if logger:
                logger.info(f"  Streaming average: {i}/{n}")
        if i % max(1, n // 5) == 0:
            gc.collect()
    if count == 0:
        raise ValueError("Cannot compute average from zero subtomograms.")
    return (total / count).astype(np.float32)


# ── Point-cloud sampling (identical to alignot_utils.py) ──────────────────────

def gaussian_lowpass_filter(volume, sigma):
    """Apply Gaussian LP. Returns unchanged volume if sigma is None or <= 0."""
    if sigma is None or sigma <= 0:
        return volume
    return gaussian_filter(volume.astype(np.float32), sigma=float(sigma))


def prepare_sampling_volume(volume, sigma=None):
    """
    Prepare a volume for TRN sampling:
      1. Optionally Gaussian LP (for sampling only, not averaging)
      2. Normalize to [0, 1]: (v - vmin) / (vmax - vmin)
      3. Invert: 1 - norm  → protein signal (dark in cryo-ET) becomes HIGH

    Returns (D, H, W) float32, values in [0, 1], signal near 1.
    """
    v = volume.astype(np.float32)
    if sigma is not None and sigma > 0:
        v = gaussian_filter(v, sigma=float(sigma))
    vmin, vmax = float(v.min()), float(v.max())
    norm = (v - vmin) / (vmax - vmin + 1e-8)
    return 1.0 - norm


def sample_volume_in_memory(prep_vol, threshold, num_points, random_seed=None):
    """
    TRN sampling via trn_rm0 + trn_iterate.

    do_log=True is REQUIRED — with do_log=False the rms array is never populated
    and the function returns all-zeros.

    Returns (num_points, 3) float32, columns = [z, y, x] (TRN centred coords).
    """
    map_th = prep_vol.copy()
    map_th[map_th < threshold] = 0.0
    if float(map_th.sum()) <= 0:
        raise ValueError(
            f"After threshold={threshold} the volume is all zeros. "
            "Lower the threshold or check normalisation."
        )

    with contextlib.redirect_stdout(io.StringIO()):
        rm0, flat, idx, xyz, _ = trn.trn_rm0(
            map_th, M=num_points, random_seed=random_seed
        )
        rms, _, _ = trn.trn_iterate(
            rm0, flat, idx, xyz,
            n_save=10,
            e0=0.3, ef=0.05,
            l0=0.005 * num_points, lf=0.5,
            tf=num_points * 8,
            do_log=True,     # MUST be True
            log_n=10,
        )

    return rms[-1].astype(np.float32)   # (num_points, 3), cols = [z, y, x]


# ── Kabsch alignment ───────────────────────────────────────────────────────────

def run_kabsch_alignment(pts_src, pts_ref, reg=5.0, sinkhorn_iters=5000):
    """
    Align a subtomogram point cloud to the reference using:
      1. Balanced Sinkhorn OT plan  (AlignOT's my_sinkhorn, ~17 ms for 500×500)
      2. Hard correspondences:  for each src point i, ref point j = argmax_j plan[i,j]
      3. Kabsch SVD on matched pairs → R (3×3), Abar (ref centroid), Bbar (src centroid)

    All point clouds are in [z, y, x] order (TRN output).
    The cost matrix is squared L2 in [z, y, x] space.

    Parameters
    ----------
    pts_src       : (N, 3) float32  — subtomogram point cloud [z, y, x]
    pts_ref       : (N, 3) float32  — reference point cloud   [z, y, x]
    reg           : float           — Sinkhorn entropy reg (default 5.0; lower = sharper plan)
    sinkhorn_iters: int             — max Sinkhorn iterations (default 5000)

    Returns
    -------
    R    : (3, 3) float64 — rotation matrix in [z, y, x] space:  src ≈ R @ (ref - Abar) + Bbar
           Equivalently: apply R^-1 = R.T to map src → ref frame.
    Abar : (3,)  float64 — centroid of matched reference points [z, y, x]
    Bbar : (3,)  float64 — centroid of matched source points    [z, y, x]
    ot_dist : float      — mean squared distance of the OT plan (proxy for alignment quality)
    """
    pts_src = pts_src.astype(np.float64)
    pts_ref = pts_ref.astype(np.float64)

    # ── Cost matrix: squared L2 ───────────────────────────────────────────────
    M = np.sum((pts_src[:, None, :] - pts_ref[None, :, :]) ** 2, axis=-1)

    # ── Sinkhorn plan ─────────────────────────────────────────────────────────
    N = len(pts_src)
    a = np.ones(N) / N
    b = np.ones(N) / N

    with contextlib.redirect_stdout(io.StringIO()):
        plan, _, _ = _aligot_utils.my_sinkhorn(
            a, b, M, reg=reg, numItermax=sinkhorn_iters
        )

    # OT distance = sum of plan * cost
    ot_dist = float(np.sum(plan * M))

    # ── Hard correspondences ──────────────────────────────────────────────────
    # For each source point i, find reference point j = argmax_j plan[i, j]
    best_j = np.argmax(plan, axis=1)   # shape (N,)

    # Keep only correspondences where the plan has positive mass
    valid = plan[np.arange(N), best_j] > 0
    if valid.sum() < 3:
        raise ValueError(
            f"Only {valid.sum()} valid correspondences found (need ≥ 3). "
            "Try lowering --reg or increasing --num_points."
        )

    A = pts_ref[best_j[valid]]   # matched reference points
    B = pts_src[valid]           # matched source points

    # ── Kabsch SVD ────────────────────────────────────────────────────────────
    Abar = A.mean(axis=0)
    Bbar = B.mean(axis=0)

    H = (A - Abar).T @ (B - Bbar)          # (3, 3) cross-covariance
    U, _, Vt = np.linalg.svd(H)

    # Ensure proper rotation (det = +1), not reflection
    d = np.linalg.det(U @ Vt)
    R = U @ np.diag([1.0, 1.0, d]) @ Vt    # rotation in [z, y, x] space

    return R, Abar, Bbar, ot_dist


def apply_rigid_transform_to_volume(volume, R, Abar, Bbar):
    """
    Apply the rigid transform (R, Abar, Bbar) to the voxel volume using
    scipy.ndimage.affine_transform (pull / inverse convention).

    Identical formula to EMPOT's apply_rigid_transform_to_volume:
        offset = Bbar + center - R_inv @ (Abar + center)

    where:
        Abar   = centroid of matched reference points [z, y, x]
        Bbar   = centroid of matched source points    [z, y, x]
        center = (N/2, N/2, N/2)  — geometric centre of the cube
        R_inv  = R.T              — inverse of orthogonal rotation matrix

    Parameters
    ----------
    volume : (D, H, W) float32 ndarray  — original subtomogram
    R      : (3, 3) float64             — rotation matrix in [z, y, x] space
    Abar   : (3,) float64               — reference centroid [z, y, x]
    Bbar   : (3,) float64               — source centroid    [z, y, x]

    Returns
    -------
    transformed : (D, H, W) float32 ndarray
    """
    N = volume.shape[0]
    center = np.array([N / 2.0, N / 2.0, N / 2.0])

    R_inv  = R.T    # R is orthogonal: R^{-1} = R^T
    offset = Bbar + center - R_inv @ (Abar + center)

    return affine_transform(
        volume.astype(np.float32),
        R_inv,
        offset=offset,
        order=3,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)


# ── Visualization ──────────────────────────────────────────────────────────────

def show_volume_slices(volume, title_prefix, save_path=None):
    """Plot central XY / XZ / YZ slices of a volume."""
    vol = np.asarray(volume, dtype=np.float32)
    cz, cy, cx = [s // 2 for s in vol.shape]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white")
    for ax, sl, lbl in zip(
        axes,
        [vol[cz, :, :], vol[:, cy, :], vol[:, :, cx]],
        ["XY", "XZ", "YZ"],
    ):
        vlo, vhi = np.percentile(sl, 2), np.percentile(sl, 98)
        ax.imshow(sl, cmap="gray", vmin=vlo, vmax=vhi, origin="lower")
        ax.set_title(f"{title_prefix}  [{lbl}]", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return fig, axes


def make_combined_plot(all_references, params_label, output_dir):
    """
    Grid plot: rows = XY / XZ / YZ,  cols = Round 0 (Initial) … Round N.
    """
    n_cols = len(all_references)
    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 12), facecolor="white")
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for col, ref in enumerate(all_references):
        ref = np.asarray(ref, dtype=np.float32)
        cz, cy, cx = [s // 2 for s in ref.shape]
        slices = [ref[cz, :, :], ref[:, cy, :], ref[:, :, cx]]
        title  = "Initial (R0)" if col == 0 else f"Round {col}"

        for row, (sl, lbl) in enumerate(zip(slices, ["XY", "XZ", "YZ"])):
            ax  = axes[row, col]
            vlo = np.percentile(sl, 2)
            vhi = np.percentile(sl, 98)
            ax.imshow(sl, cmap="gray", vmin=vlo, vmax=vhi, origin="lower")
            ax.set_title(f"{title}\n[{lbl}]", fontsize=10)
            ax.axis("off")

    fig.suptitle(
        f"AlignOT Kabsch Alignment — {n_cols - 1} rounds\n{params_label}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = Path(output_dir) / "slices_all_rounds.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def reference_change_stats(prev_ref, next_ref, round_idx):
    """Compute L2, mean-abs, max-abs difference between two reference volumes."""
    diff = np.asarray(next_ref, dtype=np.float32) - np.asarray(prev_ref, dtype=np.float32)
    return {
        "round":                     int(round_idx),
        "reference_l2_change":       float(np.linalg.norm(diff)),
        "reference_mean_abs_change": float(np.mean(np.abs(diff))),
        "reference_max_abs_change":  float(np.max(np.abs(diff))),
    }

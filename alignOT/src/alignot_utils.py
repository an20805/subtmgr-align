"""
alignot_utils.py
----------------
Utility functions for the AlignOT iterative subtomogram alignment pipeline.

Coordinate conventions
----------------------
TRN output : pts[:, 0] = z,  pts[:, 1] = y,  pts[:, 2] = x   (C-order z,y,x)
AlignOT SGD: takes separate x, y, z lists in (x, y, z) Euclidean order
Volume array: indexed as vol[z, y, x]

The quaternion recovered by SGD lives in (x, y, z) space.
apply_rotation_to_volume() permutes it to [z, y, x] before calling
scipy.ndimage.affine_transform.

TRN sampling follows exactly the approach in scripts/check_trn_sampling.py:
  - normalize -> invert so protein signal -> high values
  - threshold
  - trn_rm0 + trn_iterate with do_log=True (required for rms to be populated)
  - use rms[-1] as the final point cloud
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

# ── Add AlignOT source to Python path ─────────────────────────────────────────
_ALIGNOT_SRC = Path(__file__).resolve().parent.parent / "alignOT"
if str(_ALIGNOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ALIGNOT_SRC))

# Suppress the module-level print() calls in utils.py (prints qs, q, b on import)
with contextlib.redirect_stdout(io.StringIO()):
    import trn
    import coords as _coords
    import utils as _utils


# ── I/O helpers ───────────────────────────────────────────────────────────────

def get_cluster_paths(cluster_dir):
    """Return sorted list of .mrc paths (sorted by numeric stem, Zone.Identifier excluded)."""
    d = Path(cluster_dir)
    paths = [
        p for p in d.iterdir()
        if p.is_file() and p.suffix == ".mrc" and "Zone" not in p.name
    ]
    return sorted(paths, key=lambda p: int(p.stem))


def load_mrc(path, permissive=True):
    with mrcfile.open(str(path), permissive=permissive) as mrc:
        return np.array(mrc.data, dtype=np.float32)


def save_mrc(path, arr, voxel_size=None):
    arr = np.asarray(arr, dtype=np.float32)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(arr)
        if voxel_size is not None:
            mrc.voxel_size = float(voxel_size)
    return Path(path)


def save_inverted_mrc(path, arr, voxel_size=None):
    """Save contrast-inverted MRC density (-arr) for ChimeraX visualization."""
    inv = -np.asarray(arr, dtype=np.float32)
    return save_mrc(path, inv, voxel_size=voxel_size)


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


# ── Point-cloud sampling (exactly as in scripts/check_trn_sampling.py) ────────

def gaussian_lowpass_filter(volume, sigma):
    """Apply Gaussian low-pass filter. Returns unchanged volume if sigma is None or <= 0."""
    if sigma is None or sigma <= 0:
        return volume
    return gaussian_filter(volume.astype(np.float32), sigma=float(sigma))


def prepare_sampling_volume(volume, sigma=None):
    """
    Prepare a volume for TRN sampling:
      1. Optionally Gaussian low-pass filter (for sampling only, not averaging)
      2. Normalize to [0, 1]: (v - vmin) / (vmax - vmin)
      3. Invert: 1 - norm  => protein signal (dark in cryo-ET) becomes HIGH density

    Parameters
    ----------
    volume : (D, H, W) float32 ndarray
    sigma  : float or None — Gaussian LP sigma in voxels (applied before normalize)

    Returns
    -------
    prepared : (D, H, W) float32 ndarray, values in [0, 1], signal near 1
    """
    v = volume.astype(np.float32)
    if sigma is not None and sigma > 0:
        v = gaussian_filter(v, sigma=float(sigma))
    vmin, vmax = float(v.min()), float(v.max())
    norm = (v - vmin) / (vmax - vmin + 1e-8)
    return 1.0 - norm


def sample_volume_in_memory(prep_vol, threshold, num_points, random_seed=None):
    """
    TRN sampling directly via trn_rm0 + trn_iterate.

    IMPORTANT: do_log=True is required so that trn_iterate actually writes
    the sampled positions into rms[]. With do_log=False the save logic is
    skipped and rms stays uninitialized (returns all-zeros).

    Parameters
    ----------
    prep_vol    : (D, H, W) float32  — output of prepare_sampling_volume()
    threshold   : float              — density threshold (values below -> 0)
    num_points  : int                — M points to sample
    random_seed : int or None

    Returns
    -------
    pts : (num_points, 3) float32,  columns = [z, y, x]  (TRN centered coords)
    """
    map_th = prep_vol.copy()
    map_th[map_th < threshold] = 0.0
    if float(map_th.sum()) <= 0:
        raise ValueError(
            f"After applying threshold={threshold} the volume is all zeros. "
            "Lower the threshold or check the normalisation."
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
            do_log=True,    # MUST be True — see docstring
            log_n=10,
        )

    return rms[-1].astype(np.float32)   # (num_points, 3), cols = [z, y, x]


# ── AlignOT alignment ──────────────────────────────────────────────────────────

def run_alignot_alignment(pts_src, pts_ref, reg=30, sgd_iters=500, lr=0.00001):
    """
    Align a subtomogram point cloud to the reference using AlignOT's SGD.

    Both point clouds are centered to their respective centroids before SGD.
    The centroids (Bbar for source, Abar for reference) are returned so that
    apply_rotation_to_volume can apply the correct EMPOT-style rigid transform:
        offset = Bbar + center - R_inv @ (Abar + center)
    This encodes both the rotation AND the centroid-to-centroid translation,
    exactly as EMPOT's Kabsch SVD does.

    Parameters
    ----------
    pts_src   : (N, 3) float32 — subtomogram point cloud [z, y, x] (TRN output)
    pts_ref   : (N, 3) float32 — reference point cloud   [z, y, x]
    reg       : float          — Sinkhorn regularisation (default 30)
    sgd_iters : int            — SGD iterations          (default 500)
    lr        : float          — learning rate           (default 1e-5)

    Returns
    -------
    q         : list [w, x, y, z] — final rotation quaternion
    costs     : list[float]       — Wasserstein cost at each SGD step
    wass_dist : float             — final Wasserstein distance
    Bbar      : (3,) float64      — centroid of source cloud [z, y, x]
    Abar      : (3,) float64      — centroid of reference cloud [z, y, x]
    """
    # Compute centroids in [z, y, x] space
    Bbar = pts_src.mean(axis=0).astype(np.float64)   # source centroid
    Abar = pts_ref.mean(axis=0).astype(np.float64)   # reference centroid

    # Center both clouds before alignment
    pts_src_c = pts_src - Bbar
    pts_ref_c = pts_ref - Abar

    # TRN gives cols [z, y, x]; AlignOT SGD expects separate (x, y, z) lists
    src_x = list(pts_src_c[:, 2].astype(float))
    src_y = list(pts_src_c[:, 1].astype(float))
    src_z = list(pts_src_c[:, 0].astype(float))
    ref_x = list(pts_ref_c[:, 2].astype(float))
    ref_y = list(pts_ref_c[:, 1].astype(float))
    ref_z = list(pts_ref_c[:, 0].astype(float))

    with contextlib.redirect_stdout(io.StringIO()):
        quaternions, costs = _utils.SGD(
            src_x, src_y, src_z,
            ref_x, ref_y, ref_z,
            lr=lr,
            max_iter=sgd_iters,
            reg=reg,
            num_samples=1,
            verbose=False,
        )

    final_q = quaternions[-1]       # list [w, x, y, z]
    wass_dist = float(costs[-1])
    return final_q, costs, wass_dist, Bbar, Abar


# ── Volume rotation ────────────────────────────────────────────────────────────

def apply_rotation_to_volume(volume, q, Bbar, Abar):
    """
    Apply the rotation encoded by quaternion q to the voxel volume, with a
    centroid-based offset matching EMPOT's apply_rigid_transform_to_volume.

    EMPOT formula (Kabsch SVD):
        offset = Bbar + center - R_inv @ (Abar + center)

    where Abar = centroid of reference point cloud (where we align TO),
          Bbar = centroid of source point cloud (where we align FROM).

    This encodes both rotation AND translation in one affine transform so that
    the rotation pivot is the source centroid (Bbar) and the result is placed
    at the reference centroid (Abar).

    q is in AlignOT's (x, y, z) convention; the volume is indexed [z, y, x],
    so R must be permuted. Bbar and Abar are both in [z, y, x] order.

    Parameters
    ----------
    volume : (D, H, W) float32 ndarray
    q      : list/array [w, x, y, z]
    Bbar   : (3,) float64 — centroid of source point cloud [z, y, x]
    Abar   : (3,) float64 — centroid of reference point cloud [z, y, x]

    Returns
    -------
    transformed : (D, H, W) float32 ndarray
    """
    N = volume.shape[0]
    center = np.full(3, N / 2.0)

    # Quaternion -> rotation matrix in (x, y, z) Euclidean space
    R_xyz = _coords.quaternion_to_R(list(q))          # (3, 3)

    # Permute: P maps [z, y, x] <-> [x, y, z]
    P     = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0]], dtype=np.float64)
    R_zyx = P @ R_xyz @ P.T     # rotation matrix in [z, y, x] voxel frame

    # scipy affine_transform uses pull (inverse) convention:
    #   output[o] = input[R_inv @ (o - offset)]   (actually: R_inv @ o + offset_arg)
    # EMPOT formula: offset = Bbar + center - R_inv @ (Abar + center)
    R_inv  = R_zyx.T            # orthogonal: inverse == transpose
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
    Grid plot: rows = XY / XZ / YZ,  columns = Round 0 (Initial) … Round N.
    Saved to <output_dir>/slices_all_rounds.png.

    Parameters
    ----------
    all_references : list of (D,H,W) float32 arrays, index 0 = initial average
    params_label   : str — shown in the figure suptitle
    output_dir     : Path
    """
    n_cols = len(all_references)

    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 12), facecolor="white")
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for col, ref in enumerate(all_references):
        ref = np.asarray(ref, dtype=np.float32)
        cz, cy, cx = [s // 2 for s in ref.shape]
        slices = [ref[cz, :, :], ref[:, cy, :], ref[:, :, cx]]
        labels = ["XY", "XZ", "YZ"]
        title  = "Initial (R0)" if col == 0 else f"Round {col}"

        for row, (sl, lbl) in enumerate(zip(slices, labels)):
            ax  = axes[row, col]
            vlo = np.percentile(sl, 2)
            vhi = np.percentile(sl, 98)
            ax.imshow(sl, cmap="gray", vmin=vlo, vmax=vhi, origin="lower")
            ax.set_title(f"{title}\n[{lbl}]", fontsize=10)
            ax.axis("off")

    fig.suptitle(
        f"AlignOT Iterative Alignment — {n_cols - 1} rounds\n{params_label}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = Path(output_dir) / "slices_all_rounds.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def reference_change_stats(prev_ref, next_ref, round_idx):
    """Compute L2, mean-abs, max-abs difference between two references."""
    diff = np.asarray(next_ref, dtype=np.float32) - np.asarray(prev_ref, dtype=np.float32)
    return {
        "round":             int(round_idx),
        "reference_l2_change":       float(np.linalg.norm(diff)),
        "reference_mean_abs_change": float(np.mean(np.abs(diff))),
        "reference_max_abs_change":  float(np.max(np.abs(diff))),
    }

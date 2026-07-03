import sys
import os
import contextlib
import io
import gc
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mrcfile
import torch
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation

# Import from existing EMPOT modules
import trn
from unbalancedgw.vanilla_ugw_solver import log_ugw_sinkhorn
from unbalancedgw._vanilla_utils import l2_distortion

def get_cluster_paths(cluster_dir):
    cluster_dir = Path(cluster_dir)
    paths = [path for path in cluster_dir.iterdir() if path.is_file() and path.suffix == ".mrc" and "Zone" not in path.name]
    paths = sorted(paths, key=lambda path: int(path.stem))
    return paths

def load_mrc(path, permissive=True):
    with mrcfile.open(path, permissive=permissive) as mrc:
        return np.array(mrc.data, dtype=np.float32)

def save_mrc(path, arr):
    arr = np.asarray(arr, dtype=np.float32)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(arr)
    return Path(path)

def compute_streaming_average(paths, logger=None, progress_every=20):
    total = None
    count = 0
    for idx, path in enumerate(paths, start=1):
        arr = load_mrc(path, permissive=True)
        if total is None:
            total = np.zeros_like(arr, dtype=np.float64)
        total += arr
        count += 1
        if progress_every and (idx % progress_every == 0 or idx == len(paths)):
            if logger:
                logger.info(f"Initial averaging: loaded {idx}/{len(paths)} subtomograms")
        del arr
        if progress_every and idx % progress_every == 0:
            gc.collect()
    if count == 0:
        raise ValueError("Cannot compute an average from zero subtomograms.")
    return (total / count).astype(np.float32)

def show_volume_slices(volume, title_prefix, save_path=None):
    volume = np.asarray(volume, dtype=np.float32)
    mid_z = volume.shape[0] // 2
    mid_y = volume.shape[1] // 2
    mid_x = volume.shape[2] // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(volume[mid_z, :, :], cmap="gray")
    axes[0].set_title(f"{title_prefix}: XY")
    axes[1].imshow(volume[:, mid_y, :], cmap="gray")
    axes[1].set_title(f"{title_prefix}: XZ")
    axes[2].imshow(volume[:, :, mid_x], cmap="gray")
    axes[2].set_title(f"{title_prefix}: YZ")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    return fig, axes

def reference_change_stats(prev_ref, next_ref, round_idx):
    diff = np.asarray(next_ref, dtype=np.float32) - np.asarray(prev_ref, dtype=np.float32)
    return {
        "round": int(round_idx),
        "reference_l2_change": float(np.linalg.norm(diff)),
        "reference_mean_abs_change": float(np.mean(np.abs(diff))),
        "reference_max_abs_change": float(np.max(np.abs(diff))),
    }

def normalize_volume(volume):
    volume = volume.astype(np.float32)
    volume = volume - volume.min()
    vmax = float(volume.max())
    if vmax <= 0:
        raise ValueError("Volume is constant after normalization.")
    return volume / (vmax + 1e-8)

def prepare_sampling_volume(volume):
    norm = normalize_volume(volume)
    return 1.0 - norm

def sample_volume_in_memory(prepared_volume, threshold, num_points, random_seed=None):
    map_th = np.asarray(prepared_volume, dtype=np.float32).copy()
    map_th[map_th < threshold] = 0
    if float(map_th.sum()) <= 0:
        raise ValueError(f"Threshold {threshold} zeroes out the entire volume.")

    with contextlib.redirect_stdout(io.StringIO()):
        rm0, arr_flat, arr_idx, xyz, coords_1d = trn.trn_rm0(
            map_th, M=num_points, random_seed=random_seed,
        )
        rms, rs, ts_save = trn.trn_iterate(
            rm0, arr_flat, arr_idx, xyz,
            n_save=10, e0=0.3, ef=0.05,
            l0=0.005 * num_points, lf=0.5,
            tf=num_points * 8,
            do_log=True, log_n=10,
        )
    return rms[-1]

def run_empot_alignment(pts_ref, pts_tgt,
                         eps=10000, rho=100000, rho2=100000,
                         nits_plan=100, nits_sinkhorn=100):
    n_ref = len(pts_ref)
    n_tgt = len(pts_tgt)

    a = np.ones(n_ref) / n_ref
    b = np.ones(n_tgt) / n_tgt

    dx = np.sum((pts_ref[:, None, :] - pts_ref[None, :, :]) ** 2, axis=-1)
    dy = np.sum((pts_tgt[:, None, :] - pts_tgt[None, :, :]) ** 2, axis=-1)

    a_t = torch.from_numpy(a)
    b_t = torch.from_numpy(b)
    dx_t = torch.from_numpy(dx)
    dy_t = torch.from_numpy(dy)

    pi, gamma = log_ugw_sinkhorn(
        a_t, dx_t, b_t, dy_t, init=None, eps=eps,
        rho=rho, rho2=rho2,
        nits_plan=nits_plan, tol_plan=1e-10,
        nits_sinkhorn=nits_sinkhorn, tol_sinkhorn=1e-10,
        two_outputs=True,
    )

    dist = float(l2_distortion(pi, gamma, dx_t, dy_t))

    pi_np = np.array(pi)
    all_coup = []
    for i in range(n_tgt):
        j = int(np.argmax(pi_np[:, i]))
        if pi_np[j, i] > 0:
            all_coup.append((i, j))

    if not all_coup:
        raise ValueError("No correspondences found between point clouds.")

    A = np.array([pts_ref[c[1]] for c in all_coup])
    B = np.array([pts_tgt[c[0]] for c in all_coup])

    Abar = A.mean(axis=0)
    Bbar = B.mean(axis=0)

    H = (A - Abar).T @ (B - Bbar)
    U, S, V = np.linalg.svd(H)

    d = np.linalg.det(U @ V)
    R = U @ np.diag([1.0, 1.0, d]) @ V

    R_recovered = Rotation.from_matrix(R)
    return Abar, Bbar, R_recovered, dist

def apply_rigid_transform_to_volume(volume, R_rec, Abar, Bbar):
    N = volume.shape[0]
    center = np.array([N/2.0, N/2.0, N/2.0])
    R_inv_mat = R_rec.inv().as_matrix()
    offset = Bbar + center - R_inv_mat @ (Abar + center)
    return affine_transform(volume, R_inv_mat, offset=offset, order=3, mode='constant', cval=0.0)

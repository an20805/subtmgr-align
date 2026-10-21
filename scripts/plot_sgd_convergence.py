#!/usr/bin/env python3
"""
plot_sgd_convergence.py
-----------------------
Inspect and visualize the convergence of AlignOT SGD for individual particle alignments.
Tracks:
  1. Entropic Optimal Transport Cost (Wasserstein distance W_eps)
  2. Step-by-step angular change on SO(3) (degrees/step)
  3. Angular distance to the solution at iteration 1000
  4. Nearest-neighbor point cloud RMSD against reference
  5. Cross-particle convergence comparison (Particles 0, 1, 2)

Usage:
  python scripts/plot_sgd_convergence.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add AlignOT src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "alignOT" / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REPO_ROOT / "alignOT" / "alignOT"))

import alignot_utils


def soft_angle_degrees(q1: torch.Tensor, q2: torch.Tensor) -> float:
    """Geodesic distance on SO(3) in degrees between two unit quaternions."""
    dot = torch.clamp(torch.abs((q1 * q2).sum()), 0.0, 1.0)
    return float(2.0 * torch.acos(dot).item() * (180.0 / np.pi))


def run_tracked_sgd(
    src_xyz: np.ndarray,
    ref_xyz: np.ndarray,
    max_iter: int = 1500,
    lr: float = 0.001,
    reg: float = 100.0,
    sinkhorn_iters: int = 50,
    target_mode: str = "base",
    seed: int = 42,
    clip_grad_norm: float = None,
    device: torch.device = torch.device("cuda"),
) -> Dict[str, np.ndarray]:
    """Runs Native GPU AlignOT SGD while tracking loss and rotational trajectory at every step."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    B, N = 1, src_xyz.shape[0]
    M = ref_xyz.shape[0]

    src_b = torch.from_numpy(src_xyz).unsqueeze(0).to(device)  # (1, N, 3)
    ref = torch.from_numpy(ref_xyz).to(device)                # (M, 3)
    ref_exp = ref.unsqueeze(0)                                # (1, M, 3)

    a = torch.ones(B, N, device=device) / float(N)
    b = torch.ones(B, M, device=device) / float(M)

    q_b = torch.zeros(B, 4, device=device)
    q_b[:, 0] = 1.0  # Identity quaternion [1, 0, 0, 0]

    u_b = torch.ones(B, M, device=device) / float(M)
    v_b = torch.ones(B, N, device=device) / float(N)
    Kp_b_const = (1.0 / a).unsqueeze(1)

    costs = []
    angular_steps = []
    grad_norms = []
    quats = []
    rmsd_history = []

    for it in range(max_iter):
        w, x, y, z = q_b[:, 0], q_b[:, 1], q_b[:, 2], q_b[:, 3]

        # 3x3 rotation matrix
        R_b = torch.stack([
            x**2 + w**2 - z**2 - y**2, 2.0 * (x*y - w*z),             2.0 * (x*z + w*y),
            2.0 * (x*y + w*z),         y**2 + w**2 - x**2 - z**2, 2.0 * (y*z - w*x),
            2.0 * (x*z - w*y),         2.0 * (y*z + w*x),             z**2 + w**2 - y**2 - x**2,
        ], dim=1).reshape(B, 3, 3)

        # Rotated coordinates
        xx_b = torch.bmm(src_b, R_b.transpose(1, 2))  # (1, N, 3)

        # Cost matrix M_b (L2 squared)
        M_b = torch.cdist(ref_exp, xx_b, p=2.0) ** 2  # (1, M, N)
        K_b = torch.exp(-M_b / reg)
        Kp_b = Kp_b_const * K_b

        # Sinkhorn iterations
        for _ in range(sinkhorn_iters):
            KtransposeU = torch.bmm(K_b.transpose(1, 2), u_b.unsqueeze(-1)).squeeze(-1)
            v_b = b / (KtransposeU + 1e-16)
            u_b = 1.0 / (torch.bmm(Kp_b, v_b.unsqueeze(-1)).squeeze(-1) + 1e-16)

        T_b = u_b.unsqueeze(-1) * K_b * v_b.unsqueeze(1)
        current_cost = (T_b * M_b).sum().item()
        costs.append(current_cost)

        # Point cloud RMSD to nearest reference points
        sq_dists_all = torch.cdist(xx_b, ref_exp, p=2.0).squeeze(0)  # (N, M)
        min_dists, _ = torch.min(sq_dists_all, dim=1)
        rmsd = torch.sqrt(torch.mean(min_dists**2)).item()
        rmsd_history.append(rmsd)

        # Stochastic sample point selection (1 point per particle)
        s_indices = torch.randint(0, N, (B,), device=device)
        b_idx = torch.arange(B, device=device)
        col_b = T_b[b_idx, :, s_indices]

        if target_mode in ("base", "argmax"):
            dest_b = torch.argmax(col_b, dim=1)
        else:
            col_sum_b = col_b.sum(dim=1, keepdim=True)
            weights_b = col_b / (col_sum_b + 1e-16)
            bary_pt_b = torch.bmm(weights_b.unsqueeze(1), ref_exp).squeeze(1)
            sq_dists_b = torch.cdist(bary_pt_b.unsqueeze(1), ref_exp, p=2.0).squeeze(1)
            dest_b = torch.argmin(sq_dists_b, dim=1)

        x2_pt_b = ref[dest_b]
        x1_pt_b = src_b[b_idx, s_indices]

        x1, y1, z1 = x1_pt_b[:, 0], x1_pt_b[:, 1], x1_pt_b[:, 2]
        x2, y2, z2 = x2_pt_b[:, 0], x2_pt_b[:, 1], x2_pt_b[:, 2]

        i_vals = (x**2 + w**2 - z**2 - y**2)*x1 + 2.0*(x*y - w*z)*y1 + 2.0*(x*z + w*y)*z1
        j_vals = 2.0*(x*y + w*z)*x1 + (y**2 + w**2 - x**2 - z**2)*y1 + 2.0*(y*z - w*x)*z1
        k_vals = 2.0*(x*z - w*y)*x1 + 2.0*(y*z + w*x)*y1 + (z**2 + w**2 - y**2 - x**2)*z1

        # Analytical derivatives
        i_der = torch.stack([2.0*(w*x1 + y*z1 - z*y1), 2.0*(x*x1 + y*y1 + z*z1), 2.0*(x*y1 + w*z1 - y*x1), 2.0*(x*z1 - w*y1 - z*x1)], dim=1)
        j_der = torch.stack([2.0*(w*y1 + z*x1 - x*z1), 2.0*(y*x1 - w*z1 - x*y1), 2.0*(x*x1 + y*y1 + z*z1), 2.0*(y*z1 + w*x1 - z*y1)], dim=1)
        k_der = torch.stack([2.0*(w*z1 + x*y1 - y*x1), 2.0*(z*x1 + w*y1 - x*z1), 2.0*(z*y1 - w*x1 - y*z1), 2.0*(x*x1 + y*y1 + z*z1)], dim=1)

        t_weights = (T_b[b_idx, dest_b, s_indices] * float(N)).unsqueeze(-1)
        d_i = (i_vals - x2).unsqueeze(-1)
        d_j = (j_vals - y2).unsqueeze(-1)
        d_k = (k_vals - z2).unsqueeze(-1)
        grad_b = 2.0 * (i_der * d_i + j_der * d_j + k_der * d_k) * t_weights

        # S3 tangent space projection
        d_prod_b = (grad_b * q_b).sum(dim=1, keepdim=True)
        grad_proj_b = grad_b - d_prod_b * q_b
        gnorm = float(torch.norm(grad_proj_b).item())
        grad_norms.append(gnorm)

        if clip_grad_norm is not None and gnorm > clip_grad_norm:
            grad_proj_b = grad_proj_b * (clip_grad_norm / (gnorm + 1e-12))

        q_old = q_b.clone()
        q_b = q_b - lr * grad_proj_b
        q_b = q_b / torch.norm(q_b, dim=1, keepdim=True)

        angular_steps.append(soft_angle_degrees(q_old[0], q_b[0]))
        quats.append(q_b[0].cpu().numpy().copy())

    quats_arr = np.array(quats)
    q_1000 = torch.tensor(quats_arr[min(999, max_iter - 1)], device=device)

    # Compute distance to q_1000 for each iteration
    dist_to_1000 = [
        soft_angle_degrees(torch.tensor(q_i, device=device), q_1000)
        for q_i in quats_arr
    ]

    return {
        "costs": np.array(costs),
        "angular_steps": np.array(angular_steps),
        "grad_norms": np.array(grad_norms),
        "quats": quats_arr,
        "rmsd": np.array(rmsd_history),
        "dist_to_1000": np.array(dist_to_1000),
    }


def compute_rolling_mean(arr: np.ndarray, window: int = 40) -> np.ndarray:
    """Computes moving average with edge padding."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[:len(arr)]


def main():
    parser = argparse.ArgumentParser(description="Check AlignOT SGD convergence for one particle")
    parser.add_argument("--particle_path", type=str, default="Data/0.1/subtomograms/0/0.mrc")
    parser.add_argument("--ref_path", type=str, default="Data/0.mrc")
    parser.add_argument("--max_iter", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--reg", type=float, default=100.0)
    parser.add_argument("--sinkhorn_iters", type=int, default=50)
    parser.add_argument("--num_points", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--lowpass_sigma", type=float, default=2.0)
    parser.add_argument("--target_mode", type=str, default="base")
    parser.add_argument("--output_dir", type=str, default="experiments/sgd_convergence")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("AlignOT SGD Convergence Inspector")
    print(f"Particle : {args.particle_path}")
    print(f"Reference: {args.ref_path}")
    print(f"Params   : max_iter={args.max_iter}, lr={args.lr}, reg={args.reg}, M={args.num_points}")
    print(f"Device   : {device}")
    print("=" * 72)

    # 1. Load volumes & sample point clouds
    vol_ref = alignot_utils.load_mrc(Path(args.ref_path))
    pts_ref = alignot_utils.sample_volume_in_memory(
        alignot_utils.prepare_sampling_volume(vol_ref, sigma=args.lowpass_sigma),
        threshold=args.threshold,
        num_points=args.num_points,
        random_seed=42,
    )
    ref_xyz = (pts_ref - pts_ref.mean(axis=0))[:, [2, 1, 0]].astype(np.float32)

    vol_src = alignot_utils.load_mrc(Path(args.particle_path))
    pts_src = alignot_utils.sample_volume_in_memory(
        alignot_utils.prepare_sampling_volume(vol_src, sigma=args.lowpass_sigma),
        threshold=args.threshold,
        num_points=args.num_points,
        random_seed=43,
    )
    src_xyz = (pts_src - pts_src.mean(axis=0))[:, [2, 1, 0]].astype(np.float32)

    # 2. Run tracked SGD for primary particle
    t0 = time.perf_counter()
    res = run_tracked_sgd(
        src_xyz,
        ref_xyz,
        max_iter=args.max_iter,
        lr=args.lr,
        reg=args.reg,
        sinkhorn_iters=args.sinkhorn_iters,
        target_mode=args.target_mode,
        seed=42,
        device=device,
    )
    elapsed = time.perf_counter() - t0
    print(f"Tracked SGD finished in {elapsed:.2f}s ({elapsed * 1000.0 / args.max_iter:.2f} ms/iter)")

    # 3. Multi-particle comparison (Particles 0, 1, 2)
    print("Running cross-particle comparison (Particles 0, 1, 2)...")
    multi_res = {"Particle 0": res}
    for p_id in [1, 2]:
        p_file = Path(f"Data/0.1/subtomograms/0/{p_id}.mrc")
        if p_file.exists():
            v_p = alignot_utils.load_mrc(p_file)
            pts_p = alignot_utils.sample_volume_in_memory(
                alignot_utils.prepare_sampling_volume(v_p, sigma=args.lowpass_sigma),
                threshold=args.threshold,
                num_points=args.num_points,
                random_seed=43 + p_id,
            )
            s_xyz_p = (pts_p - pts_p.mean(axis=0))[:, [2, 1, 0]].astype(np.float32)
            multi_res[f"Particle {p_id}"] = run_tracked_sgd(
                s_xyz_p,
                ref_xyz,
                max_iter=args.max_iter,
                lr=args.lr,
                reg=args.reg,
                sinkhorn_iters=args.sinkhorn_iters,
                target_mode=args.target_mode,
                seed=42 + p_id,
                device=device,
            )

    # 4. Generate Main 4-Panel Convergence Figure
    costs = res["costs"]
    costs_smooth = compute_rolling_mean(costs, window=40)
    ang_steps = res["angular_steps"]
    ang_steps_smooth = compute_rolling_mean(ang_steps, window=40)
    dist_1000 = res["dist_to_1000"]
    dist_1000_smooth = compute_rolling_mean(dist_1000, window=40)
    rmsd = res["rmsd"]
    rmsd_smooth = compute_rolling_mean(rmsd, window=40)

    iters = np.arange(len(costs))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: OT Cost (Wasserstein distance)
    ax_a = axes[0, 0]
    ax_a.plot(iters, costs, color="#a0c4ff", alpha=0.45, lw=1.0, label="Stochastic OT Cost (per step)")
    ax_a.plot(iters, costs_smooth, color="#00509d", lw=2.2, label="Moving Average (window=40)")
    ax_a.axvline(1000, color="#d90429", linestyle="--", lw=1.8, label="Pipeline Cutoff (1000 iters)")

    min_cost = costs.min()
    min_iter = np.argmin(costs)
    ax_a.axhline(min_cost, color="#2b9348", linestyle=":", lw=1.5, label=f"Min Cost ({min_cost:.1f} at #{min_iter})")

    ax_a.set_title("(A) Entropic Optimal Transport Cost $\\mathcal{W}_{\\epsilon}$", fontsize=12, fontweight="bold")
    ax_a.set_xlabel("SGD Iteration", fontsize=11)
    ax_a.set_ylabel("Wasserstein Cost $\\langle T, M \\rangle$", fontsize=11)
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.legend(loc="upper right", fontsize=9.5, framealpha=0.9)

    # Annotate key milestones
    ax_a.annotate(
        f"Initial: {costs[0]:.1f}",
        xy=(0, costs[0]),
        xytext=(40, costs[0] + 8),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
        fontsize=9,
    )
    c_200 = costs_smooth[min(200, len(costs_smooth)-1)]
    ax_a.annotate(
        f"Iter 200: {c_200:.1f}\n(Rapid Descent)",
        xy=(200, c_200),
        xytext=(260, c_200 + 25),
        arrowprops=dict(facecolor="#00509d", shrink=0.08, width=1, headwidth=6),
        fontsize=9,
        fontweight="bold",
        color="#00509d",
    )
    c_1000 = costs_smooth[min(1000, len(costs_smooth)-1)]
    ax_a.annotate(
        f"Iter 1000: {c_1000:.1f}\n(Flat Plateau)",
        xy=(1000, c_1000),
        xytext=(950, c_1000 + 35),
        arrowprops=dict(facecolor="#d90429", shrink=0.08, width=1, headwidth=6),
        fontsize=9,
        fontweight="bold",
        color="#d90429",
    )

    # Panel B: Step-to-Step Angular Change
    ax_b = axes[0, 1]
    ax_b.plot(iters, ang_steps, color="#ffc6ff", alpha=0.45, lw=1.0, label="Step size (degrees / step)")
    ax_b.plot(iters, ang_steps_smooth, color="#7209b7", lw=2.2, label="Moving Average (window=40)")
    ax_b.axvline(1000, color="#d90429", linestyle="--", lw=1.8, label="Pipeline Cutoff (1000 iters)")
    ax_b.set_title("(B) Step-to-Step Angular Rotation Update $\\Delta \\theta$", fontsize=12, fontweight="bold")
    ax_b.set_xlabel("SGD Iteration", fontsize=11)
    ax_b.set_ylabel("Angular Step Size ($^\\circ$ / step)", fontsize=11)
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(loc="upper right", fontsize=9.5, framealpha=0.9)

    # Panel C: Angular Distance to Iteration 1000
    ax_c = axes[1, 0]
    ax_c.plot(iters, dist_1000, color="#ffd166", alpha=0.45, lw=1.0, label="Angle to $q_{1000}$ (degrees)")
    ax_c.plot(iters, dist_1000_smooth, color="#e76f51", lw=2.2, label="Moving Average (window=40)")
    ax_c.axvline(1000, color="#d90429", linestyle="--", lw=1.8, label="Pipeline Cutoff (1000 iters)")
    ax_c.axhline(0, color="gray", linestyle=":", lw=1.0)
    ax_c.set_title("(C) Geodesic Angular Distance to Iteration 1000 Solution", fontsize=12, fontweight="bold")
    ax_c.set_xlabel("SGD Iteration", fontsize=11)
    ax_c.set_ylabel("Angle to $q_{1000}$ ($^\\circ$)", fontsize=11)
    ax_c.grid(True, linestyle=":", alpha=0.6)
    ax_c.legend(loc="upper right", fontsize=9.5, framealpha=0.9)

    # Panel D: Multi-Particle Overlay Comparison
    ax_d = axes[1, 1]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for (name, p_res), col in zip(multi_res.items(), colors):
        p_costs = p_res["costs"]
        p_smooth = compute_rolling_mean(p_costs, window=40)
        ax_d.plot(iters[:len(p_smooth)], p_smooth, lw=2.0, color=col, label=f"{name} (Initial: {p_costs[0]:.0f} $\\to$ {p_costs[min(1000, len(p_costs)-1)]:.0f})")

    ax_d.axvline(1000, color="#d90429", linestyle="--", lw=1.8, label="Pipeline Cutoff (1000 iters)")
    ax_d.set_title("(D) Cross-Particle Convergence Consistency (Particles 0, 1, 2)", fontsize=12, fontweight="bold")
    ax_d.set_xlabel("SGD Iteration", fontsize=11)
    ax_d.set_ylabel("Smoothed Wasserstein Cost $\\langle T, M \\rangle$", fontsize=11)
    ax_d.grid(True, linestyle=":", alpha=0.6)
    ax_d.legend(loc="upper right", fontsize=9.5, framealpha=0.9)

    fig.suptitle(
        f"AlignOT GPU Barycentric/Argmax SGD Convergence Profile (Particle 0 vs. Ground Truth)\n"
        f"M={args.num_points} TRN Points | lr={args.lr} | reg={args.reg} | Sinkhorn Iters={args.sinkhorn_iters}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    plot_path = out_dir / "sgd_convergence_analysis.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved 4-panel convergence plot to: {plot_path.resolve()}")

    # 5. Quantitative Convergence Milestones
    initial_cost = float(costs[0])
    cost_at_100 = float(costs_smooth[100])
    cost_at_200 = float(costs_smooth[200])
    cost_at_300 = float(costs_smooth[300])
    cost_at_500 = float(costs_smooth[500])
    cost_at_1000 = float(costs_smooth[min(1000, len(costs_smooth)-1)])
    cost_at_1500 = float(costs_smooth[-1])

    drop_total = initial_cost - cost_at_1000
    pct_by_200 = 100.0 * (initial_cost - cost_at_200) / (drop_total + 1e-12)
    pct_by_300 = 100.0 * (initial_cost - cost_at_300) / (drop_total + 1e-12)
    pct_by_500 = 100.0 * (initial_cost - cost_at_500) / (drop_total + 1e-12)

    stats = {
        "initial_cost": initial_cost,
        "cost_at_100": cost_at_100,
        "cost_at_200": cost_at_200,
        "cost_at_300": cost_at_300,
        "cost_at_500": cost_at_500,
        "cost_at_1000": cost_at_1000,
        "cost_at_1500": cost_at_1500,
        "min_cost": float(min_cost),
        "min_iter": int(min_iter),
        "pct_progress_at_200": float(pct_by_200),
        "pct_progress_at_300": float(pct_by_300),
        "pct_progress_at_500": float(pct_by_500),
        "angle_change_at_1000_deg_per_step": float(ang_steps_smooth[min(1000, len(ang_steps_smooth)-1)]),
        "angle_distance_between_500_and_1000_deg": float(dist_1000_smooth[500]),
    }

    # 6. Generate Clipped vs Unclipped Comparison Plot
    print("Generating Clipped vs. Unclipped Stabilization comparison...")
    res_clipped = run_tracked_sgd(
        src_xyz,
        ref_xyz,
        max_iter=args.max_iter,
        lr=args.lr,
        reg=args.reg,
        sinkhorn_iters=args.sinkhorn_iters,
        target_mode=args.target_mode,
        seed=42,
        clip_grad_norm=20.0,
        device=device,
    )

    fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Raw vs Clipped Loss
    ax1.plot(iters, costs, color="#a0c4ff", alpha=0.4, lw=1.0, label="Baseline Unclipped (Raw)")
    ax1.plot(iters, costs_smooth, color="#d90429", lw=2.2, label="Baseline Unclipped (Moving Avg)")
    ax1.plot(iters, res_clipped["costs"], color="#2b9348", lw=2.2, label="Gradient-Clipped (max_norm=20.0)")
    ax1.axvline(1000, color="gray", linestyle="--", lw=1.5, label="1000 iters cutoff")
    ax1.set_title("Optimal Transport Cost Trajectory", fontsize=12, fontweight="bold")
    ax1.set_xlabel("SGD Iteration", fontsize=11)
    ax1.set_ylabel("Wasserstein Cost $\\langle T, M \\rangle$", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9.5)

    # Angular step sizes
    ax2.plot(iters, ang_steps, color="#d90429", alpha=0.5, lw=1.0, label=r"Baseline Unclipped ($\|\nabla\| \leq 1217$)")
    ax2.plot(iters, res_clipped["angular_steps"], color="#2b9348", alpha=0.8, lw=1.2, label=r"Gradient-Clipped ($\|\nabla\| \leq 20$)")
    ax2.axvline(1000, color="gray", linestyle="--", lw=1.5, label="1000 iters cutoff")
    ax2.set_title(r"Angular Step Size per Iteration ($\Delta \theta$ in degrees)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("SGD Iteration", fontsize=11)
    ax2.set_ylabel(r"Step Angle ($^\circ$)", fontsize=11)
    ax2.set_ylim(-0.2, 10.0)  # zoom in to show clean comparison
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9.5)

    fig_comp.suptitle(
        "Particle 0 SGD Dynamics: Baseline Stochastic Jumps vs. Gradient-Clipped Stable Convergence",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    comp_plot_path = out_dir / "sgd_stabilized_comparison.png"
    plt.savefig(comp_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig_comp)
    print(f"✓ Saved stabilized comparison plot to: {comp_plot_path.resolve()}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
plot_multi_round_fsc.py

Given an even run directory and an odd run directory, for each round:
  - Computes the overall average: (even_ref + odd_ref) / 2
  - Computes FSC(even_ref, odd_ref)

Produces a single comprehensive figure:
  Rows  = Rounds 0 … N
  Cols  = XY slice | XZ slice | YZ slice | FSC (all rounds overlaid,
                                                 current round highlighted)

Also saves the final overall average as an MRC.
"""

import argparse
from pathlib import Path
import numpy as np
import mrcfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys
sys.path.insert(0, str(Path(__file__).parent))
from compute_fsc import (
    load_mrc, save_mrc, match_sizes,
    compute_fsc, resolution_at_threshold,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def norm_display(sl):
    lo, hi = np.percentile(sl, 1), np.percentile(sl, 99)
    return np.clip((sl - lo) / (hi - lo + 1e-9), 0, 1)


def central_slices(vol):
    c = [s // 2 for s in vol.shape]
    return {
        "XY": vol[c[0], :, :],
        "XZ": vol[:, c[1], :],
        "YZ": vol[:, :, c[2]],
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--even_dir", required=True)
    parser.add_argument("--odd_dir",  required=True)
    parser.add_argument("--n_rounds", type=int, required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--apix",     type=float, default=4.0)
    args = parser.parse_args()

    even_dir = Path(args.even_dir)
    odd_dir  = Path(args.odd_dir)
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rounds  = args.n_rounds
    apix      = args.apix
    n_rows    = n_rounds + 1          # Round 0 … Round N
    n_cols    = 4                     # XY | XZ | YZ | FSC

    # ── pass 1: collect data for all rounds ─────────────────────────────────
    rounds_data = []        # list of dicts keyed by round index
    all_shell_centers = None
    all_fsc_curves    = []  # one entry per round (in order)
    all_nvox          = []
    all_res143        = []

    for r in range(n_rounds + 1):
        even_path = even_dir / f"reference_round_{r:02d}.mrc"
        odd_path  = odd_dir  / f"reference_round_{r:02d}.mrc"

        if not even_path.exists() or not odd_path.exists():
            print(f"[Warning] Round {r}: files missing, skipping.")
            continue

        vol_e = load_mrc(even_path)
        vol_o = load_mrc(odd_path)
        if vol_e.shape != vol_o.shape:
            vol_e, vol_o = match_sizes(vol_e, vol_o)

        avg = (vol_e + vol_o) / 2.0
        shell_centers, fsc_vals, n_vox = compute_fsc(vol_e, vol_o)

        res143 = resolution_at_threshold(shell_centers, fsc_vals, 0.143, apix, n_vox)
        res05  = resolution_at_threshold(shell_centers, fsc_vals, 0.5,   apix, n_vox)
        print(f"Round {r:>2d}  |  FSC=0.143 → "
              f"{f'{res143:.1f} Å' if res143 else 'n/a':>8s}  |  "
              f"FSC=0.5 → {f'{res05:.1f} Å' if res05 else 'n/a':>8s}")

        all_shell_centers = shell_centers
        all_fsc_curves.append(fsc_vals)
        all_nvox.append(n_vox)
        all_res143.append(res143)

        rounds_data.append({
            "round": r,
            "slices": central_slices(avg),
            "fsc": fsc_vals,
            "n_vox": n_vox,
            "res143": res143,
            "res05":  res05,
            "avg": avg,
        })

    n_rows_actual = len(rounds_data)

    # ── save final overall average ───────────────────────────────────────────
    final = rounds_data[-1]
    save_mrc(out_dir / "overall_average_final.mrc", final["avg"])
    print(f"\nSaved final overall average → {out_dir / 'overall_average_final.mrc'}")

    # ── FSC x-axis (resolution in Å) ────────────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        res_axis = np.where(all_shell_centers > 0,
                            apix / all_shell_centers, np.inf)
    nyquist_A = 2.0 * apix

    # ── colour palette: one colour per round ────────────────────────────────
    cmap  = plt.cm.plasma
    colors = [cmap(i / max(n_rows_actual - 1, 1)) for i in range(n_rows_actual)]

    # ── build figure ─────────────────────────────────────────────────────────
    fig_h = 2.8 * n_rows_actual + 1.0
    fig, axes = plt.subplots(
        n_rows_actual, n_cols,
        figsize=(18, fig_h),
        gridspec_kw={"width_ratios": [1, 1, 1, 2.2]},
    )
    fig.patch.set_facecolor("#0d0d0d")

    # Ensure axes is always 2-D
    if n_rows_actual == 1:
        axes = axes[np.newaxis, :]

    plane_keys = ["XY", "XZ", "YZ"]
    plane_labels = ["XY (z=c)", "XZ (y=c)", "YZ (x=c)"]

    for row_idx, rd in enumerate(rounds_data):
        r      = rd["round"]
        color  = colors[row_idx]

        # ── volume slices (cols 0-2) ─────────────────────────────────────
        for col_idx, (pkey, plabel) in enumerate(zip(plane_keys, plane_labels)):
            ax = axes[row_idx, col_idx]
            ax.imshow(norm_display(rd["slices"][pkey]),
                      cmap="gray", origin="lower", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(color); sp.set_linewidth(1.5)

            # Column headers on first row
            if row_idx == 0:
                ax.set_title(plabel, color="#cccccc", fontsize=10, pad=4)

            # Row label on the left
            if col_idx == 0:
                res_str = (f"{rd['res143']:.1f} Å"
                           if rd["res143"] else "n/a")
                ax.set_ylabel(f"Round {r}\n(0.143→{res_str})",
                              color=color, fontsize=8, labelpad=4)

        # ── FSC panel (col 3): all rounds up to this one ─────────────────
        ax_fsc = axes[row_idx, 3]
        ax_fsc.set_facecolor("#111")

        # Plot all rounds so far (grey) then current round (bright)
        for prev_idx in range(row_idx):
            valid = (all_nvox[prev_idx] > 0) & (res_axis < np.inf)
            ax_fsc.plot(res_axis[valid], all_fsc_curves[prev_idx][valid],
                        color=colors[prev_idx], linewidth=1.0, alpha=0.35)

        # Current round — thick and bright
        valid = (rd["n_vox"] > 0) & (res_axis < np.inf)
        ax_fsc.plot(res_axis[valid], rd["fsc"][valid],
                    color=color, linewidth=2.2,
                    label=f"Round {r} — {rd['res143']:.1f} Å" if rd["res143"] else f"Round {r}")

        ax_fsc.axhline(0.5,   color="#888", ls="--", lw=0.8)
        ax_fsc.axhline(0.143, color="#ccc", ls=":",  lw=0.8)
        ax_fsc.set_xlim(nyquist_A, max(res_axis[res_axis < np.inf]) * 1.08)
        ax_fsc.invert_xaxis()
        ax_fsc.set_ylim(-0.1, 1.05)
        ax_fsc.tick_params(colors="#aaa", labelsize=7)
        ax_fsc.set_ylabel("FSC", color="#aaa", fontsize=8)
        for sp in ax_fsc.spines.values():
            sp.set_edgecolor("#333")

        if row_idx == 0:
            ax_fsc.set_title("FSC (even vs odd)", color="#cccccc", fontsize=10, pad=4)
        if row_idx == n_rows_actual - 1:
            ax_fsc.set_xlabel("Resolution (Å)", color="#aaa", fontsize=8)

        # Legend for current row only (shows just current round label)
        ax_fsc.legend(fontsize=7, loc="upper right",
                      facecolor="#1a1a1a", labelcolor=color, edgecolor="#333")

        ax_fsc.grid(True, alpha=0.15)

    # ── global legend: all rounds in one place (last FSC panel) ─────────────
    legend_lines = [
        Line2D([0], [0], color=colors[i], lw=2,
               label=(f"Round {rounds_data[i]['round']} → "
                      f"{rounds_data[i]['res143']:.1f} Å"
                      if rounds_data[i]['res143']
                      else f"Round {rounds_data[i]['round']}"))
        for i in range(n_rows_actual)
    ]
    legend_lines += [
        Line2D([0], [0], color="#888", ls="--", lw=1, label="FSC = 0.5"),
        Line2D([0], [0], color="#ccc", ls=":",  lw=1, label="FSC = 0.143"),
    ]
    axes[-1, 3].legend(handles=legend_lines, fontsize=7,
                       loc="upper right", facecolor="#1a1a1a",
                       labelcolor="white", edgecolor="#444")

    # ── title ────────────────────────────────────────────────────────────────
    best_res = min((rd["res143"] for rd in rounds_data if rd["res143"]),
                   default=None)
    title = (f"EMPOT Iterative Alignment — {n_rounds} rounds  |  "
             f"apix={apix} Å  |  "
             f"Best FSC=0.143 resolution: "
             f"{f'{best_res:.1f} Å' if best_res else 'n/a'}")
    fig.suptitle(title, color="white", fontsize=12, y=1.002)

    plt.tight_layout(pad=0.6, h_pad=0.4, w_pad=0.3)
    out_path = out_dir / "multi_round_evolution.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved comprehensive evolution figure → {out_path}")


if __name__ == "__main__":
    main()

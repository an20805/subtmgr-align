#!/usr/bin/env python3
"""Gold-standard half-map FSC: the honest score for an alignment run.

WHY THIS EXISTS. The optimiser's own objective (SFSC) is computed on the very
particles being aligned, so it rises when the optimiser fits noise. Measured:

  * 40-degree cluster 4: a greedy line search reached a HIGHER SFSC (24.6215)
    than a weaker configuration (24.4511) while recovering essentially no pose
    information (0.05 deg against 12.66 deg of a 36.27 deg spread).
  * Calibrating against the run whose pose recovery WAS confirmed, genuine
    alignment buys about 0.158 SFSC per degree. On the 10-degree set the first
    accepted step bought 0.588 SFSC per degree -- 3.7x the honest rate -- and
    recovered nothing. With only 9.21 deg of true spread there, noise fitting is
    simply a better deal than alignment, so SFSC actively misleads.

The fix is the standard one in the field: split the particles into two disjoint
halves, align each half using ONLY its own data, reconstruct a map from each,
and score the FSC BETWEEN the two maps. Noise in one half cannot correlate with
noise in the other, so overfitting cannot inflate it.

Because the SFSC objective is invariant to a global rotation, two independently
aligned halves can settle in different global frames. That is removed by a rigid
3-DOF search before computing FSC -- three parameters cannot manufacture shell
correlation, they can only undo the frame ambiguity.

The geometry and FSC helpers below are adapted from
analytical/halfmap_validation.py (2026-06-29, earlier work on this project);
the dataset-driven driver and the calibration mode are new.

  python3 halfmap_score.py --dataset 40deg --cluster 4 --group 0
  python3 halfmap_score.py --dataset 10deg --cluster 0 --group 0 --maxiter 60
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# Guarded: when imported inside a benchmark cell, stdout is the harness's log
# tee, which has no reconfigure (this crashed every Check 5 cell on 2026-09-25).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import dataset_specs


def soft_sphere(D, frac_in=0.40, frac_out=0.48):
    """Smooth spherical envelope, so the FSC is not dominated by box corners."""
    c = D // 2
    ax = np.arange(D) - c
    r = np.sqrt(ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2)
    m = np.clip((frac_out * D - r) / (frac_out * D - frac_in * D), 0, 1)
    return (0.5 - 0.5 * np.cos(np.pi * m)).astype(np.float64)


def masked_corr(A, B, m):
    a = (A * m).ravel(); b = (B * m).ravel()
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def fsc_curve(A, B, mask):
    """FSC per shell between two maps, plus the shells where it crosses 0.143 / 0.5."""
    D = A.shape[-1]
    FA = np.fft.fftshift(np.fft.fftn(A * mask))
    FB = np.fft.fftshift(np.fft.fftn(B * mask))
    c = D // 2
    ax = np.arange(D) - c
    rad = np.round(np.sqrt(ax[:, None, None] ** 2 + ax[None, :, None] ** 2
                           + ax[None, None, :] ** 2)).astype(int)
    nsh = D // 2 + 1
    fsc = np.zeros(nsh)
    for s in range(nsh):
        sel = rad == s
        num = np.real(np.sum(FA[sel] * np.conj(FB[sel])))
        den = np.sqrt(np.sum(np.abs(FA[sel]) ** 2) * np.sum(np.abs(FB[sel]) ** 2))
        fsc[s] = num / den if den > 0 else 0.0

    def res_at(thr):
        for s in range(1, nsh):
            if fsc[s] < thr:
                return s - 1
        return nsh - 1

    return fsc, res_at(0.143), res_at(0.5)


def reconstruct(P, M, R):
    from rotation_utils_lie import rotate_particles_parallel_lie
    z = np.zeros((len(R), 3))
    pr, _ = rotate_particles_parallel_lie(P, M, R, loc_r=z, njobs=25,
                                          default_val=float('nan'))
    return np.nan_to_num(np.nanmean(np.asarray(pr), axis=0), nan=0.0).astype(np.float64)


def global_align(volA, volB, mask, n_coarse=300):
    """Rotate volB to best match volA. Only 3 DOF, so it removes the frame
    ambiguity between two independently aligned halves without being able to
    create shell correlation."""
    import cupy as cp
    from lie_algebra_utils import so3_exp_map
    from rotation_utils_lie import rotate_particles_parallel_lie

    D = volA.shape[-1]
    rng = np.random.default_rng(0)
    cands = [np.zeros(3)]
    for ang in (5, 10, 15, 20, 30, 45):
        for _ in range(20):
            ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
            cands.append(np.deg2rad(ang) * ax)
    for _ in range(n_coarse):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        cands.append(rng.uniform(0, np.pi) * ax)
    Rs = cp.asnumpy(so3_exp_map(cp.asarray(np.array(cands))))

    best_c, best_R = -2.0, np.eye(3)
    CH = 60
    for s in range(0, len(Rs), CH):
        chunk = Rs[s:s + CH]
        vb = np.tile(volB[None], (len(chunk), 1, 1, 1))
        mm = np.ones((len(chunk), D, D, D))
        rot, _ = rotate_particles_parallel_lie(vb, mm, chunk,
                                               loc_r=np.zeros((len(chunk), 3)),
                                               njobs=25, default_val=0)
        for i, rv in enumerate(np.asarray(rot)):
            c = masked_corr(volA, rv, mask)
            if c > best_c:
                best_c, best_R = c, chunk[i]

    for scale in (8, 3, 1):
        improved = True
        while improved:
            improved = False
            base = cp.asnumpy(so3_exp_map(cp.asarray(np.array(
                [np.deg2rad(scale) * np.eye(3)[k] * sgn
                 for k in range(3) for sgn in (1, -1)]))))
            trial = np.array([best_R @ b for b in base])
            vb = np.tile(volB[None], (len(trial), 1, 1, 1))
            mm = np.ones((len(trial), D, D, D))
            rot, _ = rotate_particles_parallel_lie(vb, mm, trial,
                                                   loc_r=np.zeros((len(trial), 3)),
                                                   njobs=25, default_val=0)
            for i, rv in enumerate(np.asarray(rot)):
                c = masked_corr(volA, rv, mask)
                if c > best_c + 1e-5:
                    best_c, best_R, improved = c, trial[i], True

    rot, _ = rotate_particles_parallel_lie(volB[None], np.ones((1, D, D, D)),
                                           best_R[None], loc_r=np.zeros((1, 3)),
                                           njobs=1, default_val=0)
    return np.asarray(rot)[0], best_c, best_R


def align_half(P, M, mask_path, spec, args, tag, outdir):
    """Align one half using ONLY that half's particles."""
    from Riemannian_Adam_SAG import RIEMANNIAN_ADAM_SAG
    d = os.path.join(outdir, f"_half_{tag}")
    os.makedirs(d, exist_ok=True)
    R, loc, val, mx, _ = RIEMANNIAN_ADAM_SAG(
        particlesPath=None, maskPath=mask_path, outputfolder=d,
        angle_step=args.angle_step or spec.default_angle_step,
        loc_step=0.75, saturationIter=20, B=args.B or spec.default_B,
        alphaRot=args.alpha_rot, alphaLoc=args.alpha_rot,
        maxIter=args.maxiter, njobsSFSC=25, njobsDerivative=25,
        cluster_label=0, originaldataconfig=None, particleCount=len(P),
        gpu_batch_size=spec.default_gpu_batch, rho1=0.91, rho2=0.95,
        type="rotation", alphaBackoff=0.5, alphaCutoff=0.0005,
        particleList=list(P), maskList=(list(M) if spec.mask_mode == "per_particle" else None),
    )
    R = R.get() if hasattr(R, "get") else np.asarray(R)
    print(f"   half {tag}: within-half SFSC {val[0]:.4f} -> {max(mx):.4f}")
    return R, max(mx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=sorted(dataset_specs.SPECS))
    p.add_argument("--cluster", type=int, required=True)
    p.add_argument("--group", type=int, default=0)
    p.add_argument("--particles", type=int, default=100,
                   help="total across BOTH halves; each half gets half of these")
    p.add_argument("--maxiter", type=int, default=100)
    p.add_argument("--B", type=int, default=None)
    p.add_argument("--angle-step", type=float, default=None)
    p.add_argument("--alpha-rot", type=float, default=0.87)
    p.add_argument("--skip-alignment", action="store_true",
                   help="score the UNALIGNED particles, as a floor to compare against")
    p.add_argument("--seed", type=int, default=0,
                   help="each half is seeded separately (seed, seed+1). Outcomes are "
                        "highly seed-dependent -- see the variance note in docs/METRICS.md "
                        "-- so a single pair is not evidence.")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    spec = dataset_specs.get(args.dataset)
    group = args.group if spec.groups else None
    outdir = args.outdir or f"/home/raghav/Desktop/RP/parth/results/halfmap/{spec.name}_c{args.cluster}_g{group}"
    os.makedirs(outdir, exist_ok=True)

    from read_write_mrc import READ_mrc
    particles, masks, ids = spec.load(args.cluster, group, args.particles)
    P = np.array(particles)

    if spec.mask_mode == "shared":
        mask_path = spec.mask_path
        M = np.array([READ_mrc(mask_path)] * len(P))
    else:
        mask_path = None
        M = np.array(masks)

    # Disjoint halves. Even/odd rather than first/second so both halves span the
    # same range of the file ordering.
    ia, ib = np.arange(0, len(P), 2), np.arange(1, len(P), 2)
    PA, MA = P[ia], M[ia]
    PB, MB = P[ib], M[ib]
    print(f"{spec.describe()}")
    print(f"half A: {len(PA)} particles | half B: {len(PB)} particles | box {P.shape[-1]}^3")
    print("The halves are disjoint and are aligned independently -- neither sees the other.")

    t0 = time.time()
    if args.skip_alignment:
        R_A = np.tile(np.eye(3), (len(PA), 1, 1)); R_B = np.tile(np.eye(3), (len(PB), 1, 1))
        sA = sB = None
        print("\n[--skip-alignment] scoring the UNALIGNED particles as a floor")
    else:
        # Seed each half separately and record it. Riemannian_Adam_SAG.py seeds
        # nothing, and the shell draw drives the outcome: on 40-degree cluster 4,
        # two 50-particle halves under identical settings gave 22 accepted steps
        # and 17.9% pose recovery, versus 3 steps and 0.8%.
        print("\naligning half A ...")
        np.random.seed(args.seed)
        R_A, sA = align_half(PA, MA, mask_path, spec, args, "A", outdir)
        print("aligning half B ...")
        np.random.seed(args.seed + 1)
        R_B, sB = align_half(PB, MB, mask_path, spec, args, "B", outdir)

    volA = reconstruct(PA, MA, R_A)
    volB = reconstruct(PB, MB, R_B)
    sphere = soft_sphere(P.shape[-1])

    fsc_pre, r143_pre, r05_pre = fsc_curve(volA, volB, sphere)
    corr_pre = masked_corr(volA, volB, sphere)
    print("\nglobally aligning half-map B onto half-map A (3 DOF) ...")
    volB_al, corr, Rg = global_align(volA, volB, sphere)
    fsc_post, r143_post, r05_post = fsc_curve(volA, volB_al, sphere)
    g_ang = float(np.degrees(np.arccos(np.clip((np.trace(Rg) - 1) / 2, -1, 1))))

    rec = dict(dataset=spec.name, cluster=args.cluster, group=group,
               n_half=len(PA), box=int(P.shape[-1]), aligned=not args.skip_alignment,
               within_half_sfsc_A=sA, within_half_sfsc_B=sB,
               masked_corr_pre=corr_pre, masked_corr_post=corr,
               global_align_angle_deg=g_ang,
               res143_pre=r143_pre, res143_post=r143_post,
               res05_pre=r05_pre, res05_post=r05_post,
               fsc_post=fsc_post.tolist(), seed=args.seed,
               wall_seconds=time.time() - t0)
    # Distinct names: the unaligned floor and the aligned run are compared against
    # each other, so they must not overwrite one another.
    fname = "halfmap_floor.json" if args.skip_alignment else "halfmap_aligned.json"
    with open(os.path.join(outdir, fname), "w") as f:
        json.dump(rec, f, indent=2)

    print()
    print("=" * 70)
    print(f"GOLD-STANDARD HALF-MAP FSC :: {spec.name} cluster {args.cluster} group {group}")
    print("=" * 70)
    if sA is not None:
        print(f"  within-half SFSC (the leaky score) : A {sA:.4f}   B {sB:.4f}")
    print(f"  masked correlation of the two maps : {corr_pre:.4f} -> {corr:.4f} after global align")
    print(f"  global frame offset removed        : {g_ang:.1f} deg")
    print(f"  FSC = 0.143 crossing shell         : {r143_pre} -> {r143_post}")
    print(f"  FSC = 0.5   crossing shell         : {r05_pre} -> {r05_post}")
    print()
    print("  Higher crossing shell = better resolution. This number cannot be")
    print("  inflated by overfitting, because the two halves never saw each other.")
    print(f"  saved {os.path.join(outdir, fname)}")


if __name__ == "__main__":
    main()

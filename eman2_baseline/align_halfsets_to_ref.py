#!/usr/bin/env python
"""
align_halfsets_to_ref.py — Align even and odd half-set averages to the
full reference so they share the same coordinate frame before FSC.

Usage (from repo root):
    ~/miniforge3/envs/eman2/bin/python eman2_baseline/align_halfsets_to_ref.py

Reads:
    eman2_baseline/baseline_avg_even.mrc
    eman2_baseline/baseline_avg_odd.mrc
    eman2_baseline/baseline_reference_91.mrc   (full baseline, all 200 particles)

Writes:
    eman2_baseline/baseline_avg_even_aligned.mrc
    eman2_baseline/baseline_avg_odd_aligned.mrc
"""
import sys
from EMAN2 import EMData

EVEN_MRC = "eman2_baseline/baseline_avg_even.mrc"
ODD_MRC  = "eman2_baseline/baseline_avg_odd.mrc"
REF_MRC  = "eman2_baseline/baseline_reference_91.mrc"   # full 200-particle average

OUT_EVEN = "eman2_baseline/baseline_avg_even_aligned.mrc"
OUT_ODD  = "eman2_baseline/baseline_avg_odd_aligned.mrc"
LP = 0.15   # same lowpass used during alignment

def prep(vol):
    """Lowpass + normalize — same preprocessing as during alignment."""
    v = vol.copy()
    v.process_inplace("normalize")
    v.process_inplace("filter.lowpass.gauss", {"cutoff_abs": LP})
    return v

def align_to_reference(vol_path, ref_prepped, out_path, label):
    print(f"  Aligning {label} to full reference...", flush=True)
    vol = EMData(vol_path)
    vol_prepped = prep(vol)

    aln = vol_prepped.align(
        "rotate_translate_3d_tree", ref_prepped,
        {}, "ccc.tomo.thresh", {}
    )
    xf   = aln["xform.align3d"]
    score = float(aln["score"])
    print(f"  {label}: alignment score = {score:.5f}", flush=True)
    print(f"  {label}: transform = {xf.get_params('eman')}", flush=True)

    # Apply transform to the FULL-RESOLUTION (not lowpassed) volume
    vol.transform(xf)
    vol.write_image(out_path, 0)
    print(f"  Saved: {out_path}", flush=True)
    return score

def main():
    print("[1/3] Loading full reference...", flush=True)
    ref = EMData(REF_MRC)
    ref_prepped = prep(ref)

    print("[2/3] Aligning even half-set average...", flush=True)
    align_to_reference(EVEN_MRC, ref_prepped, OUT_EVEN, "even")

    print("[3/3] Aligning odd half-set average...", flush=True)
    align_to_reference(ODD_MRC, ref_prepped, OUT_ODD, "odd")

    print("\n[done] Aligned half-set averages written:")
    print(f"  {OUT_EVEN}")
    print(f"  {OUT_ODD}")
    print("\nNow compute FSC with:")
    print(f"  EMPOT/empot/bin/python3 scripts/compute_fsc.py compare \\")
    print(f"    --vol1 {OUT_EVEN} --vol2 {OUT_ODD} \\")
    print(f"    --label1 'EMAN2 even (aligned to ref)' \\")
    print(f"    --label2 'EMAN2 odd (aligned to ref)' \\")
    print(f"    --output experiments/eman2_halfset_fsc_aligned --apix 4.0")

if __name__ == "__main__":
    main()

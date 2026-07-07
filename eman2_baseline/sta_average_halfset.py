#!/usr/bin/env python
"""
sta_average_halfset.py — Run EMAN2 iterative STA on an even or odd half-set.

Identical science to sta_average.py (same aligner, comparator, averager).
Adds a 'selection' argument to restrict which particle indices are processed:
  even → indices 0, 2, 4, ..., 198   (100 particles)
  odd  → indices 1, 3, 5, ..., 199   (100 particles)

Usage (from repo root):
    ~/miniforge3/envs/eman2/bin/python eman2_baseline/sta_average_halfset.py even [niter] [shrink] [nproc]
    ~/miniforge3/envs/eman2/bin/python eman2_baseline/sta_average_halfset.py odd  [niter] [shrink] [nproc]
"""
import sys, time
import multiprocessing as mp
from EMAN2 import EMData, EMUtil, Averagers, Transform

STACK   = "eman2_baseline/particles_e.hdf"   # 200 x 96^3, apix=1
INITREF = "eman2_baseline/init_ref_e.hdf"    # lowpassed raw average (seed)
OUTDIR  = "eman2_baseline"
LP      = 0.15                               # gaussian lowpass cutoff_abs

_SHRINK = None
_REFP   = None
_REFTMP = None

def _init(shrink, reftmp):
    global _SHRINK, _REFP, _REFTMP
    _SHRINK = shrink
    _REFTMP = reftmp
    _REFP = EMData(reftmp, 0)

def prep_inplace(x, shrink):
    x.process_inplace("normalize")
    x.process_inplace("filter.lowpass.gauss", {"cutoff_abs": LP})
    if shrink > 1:
        x.process_inplace("math.meanshrink", {"n": shrink})

def align_one(i):
    p = EMData(STACK, i)
    prep_inplace(p, _SHRINK)
    aln = p.align("rotate_translate_3d_tree", _REFP, {}, "ccc.tomo.thresh", {})
    xf = aln["xform.align3d"]
    return (i, xf.get_params("eman"), float(aln["score"]))

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("even", "odd"):
        print("Usage: sta_average_halfset.py <even|odd> [niter] [shrink] [nproc]")
        sys.exit(1)

    selection = sys.argv[1]          # "even" or "odd"
    niter     = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    shrink    = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    nproc     = int(sys.argv[4]) if len(sys.argv) > 4 else 14

    # Each half gets its own temp ref file to avoid conflicts
    reftmp = f"{OUTDIR}/_ref_prep_{selection}.hdf"

    # Build the filtered index list
    n_total = EMUtil.get_image_count(STACK)
    if selection == "even":
        indices = [i for i in range(n_total) if i % 2 == 0]
    else:
        indices = [i for i in range(n_total) if i % 2 == 1]

    print(f"[cfg] selection={selection} nptcl={len(indices)} niter={niter} "
          f"shrink={shrink} nproc={nproc} lowpass_abs={LP}", flush=True)

    ref = EMData(INITREF, 0)

    for it in range(niter):
        t0 = time.time()
        refp = ref.copy()
        prep_inplace(refp, shrink)
        refp.write_image(reftmp, 0)

        with mp.Pool(nproc, initializer=_init, initargs=(shrink, reftmp)) as pool:
            results = pool.map(align_one, indices, chunksize=2)

        avg = Averagers.get("mean.tomo")   # wedge-weighted Fourier averager
        scores = []
        for (i, params, sc) in results:
            scores.append(sc)
            params = dict(params)
            params["tx"] *= shrink
            params["ty"] *= shrink
            params["tz"] *= shrink
            xf = Transform(params)
            pr = EMData(STACK, i)          # RAW full-res particle
            pr.transform(xf)
            avg.add_image(pr)

        ref = avg.finish()
        ref.process_inplace("normalize")
        ref.write_image(f"{OUTDIR}/baseline_avg_{selection}_iter{it:02d}.hdf", 0)
        print(f"[iter {it}] mean score={sum(scores)/len(scores):.5f} "
              f"time={time.time()-t0:.0f}s", flush=True)

    # Save final average as both HDF and MRC
    ref.write_image(f"{OUTDIR}/baseline_avg_{selection}.hdf", 0)
    ref.write_image(f"{OUTDIR}/baseline_avg_{selection}.mrc", 0)
    print(f"[done] wrote {OUTDIR}/baseline_avg_{selection}.{{hdf,mrc}}", flush=True)


if __name__ == "__main__":
    main()

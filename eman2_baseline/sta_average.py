#!/usr/bin/env python
"""
Reference-free-seeded subtomogram alignment + averaging using EMAN2's own SPT
library primitives (missing-wedge-aware tree aligner + tomo comparator +
wedge-weighted averager). This is the same scientific core e2spt_classaverage
uses, driven in a controlled, PARALLEL loop (the high-level CLI wrappers in this
EMAN2 build have a checkinput() bug, and the per-particle tree alignment is the
bottleneck -> fan out across CPU cores).

Each worker reads its particle by index from the stack, aligns to the current
(preprocessed+shrunk) reference, and returns only the serializable transform +
score. The main process applies the (translation-rescaled) transform to the RAW
full-resolution particle and accumulates a wedge-weighted average.

Usage:
    python sta_average.py [niter] [nptcl_limit] [shrink] [nproc]
"""
import sys, time
import multiprocessing as mp
from EMAN2 import EMData, EMUtil, Averagers, Transform

STACK   = "eman2_baseline/particles_e.hdf"     # 200 x 96^3, apix=1
INITREF = "eman2_baseline/init_ref_e.hdf"      # lowpassed raw average (seed)
OUTDIR  = "eman2_baseline"
REFTMP  = "eman2_baseline/_ref_prep.hdf"       # preprocessed+shrunk ref for workers
LP      = 0.15                                  # gaussian lowpass cutoff_abs

_SHRINK = None
_REFP   = None

def _init(shrink):
    global _SHRINK, _REFP
    _SHRINK = shrink
    _REFP = EMData(REFTMP, 0)

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
    niter  = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    limit  = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    shrink = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    nproc  = int(sys.argv[4]) if len(sys.argv) > 4 else 14

    n = EMUtil.get_image_count(STACK)
    if limit > 0:
        n = min(n, limit)
    print(f"[cfg] niter={niter} nptcl={n} shrink={shrink} nproc={nproc} lowpass_abs={LP}", flush=True)

    ref = EMData(INITREF, 0)
    for it in range(niter):
        t0 = time.time()
        refp = ref.copy()
        prep_inplace(refp, shrink)
        refp.write_image(REFTMP, 0)

        with mp.Pool(nproc, initializer=_init, initargs=(shrink,)) as pool:
            results = pool.map(align_one, range(n), chunksize=2)

        avg = Averagers.get("mean.tomo")   # wedge-weighted Fourier averager
        scores = []
        for (i, params, sc) in results:
            scores.append(sc)
            params = dict(params)
            params["tx"] *= shrink         # translation measured in shrunk voxels
            params["ty"] *= shrink
            params["tz"] *= shrink
            xf = Transform(params)
            pr = EMData(STACK, i)          # RAW full-res particle
            pr.transform(xf)
            avg.add_image(pr)
        ref = avg.finish()
        ref.process_inplace("normalize")
        ref.write_image(f"{OUTDIR}/baseline_avg_iter{it:02d}.hdf", 0)
        print(f"[iter {it}] mean score={sum(scores)/len(scores):.5f} "
              f"time={time.time()-t0:.0f}s", flush=True)

    ref.write_image(f"{OUTDIR}/baseline_avg.hdf", 0)
    ref.write_image(f"{OUTDIR}/baseline_avg.mrc", 0)
    print("[done] wrote eman2_baseline/baseline_avg.{hdf,mrc}", flush=True)

if __name__ == "__main__":
    main()

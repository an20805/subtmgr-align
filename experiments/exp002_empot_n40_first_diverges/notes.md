## exp002 — EMPOT N=40, First, Post-Fix (Diverges)

**Date:** 2026-06-23  
**Status:** ❌ FAILED (method limitation, not a bug)

### What I ran
Same as exp001 but with the `do_log` bug fixed. TRN now returns real point clouds.
40 subtomograms, SNR 0.1, Cluster_0, 5 rounds.

### What I found
- `ugw_dist` values now real and large (~5000–8000 per subtomogram per round).
- `reference_l2_change` shows the average IS changing between rounds.
- **Visually: average gets blurrier each round** — the opposite of what we want.

### Root cause analysis
Three compounding factors (documented in PROJECT_CONTEXT.md):

1. **Missing wedge (dominant):** The wedge is fixed in the lab frame. Different 
   subtomograms have the wedge smear in different orientations relative to the 
   protein. TRN point clouds are dominated by wedge geometry, so EMPOT/GW aligns 
   wedge artifacts, not protein structure.

2. **SNR 0.1 → TRN samples noise:** At this SNR, the highest-density voxels after 
   thresholding are mostly noise peaks. Point clouds from two different subtomograms 
   have uncorrelated geometry. OT forces a matching on random structure → effectively 
   random rotations.

3. **Random rotations + real-space averaging = blur:** Applying a random rotation 
   to each subtomogram then averaging them together washes out all structure. Each 
   round builds on this blurrier blob → feedback divergence.

### Why the benchmark gave zero error
The benchmark rotated one subtomogram's point cloud mathematically — both point 
clouds were identical (same noise, same wedge, just rotated). OT trivially matches 
them. This validates Kabsch/SVD math only, says nothing about cross-particle alignment.

### Conclusion
EMPOT (and all real-space point-cloud OT) cannot align subtomograms at SNR 0.1 
with a missing wedge. The method is correct; the problem is unsolvable with this approach.

### Next step
- Try EMAN2-seeded reference as starting point (can OT refine a good reference?)
- Try Fourier-domain or bandlimited alignment
- Get simulation metadata (tilt range) to properly handle missing wedge

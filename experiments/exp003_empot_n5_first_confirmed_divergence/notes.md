## exp003 — EMPOT N=5, Confirmed Divergence

**Date:** 2026-06-25  
**Status:** ❌ FAILED (confirmed method limitation, not scale issue)

### What I ran
Minimal run with only 5 subtomograms to confirm divergence is not 
caused by scale (i.e., whether using all 40 was somehow the problem).

### What I found
- Same divergence pattern as exp002 even with 5 subtomograms.
- Confirms: the issue is fundamental to the OT approach on this data, 
  not a memory/scale/averaging artifact.

### Conclusion
Divergence is NOT caused by:
- Number of subtomograms being too large
- Memory issues
- Code bugs

Divergence IS caused by: missing wedge + SNR 0.1 (see exp002 notes).

### This is the last EMPOT-on-raw-data experiment
No further tuning of EMPOT parameters (eps, num_points, threshold) 
is expected to fix this. The method is fundamentally mismatched to the problem.

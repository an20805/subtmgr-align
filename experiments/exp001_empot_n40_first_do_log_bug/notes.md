## exp001 — EMPOT N=40, First, do_log Bug

**Date:** 2026-06-23  
**Status:** ❌ FAILED (code bug)

### What I ran
First iterative EMPOT run after refactoring notebook to Python script.
40 subtomograms, SNR 0.1, Cluster_0, 5 rounds.

### What I found
- `ugw_dist = 0.0` for every subtomogram every round.
- `reference_l2_change = 0.0` — average never changed at all.

### Root cause
**Bug in `iterative_utils.py`:** When moving code from notebook to script, 
`do_log=False` was passed to `trn.trn_iterate()`. The original `trn.py` code 
only saves points into `rms` when `do_log=True`, so with `do_log=False` the 
function returned an array of all zeros.

TRN returned `[[0,0,0], [0,0,0], ...]` for every subtomogram.
EMPOT received identical point clouds → zero distance, identity rotation → average unchanged.

### Fix applied
Changed `do_log=False` → `do_log=True` in `sample_volume_in_memory()`.
stdout spam suppressed with `contextlib.redirect_stdout(io.StringIO())`.

### Lesson
Always verify that TRN returns non-zero points before trusting any alignment result.
Add a sanity check assertion to `sample_volume_in_memory`.

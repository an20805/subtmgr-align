# EMAN2 Baseline Execution Log

## Preflight & Setup Log (Phase 1 & 2)

- **Date & Time**: 2026-09-21 17:30 IST
- **Platform**: Linux-6.18.33.2-microsoft-standard-WSL2-x86_64
- **Host CPU**: 13th Gen Intel(R) Core(TM) i7-13650HX (10 physical cores, 20 logical threads)
- **Host Memory**: 7.6 GB RAM allocated in WSL2
- **Conda Installation**: Miniforge3 (`/home/anshu/miniforge3`)
- **Dataset**: `Data/0.1/Cluster_0` (200 subtomograms, box size $91 \times 91 \times 91$, pixel size $4.0\text{ \AA}$)
- **Contrast**: Dark particles in raw cryo-ET $\to$ inverted (`mult(-1.0)`) for positive density alignment
- **Reference**: Reference-free (initial seeds calculated as unaligned raw streaming averages per half-set)

---

## Phase 2 & 3: Environment Installation & Verification

- **Environment**: Clean conda environment `eman2` created via Miniforge `mamba` (`eman-dev-2.99.85`).
- **Python Version**: Python 3.12.14
- **EMAN2 Version**: `EMAN 2.99.85`
- **Speed Test**: `e2speedtest.py` completed with factor $2.2740 \pm 0.0167$ ($0.0101 \pm 0.00008\text{ s}$).
- **Environment Export**: Saved to `eman2_baseline/eman2_env.yml`.
- **GUI Bugfix Applied**: Patched `empmtabwidgets.py` line 236 (`safe_int(getattr(x, 'nimg', ...))`) to ensure `e2projectmanager.py` and `e2display.py` file browsers run without `AttributeError`.

---

## Phase 4, 6 & 7: Implementation & Smoke Test

- **Script**: `eman2_baseline/run_iterative_alignment_eman2.py`
- **Runner**: `scripts/run_eman2_baseline.sh`
- **Aligner Engine**:
  - Aligner: `rotate_translate_3d_tree` (missing-wedge aware 3D tree search)
  - Comparator: `ccc.tomo.thresh`
  - Averager: `mean.tomo` (wedge-weighted Fourier average)
- **Smoke Test Verification**:
  - 4 particles, 1 round, 2 workers.
  - Round 00 raw unaligned seed: completed in $0.7\text{ s}$.
  - Round 01 Even half: score $-0.85418$, duration $48.4\text{ s}$.
  - Round 01 Odd half: score $-0.77563$, duration $48.1\text{ s}$.
  - Gold-standard FSC: $\text{FSC}@0.143 = 35.06\text{ \AA}$.
  - Output files generated cleanly: `benchmark_summary.json`, `timing_report.txt`, `fsc_all_rounds_comparison.png`, `slices_all_rounds_comparison.png`, `halfsets.json`, per-round `.mrc` volumes, and `.csv`/`.json` transforms.

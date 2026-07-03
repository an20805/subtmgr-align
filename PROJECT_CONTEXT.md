# Subtomogram Alignment Project Context

## What This Project Is

This repository is a working research/code sandbox for **subtomogram alignment and averaging** using ideas from three different optimal-transport (OT) 3D alignment papers:

- **AlignOT**: optimal-transport-based rotational alignment on sampled point clouds from 3D density maps
- **BOTalign**: Bayesian optimization / Wasserstein-EMD-style alignment on full 3D volumes
- **EMPOT**: Unbalanced Gromov-Wasserstein (UGW) OT on sampled point clouds, recovering **both rotation and translation** via Kabsch/SVD (designed for *partial* alignment of cryo-EM maps)

The project goal is to understand, adapt, and compare these methods for **subtomogram alignment** on noisy cryo-ET-like 3D volumes, especially for **iterative subtomogram averaging**.

This document is intended as an LLM handoff file for coding assistance. It focuses on:

- the paper-level intent
- the current repository structure
- how the local implementations work
- what we are trying to do with the code
- the important constraints and conventions already established here

> **Note (updated):** This document now *also* records the key experimental findings,
> conclusions, and decisions from the investigation into why the OT methods fail on
> subtomograms, plus the EMAN2 baseline-reference work. See the sections
> **"Investigation: Why the OT Methods Fail on Subtomograms"**, **"EMAN2 Baseline
> Reference"**, and **"Conclusions and Next Steps"** near the end.

---

## High-Level Goal

We want to use alignment algorithms designed for 3D density maps to align a collection of noisy subtomograms and then average them into an improved reference.

The core workflow is:

1. Start with a set of 3D subtomograms.
2. Build an initial reference, usually by simple averaging.
3. Align each subtomogram to the current reference.
4. Rotate the original full subtomogram using the recovered transform.
5. Average the rotated subtomograms to form a new reference.
6. Repeat for multiple rounds.

The current repo contains two main approaches to this problem:

- a **BOTalign-based iterative alignment pipeline** that operates on full volumes
- an **AlignOT-based iterative notebook** that uses sampled point clouds for alignment and ASPiRE for rotating full volumes before averaging

---

## The Two Papers and Their Roles

## 1. AlignOT

Paper intent:

- AlignOT is a method for **rigid rotational alignment of 3D density maps**
- It does this by converting volumes into **sampled point clouds**
- It then uses **optimal transport** between point clouds as the alignment objective
- Rotation is represented with **quaternions**
- The rotation is optimized with a custom **stochastic gradient descent** loop

What matters for this project:

- AlignOT is used here as a **rotation estimator**
- It does **not** directly rotate full subtomogram voxel grids for averaging
- In this repo, AlignOT is being adapted from cryo-EM density maps to **subtomogram data**
- It is sensitive to how the volume is normalized, thresholded, and sampled into pseudo-atoms

Local repo role:

- `alignOT/` contains the upstream-style AlignOT code and notebooks
- the current subtomogram adaptation lives mainly in `alignOT/notebooks/`

---

## 2. BOTalign

Paper intent:

- BOTalign aligns two 3D volumes directly
- It uses **Bayesian optimization** over rotation space
- The objective can use either:
  - a wavelet/Wasserstein-style embedding loss (`wemd`)
  - a simple Euclidean loss (`eu`)
- It operates on **full 3D volumes**, not point clouds

What matters for this project:

- BOTalign is more naturally suited to **volume-to-volume alignment**
- It already has a local iterative averaging runner in this repo
- It provides a useful baseline and a cleaner full-volume pipeline

Local repo role:

- `BOTalign/` contains the core implementation and the current iterative alignment runner
- this side of the repo is the more complete existing subtomogram-averaging workflow

---

## Repository Layout

Top-level relevant directories:

- `alignOT/`
- `BOTalign/`
- `EMPOT/`  
  Third OT method (UGW + Kabsch). Has its own bundled venv at `EMPOT/empot/`, an
  iterative runner, a benchmark notebook, and outputs. Now an active part of the
  subtomogram workflow. See "How EMPOT Works in This Repo" below.
- `eman2_baseline/`  
  EMAN2-based **baseline reference average** produced as an independent, non-OT
  control (see "EMAN2 Baseline Reference"). Holds `baseline_reference.mrc/.hdf`,
  `baseline_reference_91.mrc`, per-iteration averages, slice montage, and
  `sta_average.py` / `inspect_baseline.py`.
- `Data/0.1/Cluster_0/`  
  The canonical 200-subtomogram dataset (91³ volumes) used by the EMPOT runner and
  EMAN2 baseline. (Mirrored under `alignOT/notebooks/Data/...` and `BOTalign/data/...`.)
- `.venv/` and per-method venvs  
  Python environments used by the notebooks/scripts. EMAN2 lives in a separate
  conda env at `~/miniforge3/envs/eman2` (not in the repo).

### `alignOT/`

Important files:

- `alignOT/README.md`
- `alignOT/alignOT/main.py`
- `alignOT/alignOT/utils.py`
- `alignOT/alignOT/trn.py`
- `alignOT/alignOT/coords.py`
- `alignOT/notebooks/AlignOT-iterative-sbtm.ipynb`
- `alignOT/notebooks/utils.py`
- `alignOT/notebooks/trn.py`
- `alignOT/notebooks/coords.py`

Important data/output folders:

- `alignOT/notebooks/Data/0.1/Cluster_0`
- `alignOT/notebooks/iterative_alignment_outputs_cluster0`

### `BOTalign/`

Important files:

- `BOTalign/README.md`
- `BOTalign/utils_BO.py`
- `BOTalign/wemd.py`
- `BOTalign/run_botalign_iterative_alignment.py`
- `BOTalign/notebooks/botalign_iterative_alignment.ipynb`

Important data/output folders:

- `BOTalign/data/0.1/Cluster_0`
- `BOTalign/notebooks/cluster0_iterative_alignment_outputs`
- `BOTalign/notebooks/cluster0_iterative_alignment_outputs_40`

---

## Data Context

The project works with a set of **200 subtomograms** at **0.1 SNR** stored as `.mrc` volumes.

Current working dataset locations:

- `alignOT/notebooks/Data/0.1/Cluster_0`
- `BOTalign/data/0.1/Cluster_0`

The code assumes:

- each subtomogram is a 3D array
- filenames are numeric like `0.mrc`, `1.mrc`, ...
- only `.mrc` files should be processed
- Windows metadata files like `Zone.Identifier` must be ignored

The active use case is **single-cluster subtomogram averaging** for `Cluster_0`.

---

## How AlignOT Works in This Repo

The AlignOT implementation has two conceptual layers:

### A. Point-cloud alignment logic

This lives mainly in `alignOT/alignOT/utils.py` and the notebook-local copy `alignOT/notebooks/utils.py`.

Important ideas:

- volumes are converted into **sampled pseudo-atoms**
- rotations are parameterized by **quaternions**
- the matching cost is based on **optimal transport**
- the transport plan is estimated with a custom **Sinkhorn** routine
- the rotation is refined by a custom **SGD** loop

Important helper functions:

- `sample(...)`
- `perform(...)`
- `SGD(...)`
- `diff_quaternions(...)`
- `OT(...)`
- `my_sinkhorn(...)`

### B. TRN-based pseudo-atom sampling

This comes from `trn.py`.

Purpose:

- convert a thresholded 3D map into a compact point cloud
- concentrate points in informative/high-weight regions

Important functions:

- `trn_rm0(...)`
- `trn_iterate(...)`
- `trn_wrapper(...)`

### Important practical note for subtomograms

AlignOT assumes that the input volume is transformed into a sampling map where **high values correspond to important structure**.

For the subtomogram workflow in this repo, we explicitly adapted the sampling side to support **dark-is-signal** behavior:

1. normalize volume to `[0, 1]`
2. invert it with `1 - normalized_volume`
3. threshold the inverted map
4. sample pseudo-atoms from that transformed map

This adaptation is important because in some subtomogram settings, darker regions may correspond to the actual structure of interest.

---

## How BOTalign Works in This Repo

BOTalign is implemented mainly in `BOTalign/utils_BO.py`.

Core idea:

- operate directly on full 3D volumes
- evaluate candidate rotations using a loss
- explore rotation space with Bayesian optimization / manifold optimization

Important function:

- `align_BO(vol0, vol_given, para, reflect=False)`

The parameter bundle is currently used as:

- loss type: `wemd` or `eu`
- downsample level
- number of optimization iterations
- whether to refine

Other helpful functions in `utils_BO.py`:

- `q_to_rot(...)`
- `u_to_rot(...)`
- `rot_to_u(...)`
- `center(...)`

BOTalign uses ASPiRE volume rotation internally and is currently the cleaner full-volume alignment baseline in the repo.

---

## How EMPOT Works in This Repo

EMPOT (paper: arXiv 2311.00850) was designed for **partial alignment of cryo-EM
density maps**. Source lives in `EMPOT/src/` (`empot.py`, `iterative_utils.py`,
`run_iterative_alignment.py`, `trn.py`, `coords.py`, `gauss_forward_model.py`).

Core pipeline:

1. **Map → point cloud** via the same TRN pseudo-atom sampling used by AlignOT
   (`trn.trn_rm0` + `trn.trn_iterate`).
2. **Match the two clouds with Unbalanced Gromov-Wasserstein (UGW)** — solved by
   `log_ugw_sinkhorn` from the `unbalancedgw` package. This is the defining
   difference from AlignOT: GW compares the **intra-cloud pairwise-distance
   matrices** (`dx`, `dy`), not coordinates, so the objective is inherently
   rotation/translation invariant. "Unbalanced" allows mass creation/destruction,
   enabling *partial* matching.
3. **Recover the rigid transform**: from the transport plan `pi`, take per-column
   argmax to build correspondences, then **Kabsch/SVD** to get rotation `R` plus
   centroids `Abar`, `Bbar`. So EMPOT recovers **rotation AND translation**
   (AlignOT is rotation-only).

Subtomogram adaptation (in `EMPOT/src/iterative_utils.py`):

- `prepare_sampling_volume`: normalize to `[0,1]`, **invert (`1 - norm`)**
  (dark-is-signal), threshold, TRN-sample.
- `run_empot_alignment`: UGW → correspondences → Kabsch.
- `apply_rigid_transform_to_volume`: applies the recovered rigid transform to the
  full original volume with **`scipy.ndimage.affine_transform`** (EMPOT does NOT
  use ASPiRE — ASPiRE is not installed in the EMPOT venv).

Entry points:

- CLI runner `EMPOT/src/run_iterative_alignment.py` (mature, parameterized; logging,
  per-round CSVs, slice PNGs, run folders under `EMPOT/outputs/iterative_runs/`).
- Notebook `EMPOT/notebooks/EMPOT-iterative-sbtm.ipynb` (mirrors the runner).
- Benchmark notebook `EMPOT/notebooks/EMPOT-sbtm-alignment-benchmark.ipynb`
  (validates rotation/translation recovery on synthetic perturbations).

EMPOT-specific caveats discovered:

- **Very slow**: UGW builds `M×M` distance matrices and runs a torch (CPU) Sinkhorn.
  `num_points=2000` ⇒ ~400 s per subtomogram; `num_points=500` is the benchmark size.
- **Inconsistent hyperparameters** across call sites: `threshold` = 0.1 (runner) vs
  0.7 (notebook); `eps` = 10000 (callers) vs 2000 (function defaults).
- **Coordinate-frame mismatch risk** in `apply_rigid_transform_to_volume`: clouds
  are sampled in a padded **even cube (92, center 46)** frame, but the volume
  transform uses `center = 91/2 = 45.5` on the raw 91³ grid. A half-voxel/center
  error would bias every averaged volume even when `R` is correct.
- Older `outputs/iterative_runs/run_*_N40_*` show `l2_change=0.0` / `ugw_dist=0.0`
  (a broken earlier version); the `run_*_N5_*` run (2026-06-25) is the valid one.

---

## Current Subtomogram Workflows in the Repo

## 1. AlignOT-Based Iterative Notebook

File:

- `alignOT/notebooks/AlignOT-iterative-sbtm.ipynb`

Purpose:

- iterative subtomogram averaging using AlignOT as the per-subtomogram rotation estimator

Current design:

1. Load all subtomogram file paths from `Data/0.1/Cluster_0`
2. Compute an initial reference as the plain average of the original subtomograms
3. For each round:
   - prepare a sampling map from the current reference
   - sample a reference point cloud once for the round
   - for each subtomogram:
     - load the **original** volume from disk
     - prepare its sampling map in memory
     - sample pseudo-atoms
     - run AlignOT `SGD(...)` to estimate a rotation quaternion
     - convert quaternion to a rotation matrix
     - rotate the full original subtomogram using ASPiRE
     - add the rotated volume to a streaming average accumulator
   - save the new reference
4. Save per-round metrics and reference volumes

Important rule already established:

- **each round must always align the original subtomogram files**
- **the code must never reuse the previously rotated subtomogram as the next round’s alignment input**

Why ASPiRE is used here:

- AlignOT estimates the rotation on sampled point clouds
- the full voxel-grid subtomogram still needs to be rotated before averaging
- ASPiRE is used only as the **full-volume rotation backend**

Memory strategy:

- do not load all 200 subtomograms at once
- keep only:
  - the current reference
  - one original subtomogram
  - one rotated subtomogram
  - one running accumulator
- use `gc.collect()` periodically

Saved outputs:

- `reference_round_00.mrc`, `reference_round_01.mrc`, ...
- `round_metrics.csv`

---

## 2. BOTalign Iterative Runner

File:

- `BOTalign/run_botalign_iterative_alignment.py`

Purpose:

- run iterative alignment/averaging over all subtomograms using BOTalign

Current design:

1. Discover `.mrc` subtomograms from `BOTalign/data/0.1/Cluster_0`
2. Optionally repair MRC headers in place
3. Compute or load the initial reference
4. For each round:
   - align every subtomogram to the current reference with `align_BO(...)`
   - rotate the full subtomogram with ASPiRE
   - streaming-average rotated volumes into the next reference
   - save per-round reference and CSV outputs

Saved outputs include:

- round-wise reference volumes
- per-subtomogram alignment metrics
- final rotations
- progress and runtime summaries

The notebook `BOTalign/notebooks/botalign_iterative_alignment.ipynb` is currently **viewer-only**. It reads saved outputs and visualizes them; it does not execute the alignment loop itself.

---

## What We Are Trying To Do Scientifically

The scientific/engineering aim is:

- adapt published 3D alignment methods to noisy subtomogram data
- compare how well different alignment strategies support subtomogram averaging
- understand the tradeoffs between:
  - full-volume alignment
  - sampled point-cloud alignment
  - different signal preprocessing choices
  - iterative reference refinement

More specifically:

- **BOTalign** serves as a full-volume alignment baseline
- **AlignOT** is being explored as a point-cloud/OT-based alternative
- both are being used in an **iterative reference-update loop**
- the task is not just pairwise alignment; it is **alignment for averaging**

That means the actual end goal is:

- better reference estimation
- better consistency across rounds
- a sharper average after repeated align-and-average steps

Even when the code is framed as “alignment,” the real use case is **subtomogram averaging**.

---

## Key Design Decisions Already Made

These are important for future coding work.

### General

- The repo is allowed to contain multiple experimental branches/workflows.
- Notebook-local copies inside `alignOT/notebooks/` are acceptable and currently used.
- We are not forcing AlignOT into a formal Python package refactor right now.

### Data handling

- Process `.mrc` files only.
- Ignore `Zone.Identifier` and other non-data sidecar files.
- Work on one subtomogram at a time whenever possible.

### AlignOT-specific

- Use dark-is-signal preprocessing for subtomogram sampling when appropriate.
- Use in-memory normalization/sampling helpers instead of writing normalized subtomograms to disk by default.
- Use ASPiRE only to apply recovered rotations to full volumes.

### Iterative averaging

- Initial reference is a simple average.
- Reference updates are performed round-by-round.
- Every round re-aligns the original subtomograms to the current reference.
- Rotated outputs are for averaging only, not for recursive re-input.

---

## Important Current Limitations / Caveats

These are not “results,” but they are implementation realities an LLM should know before modifying code.

- AlignOT in this repo is fundamentally **rotation-only** in the way it is currently used for subtomogram averaging.
- Translation handling is not the main active mechanism in the AlignOT iterative notebook.
- AlignOT operates on **sampled point clouds**, so its success depends heavily on:
  - normalization
  - inversion choice
  - thresholding
  - number of sampled points
- The upstream AlignOT code contains notebook-local duplication (`utils.py`, `trn.py`, etc.) and import-time print behavior.
- BOTalign is more mature for full-volume iterative averaging in this workspace.
- AlignOT iterative averaging is more experimental and adaptation-heavy.

---

## How To Think About Future Coding Work

If an LLM is asked to help on this repo, it should treat the project as:

- a research engineering environment
- with two competing/complimentary alignment methods
- aimed at subtomogram alignment and iterative averaging

The LLM should preserve these priorities:

1. Do not break the existing BOTalign runner workflow.
2. Do not assume AlignOT is a clean packaged library; notebook-local code may be the actual active path.
3. Preserve the “always align original subtomograms each round” rule.
4. Preserve memory-safe streaming behavior for the 200-volume dataset.
5. Distinguish clearly between:
   - estimating a rotation
   - applying a rotation to a full volume
   - averaging rotated subtomograms into a new reference

---

## Investigation: Why the OT Methods Fail on Subtomograms

**Observed problem.** All three OT methods (AlignOT, BOTalign, EMPOT) *fail* at
iterative subtomogram averaging on the 0.1-SNR cluster: the average gets **worse**
each round rather than sharper. EMPOT in particular recovers known transforms with
~0 error in single-subtomogram self-tests, yet returns effectively random
transforms when aligning two different subtomograms.

**Key clarification on the data.** There is **no known ground-truth map** for this
dataset (it was provided by the project guide). The `emd_1717.map` files present in
`Data/` and `alignOT/Data/` are **AlignOT's bundled test map (GroEL-era), NOT the
ground truth** for these subtomograms — do not treat them as such.

**Confirmed property of the data: a missing wedge.** An FFT power-spectrum check on
`Data/0.1/Cluster_0/0.mrc` shows power strongly suppressed toward the z-axis
(log-power ~4.3 at 0–15° from z vs ~6.6 near the equator — roughly a 10× drop).
This is an unmistakable cryo-ET **missing wedge** along z. The data is also
standardized (mean 0, std 1).

### Why the single-particle self-test is misleading

Rotating one subtomogram and recovering the rotation matches a point cloud to a
*rotated copy of itself* — identical points, identical noise, identical wedge. OT/GW
finds the trivial perfect correspondence. This validates the Kabsch/SVD math only;
it says nothing about cross-particle alignment.

### The three real killers (shared by all three OT methods)

1. **Missing wedge (the big one, currently unmodeled).** The wedge is fixed in the
   **lab frame (z)**, not the particle frame. Different particle orientations put the
   anisotropic smear in different places relative to the molecule. Real-space density
   (and any point cloud or volume loss built from it) is dominated by this artifact;
   OT often "aligns the wedges" instead of the molecules. Mature STA packages handle
   this with explicit **missing-wedge masks / constrained correlation in Fourier
   space**. None of AlignOT/BOTalign/EMPOT do — they were validated on clean,
   wedge-free single-particle cryo-EM maps.
2. **SNR 0.1 ⇒ the point cloud samples noise, not structure.** TRN samples
   pseudo-atoms ∝ density; at this SNR the thresholded extrema are mostly noise, so
   the cloud geometry is uncorrelated between particles. Raising 500→2000 points just
   samples more noise (matches the observation that more points didn't help).
3. **Real-space, interpolated averaging of wrong transforms monotonically
   degrades.** Cubic-spline rotation + off-center translation (with zero padding)
   low-pass blurs and pushes mass off-grid; the next reference is rebuilt from the
   degraded average ⇒ feedback divergence. EMPOT's centroid-difference translation is
   especially destructive with bad correspondences. (AlignOT is rotation-only, so it
   avoids the translation blow-up but still injects random rotations.)

### Secondary / implementation issues to rule out

- **Density polarity / inversion.** In a normal EMDB-style map protein is
  **high-positive** density. EMPOT/AlignOT "dark-is-signal" inversion (`1 - norm`,
  keep most-negative voxels) may be sampling **background/noise**. Test both
  polarities on the actual subtomograms.
- **Centering/coordinate-frame mismatch** in EMPOT's volume transform (91 vs 92,
  center 45.5 vs 46) — verify with an identity / known-90° round-trip.
- **GW invariance to reflections/pseudo-symmetry** — a "perfect" GW match can be a
  mirror or symmetry-equivalent (the particle looks barrel/ring-like, likely
  symmetric), i.e. the wrong physical alignment.

### Bottom line

The OT methods are fine on what they were designed for (clean, wedge-free maps);
the implementations are correct (self-tests pass). They fail on subtomograms because
**real-space OT on noisy, missing-wedge data has no consistent geometry to match.**
The most promising research direction is a **missing-wedge-aware (Fourier/masked) OT
formulation**, plus denoising/filtering before sampling, plus FSC-based evaluation.

---

## EMAN2 Baseline Reference

To separate "is the data alignable?" from "do the OT methods work?", we produced an
independent, non-OT baseline average with **EMAN2** (established STA software).

**Environment.** EMAN2 2.99.72 installed via miniforge into the conda env
`~/miniforge3/envs/eman2` (CPU; the laptop has 20 cores / ~7.6 GB RAM and a very new
RTX 5060 / sm_120 GPU that prebuilt CUDA wheels don't yet support — so CPU was used).

**Why not the obvious tools.**
- *PyTom* was rejected: classic PyTom needs a painful MPI/source build; modern
  `pytom-match-pick` is GPU template-matching, not 3D averaging.
- EMAN2's **modern** `e2spt_refine_new` / `e2spt_sgd_new` pipeline does **not** apply:
  it expects particles extracted from **2D tilt series** (`class_ptcl_src` metadata)
  and reconstructs from tilts. Our data is **pre-reconstructed 3D subvolumes only**.
- The **classic** `e2spt_*` CLI wrappers (`classaverage`, `preproc`, `binarytree`,
  `hac`) all have a build bug: `options = checkinput(options)` clobbers `options` with
  a `(options, inputs)` tuple. (Also: boxes must be **even-sized**; ours are 91³.)

**What we did instead.** Drive EMAN2's **own** missing-wedge-aware SPT library
primitives directly, in a controlled, parallel loop — the same scientific core
`e2spt_classaverage` uses internally:
- aligner `rotate_translate_3d_tree`, comparator `ccc.tomo.thresh` (wedge-aware),
  averager `mean.tomo` (wedge-weighted Fourier average).
- Particles padded 91³ → **96³**, apix forced to 1.0.
- Seed = lowpassed **raw average** of all 200 particles (unbiased round-0 reference).
- Per-iteration: each particle aligned to the current (preprocessed + shrink-3)
  reference; the translation-rescaled transform applied to the **raw** full-res
  particle; wedge-weighted accumulation. 5 iterations.
- Parallelized across CPU cores with `multiprocessing` (workers read by index, return
  only the serializable transform) — ~28 s/particle serial → ~6 min/iteration with
  14 workers (~32 min total).

Scripts: `eman2_baseline/sta_average.py` (the loop) and
`eman2_baseline/inspect_baseline.py` (orthoslice montage).

**Result — it works.** Alignment score converged and *improved* (mean −0.606 →
−0.668 → −0.672 → **−0.672 (iter3, best)** → −0.667), the opposite of the OT
divergence. The orthoslice montage shows real structure emerging from a blurry,
ring-artifact seed: a clear ring (top view) and repeating side-view densities, best
at iteration 3 (iter 4 slightly overfits). Canonical outputs:
- `eman2_baseline/baseline_reference.mrc` / `.hdf` — iter03, 96³, apix 1.0
- `eman2_baseline/baseline_reference_91.mrc` — clipped to original 91³ box
- `eman2_baseline/baseline_avg_iter00..04.hdf`, `baseline_slices.png`

**Conclusion from the baseline:** the dataset **has genuine, alignable signal**.
Therefore the OT failure is a **method limitation, not a data problem** — the key
evidence for the project guide.

**Caveats:** the baseline was built **without** the true simulation missing-wedge
geometry (EMAN2 inferred it); residual XY/XZ anisotropy is partly that wedge, partly
real molecular elongation. Sharpen it once the tilt range/axis are known.

---

## Conclusions and Next Steps

**Conclusions.**
- OT methods recover transforms perfectly in self-tests but fail cross-particle on
  this data; the average diverges each round.
- Root cause is dominated by the **missing wedge** (confirmed in the data) and
  **SNR 0.1**, compounded by real-space interpolated averaging of bad transforms.
- EMAN2 (missing-wedge-aware STA) **converges to a structured average**, proving the
  data is alignable and the OT failure is methodological.

**Next steps (priority order).**
0. **Get simulation metadata from the guide**: true source structure, missing-wedge
   tilt range + axis, SNR definition, pixel/box size, whether particles are centered.
   Highest value; unblocks real FSC and the ablation ladder.
1. **Validation harness** — gold-standard half-set FSC (split 200 into halves, align
   + average independently, FSC the two), and FSC of each OT round's average vs the
   EMAN2 `baseline_reference`. Replace eyeballing with a curve.
2. **Ablation ladder** against the EMAN2 baseline (or true ground truth) as a clean
   template: rotation-only → + noise → + missing wedge, to pinpoint exactly what
   breaks OT (expectation: survives rotation, degrades at noise, dies at wedge).
3. **Cheap bug checks**: density polarity / inversion, and the EMPOT
   centering/coordinate-frame round-trip.
4. **Fixes, by expected impact**: (a) heavy low-pass / denoise before TRN sampling;
   (b) seed the OT loops with `baseline_reference_91.mrc` instead of the raw blob
   (test whether OT can *refine* a good reference even if it can't bootstrap);
   (c) **missing-wedge-aware OT objective** in Fourier/masked space (the real
   research contribution); (d) wedge-aware Fourier averaging instead of naive
   real-space averaging.

**Strategic framing:** treat EMAN2 as both the **seed** and the **benchmark to beat**.
The research question becomes "can OT-based alignment match/beat correlation-based
STA, and in what regime?" — especially a missing-wedge-aware OT formulation.

---

## Short Working Summary

This repo is about **subtomogram alignment and averaging** using three OT-based
methods:

- **AlignOT** — point-cloud OT rotational alignment (rotation-only)
- **BOTalign** — full-volume Bayesian-optimization alignment
- **EMPOT** — point-cloud Unbalanced Gromov-Wasserstein + Kabsch (rotation + translation)

Work focuses on a **200-volume, 0.1-SNR** subtomogram cluster (`Data/0.1/Cluster_0`,
91³ volumes) run through iterative align-and-average loops.

**Current status / headline finding:** all three OT methods **diverge** on this data
(the average worsens each round), while **EMAN2's missing-wedge-aware STA converges**
to a structured average (`eman2_baseline/baseline_reference.mrc`). The data is
alignable; the OT failure is a **method** problem driven by the **missing wedge** +
**SNR 0.1** + real-space interpolated averaging. The central problem is no longer
generic registration — it is **building a missing-wedge-aware, OT-based iterative
subtomogram-averaging workflow**, validated by FSC against the EMAN2 baseline. See
the Investigation, EMAN2 Baseline, and Conclusions/Next-Steps sections above.

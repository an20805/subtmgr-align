# EMAN2 Baseline Setup (WSL2) — Instructions for an AI Agent

**Goal:** Install EMAN2 in WSL2 and run a gold-standard (even/odd half-set) subtomogram alignment on ~500 pre-extracted subtomograms, producing half-maps and an FSC curve. These results are the **baseline** for comparison against the user's own subtomogram alignment method.

**Reference docs (read these if anything below is unclear or fails):**
- EMAN2 home / install index: https://blake.bcm.edu/emanwiki/doku.php?id=eman2
- Source install (Linux/WSL2): https://blake.bcm.edu/emanwiki/doku.php?id=eman2:install:sourceinstall
- Tomography tutorial: https://blake.bcm.edu/emanwiki/doku.php?id=eman2:e2tomosmall
- Newer refinement pipeline notes: https://blake.bcm.edu/emanwiki/doku.php?id=eman2:e2tomo_new
- Troubleshooting / help: Google Group `eman2`

---

## 0. Ground rules for the agent

1. **Do not guess command-line flags.** Before running any EMAN2 program for the first time, run it with `--help` and confirm every flag you use exists. Commands in this file are templates; adapt them to what `--help` shows, and note any changes in the log.
2. **Log everything.** Keep a running log at `~/eman2_baseline/LOG.md`: each phase, the exact commands run, key output, errors, and decisions.
3. **Stop and report** (do not improvise) if: a build step fails twice, a required user input below is missing, the data looks inconsistent (wrong box size, contrast, pixel size), or a step would need `sudo` in a way that could affect the Windows side.
4. **Work on the Linux filesystem** (`$HOME/...`), never under `/mnt/c/...`. Only *read* the user's input data from wherever it lives, and copy it into the project folder.
5. **Do not modify** the user's original data. Do not install packages into the `base` conda env.
6. Use **physical CPU cores** (not logical threads) wherever EMAN2 asks for a thread count.

---

## 1. Inputs the user must supply (ask before starting Phase 4)

Fill this table by asking the user; do not assume values.

| Item | Value |
|---|---|
| Path to the 500 subtomograms (single stack, or folder of files; format e.g. .mrc/.hdf/.em) | `<DATA_PATH>` |
| Voxel size (Å/pixel) | `<APIX>` |
| Box size (voxels, cubic) | `<BOX>` |
| Particle contrast: are particles **white/bright** (positive density, EMAN2 convention) or dark? | `<CONTRAST>` |
| Symmetry (e.g. c1) | `<SYM>` |
| Molecular mass in kDa (approximate; only used for isosurface scaling) | `<MASS_KDA>` |
| Initial reference map (path) — **preferred**, so both methods start from the same reference. If none, EMAN2 will generate one | `<REF_PATH or none>` |
| Mask (path), if the user's method uses one | `<MASK_PATH or none>` |
| Fixed even/odd split required to match the user's method? (yes/no; if yes, the list of indices for each half) | `<SPLIT>` |

**Assumption baked into this file:** the data are **already-extracted subtomograms (3D volumes)**, not a raw tilt series. If the user actually has raw tilt series, stop and see Appendix A.

---

## 2. Phase 1 — Preflight checks

Run and record the output of:

```bash
cat /etc/os-release | head -3          # expect Ubuntu under WSL2
uname -r                               # kernel should mention "microsoft" / WSL2
lscpu | grep -E "Model name|Socket|Core|Thread|^CPU\(s\)"   # physical cores = sockets x cores/socket
free -h                                # RAM available to WSL
df -h $HOME                            # free disk (need >= 40 GB; more if data are large)
nvidia-smi                             # GPU visible? (informational; see note below)
echo "PATH=$PATH"; echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"; echo "PYTHONPATH=$PYTHONPATH"
```

Notes:
- **GPU:** The subtomogram alignment/averaging step (`e2spt_refine.py`) is CPU-threaded. EMAN2's GPU code (TensorFlow/JAX) is for neural-network tools such as the particle picker and tomogram annotation, which this baseline does not use. Do **not** spend time on GPU setup unless the user asks (optional step in Phase 2).
- Recommended minimum: 16 GB RAM, 4 cores. If `free -h` shows much less RAM than the Windows host has, WSL2's default memory cap is limiting it; tell the user they can raise it in `%UserProfile%\.wslconfig` (this needs a Windows-side `wsl --shutdown`, so ask the user to do it).
- If `PATH`/`LD_LIBRARY_PATH`/`PYTHONPATH` contain old Anaconda/Miniconda/EMAN entries, `unset LD_LIBRARY_PATH PYTHONPATH` for the install session and warn the user. WSL `PATH` normally also contains Windows paths; that is fine.

---

## 3. Phase 2 — Install conda (Miniforge) and the EMAN2 environment

### 2.1 System packages (if missing)

```bash
sudo apt update
sudo apt install -y build-essential git curl wget \
    libgl1-mesa-dev libglu1-mesa-dev libxcb-cursor0
```

(EMAN2's GUI needs OpenGL/Qt; on Windows 11, WSLg provides display. The baseline can run headless from the command line, so GUI problems are non-blocking.)

### 2.2 Miniforge

Skip if `conda`/`mamba` already exist and work (`conda --version`, `mamba --version`); in that case go to 2.3 and create a **new, fresh** env.

```bash
cd ~
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Answer **yes** when asked to initialize conda in the shell profile. Then start a new shell (or `source ~/.bashrc`) and verify `conda --version` and `mamba --version`.

### 2.3 Configure conda (as recommended by the EMAN2 wiki)

```bash
conda config --set auto_activate_base False
conda config --set auto_update_conda False
mamba update --all -n base -y
```

### 2.4 Create the EMAN2 environment (dependencies only)

The wiki notes that since Aug 2024 the dependencies use **Python 3.12**; always create a fresh environment.

```bash
conda create -n eman2 eman-dev --only-deps -c conda-forge -c cryoem -y
```

Verify: `conda activate eman2 && python --version` (expect 3.12.x).

Do **not** install extra packages into this env until EMAN2 is built and tested (the wiki warns that extra packages can change EMAN2 dependency versions).

### 2.5 (Optional, skip by default) GPU libraries

Only if the user wants GPU-based EMAN2 tools later:

```bash
conda activate eman2
pip install "jax[cuda12]" optax
```

Requires an NVIDIA driver installed on the **Windows** side with WSL support; do not install a Linux NVIDIA driver inside WSL.

---

## 4. Phase 3 — Build and install EMAN2 from source

```bash
conda activate eman2
mkdir -p ~/src && cd ~/src
git clone https://github.com/cryoem/eman2.git
mkdir -p ~/src/eman2-build
cd ~/src/eman2-build
cmake ~/src/eman2 -DENABLE_OPTIMIZE_MACHINE=ON
make -j <PHYSICAL_CORES>
make install
```

Notes:
- `make -j` with fewer jobs is fine if RAM is tight (compilation is memory hungry). The build can take a while.
- Do **not** run `make test` (the wiki says failures there are common and not meaningful for normal installs).
- If cmake picks up wrong paths, ensure `echo $CONDA_PREFIX` points to the `eman2` env, delete `CMakeCache.txt`, and rerun cmake.

### 3.1 Verify

```bash
conda activate eman2
e2version.py
e2speedtest.py
```

`e2version.py` should print an EMAN2 2.99.x version. Record it in the log, and also `cd ~/src/eman2 && git log -1`.

### 3.2 Known issues (from the wiki)

- `ImportError: libtensorflow_cc.so.2: cannot enable executable stack ...` → try `patchelf --clear-execstack $CONDA_PREFIX/lib/libtensorflow_cc.so.2` (install `patchelf` via `mamba install patchelf -n eman2` or apt if needed). This only matters for TensorFlow-based programs, not for the baseline refinement.
- `version 'GLIBCXX_3.4.29' not found` → remove the offending `libstdc++.so*` in `$CONDA_PREFIX/lib` (back it up first), then retry.
- If GUI programs (`e2display.py`) fail, that is non-blocking for this task; note it and continue.

Save a reproducible record: `conda env export -n eman2 > ~/eman2_baseline/eman2_env.yml`.

---

## 5. Phase 4 — Project setup and data import

### 4.1 Create the project folder

```bash
mkdir -p ~/eman2_baseline/project && cd ~/eman2_baseline/project
```

Run **all EMAN2 commands from this folder** (not a subfolder). File names must contain **no spaces, no periods except the extension, and no double underscores**.

### 4.2 Inspect the input data

```bash
e2iminfo.py <DATA_PATH>          # for a single stack/file: size, box, apix, count
ls <DATA_PATH> | head            # for a folder of files
```

Confirm with the user-supplied table: 500 particles, cubic box of `<BOX>`, and voxel size `<APIX>`. If the header apix differs from the user's value, **ask which is correct**; do not silently override.

### 4.3 Convert to one HDF stack with correct metadata

Create `import_particles.py` in the project folder:

```python
#!/usr/bin/env python
"""Import subtomograms into a single EMAN2 HDF stack. Handles a folder of files or one multi-image stack."""
import argparse, glob, os, sys
from EMAN2 import EMData, EMUtil

p = argparse.ArgumentParser()
p.add_argument("--src", required=True, help="file or folder of subtomograms")
p.add_argument("--out", default="particles3d/all.hdf")
p.add_argument("--apix", type=float, required=True)
p.add_argument("--invert", action="store_true", help="multiply by -1 (use if particles are dark)")
a = p.parse_args()

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
if os.path.exists(a.out):
    os.remove(a.out)

files = sorted(glob.glob(os.path.join(a.src, "*"))) if os.path.isdir(a.src) else [a.src]
n_out = 0
for f in files:
    n = EMUtil.get_image_count(f)
    for i in range(n):
        e = EMData(f, i)
        if e["nx"] != e["ny"] or e["nx"] != e["nz"]:
            sys.exit(f"Non-cubic volume in {f} idx {i}: {e['nx']}x{e['ny']}x{e['nz']}")
        e["apix_x"] = e["apix_y"] = e["apix_z"] = a.apix
        if a.invert:
            e.mult(-1.0)
        e["source_path"] = os.path.basename(f)
        e["source_n"] = i
        e.write_image(a.out, n_out)
        n_out += 1
print(f"wrote {n_out} particles to {a.out}")
```

Run (add `--invert` only if the user confirmed particles are dark):

```bash
python import_particles.py --src <DATA_PATH> --apix <APIX> [--invert]
e2iminfo.py particles3d/all.hdf
```

Expect **500** images. If the count differs, stop and report.

**Contrast convention:** in EMAN2's tomography pipeline, reconstructed particles have **positive** contrast (protein looks bright), while tilt series/tomograms are negative. Verify contrast by checking a mean of the central region vs. the corners of the average:

```python
python - <<'EOF'
from EMAN2 import EMData
import numpy as np
# average of the first 50 particles
avg = None
for i in range(50):
    e = EMData("particles3d/all.hdf", i)
    avg = e if avg is None else avg + e
avg.mult(1/50.0)
a = avg.numpy()
c = a.shape[0]//2; r = a.shape[0]//8
print("center mean:", a[c-r:c+r, c-r:c+r, c-r:c+r].mean(), " corner mean:", a[:r,:r,:r].mean())
print("Expect center > corner for bright (positive) particles")
EOF
```

If center < corner, ask the user before re-importing with `--invert`.

### 4.4 Make a particle set (list file)

Create the set list that refinement will use:

```bash
mkdir -p sets
python - <<'EOF'
from EMAN2 import LSXFile, EMUtil
n = EMUtil.get_image_count("particles3d/all.hdf")
lsx = LSXFile("sets/all.lst", False)
for i in range(n):
    lsx.write(-1, i, "particles3d/all.hdf", "")
lsx.close()
print("wrote sets/all.lst with", n, "entries")
EOF
head -5 sets/all.lst
```

If `LSXFile` usage fails, check `python -c "from EMAN2 import LSXFile; help(LSXFile)"` for the correct signature. As a fallback, `e2spt_refine.py --help` will say whether a plain `.hdf` stack is accepted for `--ptcls`; if so, use `particles3d/all.hdf` directly.

Sanity check: `e2display.py particles3d/all.hdf` (GUI), or render a few central slices to PNG with matplotlib (in a **separate** venv, see Phase 7) to confirm the particles look sensible.

---

## 6. Phase 5 — Reference map and mask

**Preferred:** use the user's initial reference so both methods start from the same map.

1. Convert to HDF and match voxel size and box size to the particles:
   ```bash
   e2iminfo.py <REF_PATH>
   e2proc3d.py <REF_PATH> ref/ref_init.hdf --apix=<APIX> --clip=<BOX>,<BOX>,<BOX> --process=filter.lowpass.gauss:cutoff_freq=<1/RES_A>
   ```
   (Create `ref/` first. Confirm option names with `e2proc3d.py --help`. Choose `<RES_A>` around 40–50 Å unless the user says otherwise, low enough to avoid reference bias. Record the choice. If the reference is on a different voxel grid, use `--scale`/`--clip` accordingly, and verify the result with `e2iminfo.py`.)
2. The reference must have **positive contrast** and be centered in the box.

**If the user has no reference:** generate one with EMAN2's stochastic gradient descent tool.

```bash
e2spt_sgd_new.py --help
e2spt_sgd_new.py sets/all.lst --res 50 --niter 10 --shrink 2 --parallel thread:<PHYSICAL_CORES>
```

- `<BOX>` divided by `--shrink` must be an **even integer**, otherwise the program crashes.
- The output appears in `sptsgd_00/output_cls0.hdf` (updates each iteration; 3–5 iterations are often enough). **Ask the user to inspect it** (or report its shape stats) before proceeding; it may not converge to the correct structure.

**Mask:** if the user gave `<MASK_PATH>`, convert it to HDF with the same apix/box and keep it for Phase 6 (check whether `e2spt_refine.py --help` has a mask option) and for the FSC computation in Phase 7.

---

## 7. Phase 6 — Gold-standard refinement (the baseline run)

EMAN2's `--goldstandard` mode splits the set into **even and odd** halves, refines them independently, and phase-randomizes the reference beyond the given resolution so the two halves stay independent. FSC is computed between the two half-maps each iteration.

**Use the older 3D-particle refinement program `e2spt_refine.py`.** Do **not** use `e2spt_refine_new.py`: it refines 2D subtilt images and needs metadata from EMAN2's own tilt-series processing, which pre-extracted subtomograms from other sources do not have. Also, for a fair comparison against a *subtomogram alignment* method, only 3D particle alignment should be used.

### 6.1 Read the help first

```bash
e2spt_refine.py --help
```

Identify the exact names of: particle input, reference, gold-standard resolution, iterations, symmetry, mass, threads/parallel, particle keep fraction, mask, local filter, output path. (The GUI equivalent is *Subtomogram Averaging → 3D Refinement*, with fields particles, reference, goldstandard, mass, threads, localfilter, pkeep, sym.)

### 6.2 Smoke test on a 50-particle subset (must pass before the full run)

```bash
python - <<'EOF'
from EMAN2 import LSXFile
src = [l for l in open("sets/all.lst") if not l.startswith("#")]
lsx = LSXFile("sets/test50.lst", False)
for i in range(50):
    lsx.write(-1, i, "particles3d/all.hdf", "")
lsx.close()
EOF
```

Run a 1–2 iteration test, adapting flags to `--help` (template; **verify each flag**):

```bash
e2spt_refine.py --ptcls sets/test50.lst --ref ref/ref_init.hdf \
    --goldstandard <GOLD_RES_A> --sym <SYM> --mass <MASS_KDA> \
    --niters 2 --threads <PHYSICAL_CORES> --path spt_test
```

- `<GOLD_RES_A>`: resolution in Å beyond which the reference is phase-randomized (the tutorial uses 30 for ribosomes at 3.93 Å/px). It must be coarser than 2×`<APIX>` and coarser than the reference's low-pass resolution. **Ask the user for their preferred value and record it.** Use the same value for any comparison run of their method.
- Confirm the run finishes and creates a results folder with maps and FSC files. Then delete the test folder (`rm -r spt_test`) so it is not confused with the real run.

### 6.3 Full baseline run

```bash
e2spt_refine.py --ptcls sets/all.lst --ref ref/ref_init.hdf \
    --goldstandard <GOLD_RES_A> --sym <SYM> --mass <MASS_KDA> \
    --niters <N_ITERS> --threads <PHYSICAL_CORES> --path spt_baseline
```

- Use the same number of iterations the user uses for their method (ask; default 5–6 if unspecified).
- Run under `nohup` or `tmux` (`nohup ... > refine.log 2>&1 &`) so it survives terminal disconnects. The tutorial notes ~1 hr/iteration for ~1000+ particles at 4 cores; 500 particles at a moderate box size is usually faster. Report timing in the log.
- Do not add extra options beyond what the user requested (e.g. local filtering, particle keep fraction) without recording them; each one changes the baseline. If used, note them in the log.

### 6.4 Record the even/odd split

Look inside the output folder for the even/odd particle lists and half-maps (names like `*_even*` / `*_odd*`). Write which original particle indices went to each half to `~/eman2_baseline/halfsets.json`. If the user needs an identical split for their own method, report exactly how EMAN2 assigns halves, and if it can't be controlled via `--help`, tell the user.

---

## 8. Phase 7 — FSC measured independently (same tool for both methods)

To compare fairly, compute the FSC from the final even/odd half-maps with a standalone script, so the same code can be applied to the user's method. EMAN2 also writes its own FSC curves (e.g. `fsc_maskedtight_XX.txt` in the results folder); keep those too and report both.

### 7.1 Separate Python venv (keeps the EMAN2 env clean)

```bash
python3 -m venv ~/venvs/fsc
source ~/venvs/fsc/bin/activate
pip install numpy mrcfile matplotlib
```

### 7.2 Export the half-maps to MRC

Back in the EMAN2 env, for the last iteration `NN` (check the actual file names):

```bash
conda activate eman2
cd ~/eman2_baseline/project
e2proc3d.py spt_baseline/threed_NN_even.hdf fsc/even.mrc
e2proc3d.py spt_baseline/threed_NN_odd.hdf  fsc/odd.mrc
# also the full map:
e2proc3d.py spt_baseline/threed_NN.hdf fsc/full.mrc
```

(`mkdir -p fsc` first. Confirm voxel size with `e2iminfo.py` and that the MRC headers carry it.)

### 7.3 FSC script

Create `~/eman2_baseline/project/fsc/compute_fsc.py`:

```python
#!/usr/bin/env python
"""FSC between two half-maps. Usage: compute_fsc.py even.mrc odd.mrc --apix 3.93 [--mask mask.mrc] [--out prefix]"""
import argparse
import numpy as np, mrcfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("a"); p.add_argument("b")
p.add_argument("--apix", type=float, required=True)
p.add_argument("--mask", default=None)
p.add_argument("--thr", type=float, default=0.143)
p.add_argument("--out", default="fsc")
args = p.parse_args()

def load(f):
    with mrcfile.open(f, permissive=True) as m:
        return np.array(m.data, dtype=np.float64)

A, B = load(args.a), load(args.b)
assert A.shape == B.shape and len(set(A.shape)) == 1, "maps must be cubic and same size"
if args.mask:
    M = load(args.mask); assert M.shape == A.shape
    A, B = A * M, B * M

n = A.shape[0]
FA, FB = np.fft.fftn(A), np.fft.fftn(B)
freq = np.fft.fftfreq(n)                      # cycles/voxel
kx, ky, kz = np.meshgrid(freq, freq, freq, indexing="ij")
r = np.sqrt(kx**2 + ky**2 + kz**2) * n         # radius in shell units
shell = np.rint(r).astype(int)
nshell = n // 2

num = np.bincount(shell.ravel(), weights=(FA * np.conj(FB)).real.ravel(), minlength=nshell + 1)[:nshell + 1]
da = np.bincount(shell.ravel(), weights=(np.abs(FA) ** 2).ravel(), minlength=nshell + 1)[:nshell + 1]
db = np.bincount(shell.ravel(), weights=(np.abs(FB) ** 2).ravel(), minlength=nshell + 1)[:nshell + 1]
fsc = num / np.sqrt(da * db + 1e-30)
spatial = np.arange(nshell + 1) / (n * args.apix)   # 1/Angstrom

# resolution at threshold: first downward crossing, linear interpolation
res = None
for i in range(1, nshell + 1):
    if fsc[i] < args.thr <= fsc[i - 1]:
        t = (fsc[i - 1] - args.thr) / (fsc[i - 1] - fsc[i])
        f = spatial[i - 1] + t * (spatial[i] - spatial[i - 1])
        res = 1.0 / f
        break

np.savetxt(args.out + ".txt", np.c_[spatial, fsc], header="spatial_freq_1/A  FSC")
plt.figure(figsize=(6, 4))
plt.plot(spatial, fsc); plt.axhline(args.thr, ls="--", c="gray")
plt.xlabel("Spatial frequency (1/Å)"); plt.ylabel("FSC"); plt.ylim(-0.1, 1.05)
plt.title(f"FSC @ {args.thr}: " + (f"{res:.2f} Å" if res else "not crossed"))
plt.tight_layout(); plt.savefig(args.out + ".png", dpi=150)
print(f"Resolution at FSC={args.thr}: " + (f"{res:.2f} A" if res else "no crossing"))
```

Run it without and with a mask (if the user supplied one), using the venv:

```bash
source ~/venvs/fsc/bin/activate
cd ~/eman2_baseline/project/fsc
python compute_fsc.py even.mrc odd.mrc --apix <APIX> --out fsc_unmasked
python compute_fsc.py even.mrc odd.mrc --apix <APIX> --mask <MASK_MRC> --out fsc_masked   # if a mask exists
```

Report both numbers along with the FSC threshold (0.143) and the mask used. A tight mask can inflate FSC, so the mask must be identical for both methods.

---

## 9. Phase 8 — Deliverables and report to the user

Save under `~/eman2_baseline/` and report:

1. `LOG.md` with all commands and outputs.
2. `eman2_env.yml`, EMAN2 version (`e2version.py`), git commit of the source.
3. Project folder path and `spt_baseline/` results: final map (`threed_NN.hdf`), half-maps, EMAN2's FSC text files.
4. `halfsets.json` (which particle indices are in each half).
5. Independent FSC: `fsc_unmasked.txt/png` (and `fsc_masked` if applicable) with the resolution at FSC=0.143.
6. The exact parameters used: `<GOLD_RES_A>`, iterations, symmetry, reference low-pass, mask, thread count, any extra options.
7. Runtime per iteration and total.
8. Any deviations from this file, and open issues.

Finish by giving the user a short comparison checklist for their own method: same particles, same initial reference, same half-split, same starting/low-pass resolution, same mask, same FSC code (`compute_fsc.py`), same threshold (0.143).

---

## Appendix A — If the data are actually raw tilt series (not extracted subtomograms)

Stop and confirm with the user. The full EMAN2 pipeline then applies (see the tutorial): import tilt series → tilt-series alignment and tomogram reconstruction → CTF estimation → particle picking → particle extraction → build sets → initial model → `e2spt_refine_new.py` with `--goldstandard`. Full pipeline requires the raw tilt series (EMAN2 metadata cannot be created from tomograms reconstructed elsewhere). For a fair baseline against a subtomogram alignment method, use only the `p` (3D particle) iterations, e.g. `--iters p,p,p`, and consult the `e2tomosmall` and `e2tomo_new` pages for the exact options.

## Appendix B — Quick troubleshooting

| Symptom | Likely cause / action |
|---|---|
| `make` runs out of memory | Reduce `-j`; raise WSL memory via `.wslconfig` (user action) |
| `e2spt_refine.py` says box or apix mismatch | Re-check reference vs particle box size and voxel size (Phase 5) |
| Refinement ends with FSC ≈ 0 everywhere | Wrong contrast, particles off-center, or reference too different; inspect the average of the input particles |
| Crash in initial model generation | `<BOX>/shrink` must be an even integer |
| Very slow run | Confirm `--threads` = physical cores and the project is on the Linux filesystem, not `/mnt/c` |
| Qt/OpenGL errors | Non-blocking; run everything headless |

#!/usr/bin/env python
"""
Inspect the EMAN2 baseline averaging output:
  - orthoslice montage of each per-iteration average + the raw seed
  - rotational-style sanity: print std/contrast per iteration (sharpening proxy)
Outputs eman2_baseline/baseline_slices.png
Run inside the eman2 env.
"""
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from EMAN2 import EMData, EMUtil

OUTDIR = "eman2_baseline"

def emd_to_np(fname, idx=0):
    e = EMData(fname, idx)
    nz, ny, nx = e.get_zsize(), e.get_ysize(), e.get_xsize()
    a = np.array(e.numpy(), copy=True).reshape(nz, ny, nx)
    return a

# Collect: seed + each iter average
items = []
if os.path.exists(f"{OUTDIR}/init_ref_e.hdf"):
    items.append(("seed (raw avg)", f"{OUTDIR}/init_ref_e.hdf"))
for f in sorted(glob.glob(f"{OUTDIR}/baseline_avg_iter*.hdf")):
    items.append((os.path.basename(f).replace("baseline_avg_", "").replace(".hdf", ""), f))

if not items:
    print("No averages found yet.")
    raise SystemExit

fig, axes = plt.subplots(len(items), 3, figsize=(9, 3 * len(items)))
if len(items) == 1:
    axes = axes.reshape(1, 3)

for r, (label, f) in enumerate(items):
    v = emd_to_np(f)
    cz, cy, cx = [s // 2 for s in v.shape]
    axes[r, 0].imshow(v[cz, :, :], cmap="gray"); axes[r, 0].set_ylabel(label, fontsize=9)
    axes[r, 1].imshow(v[:, cy, :], cmap="gray")
    axes[r, 2].imshow(v[:, :, cx], cmap="gray")
    axes[r, 0].set_title(f"{label}  XY  (std={v.std():.3f})", fontsize=8)
    axes[r, 1].set_title("XZ", fontsize=8)
    axes[r, 2].set_title("YZ", fontsize=8)
    for c in range(3):
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

plt.tight_layout()
plt.savefig(f"{OUTDIR}/baseline_slices.png", dpi=120)
print(f"wrote {OUTDIR}/baseline_slices.png with {len(items)} rows")

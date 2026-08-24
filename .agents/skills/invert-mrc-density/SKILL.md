---
name: invert-mrc-density
description: >-
  Invert and flip density contrast in 3D MRC volume maps (e.g. converting dark protein signal in cryo-ET to positive/bright densities for UCSF ChimeraX visualization). Use this skill whenever the user asks to invert, flip contrast, or prepare an MRC map for ChimeraX.
---

# Invert MRC Density Contrast

This skill inverts or flips density values of 3D MRC volumes so that cryo-ET / subtomogram averaging maps (where protein is traditionally dark / negative) are converted into positive, bright densities ready for standard isosurface contouring in **UCSF ChimeraX**.

## Workflow Steps

### 1. Invert Volume Using Helper Script

Run the automated helper script using the project's Python environment:

```bash
/home/anshu/subtmgr-align/alignOT/alignot_env/bin/python \
    /home/anshu/subtmgr-align/.agents/skills/invert-mrc-density/scripts/invert_mrc.py \
    <path/to/input.mrc> \
    --output_dir temp/
```

### Options:
- `<input_mrc>`: Path to the input `.mrc` volume.
- `--output_dir <dir>`: Target folder to save the inverted MRC (default: `temp/`).
- `--output_mrc <path>`: Explicit destination filepath for the inverted `.mrc` file.
- `--mode <negate|min_max_flip|normalize_invert>`:
  - `negate` (default): Sets $V_{\text{out}} = -V_{\text{in}}$. Ideal for ChimeraX visualization.
  - `min_max_flip`: Flips range based on $(\max + \min) - V_{\text{in}}$.
  - `normalize_invert`: Normalizes density to $[0, 1]$ and computes $1 - V_{\text{norm}}$.

### 2. Verify Output

Confirm that the inverted file `<name>_inverted.mrc` exists in the output directory.

### 3. Provide ChimeraX Usage Commands

When presenting the result to the user, include the corresponding ChimeraX command:
```text
open <path_to_inverted.mrc>
volume #1 level 0.3
```

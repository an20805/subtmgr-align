# Subtomogram Alignment Project Handover & Context

**Repository**: `subtmgr-align`  
**Active Working Branch**: `dev`  
**Current Date**: October 2026  
**Primary Goal**: Cryo-ET Subtomogram Alignment & Averaging (STA) using Entropic Optimal Transport (Sinkhorn) + Barycentric 1-NN SGD on $S^3$.

---

## 1. Executive Summary & Current State of the Codebase

The active production method is **Native GPU-Accelerated AlignOT** implemented in:
- `alignOT/src/run_iterative_alignment_gpu_halfset.py`
- `alignOT/src/alignot_gpu_utils.py`
- `scripts/run_alignot_iterative_fsc.sh`

### Key Methodological Components:
1. **Gold-Standard Even/Odd Splitting**: Subtomograms are split into independent even and odd halves. Each half-set aligns and averages completely independently to prevent reference bias.
2. **Tangent-Space Projected SGD on $S^3$**: Analytical polynomial gradients on the unit quaternion manifold optimize 3D rotation and centroid-aware translation.
3. **Fourier-Space Wedge-Masked Averaging**: When `--wedge_mask Data/wedge_mask.mrc` is enabled, volumes and per-particle rotated wedge masks are accumulated in Fourier space with a Wiener filter to correct for missing-wedge distortion.
4. **Raised-Cosine Fourier Shell Low-Pass Filter**: Intermediate references from Round 1 onwards are filtered via `--lp_shells 23 --lp_rolloff 2` to suppress high-frequency noise amplification.

---

## 2. Key Experimental Findings & Quantitative Benchmarks

### Quantitative Comparison: AlignOT vs. EMAN2 vs. Ground Truth (`Data/0.mrc`)

Evaluating reconstructed consensus averages against noiseless Ground Truth (`Data/0.mrc`) on Protein 0 (GroEL chaperonin):

| Method / Configuration | FSC@0.143 | Pearson CC to GT ($r$) | $W_2$ Distance to GT | $W_2^2$ Cost to GT | Apical Caps in XZ? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AlignOT ($\text{th}=0.6$, 500 pts)** | $19.60\text{ \AA}$ | **0.5114** | **$11.581\text{ vox}\ (46.32\text{ \AA})$** | **134.11** | **Fully Resolved** |
| **AlignOT ($\text{th}=0.7$, 1000 pts)** | **$16.86\text{ \AA}$** | **0.5152** | $11.785\text{ vox}\ (47.14\text{ \AA})$ | $138.89$ | Truncated / weak |
| **AlignOT ($\text{th}=0.7$, 500 pts)** | $17.55\text{ \AA}$ | $0.5082$ | $11.890\text{ vox}\ (47.56\text{ \AA})$ | $141.37$ | Truncated / weak |
| **EMAN2 Baseline (Round 02)** | $17.55\text{ \AA}$ | $0.4715$ | $12.260\text{ vox}\ (49.04\text{ \AA})$ | $150.30$ | Yes |

### Core vs. Full Envelope Trade-off:
- **`threshold = 0.70`**: Concentrates TRN pseudo-atoms purely into the dense equatorial core of GroEL. Yields an artificially tight FSC ($16.86\text{ \AA}$), but cuts off the lower-density apical domain caps at the barrel ends.
- **`threshold = 0.60`**: Captures the complete molecular envelope including the apical caps. Achieves the **lowest Wasserstein transport distance to Ground Truth ($W_2^2 = 134.11$)** of any method, with an FSC of $19.60\text{ \AA}$.

---

## 3. Next Planned Experiments

1. **Increase Point Density at $\text{threshold} = 0.60$**:
   - Test `NUM_POINTS=1000` with `THRESHOLD=0.6`.
   - In the $0.7$ tests, increasing points from 500 to 1000 improved resolution from $17.55\text{ \AA} \to 16.86\text{ \AA}$. Doubling point density across the larger $0.6$ volume is expected to tighten FSC back towards $\sim 17\text{ \AA}$ while preserving the resolved apical caps.
2. **Evaluate `TARGET_MODE="barycentric"`**:
   - Compare continuous barycentric target projection against hard `argmax` on the $\text{threshold} = 0.60$ envelope.
3. **Execute on Prajna HPC**:
   - Submit via `sbatch scripts/submit_prajna.slurm` on the `l40` partition ($8\times$ NVIDIA L40S ADA GPUs).

---

## 4. Production Runner Scripts

| Script | Purpose |
| :--- | :--- |
| `scripts/run_alignot_iterative_fsc.sh` | Main automated GPU halfset alignment pipeline. |
| `scripts/submit_prajna.slurm` | SLURM submission script for Prajna cluster (`l40` partition & QoS). |
| `scripts/sync_to_prajna.sh` | Rsync code and scripts to `anshuman25@prajna.iitb.ac.in:~/subtmgr-align`. |
| `scripts/sync_from_prajna.sh` | Rsync outputs and logs from Prajna back to local machine. |
| `scripts/submit_madhava.slurm` | SLURM submission script for Kalpana/Madhava cluster. |
| `scripts/compute_fsc.py` | Standalone gold-standard FSC calculator ($0.5$ and $0.143$ cutoffs). |

---

## 5. Python Environment Setup (Clean Install)

```bash
cd subtmgr-align
python3 -m venv alignOT/alignot_env
source alignOT/alignot_env/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r alignOT/requirements.txt
```

#!/bin/bash
#SBATCH --job-name=test_cuda_empot
#SBATCH --account=sharada_proj2
#SBATCH --partition=cn2_madhava
#SBATCH --qos=madhava
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Job started on $(hostname) at $(date)"

# 1. Move to project directory and activate the uv environment
cd /users/student/pg/pg25/anshuman/subtmgr-align
source EMPOT/empot/bin/activate

# 2. Check CUDA and core dependencies
python - <<'EOF'
import torch
import numpy as np
import mrcfile
import scipy

print("=" * 40)
print("CUDA check")
print("=" * 40)
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))

print()
print("=" * 40)
print("Package versions")
print("=" * 40)
print("torch  :", torch.__version__)
print("numpy  :", np.__version__)
print("mrcfile:", mrcfile.__version__)
print("scipy  :", scipy.__version__)

print()
print("=" * 40)
print("Quick tensor test on GPU")
print("=" * 40)
if torch.cuda.is_available():
    a = torch.randn(1000, 1000, device="cuda")
    b = torch.randn(1000, 1000, device="cuda")
    c = a @ b
    print("Matrix multiply (1000x1000) on GPU: OK")
else:
    print("Skipped — no GPU available")

print()
print("All checks passed!")
EOF

echo "Job finished at $(date)"

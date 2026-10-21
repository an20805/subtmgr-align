#!/bin/bash
#SBATCH --job-name=eman2_sta
#SBATCH --account=sharada_proj2
#SBATCH --partition=cn2_madhava
#SBATCH --qos=madhava
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export TZ="Asia/Kolkata"

PROJECT_DIR="/users/student/pg/pg25/anshuman/subtmgr-align"
cd "${PROJECT_DIR}"
mkdir -p logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate eman2

echo "======================================================================"
echo "EMAN2 Gold-Standard Iterative STA (Reference-Free, Pure EMAN2)"
echo "Start: $(date)"
echo "Node : $(hostname)"
echo "CPUs : ${SLURM_CPUS_PER_TASK}"
echo "======================================================================"

python eman2_baseline/run_iterative_alignment_eman2.py \
    --cluster_dir Data/0.1/subtomograms/0 \
    --output_base_dir eman2_baseline/runs \
    --n_rounds 5 \
    --num_workers 16 \
    --shrink 3 \
    --lowpass 0.15 \
    --apix 4.0

echo "======================================================================"
echo "Job finished successfully at $(date)"
echo "======================================================================"

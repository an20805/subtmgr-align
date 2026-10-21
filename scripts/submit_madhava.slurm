#!/bin/bash
#SBATCH --job-name=alignot_0
#SBATCH --account=sharada_proj2
#SBATCH --partition=cn2_madhava
#SBATCH --qos=madhava
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export TZ="Asia/Kolkata"

PROJECT_DIR="/users/student/pg/pg25/anshuman/subtmgr-align"
cd "${PROJECT_DIR}"
mkdir -p logs

bash scripts/run_alignot_iterative_fsc.sh
# bash scripts/run_eman2_baseline.sh

echo "======================================================================"
echo "Job finished successfully at $(date)"
echo "======================================================================"

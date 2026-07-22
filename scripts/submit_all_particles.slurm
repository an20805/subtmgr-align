#!/bin/bash
#SBATCH --job-name=empot_all_ptcl
#SBATCH --account=sharada_proj2
#SBATCH --partition=cn2_madhava
#SBATCH --qos=madhava
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Job started on $(hostname) at $(date)"

# 1. Move to project directory and activate the uv environment
cd /users/student/pg/pg25/anshuman/subtmgr-align
source EMPOT/empot/bin/activate

# 2. Run all-particles alignment
bash scripts/run_empot_all_particles.sh

echo "Job finished at $(date)"

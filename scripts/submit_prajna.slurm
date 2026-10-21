#!/bin/bash
# ==============================================================================
# SLURM Job Submission Script for PRAJNA (AI/ML) HPC Cluster - IIT Bombay
# System: Rocky 9, Slurm 23.11, NVIDIA L40S / A40 / A100 GPUs
# Reference: Prajna User Manual (Pages 10, 13-17)
# ==============================================================================
#SBATCH --job-name=alignot_prajna
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# ── Partitions & QoS (per Prajna User Manual Page 10, 15-17) ──────────────────
# Available GPU partitions on Prajna:
#   - l40 (Default, 7 nodes with 8x L40S ADA GPUs)  --> QoS: l40
#   - a40 (20 nodes with 4x A40 GPUs)               --> QoS: a40
#   - a100 (DGX A100, 9 nodes with 8x A100 GPUs)    --> QoS: a100
#
# IMPORTANT (Manual Page 16): Prajna requires matching QoS for the partition:
#SBATCH --partition=l40
#SBATCH --qos=l40

# Account: If you belong to a specific allocation/group, uncomment and set:
# #SBATCH --account=<your_group_or_pi_account>

# Logs: Slurm output and error logs (%x=job-name, %j=job-id)
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export TZ="Asia/Kolkata"

# Navigate to project working directory (Manual Page 14)
PROJECT_DIR="${SLURM_SUBMIT_DIR:-${HOME}/subtmgr-align}"
cd "${PROJECT_DIR}"
mkdir -p logs

echo "======================================================================"
echo "Starting AlignOT GPU on PRAJNA (AI/ML) HPC Cluster"
echo "Job ID        : ${SLURM_JOB_ID:-Manual}"
echo "Node          : $(hostname)"
echo "Partition/QoS : l40 / l40"
echo "CPUs Allocated: ${SLURM_CPUS_PER_TASK}"
echo "Start Time    : $(date)"
echo "======================================================================"

# Optional: Load Spack environment if needed (Manual Page 19)
if [[ -f "/lustre-flash/apps/spack/share/spack/setup-env.sh" ]]; then
    source /lustre-flash/apps/spack/share/spack/setup-env.sh || true
fi

# Run AlignOT Native GPU Gold-Standard Iterative Alignment
bash scripts/run_alignot_iterative_fsc.sh

echo "======================================================================"
echo "Job finished successfully at $(date)"
echo "======================================================================"

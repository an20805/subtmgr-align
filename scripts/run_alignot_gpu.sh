#!/usr/bin/env bash
# run_alignot_gpu.sh
# ------------------
# Launch the End-to-End PyTorch GPU accelerated AlignOT (Barycentric SGD) iterative alignment.
#
# Usage:
#   bash scripts/run_alignot_gpu.sh
#
# Outputs land in:
#   alignOT/outputs/gpu_runs/run_<timestamp>_<selection_mode>_gpu_barycentric/

set -euo pipefail
export TZ="Asia/Kolkata"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO}/alignOT/alignot_env/bin/python"
SCRIPT="${REPO}/alignOT/src/run_iterative_alignment_gpu.py"

CLUSTER_DIR="${REPO}/Data/0.1/Cluster_0"
OUTPUT_DIR="${REPO}/alignOT/outputs/gpu_runs"

# ── Alignment loop & Reference ────────────────────────────────────────────────
N_ROUNDS=3
INIT_REFERENCE=""   # Set to initial reference MRC ${REPO}/Data/0.mrc , or leave empty to compute streaming average

# ── Particle selection ────────────────────────────────────────────────────────
SELECTION_MODE=all      # all | first | random | even | odd
# N_SBTMS=40           # uncomment and set --n_sbtms below if using first/random

# ── Sampling ──────────────────────────────────────────────────────────────────
NUM_POINTS=500          # TRN points per volume
THRESHOLD=0.7           # Density threshold after inversion
LOWPASS_SIGMA=2.0       # Gaussian LP sigma (voxels) before sampling only; set to 0 to disable

# ── AlignOT SGD Hyperparameters ───────────────────────────────────────────────
REG=100                 # Sinkhorn entropic regularization
SGD_ITERS=500           # SGD iterations per particle
SGD_LR=0.001            # Learning rate

# ── Hardware & Workers ────────────────────────────────────────────────────────
NUM_WORKERS=16          # Parallel CPU workers for exact TRN sampling & OT
DEVICE=cuda             # 'cuda' or 'cpu' for exact 3D volume rigid rotation
SEED=42

mkdir -p "${OUTPUT_DIR}"

echo "======================================================================"
echo "Launching AlignOT High-Performance GPU-Accelerated Alignment"
echo "Python Interpreter: ${PYTHON}"
echo "Script            : ${SCRIPT}"
echo "Subtomograms Dir  : ${CLUSTER_DIR}"
echo "Output Base Dir   : ${OUTPUT_DIR}"
echo "CPU Workers       : ${NUM_WORKERS}"
echo "Device            : ${DEVICE}"
echo "======================================================================"

ARGS=(
    --cluster_dir         "${CLUSTER_DIR}"
    --output_base_dir     "${OUTPUT_DIR}"
    --selection_mode      "${SELECTION_MODE}"
    --n_rounds            "${N_ROUNDS}"
    --num_workers         "${NUM_WORKERS}"
    --num_points          "${NUM_POINTS}"
    --threshold           "${THRESHOLD}"
    --lowpass_sigma       "${LOWPASS_SIGMA}"
    --reg                 "${REG}"
    --sgd_iters           "${SGD_ITERS}"
    --sgd_lr              "${SGD_LR}"
    --device              "${DEVICE}"
    --seed                "${SEED}"
)

if [ -n "${N_SBTMS:-}" ]; then
    ARGS+=(--n_sbtms "${N_SBTMS}")
fi

if [ -n "${INIT_REFERENCE:-}" ] && [ -f "${INIT_REFERENCE}" ]; then
    ARGS+=(--init_reference "${INIT_REFERENCE}")
fi

"${PYTHON}" "${SCRIPT}" "${ARGS[@]}"

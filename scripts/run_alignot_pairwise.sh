#!/usr/bin/env bash
# run_alignot_pairwise.sh
# -----------------------
# Launch the Joint Pairwise Wasserstein SGD reference-free alignment.
#
# Objective: min_{q_i} sum_{i<j} W_eps(R(q_i)*Xi, R(q_j)*Xj)
#
# Usage:
#   bash scripts/run_alignot_pairwise.sh
#
# Outputs land in:
#   alignOT/outputs/pairwise_runs/run_<timestamp>_<mode><N>_pairwise_wass/

set -euo pipefail
export TZ="Asia/Kolkata"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}}")/.."; pwd)"
PYTHON="${REPO}/alignOT/alignot_env/bin/python"
SCRIPT="${REPO}/alignOT/src/run_pairwise_alignment_gpu.py"

# CLUSTER_DIR="${REPO}/Data/0.1/Cluster_0"
# OUTPUT_DIR="${REPO}/alignOT/outputs/pairwise_runs"

CLUSTER_DIR="${REPO}/Data/synthetic_clean_40"
OUTPUT_DIR="${REPO}/alignOT/outputs/synthetic_pairwise_runs"

# ── Particle selection ────────────────────────────────────────────────────────
SELECTION_MODE=first   # first | random | all | even | odd
N_SBTMS=40             # number of particles (used for first / random mode)

# ── Sampling ──────────────────────────────────────────────────────────────────
NUM_POINTS=500
THRESHOLD=0.7
LOWPASS_SIGMA=2.0

# ── Pairwise SGD Hyperparameters ──────────────────────────────────────────────
REG=100.0              # Sinkhorn entropic regularization
SGD_ITERS=500          # Number of SGD iterations
SGD_LR=0.001           # Learning rate
SINKHORN_ITERS=15      # Sinkhorn scaling iterations per SGD step
ANCHOR_IDX=0           # Particle index fixed at identity (gauge anchor)
LOG_EVERY=50           # Log mean W2 every N iterations

# ── Hardware ──────────────────────────────────────────────────────────────────
NUM_WORKERS=16
DEVICE=cuda
SEED=42

mkdir -p "${OUTPUT_DIR}"

echo "======================================================================"
echo " Joint Pairwise Wasserstein SGD — Reference-Free Alignment"
echo "======================================================================"
echo "  Script      : ${SCRIPT}"
echo "  Cluster dir : ${CLUSTER_DIR}"
echo "  Mode        : ${SELECTION_MODE} (N=${N_SBTMS})"
echo "  Pairs       : C(${N_SBTMS},2) = $((N_SBTMS*(N_SBTMS-1)/2)) pairs per step"
echo "  SGD iters   : ${SGD_ITERS}  |  lr: ${SGD_LR}  |  reg: ${REG}"
echo "  Anchor      : particle ${ANCHOR_IDX} (identity, fixed)"
echo "  Device      : ${DEVICE}"
echo "======================================================================"

"${PYTHON}" "${SCRIPT}" \
    --cluster_dir      "${CLUSTER_DIR}" \
    --output_base_dir  "${OUTPUT_DIR}" \
    --selection_mode   "${SELECTION_MODE}" \
    --n_sbtms          "${N_SBTMS}" \
    --num_workers      "${NUM_WORKERS}" \
    --num_points       "${NUM_POINTS}" \
    --threshold        "${THRESHOLD}" \
    --lowpass_sigma    "${LOWPASS_SIGMA}" \
    --reg              "${REG}" \
    --sgd_iters        "${SGD_ITERS}" \
    --sgd_lr           "${SGD_LR}" \
    --sinkhorn_iters   "${SINKHORN_ITERS}" \
    --anchor_idx       "${ANCHOR_IDX}" \
    --log_every        "${LOG_EVERY}" \
    --device           "${DEVICE}" \
    --seed             "${SEED}"

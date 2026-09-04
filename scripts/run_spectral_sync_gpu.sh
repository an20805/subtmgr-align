#!/usr/bin/env bash
# run_spectral_sync_gpu.sh
# ------------------------
# Launch the Two-Stage Rotation Synchronization pipeline:
#   Stage 1 — Decoupled Pairwise OT (GPU):
#             C(N,2) independent quaternions, one per pair, no gradient tug-of-war.
#   Stage 2 — Spectral Rotation Synchronization (<0.05 s):
#             3N×3N block matrix → top-3 eigenvectors → SO(3) rounding.
#   Stage 3 — Iterative Refinement (optional):
#             Reference-guided Barycentric SGD warm-started from spectral sync result.
#
# Usage:
#   bash scripts/run_spectral_sync_gpu.sh
#
# Outputs land in:
#   alignOT/outputs/spectral_sync_runs/run_<timestamp>_<mode><N>_spectral_sync/

set -euo pipefail
export TZ="Asia/Kolkata"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO}/alignOT/alignot_env/bin/python"
SCRIPT="${REPO}/alignOT/src/run_spectral_sync_alignment_gpu.py"

# ── Data ──────────────────────────────────────────────────────────────────────
# CLUSTER_DIR="${REPO}/Data/0.1/Cluster_0"
# OUTPUT_DIR="${REPO}/alignOT/outputs/spectral_sync_runs"

CLUSTER_DIR="${REPO}/Data/synthetic_clean_40"
OUTPUT_DIR="${REPO}/alignOT/outputs/synthetic_pairwise_runs"

GT_CSV=""


# ── Particle selection ────────────────────────────────────────────────────────
SELECTION_MODE=first   # first | random | all | even | odd
N_SBTMS=40             # used for first / random mode

# ── TRN Sampling ─────────────────────────────────────────────────────────────
NUM_POINTS=500         # TRN points per volume
THRESHOLD=0.7          # Density threshold after inversion
LOWPASS_SIGMA=2.0      # Gaussian LP sigma (voxels); set to 0 to disable
NUM_WORKERS=16         # Parallel CPU workers for TRN sampling

# ── Stage 1: Decoupled Pairwise OT ───────────────────────────────────────────
REG=100.0              # Sinkhorn entropic regularization epsilon
SGD_ITERS=500          # SGD iterations per pair
SGD_LR=0.001           # Riemannian SGD learning rate on S^3
SINKHORN_ITERS=15      # Sinkhorn inner iterations per SGD step
PAIR_BATCH_SIZE=200    # Pairs per GPU mini-batch (tune for your VRAM)
                       #   N=40  → 780 pairs,  batch=200 → ~4 mini-batches (~200 MB)
                       #   N=200 → 19900 pairs, batch=100 → ~200 mini-batches (~100 MB)

# ── Stage 3: Iterative Refinement (set to 0 to skip) ─────────────────────────
N_REFINE_ROUNDS=3      # Reference-guided GPU Barycentric SGD rounds after sync

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE=cuda            # cuda | cpu
SEED=42

# ── Derived info ──────────────────────────────────────────────────────────────
N_PAIRS=$((N_SBTMS * (N_SBTMS - 1) / 2))

mkdir -p "${OUTPUT_DIR}"

echo "======================================================================"
echo " Two-Stage Rotation Synchronization: Decoupled Pairwise OT + Spectral Sync"
echo "======================================================================"
echo "  Script        : ${SCRIPT}"
echo "  Data          : ${CLUSTER_DIR}"
echo "  Output        : ${OUTPUT_DIR}"
echo "  Particles     : ${SELECTION_MODE} (N=${N_SBTMS})"
echo "  Pairs         : C(${N_SBTMS},2) = ${N_PAIRS}"
echo "----------------------------------------------------------------------"
echo "  [Stage 1]  Decoupled Pairwise OT"
echo "    iters=${SGD_ITERS}  lr=${SGD_LR}  eps=${REG}  pair_batch=${PAIR_BATCH_SIZE}"
echo "  [Stage 2]  Spectral Rotation Sync  (target gap >> 1)"
echo "  [Stage 3]  Iterative Refinement  (${N_REFINE_ROUNDS} rounds)"
echo "----------------------------------------------------------------------"
echo "  Device        : ${DEVICE}"
echo "  Workers       : ${NUM_WORKERS}"
if [ -n "${GT_CSV}" ] && [ -f "${GT_CSV}" ]; then
    echo "  Ground truth  : ${GT_CSV}"
fi
echo "======================================================================"
echo ""

ARGS=(
    --cluster_dir       "${CLUSTER_DIR}"
    --output_base_dir   "${OUTPUT_DIR}"
    --selection_mode    "${SELECTION_MODE}"
    --n_sbtms           "${N_SBTMS}"
    --num_workers       "${NUM_WORKERS}"
    --num_points        "${NUM_POINTS}"
    --threshold         "${THRESHOLD}"
    --lowpass_sigma     "${LOWPASS_SIGMA}"
    --reg               "${REG}"
    --sgd_iters         "${SGD_ITERS}"
    --sgd_lr            "${SGD_LR}"
    --sinkhorn_iters    "${SINKHORN_ITERS}"
    --pair_batch_size   "${PAIR_BATCH_SIZE}"
    --n_refine_rounds   "${N_REFINE_ROUNDS}"
    --device            "${DEVICE}"
    --seed              "${SEED}"
)

# Append ground-truth CSV only if set and the file exists
if [ -n "${GT_CSV:-}" ] && [ -f "${GT_CSV}" ]; then
    ARGS+=(--gt_csv "${GT_CSV}")
fi

"${PYTHON}" "${SCRIPT}" "${ARGS[@]}"

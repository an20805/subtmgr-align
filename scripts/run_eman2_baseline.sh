#!/usr/bin/env bash
# =============================================================================
# run_eman2_baseline.sh — EMAN2 Gold-Standard Iterative Subtomogram Alignment
#                         (Reference-Free, Even/Odd Half-Set Protocol)
#
# Usage:
#   bash scripts/run_eman2_baseline.sh [DATA_DIR] [N_ROUNDS] [NUM_WORKERS]
#
# Examples:
#   bash scripts/run_eman2_baseline.sh Data/0.1/Cluster_0 5 10
#   bash scripts/run_eman2_baseline.sh Data/0.1/Cluster_0 5 16
# =============================================================================

set -euo pipefail

export TZ="Asia/Kolkata"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYTHON="${EMAN2_PYTHON:-${HOME}/miniforge3/envs/eman2/bin/python}"
SCRIPT="${REPO}/eman2_baseline/run_iterative_alignment_eman2.py"

DATA_DIR="${1:-${REPO}/Data/0.1/Cluster_0}"
N_ROUNDS="${2:-5}"
NUM_WORKERS="${3:-16}"
SHRINK=3
LOWPASS=0.15
APIX=4.0
LIMIT=-1
OUTPUT_DIR="${REPO}/eman2_baseline/runs"

mkdir -p "${OUTPUT_DIR}"

echo "======================================================================"
echo "EMAN2 Gold-Standard Iterative Subtomogram Alignment (Reference-Free)"
echo "======================================================================"
echo "Python Interpreter: ${PYTHON}"
echo "Script            : ${SCRIPT}"
echo "Data Directory    : ${DATA_DIR}"
echo "Initial Reference : NONE (raw streaming average per half-set)"
echo "Output Directory  : ${OUTPUT_DIR}"
echo "Rounds            : ${N_ROUNDS}"
echo "CPU Workers       : ${NUM_WORKERS} (physical cores)"
echo "Shrink Factor     : ${SHRINK} (alignment box: ~30^3)"
echo "Lowpass Cutoff    : ${LOWPASS}"
echo "Pixel Size        : ${APIX} Å"
echo "Subtomogram Limit : ${LIMIT}"
echo "======================================================================"

"${PYTHON}" "${SCRIPT}" \
    --cluster_dir    "${DATA_DIR}" \
    --output_base_dir "${OUTPUT_DIR}" \
    --n_rounds       "${N_ROUNDS}" \
    --num_workers    "${NUM_WORKERS}" \
    --n_sbtms        "${LIMIT}" \
    --shrink         "${SHRINK}" \
    --lowpass        "${LOWPASS}" \
    --apix           "${APIX}"

echo ""
echo "======================================================================"
echo "EMAN2 Pipeline Finished at $(date)"
echo "Outputs: ${OUTPUT_DIR}"
echo "======================================================================"

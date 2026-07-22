#!/usr/bin/env bash
# run_empot_all_particles.sh
#
# Iterative EMPOT alignment on ALL 200 subtomograms (no even/odd split).
# Produces a single evolving average reference per round.
# No FSC — just alignment quality via self-consistency of the average.
#
# Usage:
#   bash scripts/run_empot_all_particles.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO}/EMPOT/empot/bin/python3"
SCRIPT="${REPO}/EMPOT/src/run_iterative_alignment.py"

CLUSTER_DIR="${REPO}/Data/0.1/Cluster_0" 
OUTPUT_DIR="${REPO}/EMPOT/outputs/all_particles_runs"

# Hyperparameters
N_ROUNDS=5
THRESHOLD=0.75
NUM_POINTS=500
LOWPASS_SIGMA=2
EPS=10000
SEED=42

echo "================================================================"
echo " EMPOT iterative alignment — ALL 200 subtomograms"
echo "================================================================"
echo "  Cluster dir : ${CLUSTER_DIR}"
echo "  Init ref    : (none — computed from all 200 subtomograms)"
echo "  Rounds      : ${N_ROUNDS}"
echo "  Threshold   : ${THRESHOLD}"
echo "  Points      : ${NUM_POINTS}"
echo "  Lowpass σ   : ${LOWPASS_SIGMA} vox"
echo "  Output dir  : ${OUTPUT_DIR}"
echo "================================================================"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" "${SCRIPT}" \
    --cluster_dir      "${CLUSTER_DIR}" \
    --output_base_dir  "${OUTPUT_DIR}" \
    --selection_mode   "all" \
    --n_rounds         "${N_ROUNDS}" \
    --threshold        "${THRESHOLD}" \
    --num_points       "${NUM_POINTS}" \
    --lowpass_sigma    "${LOWPASS_SIGMA}" \
    --eps              "${EPS}" \
    --seed             "${SEED}"

echo ""
echo "Done. Results saved in ${OUTPUT_DIR}"

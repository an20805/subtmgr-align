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
export TZ="Asia/Kolkata"


REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO}/EMPOT/empot/bin/python3"
SCRIPT="${REPO}/EMPOT/src/run_iterative_alignment.py"

CLUSTER_DIR="${REPO}/Data/0.1/Cluster_0" 
OUTPUT_DIR="${REPO}/EMPOT/outputs/all_particles_runs"
INIT_REF=""   # ${REPO}/Data/0.mrc

# Hyperparameters
N_ROUNDS=3
THRESHOLD=0.75
NUM_POINTS=500
LOWPASS_SIGMA=2
EPS=2000
SEED=42
REJECTION_FRACTION=0.0   # drop worst 20% by UGW distance each round (0.0 = keep all)
SYMMETRY=C1              # multi-hypothesis alignment: try all 6 C6 rotations of ref per particle

echo "================================================================"
echo " EMPOT iterative alignment — ALL 200 subtomograms with true sructure "
echo "================================================================"
echo "  Cluster dir : ${CLUSTER_DIR}"
echo "  Init ref    : ${INIT_REF}"
echo "  Rounds      : ${N_ROUNDS}"
echo "  Threshold   : ${THRESHOLD}"
echo "  Points      : ${NUM_POINTS}"
echo "  Lowpass σ   : ${LOWPASS_SIGMA} vox"
echo "  Rejection   : ${REJECTION_FRACTION} (worst fraction dropped per round)"
echo "  Symmetry    : ${SYMMETRY} (multi-hypothesis alignment)"
echo "  Output dir  : ${OUTPUT_DIR}"
echo "================================================================"

    # --init_reference   "${INIT_REF}" \
    # --symmetry            "${SYMMETRY}" \
mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" "${SCRIPT}" \
    --cluster_dir      "${CLUSTER_DIR}" \
    --output_base_dir  "${OUTPUT_DIR}" \
    --selection_mode   "all" \
    --n_rounds         "${N_ROUNDS}" \
    --threshold        "${THRESHOLD}" \
    --num_points       "${NUM_POINTS}" \
    --lowpass_sigma       "${LOWPASS_SIGMA}" \
    --rejection_fraction  "${REJECTION_FRACTION}" \
    --eps                 "${EPS}" \
    --seed                "${SEED}"

echo ""
echo "Done. Results saved in ${OUTPUT_DIR}"

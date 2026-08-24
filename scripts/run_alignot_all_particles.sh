#!/usr/bin/env bash
# run_alignot_all_particles.sh
# ----------------------------
# Run the AlignOT iterative subtomogram alignment on all 200 particles.
#
# Usage:
#   bash scripts/run_alignot_all_particles.sh
#
# Outputs land in:
#   alignOT/outputs/all_particles_runs/run_<timestamp>_all/

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO}/alignOT/alignot_env/bin/python"
# SCRIPT="${REPO}/alignOT/src/run_iterative_alignment.py"
SCRIPT="${REPO}/alignOT/src/run_iterative_alignment_barycentric.py"

CLUSTER_DIR="${REPO}/Data/0.1/Cluster_0"
OUTPUT_DIR="${REPO}/alignOT/outputs/all_particles_runs"

# ── Sampling ──────────────────────────────────────────────────────────────────
NUM_POINTS=500          # TRN points per volume
THRESHOLD=0.7           # density threshold after inversion
LOWPASS_SIGMA=2.0       # Gaussian LP sigma (voxels) before sampling only; set to "" to disable

# ── AlignOT SGD (values from basic_tour.ipynb) ────────────────────────────────
REG=100
SGD_ITERS=500
SGD_LR=0.001

# ── Alignment loop ────────────────────────────────────────────────────────────
N_ROUNDS=1
INIT_REFERENCE="${REPO}/Data/0.mrc"

# ── Particle selection ────────────────────────────────────────────────────────
SELECTION_MODE=all      # all | first | random | even | odd
# N_SBTMS=100           # uncomment and set --n_sbtms below if using first/random
# --n_sbtms             "${N_SBTMS}" \

# ── Misc ──────────────────────────────────────────────────────────────────────
SEED=42

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" "${SCRIPT}" \
    --cluster_dir         "${CLUSTER_DIR}" \
    --output_base_dir     "${OUTPUT_DIR}" \
    --init_reference      "${INIT_REFERENCE}" \
    --selection_mode      "${SELECTION_MODE}" \
    --n_rounds            "${N_ROUNDS}" \
    --num_points          "${NUM_POINTS}" \
    --threshold           "${THRESHOLD}" \
    --lowpass_sigma       "${LOWPASS_SIGMA}" \
    --reg                 "${REG}" \
    --sgd_iters           "${SGD_ITERS}" \
    --sgd_lr              "${SGD_LR}" \
    --seed                "${SEED}"

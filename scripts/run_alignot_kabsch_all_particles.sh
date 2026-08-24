#!/usr/bin/env bash
# run_alignot_kabsch_all_particles.sh
# ------------------------------------
# Run Kabsch-based AlignOT iterative subtomogram alignment on all 200 particles.
# Uses Sinkhorn OT plan + Kabsch SVD instead of SGD (~150x faster per particle).
#
# Usage:
#   bash scripts/run_alignot_kabsch_all_particles.sh
#
# Outputs land in:
#   alignOT/outputs/kabsch_runs/run_<timestamp>_all_kabsch/

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO}/alignOT/alignot_env/bin/python"
SCRIPT="${REPO}/alignOT/src/run_iterative_alignment_kabsch.py"

CLUSTER_DIR="${REPO}/Data/0.1/Cluster_0"
OUTPUT_DIR="${REPO}/alignOT/outputs/kabsch_runs"

# ── TRN Sampling ──────────────────────────────────────────────────────────────
NUM_POINTS=500          # TRN points per volume
THRESHOLD=0.7           # density threshold after inversion
LOWPASS_SIGMA=2.0       # Gaussian LP sigma (voxels) before TRN only; 0 = disable

# ── Sinkhorn + Kabsch ─────────────────────────────────────────────────────────
REG=100               # Sinkhorn entropy reg (lower = sharper correspondences)
SINKHORN_ITERS=500     # max Sinkhorn iterations

# ── Alignment loop ────────────────────────────────────────────────────────────
N_ROUNDS=3
INIT_REFERENCE="${REPO}/Data/0.mrc"

# ── Particle selection ────────────────────────────────────────────────────────
SELECTION_MODE=even      # all | first | random | even | odd
# N_SBTMS=10            # uncomment + add --n_sbtms below for first/random mode

# ── Misc ──────────────────────────────────────────────────────────────────────
SEED=42

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" "${SCRIPT}" \
    --cluster_dir          "${CLUSTER_DIR}" \
    --output_base_dir      "${OUTPUT_DIR}" \
    --init_reference       "${INIT_REFERENCE}" \
    --selection_mode       "${SELECTION_MODE}" \
    --n_rounds             "${N_ROUNDS}" \
    --num_points           "${NUM_POINTS}" \
    --threshold            "${THRESHOLD}" \
    --lowpass_sigma        "${LOWPASS_SIGMA}" \
    --reg                  "${REG}" \
    --sinkhorn_iters       "${SINKHORN_ITERS}" \
    --seed                 "${SEED}"

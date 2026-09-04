#!/usr/bin/env bash
# run_coarse_to_fine_gpu.sh
# --------------------------
# Reference-Guided Multi-Level Coarse-to-Fine Search on SO(3).
#
# Supports Objective Metrics:
#   1. METRIC="l1" — Optimal Transport with L1 Manhattan ground cost on TRN points.
#   2. METRIC="w2" — Optimal Transport with L2² Squared Euclidean ground cost on TRN points.
#   3. METRIC="l2" — Optimal Transport with L2 Euclidean ground cost on TRN points.
#   4. METRIC="cc" — Normalized Cross-Correlation (NCC) on raw 3D volume densities.
#
# ROTATION SAMPLING:
#   Level 0    — ZYZ Euler angles with Haar measure (cos θ uniform, φ sin-weighted).
#                 Global exhaustive: evaluates ALL K₀ rotations × ALL N particles.
#   Levels 1-N — Rotation vectors (so(3) Lie algebra, exponential map).
#                 Local cone ball: samples a regular 3D grid in R³ within the cone,
#                 converted via Rodrigues formula. No Euler singularities.
#
# MULTI-LEVEL CASCADE: 60° → 20° → 7° → 2.5° → 1° (→ 0.5°)
#   L0: 48 global rotations,  ZYZ Euler
#   L1: 179 local/candidate,  rotation vector
#   L2: 203 local/candidate,  rotation vector
#   L3: 203 local/candidate,  rotation vector
#   L4: 203 local/candidate,  rotation vector
#
# Usage:
#   bash scripts/run_coarse_to_fine_gpu.sh
#
# Outputs:
#   alignOT/outputs/ctf_runs/run_<timestamp>_<mode><N>_ctf_<metric>_L<levels>/

set -euo pipefail
export TZ="Asia/Kolkata"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO}/alignOT/alignot_env/bin/python"
SCRIPT="${REPO}/alignOT/src/run_coarse_to_fine_gpu.py"

# ── Metric Selection ──────────────────────────────────────────────────────────
METRIC="w2"            # "l1" (Manhattan OT on TRN) | "w2" (Squared Euclidean OT on TRN) | "l2" (Euclidean OT on TRN) | "cc" (Cross-Correlation on Volumes)
COST_METRIC="l1"       # "l1", "l2", or "l2_sq"

# ── Data ──────────────────────────────────────────────────────────────────────
# Synthetic benchmark (with ground-truth evaluation):
# CLUSTER_DIR="${REPO}/Data/synthetic_clean_40"
# OUTPUT_DIR="${REPO}/alignOT/outputs/ctf_runs"
# GT_CSV="${REPO}/Data/synthetic_clean_40/ground_truth_rotations.csv"

# Real cryo-ET data (uncomment and comment the 3 lines above):
CLUSTER_DIR="${REPO}/Data/0.1/Cluster_0"
OUTPUT_DIR="${REPO}/alignOT/outputs/ctf_runs"
GT_CSV=""

# Optional initial reference MRC (leave empty to use streaming average):
INIT_REFERENCE=""
# INIT_REFERENCE="${REPO}/Data/0.mrc"

# ── Particle selection ────────────────────────────────────────────────────────
SELECTION_MODE=all     # all | first | random | even | odd
N_SBTMS=40             # used for first / random mode

# ── TRN Sampling (used when METRIC is OT: "w2" / "l1" / "l2") ─────────────────
NUM_POINTS=500
THRESHOLD=0.7
LOWPASS_SIGMA=2.0
NUM_WORKERS=16

# ── Multi-Level Cascade (comma-separated, no spaces) ──────────────────────────
# Steps: angular resolution of each level in degrees
CASCADE_STEPS="60,20,7,2.5,1"
# Sinkhorn iterations per level (only used when METRIC is OT: w2 / l1 / l2)
CASCADE_SINKHORN="5,8,10,12,15"

# Top-k candidates kept at each INTERMEDIATE level (last level always → top-1)
TOP_K=3

# Rotation batch size for local levels (memory knob):
FINE_ROT_BATCH=200

# ── Sinkhorn regularization (only used when METRIC is OT) ─────────────────────
REG=100.0

# ── Iterative rounds ──────────────────────────────────────────────────────────
N_ROUNDS=3             # reference improvement rounds

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE=cuda
SEED=42

# ── Banner ────────────────────────────────────────────────────────────────────
echo "======================================================================"
echo " Coarse-to-Fine Search on SO(3) — Reference-Guided STA"
echo "======================================================================"
echo "  Script       : ${SCRIPT}"
echo "  Data         : ${CLUSTER_DIR}"
echo "  Particles    : ${SELECTION_MODE} (N=${N_SBTMS})"
echo "  Metric       : ${METRIC^^} ($([ "${METRIC}" != "cc" ] && echo "Optimal Transport (${METRIC^^}) on TRN Points" || echo "Cross-Correlation on 3D Volumes"))"
echo "----------------------------------------------------------------------"
echo "  Cascade      : ${CASCADE_STEPS}"
if [ "${METRIC}" != "cc" ]; then
  echo "  Sinkhorn     : ${CASCADE_SINKHORN}  |  reg=${REG}  |  cost=${COST_METRIC}"
fi
echo "  Top-k        : ${TOP_K}"
echo "  Sampling     :"
echo "    L0 → ZYZ Euler triplets (cos-θ uniform, φ sin-weighted Haar measure)"
echo "    L1-LN → Rotation vectors (so(3) Lie algebra, Rodrigues exp map)"
echo "  Rounds       : ${N_ROUNDS}"
echo "----------------------------------------------------------------------"
echo "  Device       : ${DEVICE}  |  Workers: ${NUM_WORKERS}"
if [ -n "${GT_CSV:-}" ] && [ -f "${GT_CSV:-}" ]; then
    echo "  Ground truth : ${GT_CSV}"
fi
echo "======================================================================"
echo ""

mkdir -p "${OUTPUT_DIR}"

ARGS=(
    --cluster_dir       "${CLUSTER_DIR}"
    --output_base_dir   "${OUTPUT_DIR}"
    --metric            "${METRIC}"
    --cost_metric       "${COST_METRIC}"
    --selection_mode    "${SELECTION_MODE}"
    --n_sbtms           "${N_SBTMS}"
    --num_workers       "${NUM_WORKERS}"
    --num_points        "${NUM_POINTS}"
    --threshold         "${THRESHOLD}"
    --lowpass_sigma     "${LOWPASS_SIGMA}"
    --cascade_steps     "${CASCADE_STEPS}"
    --cascade_sinkhorn  "${CASCADE_SINKHORN}"
    --top_k             "${TOP_K}"
    --fine_rot_batch    "${FINE_ROT_BATCH}"
    --reg               "${REG}"
    --n_rounds          "${N_ROUNDS}"
    --device            "${DEVICE}"
    --seed              "${SEED}"
)

if [ -n "${INIT_REFERENCE:-}" ] && [ -f "${INIT_REFERENCE}" ]; then
    ARGS+=(--init_reference "${INIT_REFERENCE}")
fi
if [ -n "${GT_CSV:-}" ] && [ -f "${GT_CSV}" ]; then
    ARGS+=(--gt_csv "${GT_CSV}")
fi

"${PYTHON}" "${SCRIPT}" "${ARGS[@]}"

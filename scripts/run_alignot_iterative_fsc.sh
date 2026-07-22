#!/usr/bin/env bash
# run_alignot_iterative_fsc.sh
#
# Gold-standard iterative AlignOT alignment:
#   1. Run AlignOT iterative on even particles  → even references (rounds 0-N)
#   2. Run AlignOT iterative on odd  particles  → odd  references (rounds 0-N)
#   3. For each round: compute FSC(even_ref, odd_ref) and plot overall average slices
#
# Output: experiments/alignot_iterative_fsc/multi_round_evolution.png
#
set -euo pipefail

REPO=/home/anshu/subtmgr-align
PYTHON="${REPO}/EMPOT/empot/bin/python3"
SCRIPT_DIR="${REPO}/scripts"
SRC_DIR="${REPO}/alignOT/src"

# ── data ────────────────────────────────────────────────────────────────────────
CLUSTER_DIR="${REPO}/Subtomograms_40_deg_uniform/0.1/Cluster_0"
OUTPUT_BASE="${REPO}/EMPOT/outputs/alignot_runs"

# ── hyperparameters ──────────────────────────────────────────────────────────────
N_ROUNDS=3
THRESHOLD=0.7      # darkest ~1% of voxels = protein signal
NUM_POINTS=500
LR=1e-5            # quaternion SGD learning rate (matches AlignOT notebook)
MAX_ITER=200       # SGD iterations per subtomogram
REG=30             # Sinkhorn regularisation (matches AlignOT notebook)

# ── output ───────────────────────────────────────────────────────────────────────
FSC_OUT="${REPO}/experiments/alignot_iterative_fsc"
APIX=4.0

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== AlignOT Iterative FSC Pipeline ==="
log "N_ROUNDS : ${N_ROUNDS}   LR: ${LR}   MAX_ITER: ${MAX_ITER}   REG: ${REG}"
log "Threshold: ${THRESHOLD}  NUM_POINTS: ${NUM_POINTS}"

# ── Step 1: align even half-set ──────────────────────────────────────────────────
log "--- Step 1: AlignOT on EVEN half-set ---"
"${PYTHON}" "${SRC_DIR}/run_alignot_iterative.py" \
    --cluster_dir       "${CLUSTER_DIR}" \
    --output_base_dir   "${OUTPUT_BASE}" \
    --selection_mode    even \
    --n_rounds          "${N_ROUNDS}" \
    --threshold         "${THRESHOLD}" \
    --num_points        "${NUM_POINTS}" \
    --lr                "${LR}" \
    --max_iter          "${MAX_ITER}" \
    --reg               "${REG}"

EVEN_RUN=$(ls -td "${OUTPUT_BASE}"/run_*_even | head -1)
log "Even run dir: ${EVEN_RUN}"

# ── Step 2: align odd half-set ───────────────────────────────────────────────────
log "--- Step 2: AlignOT on ODD half-set ---"
"${PYTHON}" "${SRC_DIR}/run_alignot_iterative.py" \
    --cluster_dir       "${CLUSTER_DIR}" \
    --output_base_dir   "${OUTPUT_BASE}" \
    --selection_mode    odd \
    --n_rounds          "${N_ROUNDS}" \
    --threshold         "${THRESHOLD}" \
    --num_points        "${NUM_POINTS}" \
    --lr                "${LR}" \
    --max_iter          "${MAX_ITER}" \
    --reg               "${REG}"

ODD_RUN=$(ls -td "${OUTPUT_BASE}"/run_*_odd | head -1)
log "Odd run dir: ${ODD_RUN}"

# ── Step 3: FSC + evolution plot ─────────────────────────────────────────────────
log "--- Step 3: Computing FSC and generating evolution figure ---"
mkdir -p "${FSC_OUT}"

"${PYTHON}" "${SCRIPT_DIR}/plot_multi_round_fsc.py" \
    --even_dir  "${EVEN_RUN}" \
    --odd_dir   "${ODD_RUN}" \
    --n_rounds  "${N_ROUNDS}" \
    --output    "${FSC_OUT}" \
    --apix      "${APIX}"

log "Pipeline complete! Results saved in ${FSC_OUT}"
log "  → ${FSC_OUT}/multi_round_evolution.png"
log "  → ${FSC_OUT}/overall_average_final.mrc"

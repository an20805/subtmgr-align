#!/usr/bin/env bash
# =============================================================================
# run_empot_eman2ref_fsc.sh
#
# Evaluate EMPOT alignment quality when seeded with the EMAN2 half-set averages.
#
# Strategy (gold-standard half-set FSC):
#   1. Align even particles using EMPOT, seeded with the EMAN2 even-half average
#   2. Align odd  particles using EMPOT, seeded with the EMAN2 odd-half  average
#   3. Compute FSC between the two resulting EMPOT averages
#
# Both halves use a reference derived ONLY from their own half (EMAN2 even→EMPOT even,
# EMAN2 odd→EMPOT odd), so this is fully gold-standard with no cross-contamination.
#
# Usage (from repo root):
#   bash scripts/run_empot_eman2ref_fsc.sh
# =============================================================================
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

EMPOT_PYTHON=EMPOT/empot/bin/python3
ALIGN_SCRIPT=EMPOT/src/run_iterative_alignment.py
FSC_SCRIPT=scripts/compute_fsc.py

CLUSTER_DIR=Subtomograms_40_deg_uniform/0.1/Cluster_0
OUTPUT_BASE=EMPOT/outputs/iterative_runs

# EMAN2 half-set references (96³, will be auto-cropped to 91³ by alignment script)
EVEN_REF=eman2_baseline/baseline_avg_even.mrc
ODD_REF=eman2_baseline/baseline_avg_odd.mrc

FSC_OUT=experiments/empot_eman2ref_fsc
APIX=4.0

# EMPOT alignment hyperparameters
N_ROUNDS=1
THRESHOLD=0.7
NUM_POINTS=500
EPS=10000
# Low-pass sigma in voxels. Cutoff ≈ 2.35 * sigma * apix Å
# sigma=2 at 4 Å/px ≈ 19 Å — matches EMAN2's effective alignment resolution
LOWPASS_SIGMA=2

# ─────────────────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== EMPOT + EMAN2-Reference Half-Set FSC Pipeline ==="
log "EMPOT python  : ${EMPOT_PYTHON}"
log "N_ROUNDS      : ${N_ROUNDS}    EPS: ${EPS}    NUM_POINTS: ${NUM_POINTS}"
log "LOWPASS_SIGMA : ${LOWPASS_SIGMA} voxels  (≈ $(echo "2.35 * ${LOWPASS_SIGMA} * 4" | bc -l | xargs printf '%.0f') Å cutoff)"
log "Cluster dir   : ${CLUSTER_DIR}"
log "Even reference: ${EVEN_REF}"
log "Odd reference : ${ODD_REF}"
echo ""

# Sanity-check input files
for f in "${EVEN_REF}" "${ODD_REF}"; do
    if [[ ! -f "$f" ]]; then
        echo "[ERROR] Required file not found: $f"; exit 1
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — EMPOT alignment of EVEN particles (seeded with EMAN2 even reference)
# ─────────────────────────────────────────────────────────────────────────────

log "STEP 1/3 — Aligning EVEN particles (EMAN2 even-half reference as seed)..."
"${EMPOT_PYTHON}" "${ALIGN_SCRIPT}" \
    --cluster_dir     "${CLUSTER_DIR}" \
    --output_base_dir "${OUTPUT_BASE}" \
    --selection_mode  even \
    --n_rounds        "${N_ROUNDS}" \
    --threshold       "${THRESHOLD}" \
    --num_points      "${NUM_POINTS}" \
    --eps             "${EPS}" \
    --lowpass_sigma   "${LOWPASS_SIGMA}" \
    --init_reference  "${EVEN_REF}"

# Locate the output directory (most recently created run_*_even folder)
EVEN_RUN_DIR=$(ls -dt "${OUTPUT_BASE}"/run_*_even/ 2>/dev/null | head -1)
if [[ -z "${EVEN_RUN_DIR}" ]]; then
    echo "[ERROR] Could not find EMPOT even run output directory under ${OUTPUT_BASE}"; exit 1
fi
log "  Even run dir: ${EVEN_RUN_DIR}"

# Find the final reference MRC (highest round number)
EVEN_AVG=$(ls -v "${EVEN_RUN_DIR}"reference_round_??.mrc | grep -v "00" | tail -1)
if [[ -z "${EVEN_AVG}" ]]; then
    echo "[ERROR] No final reference MRC found in ${EVEN_RUN_DIR}"; exit 1
fi
log "  Even average: ${EVEN_AVG}"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EMPOT alignment of ODD particles (seeded with EMAN2 odd reference)
# ─────────────────────────────────────────────────────────────────────────────

log "STEP 2/3 — Aligning ODD particles (EMAN2 odd-half reference as seed)..."
"${EMPOT_PYTHON}" "${ALIGN_SCRIPT}" \
    --cluster_dir     "${CLUSTER_DIR}" \
    --output_base_dir "${OUTPUT_BASE}" \
    --selection_mode  odd \
    --n_rounds        "${N_ROUNDS}" \
    --threshold       "${THRESHOLD}" \
    --num_points      "${NUM_POINTS}" \
    --eps             "${EPS}" \
    --lowpass_sigma   "${LOWPASS_SIGMA}" \
    --init_reference  "${ODD_REF}"

# Locate the output directory (most recently created run_*_odd folder)
ODD_RUN_DIR=$(ls -dt "${OUTPUT_BASE}"/run_*_odd/ 2>/dev/null | head -1)
if [[ -z "${ODD_RUN_DIR}" ]]; then
    echo "[ERROR] Could not find EMPOT odd run output directory under ${OUTPUT_BASE}"; exit 1
fi
log "  Odd run dir: ${ODD_RUN_DIR}"

ODD_AVG=$(ls -v "${ODD_RUN_DIR}"reference_round_??.mrc | grep -v "00" | tail -1)
if [[ -z "${ODD_AVG}" ]]; then
    echo "[ERROR] No final reference MRC found in ${ODD_RUN_DIR}"; exit 1
fi
log "  Odd average: ${ODD_AVG}"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FSC between even and odd EMPOT averages
# ─────────────────────────────────────────────────────────────────────────────

log "STEP 3/3 — Computing FSC between even and odd EMPOT averages..."
mkdir -p "${FSC_OUT}"

"${EMPOT_PYTHON}" "${FSC_SCRIPT}" compare \
    --vol1   "${EVEN_AVG}" \
    --vol2   "${ODD_AVG}" \
    --label1 "EMPOT even (EMAN2 seed, ${N_ROUNDS}R)" \
    --label2 "EMPOT odd  (EMAN2 seed, ${N_ROUNDS}R)" \
    --output "${FSC_OUT}" \
    --apix   "${APIX}"

# ─────────────────────────────────────────────────────────────────────────────

echo ""
log "=== Done ==="
echo ""
echo "  Results:"
echo "    ${FSC_OUT}/fsc_compare.png      ← FSC curve"
echo "    ${FSC_OUT}/fsc_compare.csv      ← raw numbers"
echo "    ${FSC_OUT}/summary.txt          ← resolution in Å"
echo ""
cat "${FSC_OUT}/summary.txt"
echo ""
echo "  EMAN2 baseline for comparison: 23.3 Å (FSC=0.5)  /  19.7 Å (FSC=0.143)"

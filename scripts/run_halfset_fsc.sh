#!/usr/bin/env bash
# =============================================================================
# run_halfset_fsc.sh
#
# Full half-set FSC pipeline for one experiment:
#   1. Run alignment on even-numbered subtomograms  (1 round)
#   2. Run alignment on odd-numbered subtomograms   (1 round)
#   3. Compute raw half-set FSC (no alignment floor) — skipped if already done
#   4. Compute FSC between the two aligned halves   (true method resolution)
#
# Usage (from the repo root):
#   bash scripts/run_halfset_fsc.sh
#
# Before running:
#   - Set EXPERIMENT_NAME to describe this run (e.g. exp004_empot_lowpass)
#   - Set CLUSTER_DIR to the absolute path of your subtomogram folder
#   - Adjust NUM_POINTS, THRESHOLD, EPS if needed
#   - Make sure your venv python is correct in PYTHON
# =============================================================================
set -euo pipefail   # exit on any error, undefined variable, or pipe failure

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these before every experiment run
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT_NAME="exp004_empot_halfset"           # used to name output folders

CLUSTER_DIR="/home/anshu/subtmgr-align/Subtomograms_40_deg_uniform/0.1/Cluster_0"
BASELINE="eman2_baseline/baseline_reference_91.mrc"
APIX=4.0

# EMPOT alignment parameters
N_ROUNDS=1
THRESHOLD=0.1
NUM_POINTS=2000
EPS=10000
SEED=42

# Output locations (relative to repo root, which is where you should run this)
EVEN_OUT_BASE="EMPOT/outputs/halfset/${EXPERIMENT_NAME}_even"
ODD_OUT_BASE="EMPOT/outputs/halfset/${EXPERIMENT_NAME}_odd"
RAW_FSC_OUT="experiments/halfset_fsc_raw_snr01"    # computed once, reused
HALFSET_FSC_OUT="experiments/${EXPERIMENT_NAME}_fsc"

# Python executable and script paths
PYTHON="EMPOT/empot/bin/python3"
RUNNER="EMPOT/src/run_iterative_alignment.py"
FSC_SCRIPT="scripts/compute_fsc.py"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*"; }

check_file() {
    if [[ ! -f "$1" ]]; then
        echo "[ERROR] Required file not found: $1"
        exit 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# PRE-FLIGHT CHECKS
# ─────────────────────────────────────────────────────────────────────────────

log "=== Half-Set FSC Pipeline: ${EXPERIMENT_NAME} ==="
log "Cluster dir : ${CLUSTER_DIR}"
log "Baseline    : ${BASELINE}"
log "Pixel size  : ${APIX} Å"
echo ""

check_file "${PYTHON}"
check_file "${RUNNER}"
check_file "${FSC_SCRIPT}"
check_file "${BASELINE}"

if [[ ! -d "${CLUSTER_DIR}" ]]; then
    echo "[ERROR] Cluster directory not found: ${CLUSTER_DIR}"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Run alignment on even-numbered subtomograms
# ─────────────────────────────────────────────────────────────────────────────

log "STEP 1/4 — Aligning even-numbered subtomograms..."
mkdir -p "${EVEN_OUT_BASE}"

"${PYTHON}" "${RUNNER}" \
    --cluster_dir    "${CLUSTER_DIR}" \
    --output_base_dir "${EVEN_OUT_BASE}" \
    --selection_mode  even \
    --n_rounds        "${N_ROUNDS}" \
    --threshold       "${THRESHOLD}" \
    --num_points      "${NUM_POINTS}" \
    --eps             "${EPS}" \
    --seed            "${SEED}"

# Find the run directory that was just created (most recent)
EVEN_RUN_DIR=$(ls -td "${EVEN_OUT_BASE}"/run_* 2>/dev/null | head -1)
EVEN_REF="${EVEN_RUN_DIR}/reference_round_01.mrc"

if [[ ! -f "${EVEN_REF}" ]]; then
    echo "[ERROR] Even alignment did not produce ${EVEN_REF}"
    exit 1
fi
log "  Even reference: ${EVEN_REF}"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Run alignment on odd-numbered subtomograms
# ─────────────────────────────────────────────────────────────────────────────

log "STEP 2/4 — Aligning odd-numbered subtomograms..."
mkdir -p "${ODD_OUT_BASE}"

"${PYTHON}" "${RUNNER}" \
    --cluster_dir     "${CLUSTER_DIR}" \
    --output_base_dir "${ODD_OUT_BASE}" \
    --selection_mode  odd \
    --n_rounds        "${N_ROUNDS}" \
    --threshold       "${THRESHOLD}" \
    --num_points      "${NUM_POINTS}" \
    --eps             "${EPS}" \
    --seed            "${SEED}"

ODD_RUN_DIR=$(ls -td "${ODD_OUT_BASE}"/run_* 2>/dev/null | head -1)
ODD_REF="${ODD_RUN_DIR}/reference_round_01.mrc"

if [[ ! -f "${ODD_REF}" ]]; then
    echo "[ERROR] Odd alignment did not produce ${ODD_REF}"
    exit 1
fi
log "  Odd reference: ${ODD_REF}"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Raw half-set FSC (no alignment floor) — skip if already computed
# ─────────────────────────────────────────────────────────────────────────────

if [[ -f "${RAW_FSC_OUT}/fsc_halfset_raw.png" ]]; then
    log "STEP 3/4 — Raw FSC already computed at ${RAW_FSC_OUT}, skipping."
else
    log "STEP 3/4 — Computing raw half-set FSC (no alignment floor)..."
    mkdir -p "${RAW_FSC_OUT}"
    "${PYTHON}" "${FSC_SCRIPT}" halfset \
        --cluster_dir "${CLUSTER_DIR}" \
        --output      "${RAW_FSC_OUT}" \
        --apix        "${APIX}"
    log "  Raw FSC saved to: ${RAW_FSC_OUT}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — FSC between even-aligned and odd-aligned halves
# ─────────────────────────────────────────────────────────────────────────────

log "STEP 4/4 — Computing aligned half-set FSC..."
mkdir -p "${HALFSET_FSC_OUT}"

"${PYTHON}" "${FSC_SCRIPT}" compare \
    --vol1   "${EVEN_REF}" \
    --vol2   "${ODD_REF}" \
    --label1 "even half (aligned)" \
    --label2 "odd half (aligned)" \
    --output "${HALFSET_FSC_OUT}" \
    --apix   "${APIX}"

# ─────────────────────────────────────────────────────────────────────────────
# DONE — Print summary of all output locations
# ─────────────────────────────────────────────────────────────────────────────

echo ""
log "=== Pipeline complete ==="
echo ""
echo "  Outputs:"
echo "    Even alignment : ${EVEN_RUN_DIR}"
echo "    Odd alignment  : ${ODD_RUN_DIR}"
echo "    Raw FSC (floor): ${RAW_FSC_OUT}"
echo "    Aligned FSC    : ${HALFSET_FSC_OUT}"
echo ""
echo "  Key files to check:"
echo "    ${HALFSET_FSC_OUT}/fsc_compare.png     ← your method's resolution"
echo "    ${HALFSET_FSC_OUT}/fsc_compare.csv     ← raw numbers"
echo "    ${HALFSET_FSC_OUT}/summary.txt         ← resolution in Å"
echo "    ${RAW_FSC_OUT}/fsc_halfset_raw.png     ← floor (no alignment)"
echo ""

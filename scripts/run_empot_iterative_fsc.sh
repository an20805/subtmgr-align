#!/bin/bash
set -e

# ─────────────────────────────────────────────────────────────────────────────
# EMPOT Iterative Alignment (From Scratch) + Multi-Round FSC
# ─────────────────────────────────────────────────────────────────────────────
# 1. Runs EMPOT iterative alignment on the EVEN half-set for N rounds.
# 2. Runs EMPOT iterative alignment on the ODD half-set for N rounds.
# 3. Runs a python script to compute FSC for each round and plot them.
# 4. Generates the final overall average (even+odd)/2.

# Environment
EMPOT_PYTHON="EMPOT/empot/bin/python3"
ALIGN_SCRIPT="EMPOT/src/run_iterative_alignment.py"
MULTI_FSC_SCRIPT="scripts/plot_multi_round_fsc.py"

# Data
CLUSTER_DIR="Subtomograms_40_deg_uniform/0.1/Cluster_0"
OUTPUT_BASE="EMPOT/outputs/iterative_runs"
APIX=4.0

# EMPOT alignment hyperparameters
N_ROUNDS=6
THRESHOLD=0.7     # FIXED: Only sampling darkest 30% of voxels
NUM_POINTS=500
EPS=10000
LOWPASS_SIGMA=2   # Apply LP filter only before point-cloud sampling

# ─────────────────────────────────────────────────────────────────────────────

log() { echo -e "\n\033[1;34m[INFO]\033[0m $1"; }

log "=== EMPOT Iterative Alignment (From Scratch) ==="
log "N_ROUNDS      : ${N_ROUNDS}"
log "THRESHOLD     : ${THRESHOLD}"
log "NUM_POINTS    : ${NUM_POINTS}"
log "EPS           : ${EPS}"
log "LOWPASS_SIGMA : ${LOWPASS_SIGMA}"

# --- EVEN HALF ---
log "Running EMPOT on EVEN half-set..."
"${EMPOT_PYTHON}" "${ALIGN_SCRIPT}" \
    --cluster_dir     "${CLUSTER_DIR}" \
    --output_base_dir "${OUTPUT_BASE}" \
    --selection_mode  even \
    --n_rounds        "${N_ROUNDS}" \
    --threshold       "${THRESHOLD}" \
    --num_points      "${NUM_POINTS}" \
    --eps             "${EPS}" \
    --lowpass_sigma   "${LOWPASS_SIGMA}"
# Note: no --init_reference! Starts from raw even average.

# Find the newly created even run directory
RUN_DIR_EVEN=$(ls -dt "${OUTPUT_BASE}"/run_*_even | head -n1)
log "Even output directory: ${RUN_DIR_EVEN}"


# --- ODD HALF ---
log "Running EMPOT on ODD half-set..."
"${EMPOT_PYTHON}" "${ALIGN_SCRIPT}" \
    --cluster_dir     "${CLUSTER_DIR}" \
    --output_base_dir "${OUTPUT_BASE}" \
    --selection_mode  odd \
    --n_rounds        "${N_ROUNDS}" \
    --threshold       "${THRESHOLD}" \
    --num_points      "${NUM_POINTS}" \
    --eps             "${EPS}" \
    --lowpass_sigma   "${LOWPASS_SIGMA}"

# Find the newly created odd run directory
RUN_DIR_ODD=$(ls -dt "${OUTPUT_BASE}"/run_*_odd | head -n1)
log "Odd output directory: ${RUN_DIR_ODD}"


# --- COMPUTE MULTI-ROUND FSC & OVERALL AVERAGE ---
log "Computing round-by-round FSC and final overall average..."
FSC_OUT="experiments/empot_iterative_fsc"
mkdir -p "${FSC_OUT}"

"${EMPOT_PYTHON}" "${MULTI_FSC_SCRIPT}" \
    --even_dir "${RUN_DIR_EVEN}" \
    --odd_dir  "${RUN_DIR_ODD}" \
    --n_rounds "${N_ROUNDS}" \
    --output   "${FSC_OUT}" \
    --apix     "${APIX}"

log "Pipeline complete! Results saved in ${FSC_OUT}"
log "  → ${FSC_OUT}/multi_round_evolution.png"
log "  → ${FSC_OUT}/overall_average_final.mrc"

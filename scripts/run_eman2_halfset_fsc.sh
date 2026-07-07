#!/usr/bin/env bash
# =============================================================================
# run_eman2_halfset_fsc.sh
#
# Compute the gold-standard half-set FSC for the EMAN2 baseline:
#   1. Run EMAN2 STA on even-numbered particles
#   2. Run EMAN2 STA on odd-numbered particles
#   3. Compute FSC between the two final averages
#
# Usage (from repo root):
#   bash scripts/run_eman2_halfset_fsc.sh
#
# Runtime: ~30 min per half (~60 min total, 14 cores)
# =============================================================================
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

EMAN2_PYTHON=~/miniforge3/envs/eman2/bin/python
FSC_PYTHON=EMPOT/empot/bin/python3
FSC_SCRIPT=scripts/compute_fsc.py

EMAN2_SCRIPT=eman2_baseline/sta_average_halfset.py
EVEN_MRC=eman2_baseline/baseline_avg_even.mrc
ODD_MRC=eman2_baseline/baseline_avg_odd.mrc

FSC_OUT=experiments/eman2_halfset_fsc
APIX=4.0

NITER=2      # number of alignment rounds (same as full baseline run)
SHRINK=3     # downsampling factor for alignment (same as full baseline run)
NPROC=14     # CPU cores

# ─────────────────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== EMAN2 Half-Set FSC Pipeline ==="
log "EMAN2 python : ${EMAN2_PYTHON}"
log "Niter        : ${NITER}   Shrink: ${SHRINK}   Nproc: ${NPROC}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — EMAN2 alignment on even-numbered particles (0,2,4,...)
# ─────────────────────────────────────────────────────────────────────────────

log "STEP 1/3 — Aligning even-numbered particles (this takes ~30 min)..."
"${EMAN2_PYTHON}" "${EMAN2_SCRIPT}" even "${NITER}" "${SHRINK}" "${NPROC}"

if [[ ! -f "${EVEN_MRC}" ]]; then
    echo "[ERROR] Even half average not found: ${EVEN_MRC}"; exit 1
fi
log "  Even average: ${EVEN_MRC}"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EMAN2 alignment on odd-numbered particles (1,3,5,...)
# ─────────────────────────────────────────────────────────────────────────────

log "STEP 2/3 — Aligning odd-numbered particles (this takes ~30 min)..."
"${EMAN2_PYTHON}" "${EMAN2_SCRIPT}" odd "${NITER}" "${SHRINK}" "${NPROC}"

if [[ ! -f "${ODD_MRC}" ]]; then
    echo "[ERROR] Odd half average not found: ${ODD_MRC}"; exit 1
fi
log "  Odd average: ${ODD_MRC}"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FSC between the two half-set averages
# ─────────────────────────────────────────────────────────────────────────────

log "STEP 3/3 — Computing FSC between even and odd half averages..."
mkdir -p "${FSC_OUT}"

"${FSC_PYTHON}" "${FSC_SCRIPT}" compare \
    --vol1   "${EVEN_MRC}" \
    --vol2   "${ODD_MRC}" \
    --label1 "EMAN2 even half (${NITER} rounds)" \
    --label2 "EMAN2 odd half (${NITER} rounds)" \
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

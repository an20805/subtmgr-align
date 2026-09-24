set -euo pipefail
export TZ="Asia/Kolkata"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYTHON="${REPO}/alignOT/alignot_env/bin/python"
SCRIPT="${REPO}/alignOT/src/run_iterative_alignment_gpu_halfset.py"

# ── Configuration & Defaults (matching run_alignot_gpu.sh) ────────────────────
DATA_DIR="${1:-${REPO}/Data/0.1/subtomograms/0}"
OUTPUT_DIR="${REPO}/alignOT/outputs/gpu_runs"
N_ROUNDS="${2:-7}"
NUM_WORKERS="${3:-16}"
LIMIT="${4:--1}"
INIT_REFERENCE="${5:-}"    # Set path to initial reference, or leave empty for reference-free raw average

# ── AlignOT GPU Parameters ───────────────────────────────────────────────────
TARGET_MODE="base"         # "base" (argmax) or "barycentric" (1-NN)
COST_METRIC="l2_sq"        # "l2_sq", "l1", or "l2"
REG=100.0                  # Sinkhorn entropic regularisation
SINKHORN_ITERS=50          # Sinkhorn scaling iterations
SGD_ITERS=1000              # SGD iterations per particle
SGD_LR=0.001               # SGD learning rate
NUM_POINTS=500             # TRN points per volume
THRESHOLD=0.7              # Density threshold after inversion
LOWPASS_SIGMA=2.0          # Gaussian LP sigma (voxels) before sampling only
APIX=4.0                   # Voxel size in Angstroms
DEVICE="cuda"              # GPU device
SEED=42

# ── Wedge-Mask Averaging (optional) ─────────────────────────────────────────
# Leave empty for the original real-space sum (default / backward-compatible).
# Set to the path of wedge_mask.mrc to enable Fourier-space wedge-masked averaging.
WEDGE_MASK="${REPO}/Data/wedge_mask.mrc"              # e.g. "${REPO}/Data/wedge_mask.mrc"

mkdir -p "${OUTPUT_DIR}"

echo "======================================================================"
echo "Starting AlignOT Native GPU Gold-Standard Iterative Alignment"
echo "======================================================================"
echo "Python Interpreter: ${PYTHON}"
echo "Script            : ${SCRIPT}"
echo "Data Directory    : ${DATA_DIR}"
echo "Output Directory  : ${OUTPUT_DIR}"
echo "Rounds            : ${N_ROUNDS}"
echo "Target Mode       : ${TARGET_MODE} (argmax)"
echo "Cost Metric       : ${COST_METRIC}"
echo "Sinkhorn Iters    : ${SINKHORN_ITERS} (reg=${REG})"
echo "SGD Iters / LR    : ${SGD_ITERS} / ${SGD_LR}"
echo "TRN Points / LP   : ${NUM_POINTS} / σ=${LOWPASS_SIGMA} vox"
echo "Device            : ${DEVICE}"
echo "TRN CPU Workers   : ${NUM_WORKERS}"
echo "Pixel Size        : ${APIX} Å"
echo "Subtomogram Limit : ${LIMIT}"
echo "Initial Reference : ${INIT_REFERENCE:-[Reference-Free Unaligned Seed]}"
echo "Wedge Mask        : ${WEDGE_MASK:-[None — real-space sum]}"
echo "======================================================================"

CMD=(
    "$PYTHON" "$SCRIPT"
    --cluster_dir       "$DATA_DIR"
    --output_base_dir   "$OUTPUT_DIR"
    --n_rounds          "$N_ROUNDS"
    --num_workers       "$NUM_WORKERS"
    --target_mode       "$TARGET_MODE"
    --cost_metric       "$COST_METRIC"
    --reg               "$REG"
    --sinkhorn_iters    "$SINKHORN_ITERS"
    --sgd_iters         "$SGD_ITERS"
    --sgd_lr            "$SGD_LR"
    --num_points        "$NUM_POINTS"
    --threshold         "$THRESHOLD"
    --lowpass_sigma     "$LOWPASS_SIGMA"
    --apix              "$APIX"
    --device            "$DEVICE"
    --seed              "$SEED"
    --n_sbtms           "$LIMIT"
)

if [[ -n "$INIT_REFERENCE" ]]; then
    CMD+=(--init_reference "$INIT_REFERENCE")
fi

if [[ -n "$WEDGE_MASK" ]]; then
    CMD+=(--wedge_mask "$WEDGE_MASK")
fi

"${CMD[@]}"

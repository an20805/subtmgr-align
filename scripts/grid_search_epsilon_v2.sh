#!/usr/bin/env bash
# grid_search_epsilon_v2.sh
# -------------------------
# Grid search over Sinkhorn regularisation (epsilon) for AlignOT GPU.
#
# For each epsilon in EPSILONS:
#   1. Runs 3-round iterative GPU alignment with sinkhorn_iters=100.
#   2. Extracts final round W2 from round_metrics_summary.csv.
#   3. Computes native 3D box CC using scripts/compute_standard_metrics.py.
#
# Prints a unified summary table at the end.
#
# Usage:
#   bash scripts/grid_search_epsilon_v2.sh
#
# Outputs land in:
#   alignOT/outputs/grid_search_epsilon_v2/eps_<value>/

set -euo pipefail
export TZ="Asia/Kolkata"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
PYTHON="${REPO}/alignOT/alignot_env/bin/python"
ALIGN_SCRIPT="${REPO}/alignOT/src/run_iterative_alignment_gpu.py"
METRICS_SCRIPT="${REPO}/scripts/compute_standard_metrics.py"

# ── Inputs ────────────────────────────────────────────────────────────────────
CLUSTER_DIR="${REPO}/Data/0.1/Cluster_0"
OUTPUT_BASE="${REPO}/alignOT/outputs/grid_search_epsilon_v2"
SELECTION_MODE="all"
N_ROUNDS=3
NUM_WORKERS=16
DEVICE="cuda"
NUM_POINTS=500
THRESHOLD=0.7
LOWPASS_SIGMA=2.0
SGD_ITERS=500
SGD_LR=0.001
SINKHORN_ITERS=100   # Fixed for this search
SEED=42

# ── Grid ──────────────────────────────────────────────────────────────────────
# EPSILONS=(3 5 8 10 50 100 200)
EPSILONS=(25)

mkdir -p "${OUTPUT_BASE}"

echo "======================================================================"
echo " AlignOT GPU Epsilon Grid Search"
echo " Epsilon values : ${EPSILONS[*]}"
echo " Sinkhorn iters : ${SINKHORN_ITERS}"
echo " SGD iters      : ${SGD_ITERS}  LR: ${SGD_LR}"
echo " Output base    : ${OUTPUT_BASE}"
echo "======================================================================"

# Arrays to collect results for the final summary table
declare -a RESULT_EPS
declare -a RESULT_W2
declare -a RESULT_MEAN_CC
declare -a RESULT_CC_RANGE

# ── Run alignment for each epsilon ────────────────────────────────────────────
for EPS in "${EPSILONS[@]}"; do
    EPS_DIR="${OUTPUT_BASE}/eps_${EPS}"
    mkdir -p "${EPS_DIR}"

    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo " Running AlignOT GPU  ε = ${EPS}  (sinkhorn_iters=${SINKHORN_ITERS})"
    echo "────────────────────────────────────────────────────────────────────"

    "${PYTHON}" "${ALIGN_SCRIPT}" \
        --cluster_dir       "${CLUSTER_DIR}" \
        --output_base_dir   "${EPS_DIR}" \
        --selection_mode    "${SELECTION_MODE}" \
        --n_rounds          "${N_ROUNDS}" \
        --num_workers       "${NUM_WORKERS}" \
        --device            "${DEVICE}" \
        --num_points        "${NUM_POINTS}" \
        --threshold         "${THRESHOLD}" \
        --lowpass_sigma     "${LOWPASS_SIGMA}" \
        --reg               "${EPS}" \
        --sinkhorn_iters    "${SINKHORN_ITERS}" \
        --sgd_iters         "${SGD_ITERS}" \
        --sgd_lr            "${SGD_LR}" \
        --seed              "${SEED}"

    # ── Find the run directory created by this invocation ───────────────────
    RUN_DIR="$(ls -td "${EPS_DIR}"/run_* 2>/dev/null | head -1)"

    if [ -z "${RUN_DIR}" ]; then
        echo "[ERROR] No run directory found under ${EPS_DIR}. Skipping."
        continue
    fi

    REF_PATH="${RUN_DIR}/reference_round_03.mrc"
    QUATS_PATH="${RUN_DIR}/quaternions.csv"
    if [ ! -f "${QUATS_PATH}" ] && [ -f "${RUN_DIR}/metrics_particles_round_03.csv" ]; then
        QUATS_PATH="${RUN_DIR}/metrics_particles_round_03.csv"
    fi
    METRICS_CSV="${RUN_DIR}/round_metrics_summary.csv"

    if [ ! -f "${REF_PATH}" ]; then
        echo "[ERROR] Reference round 03 not found: ${REF_PATH}. Skipping."
        continue
    fi

    # ── Extract final round W2 from round_metrics_summary.csv ───────────────
    FINAL_W2="$("${PYTHON}" -c "
import pandas as pd
df = pd.read_csv('${METRICS_CSV}')
print(f\"{df.iloc[-1]['mean_wass_dist']:.2f}\")
")"

    echo "  ε=${EPS} | Final W2 (from log, own epsilon) = ${FINAL_W2}"

    # ── Compute native 3D box CC using compute_standard_metrics.py ──────────
    CC_OUTPUT="$("${PYTHON}" "${METRICS_SCRIPT}" \
        --ref         "${REF_PATH}" \
        --quats       "${QUATS_PATH}" \
        --cluster_dir "${CLUSTER_DIR}" \
        --reg         100.0 \
        --device      "${DEVICE}" 2>&1)"

    MEAN_CC="$(echo "${CC_OUTPUT}"    | grep 'Mean CC'   | awk '{print $NF}')"
    CC_RANGE="$(echo "${CC_OUTPUT}"   | grep 'CC Range'  | sed 's/.*CC Range.*: //')"

    echo "  ε=${EPS} | Mean CC (Native 3D Box) = ${MEAN_CC}  Range: ${CC_RANGE}"

    # Store results
    RESULT_EPS+=("${EPS}")
    RESULT_W2+=("${FINAL_W2}")
    RESULT_MEAN_CC+=("${MEAN_CC}")
    RESULT_CC_RANGE+=("${CC_RANGE}")
done

# ── Final Summary Table ───────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo " EPSILON GRID SEARCH SUMMARY  (sinkhorn_iters=${SINKHORN_ITERS})"
echo " W2  : from each run's own round_metrics_summary.csv (own epsilon)"
echo " CC  : native 3D continuous box cross-correlation (reg=100 for W2 eval)"
echo "======================================================================"
printf "%-10s | %-20s | %-12s | %s\n" \
       "Epsilon" "Mean W2 (own eps)" "Mean CC" "CC Range [Min, Max]"
printf "%-10s-+-%-20s-+-%-12s-+-%-30s\n" \
       "----------" "--------------------" "------------" "------------------------------"
for i in "${!RESULT_EPS[@]}"; do
    printf "%-10s | %-20s | %-12s | %s\n" \
        "${RESULT_EPS[$i]}" \
        "${RESULT_W2[$i]}" \
        "${RESULT_MEAN_CC[$i]}" \
        "${RESULT_CC_RANGE[$i]}"
done
echo "======================================================================"

# ── Save summary CSV ─────────────────────────────────────────────────────────
SUMMARY_CSV="${OUTPUT_BASE}/grid_search_summary.csv"
"${PYTHON}" - \
    "${OUTPUT_BASE}" \
    "${CLUSTER_DIR}" \
    "${SINKHORN_ITERS}" \
    "${DEVICE}" \
    "${METRICS_SCRIPT}" \
<< 'PYEOF'
import sys, os, re, glob, subprocess
from pathlib import Path
import pandas as pd

base         = Path(sys.argv[1])
cluster_dir  = sys.argv[2]
sink_iters   = sys.argv[3]
device       = sys.argv[4]
metrics_scr  = sys.argv[5]
python       = sys.executable

epsilons = [float(x) for x in sorted([p.name.replace("eps_", "") for p in base.glob("eps_*") if p.is_dir()], key=lambda s: float(s))]
rows = []

for eps_val in epsilons:
    eps_s = f"{eps_val:g}"
    eps_dir = base / f"eps_{eps_s}"
    run_dirs = sorted(eps_dir.glob("run_*")) if eps_dir.is_dir() else []
    if not run_dirs:
        continue
    run_dir = run_dirs[-1]
    metrics_csv = run_dir / "round_metrics_summary.csv"
    quats_csv   = run_dir / "quaternions.csv"
    if not quats_csv.is_file():
        quats_csv = run_dir / "metrics_particles_round_03.csv"
    ref_mrc     = run_dir / "reference_round_03.mrc"
    if not (metrics_csv.is_file() and ref_mrc.is_file() and quats_csv.is_file()):
        continue

    df = pd.read_csv(metrics_csv)
    final_w2 = float(df.iloc[-1]["mean_wass_dist"])

    result = subprocess.run(
        [python, metrics_scr,
         "--ref",         str(ref_mrc),
         "--quats",       str(quats_csv),
         "--cluster_dir", cluster_dir,
         "--reg",         "100.0",
         "--device",      device],
        capture_output=True, text=True
    )
    mean_cc  = float(re.search(r"Mean CC\s*:\s*([-0-9.]+)", result.stdout).group(1))
    cc_range = re.search(r"CC Range\s*:\s*(\[.+\])", result.stdout).group(1)

    rows.append({
        "epsilon":           eps_val,
        "sinkhorn_iters":    int(sink_iters),
        "final_mean_w2":     final_w2,
        "mean_cc":           mean_cc,
        "cc_range":          cc_range,
        "run_dir":           str(run_dir),
    })

out = pd.DataFrame(rows)
out_path = base / "grid_search_summary.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved summary CSV to: {out_path}")
PYEOF

echo ""
echo "All grid search runs complete."
echo "Outputs: ${OUTPUT_BASE}"

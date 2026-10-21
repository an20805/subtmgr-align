#!/bin/bash
# ==============================================================================
# CONFIGURATION
# ==============================================================================
USERNAME="anshuman"
SERVER="${USERNAME}@kalpana.minds.iitb.ac.in"
REMOTE_DIR="/users/student/pg/pg25/anshuman/subtmgr-align"
# ==============================================================================

echo "Syncing results FROM ${SERVER} to your laptop..."

# rsync -avz "${SERVER}:${REMOTE_DIR}/EMPOT/outputs/" ./EMPOT/outputs/  

# rsync -avz "${SERVER}:${REMOTE_DIR}/logs/" ./logs/

# rsync -avz "${SERVER}:${REMOTE_DIR}/alignOT/outputs/gpu_runs/" ./alignOT/outputs/gpu_runs/

rsync -avz --exclude='*.hdf' "${SERVER}:${REMOTE_DIR}/eman2_baseline/runs/" ./eman2_baseline/runs/

# rsync -avz --exclude='*.hdf' "${SERVER}:${REMOTE_DIR}/eman2_baseline/runs/run_20260928_130243_snr0.001_eman2_5rounds_16cpu/" ./eman2_baseline/runs/run_20260928_130243_snr0.001_eman2_5rounds_16cpu

echo "------------------------------------------------------"
echo "✅ Download complete! The results are now on your laptop."
echo "------------------------------------------------------"

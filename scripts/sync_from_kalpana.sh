#!/bin/bash
# ==============================================================================
# CONFIGURATION
# ==============================================================================
USERNAME="anshuman"
SERVER="${USERNAME}@kalpana.minds.iitb.ac.in"
REMOTE_DIR="/users/student/pg/pg25/anshuman/subtmgr-align"
# ==============================================================================

echo "Syncing results FROM ${SERVER} to your laptop..."

# 1. Sync the EMPOT outputs (which includes the all_particles_runs)
rsync -avz "${SERVER}:${REMOTE_DIR}/EMPOT/outputs/" ./EMPOT/outputs/

# 2. Sync the Slurm logs (so we can see the exact terminal output/errors)
rsync -avz "${SERVER}:${REMOTE_DIR}/logs/" ./logs/

# 3. Sync the experiments folder (if you ran the FSC scripts)
rsync -avz "${SERVER}:${REMOTE_DIR}/experiments/" ./experiments/

echo "------------------------------------------------------"
echo "✅ Download complete! The results are now on your laptop."
echo "------------------------------------------------------"

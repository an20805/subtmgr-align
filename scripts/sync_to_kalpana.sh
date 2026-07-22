#!/bin/bash

# ==============================================================================
# CONFIGURATION
# ==============================================================================
USERNAME="anshuman"
SERVER="${USERNAME}@kalpana.minds.iitb.ac.in"

# The remote path where your code will live.
REMOTE_DIR="/users/student/pg/pg25/anshuman/subtmgr-align"
# ==============================================================================

echo "Syncing code to ${SERVER}..."

# We use rsync to push changes.
# We EXCLUDE datasets and output folders so we don't waste time/bandwidth uploading them.     --exclude 'experiments/' \

rsync -avz \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '.venv/' \
    --exclude 'envs/' \
    --exclude 'EMPOT/empot/' \
    --exclude 'Subtomograms_40_deg_uniform/' \
    --exclude 'eman2_baseline/' \
    --exclude 'EMPOT/outputs/' \
    ./ ${SERVER}:${REMOTE_DIR}

echo "------------------------------------------------------"
echo "✅ Sync complete!"
echo "To run your job, SSH into the server:"
echo "    ssh ${SERVER}"
echo "Then navigate to the directory and submit the job:"
echo "    sbatch scripts/submit_madhava.slurm"
echo "------------------------------------------------------"

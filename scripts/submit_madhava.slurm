#!/bin/bash
#SBATCH --job-name=empot_align
#SBATCH --account=sharada_proj2
#SBATCH --partition=cn2_madhava
#SBATCH --qos=madhava
#SBATCH --gres=gpu:1           # Request 1 GPU (adjust if EMPOT supports multi-GPU)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      # Adjust based on the node's CPU/GPU ratio limits!
#SBATCH --mem=32G              # Memory requested
#SBATCH --time=04:00:00        # Maximum time the job can run
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Job started on $(hostname) at $(date)"

# 1. Move to your project directory on the server
cd /users/student/pg/pg25/anshuman/subtmgr-align

# 2. Activate the uv environment
source EMPOT/empot/bin/activate

# 3. Run the iterative pipeline
# We can just call the bash script we already wrote!
chmod +x scripts/run_empot_iterative_fsc.sh
./scripts/run_empot_iterative_fsc.sh

echo "Job completed at $(date)"

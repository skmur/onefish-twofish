#!/bin/bash
#SBATCH --job-name=starlingLM_color_completion
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=0-12:00
#SBATCH --mem=100G
#SBATCH --output=starlingLM-none-completion-output.out
#SBATCH --error=starlingLM-none-completion-error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=soniamurthy@g.harvard.edu

# Load modules
module load python/3.10.9-fasrc01 
module load Anaconda2/2019.10-fasrc01

# Activate conda environment (optional)
conda activate jennlms

# Run the job
python starlingLM.py none
#!/bin/bash
#SBATCH --job-name=modelsCRP_temp2.0
#SBATCH --account=ullman_lab
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-00:20
#SBATCH --mem=16G
#SBATCH --output=../../sbatch-logs/models-CRP-temp2.0-%A_%a.out
#SBATCH --error=../../sbatch-logs/models-CRP-temp2.0-%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=soniamurthy@g.harvard.edu
#SBATCH --array=0-7  # Array indices for four jobs

# Load modules
module load python/3.10.9-fasrc01 
module load Anaconda2/2019.10-fasrc01

# Activate conda environment (optional)
conda activate jennlms
conda deactivate
conda activate jennlms

# Define prompt conditions in an array
MODEL_NAMES=("openchat" "starling" "gemma-instruct" "zephyr-gemma" "mistral-instruct" "zephyr-mistral" "llama2" "llama2-chat", "tulu", "tulu-dpo")

# Run the job based on the array index
python Main.py\
  --model="${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}"\
  --prompt="none"\
  --temperature="2.0"
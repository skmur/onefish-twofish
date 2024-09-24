#!/bin/bash
#SBATCH --job-name=zephyrGemma_concepts
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=0-03:00
#SBATCH --mem=100G
#SBATCH --output=./sbatch-logs/zephyrGemma-%A_%a.out
#SBATCH --error=./sbatch-logs/zephyrGemma-%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=soniamurthy@g.harvard.edu
#SBATCH --array=0-3  # Array indices for four jobs

# Load modules
module load python/3.10.9-fasrc01 
module load Anaconda2/2019.10-fasrc01

# Activate conda environment (optional)
conda activate jennlms
conda deactivate
conda activate jennlms

# Define prompt conditions in an array
PROMPT_CONDITIONS=("none" "random" "nonsense" "identity")

# Run the job based on the array index
python run_experiments.py --task="concepts" \
  --concept_category="animals"\
  --model_name="zephyr-gemma" \
  --model_path="HuggingFaceH4/zephyr-7b-gemma-v0.1" \
  --prompt_condition="${PROMPT_CONDITIONS[$SLURM_ARRAY_TASK_ID]}" \
  --hf_token="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" \
  --num_subjects=150 \
  --batch_size=128

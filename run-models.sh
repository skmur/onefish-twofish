#!/bin/bash
#SBATCH --job-name=tulu_concept
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_requeue
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0-03:00
#SBATCH --mem=100G
#SBATCH --output=tulu-prompts-concepts-politicians-%A_%a.out
#SBATCH --error=tulu-prompts-concepts-politicians-%A_%a.err
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

PROMPT_CONDITIONS=("none" "random" "nonsense" "identity")

# Run the job based on the array index
# no specified temperature = "default", add --temperature="1.5" or "2.0" (making sure prompt = "none")
python run-experiments.py\
  --task="concepts"\
  --concept_category="politicians"\
  --model_name="tulu"\
  --model_path="allenai/tulu-2-7b"\
  --prompt_condition="${PROMPT_CONDITIONS[$SLURM_ARRAY_TASK_ID]}" \
  --hf_token="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"\
  --batch_size=64


python run-experiments.py\
 --task="colors"\
 --model_name="tulu"\
 --model_path="allenai/tulu-2-7b"\
 --prompt_condition="${PROMPT_CONDITIONS[$SLURM_ARRAY_TASK_ID]}"\
 --hf_token="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"\
 --num_words=199\
 --num_subjects=150\
 --batch_size=128
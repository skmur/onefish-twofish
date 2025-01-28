# 🐟 One fish, two fish, but not the whole sea: Alignment reduces language models' conceptual diversity

This repository contains code and data for the paper ["One fish, two fish, but not the whole sea: Alignment reduces language models' conceptual diversity"](https://arxiv.org/abs/2411.04427) by Sonia K. Murthy, Tomer Ullman, and Jennifer Hu (NAACL 2025)

## 0. Code structure

The repository is structured with the following main folders:

- `concept-task`: Chinese Restaurant Process model and related files for Conceptual similarity judgements domain
- `figures`: intermediate plots and rendered figures used in the paper
- `input-data`: human baseline data for both tasks, plus code to generate the prompt conditions used in our experiments
- `analysis`: Python scripts for processing experiment outputs, analyses, reproducing figures, and summary statistics found in paper
- `src`: Python scripts for implementing the experiments
- `output`: CSV files with outputs from model runs

## 1. Reproducing results

### a. Depedencies

The packages used in our experiments can be found in the `requirements.txt` file.
To create a new environment based on these requirements, please run:

```bash
conda create --name onefishtwofish --file requirements.txt
```
Then activate the environment with:
```bash
conda activate onefishtwofish
```

### a. Running experiments
To reproduce results, run files in this order:

#### i. Obtaining prompting conditions for simulating unique human subjects: `./input-data/get-prompts.py`:
- Filters English Wikipedia to remove people and disambiguation pages
- Outputs 150 prompts (color task) or 1800 prompts (concept task) for each prompting strategy to task-specific directory in `./input-data/`

#### ii. Main experiment script for both tasks: `./src/run-experiments.py`
- __Word-color association task__: queries model for two color associations for each word in `./input-data/colorref.csv` and computes internal ΔE for these responses
Example with command-line arguments:
```bash
python ./src/run_experiments.py 
    --task="colors" 
    --model_name="starling" 
    --model_path="berkeley-nest/Starling-LM-7B-alpha" 
    --prompt_condition="random" 
    --hf_token="{YOUR_TOKEN}" 
    --num_words=199 
    --num_subjects=100 
    --batch_size=64
```
- ___Conceptual similarity judgements task__: queries model for which one of two word choices is more similar to a target
Example with command-line arguments:
```bash
python ./src/run-experiments.py
    --task="concepts" 
    --concept_category="politicians" 
    --model_name="tulu" 
    --model_path="allenai/tulu-2-7b" 
    --prompt_condition="random" 
    --hf_token="{YOUR_TOKEN}" 
    --batch_size=64
```
- Outputs data to task-specific directory in user-specified storage directory

### b. Processing and analyzing experiment results

#### i. Compiling results of all prompt and temperature experiments: `./analysis/process-data.py`
- Run with flag for "concept" or "color" task to concatenate all prompt and temperature versions of the model output into a single file
- Outputs data to `./output-data/[TASK]/`

#### ii. Word-color association task-specific processing
1. `./analysis/analyze-color-data.py`: Computes the mean and variance of word response distributions, Jensen-Shannon divergence, population ΔE based off block1 and block2 associations, distance of internal and population deltaE points from diagonal (line of unity) and corresponding regression, number of valid and invalid responses
    - Outputs compiled statistics to `./output-data/color-task/all/`
2. `./analysis/plot-color-data.py`: Plots response counts, population homogeneity (i.e., distance from diagonal), ΔE, word ratings, JS divergence, and color response bars figures using flag for plot type 

#### iii. Conceptual similarity judgements task-specific processing
1. Data Formatting for CRP Model: `./concept-task/input-data/format-CRP-input.py`
    - Takes in model responses from the previous step and reformats it for the Chinese Restaurant Process (CRP) model.
    - Outputs data to `./concept-task/input-data/` in the format:
        - Concept = target
        - ID = subject
        - Question = choice1 + " " + choice2
        - ChoiceNumber = binary representation of subject's response as either choice1 (0) or choice2 (1)
2. Running Model: `./concept-task/Chinese Restaurant Model/Main.py`
    - Takes in formatted response data from the previous step and runs CRP (importing Model.py)
    - Outputs main clustering results to `./concept-task/Chinese Restaurant Model/output-data-{num_interations}iterations/`
    <!-- - Outputs each participant's MAP Table in format ["ID", "Table", "Concept"] to `./concept-task/Chinese Restaurant Model/output-data-{num_interations}iterations/` under some conditions (see line 338 of `Model.py`) -->
3. Compiling output of CRP model and human data for analysis: `./concept-task/output-data-500iterations/format-CRP-output.py`
    - Selects relevant parameters from human baseline data: prior="Simplicity", Iteration=499
    - Compiles all prompt and temperature manipulation data files for LLMs CRP clustering results from previous step and computes number of valid responses for the model data
    - Outputs compiled results for all models and human baseline to `../../output-data/concept-task/all/`
4. `plot-concept-data.py`: Plots response counts, P(multiple concepts) figures using flag for plot type
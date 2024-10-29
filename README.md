## 🐟 One fish, two fish, or the whole sea: Do LLM judgements capture population variability? 

RQ: Is there a difference between simulating multiple unique individuals by: increasing temperature vs. prompting to elicit different distributions vs. using different models?
- H1: One LLM = one individual
    - Sampling from a model’s distribution (with sufficiently high temperature OR prompting) is like sampling responses from an individual.
- H2: One LLM = one population
    - Sampling from a model’s distribution (with sufficiently high temperature OR prompting) is like sampling responses across individuals in a population.

### key files

- [experiment tracking](https://docs.google.com/spreadsheets/d/1iNdMq4soYBgeUVpqwTTH2vFBYybOZfNAQBCxXVBrQJA/edit?usp=sharing)
- [master doc](https://docs.google.com/document/d/12nD7cuF-pl3CeRP1UV-OZrv-aQK0_BCJoISCVaiKv1E/edit?usp=sharing)

Results logs:
- [update 1](https://docs.google.com/presentation/d/1aUmKyZmHWkECU8u3egIrzLDMrOuMDG2zY1ycc1KQL8Y/edit?usp=sharing)
- [update 2](https://docs.google.com/presentation/d/1c_mrbb26wBy3QlQUV7bZioZxWlghNOP8g9OUW2B5iYI/edit?usp=sharing)

### replication

To reproduce, run files in this order:
1. `get-prompts.py`: filters English Wikipedia to remove people and disambiguation pages, outputs 150 prompts (color task) or 1800 prompts (concept task) for each prompting strategy
- Outputs prompts to task directory `./input-data/`

## 2. Color Task

### a. `run-experiments.py`
- Queries model for two color associations for each word in `./input-data/colorref.csv`
- Computes internal deltaE for these responses
- Queries model for expected population agreement for 2nd sample of association
- Outputs data to lab storage directory

### b. `process-data.py`
- Run this with flag for "concept" or "color" task to concatenate all prompt and temperature versions of the model output into a single file
- Outputs data to `./output-data/color-task/`

### c. `analyze-color-data-v2.py`
- Computes:
    - mean and variance of word response distributions
    - Jensen-Shannon divergence, population deltaE based off block1 and block2 associations
    - distance of internal and population deltaE points from the diagonal
    - number of valid and invalid responses
- Outputs compiled stats to `./output-data/color-task/all/`

### d. `plot-color-data-v2.py`
- Plots figures using flag for plot type (response counts, population homogeneity (i.e., distance from diagonal), deltaE, word ratings, and JS divergence)

## 3. Concept Task

### a. Data Generation: `run_experiments.py`
- Queries model for which one of two word choices is more similar to a target
- Outputs data to `./output-data/concept-task/` in the format of: ['model_name', 'temperature', 'subject_num', 'concept_category', 'target', 'choice1', 'choice2', 'generation1', 'answer1', 'generation2', 'answer2', 'prompt']

### b. [SKIPPED] Determining Selected Answer Choice for Model Generations: `gpt4o.py`
- Some of the models' generations do not match either choice1 or choice2, so we use gpt-4o to process these generations to determine the model's choice.
- If there's a mismatch between previously determined 'answer1' or 'answer2' values:
- Impute with gpt-4o response
- Set 'gpt_response1', 'gpt_response2' to TRUE accordingly
- Outputs data to `./output-data/concept-task/` as `[gpt-imputed]{filename}`

### c. Data Formatting for CRP Model: `./concept-task/input-data/format-CRP-input.py`
- Takes in model responses from the previous step and reformats it for the Chinese Restaurant Process (CRP) model.
- Outputs data to `./concept-task/input-data/` in the format:
    - Concept = target
    - ID = subject
    - Question = choice1 + " " + choice2
    - ChoiceNumber = binary representation of subject's response as either choice1 (0) or choice2 (1)

### d. Running Model: `./concept-task/Chinese Restaurant Model/Main.py`
- Takes in formatted response data from the previous step and runs CRP (importing Model.py)
- Outputs main clustering results in the format: ["Concept", "Iteration", "S_Chao1", "NumberOfPeople", "NumberOfTrials", "Prior", "Tables", "Alpha", "Posterior", "Chain", "ProbabilityOfSameTable"] to `./concept-task/Chinese Restaurant Model/output-data-{num_interations}iterations/`
- Outputs each participant's MAP Table (for use in TSNE visualization) in format ["ID", "Table", "Concept"] to `./concept-task/Chinese Restaurant Model/output-data-{num_interations}iterations/` under some conditions (see line 338 of `Model.py`)

### e. Compiling output of CRP model and human data for analysis: `./concept-task/output-data-500iterations/format-CRP-output.py`
- Selects relevant parameters from human baseline data: prior="Simplicity", Iteration=499
- Compiles all prompt and temperature manipulation data files for LLMs CRP clustering results from previous step
- Computes number of valid responses for the model data
- Outputs compiled results for all models and human baseline to `../../output-data/concept-task/all/`

### d. `plot-concept-data.py`
- Plots figures using flag for plot type (response counts, P(multiple concepts))
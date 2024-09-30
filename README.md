## 🐟 One fish, two fish, or the whole sea: Do LLM judgements capture population variability? 

RQ: Is there a difference between simulating multiple unique individuals by: increasing temperature vs. prompting to elicit different distributions vs. using different models?
- H1: One LLM = one individual
    - Sampling from a model’s distribution (with sufficiently high temperature OR prompting) is like sampling responses from an individual.
- H2: One LLM = one population
    - Sampling from a model’s distribution (with sufficiently high temperature OR prompting) is like sampling responses across individuals in a population.

### secrets file

- duplicate `secrets_example.json` + rename `secrets.json`
- copy and paste API keys in `secrets.json`

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
2. Color task:
    a. `run_experiments.py`: queries model for two color associations for each word in `./input-data/colorref.csv`, computes internal deltaE for these responses, and queries model for expected population agreement for 2nd sample of association
    b. `process-color`
    c. `analyze-color-data.py`: compute population deltaE based off block1 and block2 associations, plot population deltaE vs internal deltaE
3. Concept task:
    a. Data generation: `run_experiments.py` queries model for which one of two word choices is more similar to a target
        - Outputs data to `./output-data/concept-task` in format of ['model_name', 'temperature', 'subject_num', 'concept_category', 'target', 'choice1', 'choice2', 'generation1', 'answer1', 'generation2', 'answer2', 'prompt']
    b. Determining selected answer choice for models generations: `gpt4o.py`. Some of the models' generations do not match either choice1 or choice2, so we use gpt-4o to process these generations to determine the model's choice.
        - If there's a mismatch between previously determined 'answer1' or 'answer2' values, impute with gpt-4o response and set 'gpt_response1', 'gpt_response2' to TRUE accordingly
    c. Data formatting for CRP model: `./replicate concept task/human-concepts-replication.Rmd` takes in model responses from previous step and reformats it for the Chinese Restaurant Process (CRP) model.
        - Outputs data to `./replicate concept task/output-data/` in format: Index,Concept,ID,Question,ChoiceNumber, where Concept = target, ID=subject, Question = choice1 + " " + choice 2, ChoiceNumber = binary representation of subject's response as either choice1 (0) or choice2 (1)
    d. Running model: `./replicate concept task/Chinese Restaurant Model/Main.py` takes in formatted response data from previous step and runs CRP (using Model.py)
        - Outputs main clustering results in format ["Concept", "Iteration", "S_Chao1", "NumberOfPeople", "NumberOfTrials", "Prior", "Tables", "Alpha", "Posterior", "Chain", "ProbabilityOfSameTable"] to `./replicate concept task/Chinese Restaurant Model/`
        - Outputs each participant's MAP Table (for use in TSNE visualization) in format ["ID", "Table", "Concept"] to `./replicate concept task/Chinese Restaurant Model/`
    e. Analysis: `./replicate concept task/human-concepts-replication.Rmd`



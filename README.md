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
1. `./input-data/get-prompts.py`: filters English Wikipedia to remove people and disambiguation pages, outputs 100 prompts for each prompting strategy
    - Color task:
        1. `./replicate color task/[model_name].py`: queries model for two color associations for each word in `./input-data/colorref.csv`, computes internal deltaE for these responses, and queries model for expected population agreement for 2nd sample of association
        2. `./replicate color task/analyze-data.py`: compute population deltaE based off block1 and block2 associations, plot population deltaE vs internal deltaE
    - Concept task:
        1. Data generation: `./replicate concept task/[model_name].py` queries model for which one of two word choices is more similar to a target
        - Outputs data to `./replicate concept task/output-data/` in format of subject,condition,category,target,choice1,choice2,response1,response2
        2. Data formatting: `./replicate concept task/human-concepts-replication.Rmd` takes in model responses from previous step and reformats it for the Chinese Restaurant Process (CRP) model.
        - Outputs data to `./replicate concept task/output-data/ in format: Index,Concept,ID,Question,ChoiceNumber, where Concept = target, ID=subject, Question = choice1 + " " + choice 2, ChoiceNumber = binary representation of subject's response as either choice1 (0) or choice2 (1)
        3. Running model: `./replicate concept task/Chinese Restaurant Model/Main.py` takes in formatted response data from previous step and runs CRP (using Model.py)
        - Outputs main clustering results in format ["Concept", "Iteration", "S_Chao1", "NumberOfPeople", "NumberOfTrials", "Prior", "Tables", "Alpha", "Posterior", "Chain", "ProbabilityOfSameTable"] to `./replicate concept task/Chinese Restaurant Model/`
        - Outputs each participant's MAP Table (for use in TSNE visualization) in format ["ID", "Table", "Concept"] to `./replicate concept task/Chinese Restaurant Model/`



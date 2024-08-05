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


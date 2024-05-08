<code>./input-data/get-prompts.py</code>
1. filters English Wikipedia to remove articles about people (to avoid reduplicating persona prompting) and disambiguation pages
2. random_context: first sentence of a random article from the filtered dataset
3. nonsense_context: shuffled word order version of random_context
4. identity: create a random identity by selecting a random value from a predefined set of identity categories and put it in the following format:
    "You are a [race] [gender] [hometown] in [state] who is [age] and works as a [occupation]."

output: <code>./input-data/promts.pkl

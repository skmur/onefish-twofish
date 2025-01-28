import pandas as pd
import pickle
import statsmodels.api as sm


def compute_human_baseline(df, task, metric, category):
    """ Compute human baseline for color task """

    # select data for model_name = human and concept_category if task = concept
    if task == "concept":
        human = df[(df['model_name'] == 'human') & (df['concept_category'] == category)]
    else:
        human = df[df['model_name'] == 'human']

    # compute mean and std deviation of dist_from_diagonal
    mean = human[metric].mean()
    std = human[metric].std()
    # compute 95% confidence interval
    n = len(human)
    t = 1.96
    ci = t * std / (n ** 0.5)
    # upper and lower bounds
    ci_upper = mean + ci
    ci_lower = mean - ci
    print(f"[{task} task, {metric}, {category}] human data, mean: {mean}")
    print(f"[{task} task, {metric}, {category}] human data, 95% CI: [{ci_lower}, {ci_upper}]")

def add_model_meta_data(df):
    family_map = {
        "openchat": "openchat_starling",
        "starling": "openchat_starling",
        "gemma-instruct": "gemma",
        "zephyr-gemma": "gemma",
        "mistral-instruct": "mistral",
        "zephyr-mistral": "mistral",
        "llama2": "llama",
        "llama2-chat": "llama",
        "tulu": "llama",
        "tulu-dpo": "llama",
        "human": "human"
    }
    align_map = {
        "openchat": False,
        "starling": True,
        "gemma-instruct": False,
        "zephyr-gemma": True,
        "mistral-instruct": False,
        "zephyr-mistral": True,
        "llama2": False,
        "llama2-chat": True,
        "tulu": False,
        "tulu-dpo": True,
        "human": None
    }
    df["model_family"] = df.model_name.map(family_map)
    df["aligned"] = df.model_name.map(align_map)
    return df

def mixed_effects_model(df, task, x_values, y, random_effect):
    """ Compute mixed effects model for task results """
    # remove human data
    df = df[df.model_name != 'human']

    # if task == color, remove data for llama and tulu 
    if task == "color":
        df = df[(df.model_name != 'llama2') & (df.model_name != 'tulu')]
    # if task == concept, remove data for llama
    if task == "concept":
        df = df[df.model_name != 'llama2']

    formula = f"{y} ~"

    # if x1 is prompt or temperature, select appropriate data
    if x_values == ["prompt"]:
        # select where temperature is default
        df = df[df.temperature == "default"]
    
    for x in x_values:
        # make all columns categorical
        df[x] = pd.Categorical(df[x]).codes
        # add to formula
        formula += f" + {x}"

    df[random_effect] = pd.Categorical(df[random_effect]).codes

    print(formula)

    model = sm.MixedLM.from_formula(formula, data=df, groups=random_effect, missing="drop")
    result = model.fit()

    print(f"[{task} task, predicting: {y}, fixed effects: {x_values}, random effect: {random_effect}]")
    # Print the summary
    print(result.summary())

def valid_responses(df, task):
    """ Compute average percentage of valid responses for each model """
    valid_responses = df.groupby("model_name")["percent_invalid"].mean()
    # sort by average percentage of invalid responses
    valid_responses = valid_responses.sort_values(ascending=False)
    print(f"[{task} task] Average percentage of invalid responses")
    print(valid_responses)



        


#-------------------------------------------------------------

df_color = pd.read_pickle("../output-data/color-task/all/word-stats.pickle")
df_color = add_model_meta_data(df_color)
df_concept = pd.read_pickle("../output-data/concept-task/all/all-ClusteringResults.pickle")
df_concept = add_model_meta_data(df_concept)

color_responsecounts = pd.read_pickle("../output-data/color-task/all/valid-response-counts.pickle")
concept_responsecounts = pd.read_pickle("../output-data/concept-task/all/model-data.pickle")

# in paper we report P(multiple concepts), so create column for 1-P(single concept)
df_concept["p_multiple_concepts"] = 1 - df_concept["ProbabilityOfSameTable"]


# compute human baselines for metrics on color and concept tasks
compute_human_baseline(df_color, "color", "dist_from_diagonal", None)
compute_human_baseline(df_concept, "concept", "p_multiple_concepts", "animals")
compute_human_baseline(df_concept, "concept", "p_multiple_concepts", "politicians")

print("-----------------")
print("Color task")
print("-----------------")


# compute mixed effects model for effect of alignment with random effect of model_family:
print("metric ~ aligned[boolean] + (1|model_family)")
mixed_effects_model(df_color, "color", ["aligned"], "dist_from_diagonal", "model_family")

# # compute mixed effects model for effect of prompt with random effect of model_name: 
# print("metric ~ prompt[none, persona, random, nonsense] + (1|model_name)")
# mixed_effects_model(df_color, "color", ["prompt"], "dist_from_diagonal", "model_name")

# # compute mixed effects model for effect of temperature with random effect of model_name: 
# print("metric ~ temperature[default, 1.5, 2.0] + (1|model_name)")
# mixed_effects_model(df_color, "color", ["temperature"], "dist_from_diagonal", "model_name")

# combined model for prompt and temperature
print("metric ~ temperature[default, 1.5, 2.0] +  prompt[none, persona, random, nonsense] + (1|model_name)")
mixed_effects_model(df_color, "color", ["temperature", "prompt"], "dist_from_diagonal", "model_name")

print("-----------------")
print("Concept task")
print("-----------------")

# compute mixed effects model for effect of alignment with random effect of model_family:
print("metric ~ aligned[boolean] + (1|model_family)")
mixed_effects_model(df_concept, "concept", ["aligned"],"p_multiple_concepts", "model_family")

# # compute mixed effects model for effect of prompt with random effect of model_name: 
# print("metric ~ prompt[none, persona, random, nonsense] + (1|model_name)")
# mixed_effects_model(df_concept, "concept", ["prompt"],"p_multiple_concepts", "model_name")

# # compute mixed effects model for effect of temperature with random effect of model_name: 
# print("metric ~ temperature[default, 1.5, 2.0] + (1|model_name)")
# mixed_effects_model(df_concept, "concept",["temperature"],"p_multiple_concepts", "model_name")

# combined model for prompt and temperature
print("metric ~ temperature[default, 1.5, 2.0] +  prompt[none, persona, random, nonsense] + (1|model_name)")
mixed_effects_model(df_concept, "concept", ["temperature", "prompt"], "p_multiple_concepts", "model_name")

print("-----------------")

valid_responses(color_responsecounts, "color")
valid_responses(concept_responsecounts, "concept")
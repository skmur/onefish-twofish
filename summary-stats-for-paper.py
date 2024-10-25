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

def mixed_effects_model(df, task, x1, x2, y, random_effect):
    """ Compute mixed effects model for task results """
    # remove human data
    df = df[df.model_name != 'human']

    # make all columns categorical
    df[x1] = pd.Categorical(df[x1]).codes
    df[x2] = pd.Categorical(df[x2]).codes
    df[random_effect] = pd.Categorical(df[random_effect]).codes

    model = sm.MixedLM.from_formula(f"{y} ~ {x1} * {x2}", data=df, groups=random_effect, missing="drop")
    result = model.fit()

    print(f"[{task} task, predicting: {y}, fixed effects: {x1}, {x2}, random effect: {random_effect}]")
    # Print the summary
    print(result.summary())




#-------------------------------------------------------------

df_color = pd.read_pickle("./output-data/color-task/all/word-stats.pickle")
df_color = add_model_meta_data(df_color)
df_concept = pd.read_pickle("./output-data/concept-task/all/all-ClusteringResults.pickle")
df_concept = add_model_meta_data(df_concept)

# in paper we report P(multiple concepts), so create column for 1-P(single concept)
df_concept["p_multiple_concepts"] = 1 - df_concept["ProbabilityOfSameTable"]


# compute human baselines for metrics on color and concept tasks
compute_human_baseline(df_color, "color", "dist_from_diagonal", None)
compute_human_baseline(df_concept, "concept", "p_multiple_concepts", "animals")
compute_human_baseline(df_concept, "concept", "p_multiple_concepts", "politicians")

print("-----------------")

# compute mixed effects model for color and concept tasks
mixed_effects_model(df_color, "color", "aligned", "prompt", "dist_from_diagonal", "model_family")
mixed_effects_model(df_concept, "concept", "aligned", "prompt", "p_multiple_concepts", "model_family")

mixed_effects_model(df_color, "color", "aligned", "temperature", "dist_from_diagonal", "model_family")
mixed_effects_model(df_concept, "concept", "aligned", "temperature", "p_multiple_concepts", "model_family")



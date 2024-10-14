
import pandas as pd
import pickle
import math
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from tqdm import tqdm

CRP_data_dir = "./"
CRP_output_dir = "./compiled/"
raw_data_dir = "../../output-data/concept-task/"
raw_output_dir = "../../output-data/concept-task/all/"

# - - - - - - - - - - - - - - - - - - -
models = ["openchat", "starling", "gemma-instruct", "zephyr-gemma", "mistral-instruct", "zephyr-mistral", "llama2", "llama2-chat", "tulu", "tulu-dpo"]
param_combos = [["none", "default"], ["none", "1.5"], ["none", "2.0"], ["random", "default"], ["nonsense", "default"], ["identity", "default"]]
results = ["ClusteringResults", "MAPTables"]
animals = ["a finch", "a robin", "a chicken", "an eagle", "an ostrich", "a penguin", "a salmon", "a seal", "a dolphin", "a whale"]
politicians = ["Abraham Lincoln", "Barack Obama", "Bernie Sanders", "Donald Trump", "Elizabeth Warren", "George W. Bush", "Hillary Clinton", "Joe Biden", "Richard Nixon", "Ronald Reagan"] 

# - - - - - - - - - - - - - - - - - - -

# human coding for animals and politicians removes articles from animals and removes first names from politicians
animals_humandata = ["finch", "robin", "chicken", "eagle", "ostrich", "penguin", "salmon", "seal", "dolphin", "whale"]
politicians_humandata = ["Lincoln", "Obama", "Sanders", "Trump", "Warren", "Bush", "Clinton", "Biden", "Nixon", "Reagan"]
# map the humandata lists to the animals and politicians lists
animals_map = dict(zip(animals_humandata, animals))
politicians_map = dict(zip(politicians_humandata, politicians))
# combine the two dictionaries
concept_map = {**animals_map, **politicians_map}
print(concept_map)
# - - - - - - - - - - - - - - - - - - -

# read in human data
human_ClusteringResults = pd.read_csv("../../input-data/concept-task/ClusteringResults.csv")
print(human_ClusteringResults["Concept"].unique())

# add columns for model_name, prompt, and temperature
human_ClusteringResults['model_name'] = "human"
human_ClusteringResults['prompt'] = "na"
human_ClusteringResults['temperature'] = "na"
# select where iteration is 499
human_ClusteringResults = human_ClusteringResults[human_ClusteringResults["Iteration"] == 499]
# replace the humandata concept names with the original concept names
human_ClusteringResults["Concept"] = human_ClusteringResults["Concept"].map(concept_map)
# check if concept is in animals or politicians and add column to df for concept type
human_ClusteringResults["concept_category"] = np.where(human_ClusteringResults["Concept"].isin(animals), "animals", "politicians")
print(human_ClusteringResults["Concept"].unique())

print("Processed human data.")
# - - - - - - - - - - - - - - - - - - -

print("Compiling data...")

all_ClusteringResults = pd.DataFrame()
all_MAPTables = pd.DataFrame()
all_data = pd.DataFrame()

for model in models:
    model_ClusteringResults = pd.DataFrame()
    model_MAPTables = pd.DataFrame()
    df_raw = pd.read_pickle(f"{raw_data_dir}concept-{model}.pickle")
    all_data = pd.concat([all_data, df_raw])

    for prompt, temperature in param_combos:
        for result in results:
            # open the csv file for the model, prompt, and temperature combination
            # add column for each parameter to the dataframe
            df = pd.read_csv(f"{CRP_data_dir}{model}-prompt={prompt}-temp={temperature}-{result}.csv")
            df['model_name'] = model
            df['prompt'] = prompt
            df['temperature'] = temperature

            # check if concept is in animals or politicians and add column to df for concept type
            df["concept_category"] = np.where(df["Concept"].isin(animals), "animals", "politicians")

            if result == "ClusteringResults":
                # select data for last iteration
                df = df[df["Iteration"] == 499]
                model_ClusteringResults = pd.concat([model_ClusteringResults, df])
            else:
                model_MAPTables = pd.concat([model_MAPTables, df])

    # save compiled version of the data to output_dir
    model_ClusteringResults.to_csv(f"{CRP_output_dir}{model}-ClusteringResults.csv", index=False)
    all_ClusteringResults = pd.concat([all_ClusteringResults, model_ClusteringResults])
    model_MAPTables.to_csv(f"{CRP_output_dir}{model}-MAPTables.csv", index=False)
    all_MAPTables = pd.concat([all_MAPTables, model_MAPTables])

# add human data to all_ClusteringResults
all_ClusteringResults = pd.concat([all_ClusteringResults, human_ClusteringResults])
all_ClusteringResults.to_pickle(f"{raw_output_dir}all-ClusteringResults.pickle")

all_MAPTables.to_pickle(f"{raw_output_dir}all-MAPTables.pickle")
        
# ----------------------------------------
# group by model_name, prompt, temperature, and count number of responses
group = all_data.groupby(['model_name', 'prompt', 'temperature'])
all_data['total_responses'] = group['answer1'].transform('count')

# create a mask for invalid responses where either answer1 or answer2 is -1
mask = (all_data['answer1'] != -1) & (all_data['answer2'] != -1)
all_data['valid_responses'] = mask
# group by model_name, prompt, temperature, and count number of valid responses
group = all_data.groupby(['model_name', 'prompt', 'temperature'])
all_data['valid_responses'] = group['valid_responses'].transform('sum')
all_data['invalid_responses'] = all_data['total_responses'] - all_data['valid_responses']

all_data.to_pickle(f"{raw_output_dir}model-data.pickle")

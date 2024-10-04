import pandas as pd
import pickle
import math
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_by_model(df, version, metric, plot_type):
    palette = {'none': "#3f9f7f", 'identity': "#dde300", 'random': "#17577e", 'nonsense': "#141163"}
    prompt_order = ['none', 'identity', 'random', 'nonsense']
    sns.set_theme(style="whitegrid")
    if plot_type == "point":
        plt.figure(figsize=(6, 6))
        sns.pointplot(x='model_name', y=metric, hue="prompt", hue_order=prompt_order, palette=palette, data=df, alpha=0.8)
    elif plot_type == "bar":
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model_name', y=metric, hue="prompt", hue_order=prompt_order, palette=palette, data=df, errorbar=None)
    plt.axhline(y=human_v2[metric].values[0], color='k', linestyle='--')
    # rotate x-axis labels
    plt.xticks(rotation=45)
    if metric == "rss":
        plt.ylim(0, 60000)
    elif metric == "mse":
        plt.ylim(0, 300)
    
    plt.title(f"{metric}")
    plt.tight_layout()
    plt.savefig(f"./{figure_dir}/{metric}-{version}-{plot_type}-byModel.png")
    plt.clf()

def plot_by_prompt(df, version, metric, plot_type):
    model_order = ["openchat", "starling", "gemma-instruct", "zephyr-gemma", "mistral-instruct", "zephyr-mistral", "llama2", "llama2-chat"]
    colors = ['#006400', '#66CDAA', '#003366', '#66B2FF', '#8B0000', '#FF7F7F', '#FF8C00', '#ffa554']

    # map model names to colors
    palette = dict(zip(model_order, colors))

    sns.set_theme(style="whitegrid")
    if plot_type == "point":
        plt.figure(figsize=(6, 6))
        sns.pointplot(x='prompt', y=metric, hue="model_name", hue_order=model_order, palette=palette, data=df,  alpha=0.8)
           

    elif plot_type == "bar":
        plt.figure(figsize=(10, 6))
        sns.barplot(x='prompt', y=metric, hue="model_name", hue_order=model_order, palette=palette, data=df, errorbar=None)

    plt.axhline(y=human_v2[metric].values[0], color='k', linestyle='--')
    # rotate x-axis labels
    plt.xticks(rotation=45)
    if metric == "rss":
        plt.ylim(0, 60000)
    elif metric == "mse":
        plt.ylim(0, 300)
    
    plt.title(f"{metric}")
    plt.tight_layout()
    plt.savefig(f"./{figure_dir}/{metric}-{version}-{plot_type}-byPrompt.png")
    plt.clf()



data_dir = "./output-data/color-task/all"
figure_dir = "./figures/color-task"
metric = "mse"

# Load the data
data_v1 = load_data(f"{data_dir}/rss_values-scriptv1.pickle")
data_v2 = load_data(f"{data_dir}/model-stats.pickle")


# extract human data value from each dataset
human_v1 = data_v1[data_v1['model_name'] == 'human']
human_v2 = data_v2[data_v2['model_name'] == 'human']

# remove llama2 and human from both datasets
data_v1 = data_v1[data_v1['model_name'] != 'human']
data_v2 = data_v2[data_v2['model_name'] != 'human']

plot_by_model(data_v2, "v2", "rss", "bar")

plot_by_prompt(data_v2, "v2", "rss", "bar")
plot_by_prompt(data_v2, "v2", "mse", "bar")
plot_by_prompt(data_v2, "v2", "mse", "point")
plot_by_prompt(data_v2, "v2", "rss", "point")



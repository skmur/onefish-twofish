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

# def plot_by_model(df, metric, plot_type):
#     palette = {'none': "#3f9f7f", 'identity': "#dde300", 'random': "#17577e", 'nonsense': "#141163"}
#     prompt_order = ['none', 'identity', 'random', 'nonsense']
#     sns.set_theme(style="whitegrid")
#     if plot_type == "point":
#         plt.figure(figsize=(6, 6))
#         sns.pointplot(x='model_name', y=metric, hue="prompt", hue_order=prompt_order, palette=palette, data=df, alpha=0.8)
#     elif plot_type == "bar":
#         plt.figure(figsize=(10, 6))
#         sns.barplot(x='model_name', y=metric, hue="prompt", hue_order=prompt_order, palette=palette, data=df, errorbar=None)
    
#     # plt.axhline(y=human_v2[metric].values[0], color='k', linestyle='--')
#     # rotate x-axis labels
#     plt.xticks(rotation=45)
#     if metric == "rss":
#         plt.ylim(0, 60000)
#     elif metric == "mse":
#         plt.ylim(0, 300)
    
#     plt.title(f"{metric}")
#     plt.tight_layout()
#     plt.savefig(f"./{figure_dir}/{metric}-{plot_type}-byModel.png")
#     plt.clf()

def plot_by_prompt(df, metric, plot_type):
    # get human value for metric from df
    human_metric = df[df['model_name'] == 'human'][metric].values[0]
    # remove human data from df for plotting
    df = df[df['model_name'] != 'human']
    
    if plot_type == "point":
        df = df[df['model_name'] != 'llama2'] # remove llama2 for pointplot

    # combine the prompt and temperature columns
    df['combined'] = df['prompt'] + "-" + df['temperature'].astype(str)

    model_order = ["openchat", "starling", "gemma-instruct", "zephyr-gemma", "mistral-instruct", "zephyr-mistral", "llama2", "llama2-chat"]
    colors = ['#006400', '#66CDAA', '#003366', '#66B2FF', '#8B0000', '#FF7F7F', '#914900', '#ffa554']    

    prompt_order = ['none-default', 'none-1.5', 'none-2.0', 'identity-default', 'random-default', 'nonsense-default']
    # map model names to colors
    palette = dict(zip(model_order, colors))

    sns.set_theme(style="whitegrid")
    if plot_type == "point":
        plt.figure(figsize=(6, 6))
        sns.pointplot(x='combined', y=metric, hue="model_name", hue_order=model_order, palette=palette, data=df,  alpha=0.8, order=prompt_order)
           
    elif plot_type == "bar":
        plt.figure(figsize=(10, 6))
        sns.barplot(x='combined', y=metric, hue="model_name", hue_order=model_order, palette=palette, data=df, errorbar=None, order=prompt_order)

    plt.axhline(y=human_metric, color='k', linestyle='--')
    # annotate human baseline
    plt.text(5.5, human_metric, "human", fontsize=8, ha='right')

    # rotate x-axis labels
    plt.xticks(rotation=45)
    if metric == "rss":
        plt.ylim(0, 60000)
    elif metric == "mse":
        plt.ylim(0, 300)
    
    plt.title(f"{metric}")
    plt.tight_layout()
    plt.savefig(f"./{figure_dir}/{metric}-{plot_type}-byPrompt.png")
    plt.clf()

def plot_response_counts(df, metric):
    # combine the prompt and temperature columns
    df['combined'] = df['prompt'] + "-" + df['temperature'].astype(str)
    # remove human data from response_counts
    df = df[df['model_name'] != 'human']

    prompt_order = ['none-default', 'none-1.5', 'none-2.0', 'identity-default', 'random-default', 'nonsense-default']

    plt.figure(figsize=(10, 6))
    sns.barplot(x='model_name', y=f'{metric}_responses', hue='combined', hue_order=prompt_order, data=df, dodge=True)
    plt.title(f"Number of {metric} responses per model, param combo")
    plt.xticks(rotation=45)
    # plot number of responses as horizontal line
    plt.axhline(y=df['total_responses'].values[0], color='k', linestyle='--')
    plt.tight_layout()
    plt.savefig(f"./{figure_dir}/{metric}-response-counts.png")
    plt.clf()

def plot_scatterplots(models, axs, word_stats, x_var, y_var, figure_dir, title):
    for i, model in enumerate(models):
        ax = axs[i] if len(models) > 1 else axs
        model_data = word_stats[word_stats['model_name'] == model]
        
        sns.scatterplot(data=model_data, x=x_var, y=y_var, hue="prompt", style="temperature", alpha=0.6, ax=ax)
        ax.plot([0,100], [0,100], 'k--', alpha=0.5)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 100)

        # Uncomment and modify these sections if you want to include error bars, correlation line, or word annotations
        # # plot error bars for each point based on the 95% confidence interval
        # for j, txt in enumerate(model_data['word'].tolist()):
        #     ax.errorbar(model_data[x_var].iloc[j], model_data[y_var].iloc[j], 
        #                 xerr=model_data['ci_upper_populationDeltaE'].iloc[j]-model_data['ci_lower_populationDeltaE'].iloc[j], 
        #                 yerr=model_data['ci_upper_internalDeltaE'].iloc[j]-model_data['ci_lower_internalDeltaE'].iloc[j], 
        #                 fmt='o', color='black', alpha=0.5)

        # # plot correlation line on the graph
        # corr = model_data[x_var].corr(model_data[y_var])
        # ax.text(10, 90, f"internal vs. population deltaE r = {corr:.2f}")

        # # plot the words
        # for j, txt in enumerate(model_data['word'].tolist()):
        #     ax.annotate(txt, (model_data[x_var].iloc[j], model_data[y_var].iloc[j]))

        ax.set_title(f"{model}")
        ax.set_xlabel("Internal ΔE")
        ax.set_ylabel("Population ΔE")
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f"{figure_dir}/{title}.png")
    plt.close()



data_dir = "./output-data/color-task/all"
figure_dir = "./figures/color-task"

#---------------------------------------------------------#
# 1) PLOT RESPONSE COUNTS
response_counts = load_data(f"{data_dir}/valid-response-counts.pickle")

plot_response_counts(response_counts, "invalid")
plot_response_counts(response_counts, "valid")
#---------------------------------------------------------#
# 2) PLOT WORD LEVEL METRICS

model_stats = load_data(f"{data_dir}/model-stats.pickle")
word_stats = load_data(f"{data_dir}/word-stats.pickle")
# group word stats by model and get mean, std, count of dist_from_diagonal
dists = word_stats.groupby(['model_name', 'prompt', 'temperature']).apply(lambda x: x['dist_from_diagonal'].agg(['sum', 'mean', 'std', 'count'])).reset_index()

# #- - - - - - - - - - - -
# # RSS and MSE from the diagonal line: how much the model responses deviate from a 
# # homogenous population where internal variability == population variability

# plot_by_prompt(word_stats, "dist_from_diagonal", "point")
# plot_by_prompt(word_stats, "dist_from_diagonal", "bar")

#- - - - - - - - - - - -

models = word_stats['model_name'].unique()
models = models[models != 'human']
fig, axs = plt.subplots(1, len(models), figsize=(5*len(models), 5))
plot_scatterplots(models, axs, word_stats, "mean_populationDeltaE", "mean_internalDeltaE", figure_dir, "models-internal-vs-population-deltaE")

# plot human data separately
models = ['human']
fig, axs = plt.subplots(1, len(models), figsize=(5*len(models), 5))
plot_scatterplots(models, axs, word_stats, "mean_populationDeltaE", "mean_internalDeltaE", figure_dir, "human-internal-vs-population-deltaE")

# 3) PLOT MODEL LEVEL METRICS

# #- - - - - - - - - - - -
# # sum of distances from diagonal line
# plot_by_prompt(dists, "sum", "bar")
# plot_by_prompt(dists, "sum", "point")



#- - - - - - - - - - - -








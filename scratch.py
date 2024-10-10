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

from scipy.stats import multivariate_normal
from scipy.spatial.distance import jensenshannon


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_by_prompt(df, metric, plot_type, manipulation):
    df = df[df['model_name'] != 'llama2'] # remove llama2 for both plots

    # get human value for metric from df
    human_metric = df[df['model_name'] == 'human'][metric].values[0]
    # remove human data from df for plotting
    df = df[df['model_name'] != 'human']

    # combine the prompt and temperature columns
    df['combined'] = df['prompt'] + "-" + df['temperature'].astype(str)

    model_order = ["openchat", "starling", "gemma-instruct", "zephyr-gemma", "mistral-instruct", "zephyr-mistral", "llama2-chat"]
    colors = ['#006400', '#66CDAA', '#003366', '#66B2FF', '#8B0000', '#FF7F7F', '#ffa554']    

    if manipulation == "prompt":
        order = ['none', 'identity', 'random', 'nonsense']
    elif manipulation == "temperature":
        order = ['default', '1.5', '2.0']

    # map model names to colors
    palette = dict(zip(model_order, colors))

    sns.set_theme(style="whitegrid")
    if plot_type == "point":
        plt.figure(figsize=(6, 6))
        sns.pointplot(x=manipulation, y=metric, hue="model_name", hue_order=model_order, palette=palette, data=df,  alpha=0.8, order=order)
           
    elif plot_type == "bar":
        plt.figure(figsize=(10, 6))
        sns.barplot(x=manipulation, y=metric, hue="model_name", hue_order=model_order, palette=palette, data=df, errorbar=None, order=order)

    plt.axhline(y=human_metric, color='k', linestyle='--')
    # annotate human baseline
    plt.text(1, human_metric, "human", fontsize=8, ha='right')

    # rotate x-axis labels
    plt.xticks(rotation=45)
    if metric == "rss":
        plt.ylim(0, 60000)
    elif metric == "mse":
        plt.ylim(0, 300)
    
    plt.title(f"{metric}")
    plt.tight_layout()
    plt.savefig(f"./{figure_dir}/{metric}-{manipulation}-{plot_type}.png")
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
    # put legend outside of plot
    plt.legend(title='Prompt-Temperature', bbox_to_anchor=(1.05, 1), loc='center left')
    plt.tight_layout()
    plt.savefig(f"./{figure_dir}/{metric}-response-counts.png")
    plt.clf()

def plot_deltaE_subplots(models, word_stats, x_var, y_var, manipulation, figure_dir, title):
    word_stats = word_stats[word_stats['model_name'].isin(models)]

    if manipulation == "prompt":
        word_stats = word_stats[word_stats['temperature'] == 'default']
        order = ['none', 'nonsense', 'random', 'identity']
        color_list = ["#87CEEB", "#1E90FF", "#4169E1", "#000080"]
        palette = dict(zip(order, color_list))
    elif manipulation == "temperature":
        order = ['default', '1.5', '2.0']
        color_list = ["#FFAE42", "#FFA500", "#FF0000"]
        palette = dict(zip(order, color_list))
    
    if models == ['human']:
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    else: 
        # Create a 2x4 grid of subplots
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        axs = axs.flatten()  # Flatten the 2D array of axes to make it easier to iterate over
    
    lim = 120
    for i, model in enumerate(models):
        model_data = word_stats[word_stats['model_name'] == model]

        if models == ['human']:
            ax = axs
            sns.scatterplot(data=model_data, x=x_var, y=y_var, color="k", alpha=0.8, ax=ax, s=10)        
            # Add error bars for all points
            for idx in model_data.index:
                ax.errorbar(model_data.loc[idx, x_var], model_data.loc[idx, y_var],
                            xerr=model_data.loc[idx, 'ci_upper_populationDeltaE'] - model_data.loc[idx, 'ci_lower_populationDeltaE'],
                            yerr=model_data.loc[idx, 'ci_upper_internalDeltaE'] - model_data.loc[idx, 'ci_lower_internalDeltaE'],
                            fmt='none', alpha=0.2, capsize=2)  
        else: 
            ax = axs[i]
            sns.scatterplot(data=model_data, x=x_var, y=y_var, hue=manipulation, style=manipulation, palette=palette, hue_order=order, alpha=0.8, ax=ax, s=10)

        ax.plot([0,lim], [0,lim], 'k--', alpha=0.5)
        ax.set_ylim(0, lim)
        ax.set_xlim(0, lim)

        # Sort words by internal ΔE
        sorted_data = model_data.sort_values(by=y_var, ascending=False)
        
        # Select top and bottom 40 words
        top_bottom_words = pd.concat([sorted_data.head(40), sorted_data.tail(40)])
        
        # Annotate every 5th word
        for j, (idx, row) in enumerate(top_bottom_words.iterrows()):
            if j % 5 == 0:
                ax.annotate(row['word'], (row[x_var] + 2, row[y_var]), fontsize=6)

        # get correlation line and add to plot title
        corr = model_data[x_var].corr(model_data[y_var])
        ax.set_title(f"{model}, r = {corr:.2f}", fontsize=10)

        # Create a single legend for the entire figure
        ax.set_xlabel("Population ΔE", fontsize=8)
        ax.set_ylabel("Internal ΔE", fontsize=8)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=8)
        ax.legend().set_visible(False)

    # Create a single legend for the entire figure
    if models != ['human']:
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', title=manipulation, bbox_to_anchor=(1.05, 0.5))

    plt.tight_layout()
    plt.savefig(f"{figure_dir}/{title}-{manipulation}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_deltaE_facetgrid(models, word_stats, x_var, y_var, x_label, y_label, title):
    # select data for the models
    word_stats = word_stats[word_stats['model_name'].isin(models)]
    # select data where temperature is default
    word_stats = word_stats[word_stats['temperature'] == 'default']

    lim = 120
    g = sns.FacetGrid(word_stats, col="prompt",  row="model_name",  hue="prompt", margin_titles=True, despine=False, col_order=["none", "identity", "random", "nonsense"])
    g.map(sns.scatterplot, x_var, y_var, s=6)

    # Get unique values for model_name and prompt
    model_names = word_stats['model_name'].unique()
    prompts = word_stats['prompt'].unique()

    for i, ax in enumerate(g.axes.flatten()):
        ax.plot([0, lim], [0, lim], color='k', linestyle='--', alpha=0.5)
        
        # Calculate row and column index
        row_idx = i // len(prompts)
        col_idx = i % len(prompts)
        
        # Get the corresponding model_name and prompt
        model_name = model_names[row_idx]
        prompt = prompts[col_idx]
        
        # Get data for the model and prompt
        data = word_stats[(word_stats['model_name'] == model_name) & (word_stats['prompt'] == prompt)]
        
        for j, txt in enumerate(data['word']):
            if j % 20==0 and (data[y_var].iloc[j] < 30 or data[y_var].iloc[j] > 70):
                ax.annotate(txt, (data[x_var].iloc[j] + 5, data[y_var].iloc[j]), fontsize=6)

    g.set(ylim=(0, lim))
    g.set(xlim=(0, lim))
    g.set_axis_labels(x_label, y_label)
    g.set(yscale='linear')
    g.set(xscale='linear')

    plt.tight_layout()
    plt.savefig(f"{figure_dir}/{title}.png")
    plt.clf()

# DOESN'T GIVE A SQUARE PLOT
def plot_word_ratings_subplots(models, word_stats, x_var, y_var, manipulation, figure_dir, title):
    word_stats = word_stats[word_stats['model_name'].isin(models)]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.scatterplot(data=word_stats, x=x_var, y=y_var)
    plt.xlabel(x_var)
    plt.ylabel("Internal ΔE")
    plt.title(models[0])
    # set aspect ratio to be equal
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/{title}-{manipulation}.png", dpi=300, bbox_inches='tight')
    plt.clf()

def plot_word_ratings_facetgrid(models, word_stats, x_var, y_var, x_label, y_label, title):
    # select data for the models
    word_stats = word_stats[word_stats['model_name'].isin(models)]
    # select data where temperature is default
    word_stats = word_stats[word_stats['temperature'] == 'default']

    g = sns.FacetGrid(word_stats, col="prompt",  row="model_name", hue="prompt", margin_titles=True, despine=False, col_order=["none", "identity", "random", "nonsense"])
    g.map(sns.regplot, x_var, y_var, scatter_kws={'s':2})

    # Get unique values for model_name and prompt
    model_names = word_stats['model_name'].unique()
    prompts = word_stats['prompt'].unique()

    for i, ax in enumerate(g.axes.flatten()):        
        # Calculate row and column index
        row_idx = i // len(prompts)
        col_idx = i % len(prompts)
        
        # Get the corresponding model_name and prompt
        model_name = model_names[row_idx]
        prompt = prompts[col_idx]
        
        # Get data for the model and prompt
        data = word_stats[(word_stats['model_name'] == model_name) & (word_stats['prompt'] == prompt)]
        
        for j, txt in enumerate(data['word']):
            if j % 20 == 0:
                ax.annotate(txt, (data[x_var].iloc[j], data[y_var].iloc[j]), fontsize=6)

    g.set(xlim=(0, 8))
    g.set(ylim=(0, 100))
    g.set_axis_labels(x_label, y_label)
    g.set(yscale='linear')
    g.set(xscale='linear')
    
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/{title}.png")
    plt.clf()


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot data from color task")

    parser.add_argument("--plot", type=str, choices=["response counts", "population homogeneity", "deltaE", "JS divergence", "word ratings", "color bars"], required=True, help="Which set of plots to generate")

    args = parser.parse_args()

    # - - - - - - - - - - - - - - - - - - -
    data_dir = "./output-data/color-task/all"
    # add subfolder for this plot type if it doesn't exist
    figure_dir = "./figures/color-task/" + args.plot + "/"
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    # - - - - - - - - - - - - - - - - - - -
    # load data
    response_counts = load_data(f"{data_dir}/valid-response-counts.pickle")
    model_stats = load_data(f"{data_dir}/model-stats.pickle")
    word_stats = load_data(f"{data_dir}/word-stats.pickle")
    # group word stats by model and get mean, std, count of dist_from_diagonal
    dists = word_stats.groupby(['model_name', 'prompt', 'temperature']).apply(lambda x: x['dist_from_diagonal'].agg(['sum', 'mean', 'std', 'count']), include_groups=False).reset_index()

    # load word ratings data
    word_ratings = pd.read_csv("./input-data/color-task/compiled-variance-entropy-glasgowRatings.csv")
    # merge imageability and concreteness ratings with word_stats
    word_stats = word_stats.merge(word_ratings[['word', 'imageability', 'concreteness']], on='word', how='left')

    print(word_stats.columns)
    # - - - - - - - - - - - - - - - - - - -
    # [DONE] plot number of valid and invalid responses per model, param combo
    if args.plot == "response counts":
        plot_response_counts(response_counts, "invalid")
        plot_response_counts(response_counts, "valid")

    # [DONE] how much the model responses deviate from a homogenous population 
    # (i.e. where internal variability = population variability)
    elif args.plot == "population homogeneity":
        plot_by_prompt(word_stats, "dist_from_diagonal", "point", "prompt")
        plot_by_prompt(word_stats, "dist_from_diagonal", "bar", "prompt")
        plot_by_prompt(word_stats, "dist_from_diagonal", "point", "temperature")
        plot_by_prompt(word_stats, "dist_from_diagonal", "bar", "temperature")

        # sum of distances from diagonal line
        plot_by_prompt(dists, "sum", "bar", "prompt")
        plot_by_prompt(dists, "sum", "point", "prompt")
        plot_by_prompt(dists, "sum", "bar", "temperature")
        plot_by_prompt(dists, "sum", "point", "temperature")

    # plot population vs. internal deltaE for each word
    elif args.plot == "deltaE":
        # make 2x4 plot with all prompt-temperature combinations overlaid
        models = ["openchat","mistral-instruct", "gemma-instruct", "llama2", 
                  "starling", "zephyr-mistral", "zephyr-gemma", "llama2-chat"]
        plot_deltaE_subplots(models, word_stats, "mean_populationDeltaE", "mean_internalDeltaE", "prompt", figure_dir, "models-internal-vs-population-deltaE")
        plot_deltaE_subplots(models, word_stats, "mean_populationDeltaE", "mean_internalDeltaE", "temperature", figure_dir, "models-internal-vs-population-deltaE")

        # plot human data separately
        models = ['human']
        plot_deltaE_subplots(models, word_stats, "mean_populationDeltaE", "mean_internalDeltaE", "na", figure_dir, "human-internal-vs-population-deltaE")

        # facetGrid plot as rows=models, cols=prompts 
        models = ["openchat", "starling", 
                  "mistral-instruct", "zephyr-mistral", 
                  "gemma-instruct", "zephyr-gemma", 
                  "llama2", "llama2-chat"]
        plot_deltaE_facetgrid(models, word_stats, "mean_populationDeltaE", "mean_internalDeltaE", "Population ΔE", "Internal ΔE", "models-internal-vs-population-deltaE")

    # [DONE] plot representational alignment: internal ΔE as a function of word imageability and concreteness
    elif args.plot == "word ratings":
        # NOT SQUARE :(
        models = ['human']
        plot_word_ratings_subplots(models, word_stats, "imageability", "mean_internalDeltaE", "na", figure_dir, "human-imageability-vs-internal-deltaE")
        plot_word_ratings_subplots(models, word_stats, "concreteness", "mean_internalDeltaE", "na", figure_dir, "human-concreteness-vs-internal-deltaE")

        # WORKS
        models = ["openchat", "starling", 
                  "mistral-instruct", "zephyr-mistral", 
                  "gemma-instruct", "zephyr-gemma", 
                  "llama2", "llama2-chat"]
        plot_word_ratings_facetgrid(models, word_stats, "imageability", "mean_internalDeltaE", "Imageability Rating", "Internal ΔE", "models-imageability-vs-internal-deltaE")
        plot_word_ratings_facetgrid(models, word_stats, "concreteness", "mean_internalDeltaE", "Concreteness Rating", "Internal ΔE", "models-concreteness-vs-internal-deltaE")

    # plot Jensen-Shannon divergences between human and model responses for each word
    elif args.plot == "JS divergence":
        pass







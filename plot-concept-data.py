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

def plot_response_counts(models, df, metric):
    # combine the prompt and temperature columns
    df['combined'] = df['prompt'] + ", " + df['temperature'].astype(str)
    # remove human data from response_counts
    df = df[df['model_name'] != 'human']

    prompt_order = ['none, default', 'none, 1.5', 'none, 2.0', 'identity, default', 'random, default', 'nonsense, default']

    plt.figure(figsize=(10, 6))

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    sns.barplot(x='model_name', y=f'{metric}_responses', hue='combined', hue_order=prompt_order, data=df, dodge=True, order=models)
    
    plt.title(f"Number of {metric} responses per model, param combo")
    plt.xticks(rotation=45)
    # plot number of responses as horizontal line
    plt.axhline(y=df['total_responses'].values[0], color='k', linestyle='--')
    # put legend outside of plot
    plt.legend(title='Prompt-Temperature', bbox_to_anchor=(1.05, 1), loc='center left')
    plt.tight_layout()
    plt.savefig(f"./{figure_dir}/{metric}-response-counts.pdf")
    plt.clf()


# plot # of subjects sampled vs probability of same table (concept)
def plot_facetgrid_regplot(df, x_var, y_var, x_label, y_label, title):
    g = sns.FacetGrid(df, col="prompt",  row="model_name",  hue="concept_category", margin_titles=True, despine=False, 
                      col_order=["none", "identity", "random", "nonsense"])
    g.map(sns.catplot, x_var, y_var,  kind='bar')

    # # annotate points with concept name
    # for ax in tqdm(g.axes.flat):
    #     for i, txt in enumerate(df["Concept"]):
    #         ax.annotate(txt, (df[x_var].iloc[i], df[y_var].iloc[i]), fontsize=6)

    g.set_axis_labels(x_label, y_label)
    g.set(yscale='linear')
    g.set(xscale='linear')
    #show legend
    g.add_legend()


    plt.tight_layout()
    plt.savefig(f"{figure_dir}/{title}.pdf")
    plt.clf()


def plot_subplots(models, df, x_var, y_var, figure_dir, title, prompt, temperature):
    if models == ['human']:
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    else: 
        # Create a 2x5 grid of subplots
        fig, axs = plt.subplots(2, 5, figsize=(20, 10))
        axs = axs.flatten()  # Flatten the 2D array of axes to make it easier to iterate over
    
    # Find the global y-axis limits
    y_min = df[y_var].min()
    y_max = df[y_var].max()
    y_range = y_max - y_min
    y_padding = y_range * 0.1  # Add 10% padding
    global_y_min = max(0, y_min - y_padding)  # Ensure it doesn't go below 0
    global_y_max = y_max

    order = ["a finch", "a robin", "a chicken", "an eagle", "an ostrich", "a penguin", "a salmon", "a seal", "a dolphin", "a whale", "Abraham Lincoln", "Barack Obama", "Bernie Sanders", "Donald Trump", "Elizabeth Warren", "George W. Bush", "Hillary Clinton", "Joe Biden", "Richard Nixon", "Ronald Reagan"]

    for i, model in enumerate(models):
        print(f"Plotting {model}...")
        model_data = df[df['model_name'] == model]

        if models == ['human']:
            ax = axs
            print(model_data["Concept"])
            # sns.barplot(data=model_data, x=x_var, y=y_var, ax=ax, order=order, hue="concept_category", hue_order=["animals", "politicians"], legend=False, dodge=True)        
        else: 
            ax = axs[i]

        sns.barplot(data=model_data, x=x_var, y=y_var, ax=ax, err_kws={'linewidth': 0.7}, order=order,  hue="concept_category", hue_order=["animals", "politicians"], dodge=True, legend=False)

        ax.set_xlabel("Concept", fontsize=8)
        ax.set_ylabel("P(1 concept)", fontsize=8)
        ax.tick_params(labelsize=8)
        # ax.legend().set_visible(False)

        # Set consistent y-axis limits
        ax.set_ylim(global_y_min, global_y_max)

        # Make the plot square
        ax.set_box_aspect(1)

        # Rotate x-axis labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title(model, fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{figure_dir}/{title}-{prompt}-{temperature}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot data from color task")

    parser.add_argument("--plot", type=str, choices=["P(1 concept)", "response counts"], required=True, help="Which set of plots to generate")

    args = parser.parse_args()

    data_dir = "./output-data/concept-task/all/"
    # add subfolder for this plot type if it doesn't exist
    figure_dir = "./figures/concept-task/" + args.plot + "/"
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    print("Loading compiled data...")
    all_ClusteringResults = pd.read_pickle(f"{data_dir}all-ClusteringResults.pickle")
    all_MAPTables = pd.read_pickle(f"{data_dir}all-MAPTables.pickle")
    stats = pd.read_pickle(f"{data_dir}model-data.pickle")

    param_combos = [["none", "default"], ["none", "1.5"], ["none", "2.0"], ["random", "default"], ["nonsense", "default"], ["identity", "default"]]

    if args.plot == "P(1 concept)":
        print("Plotting P(1 concept)...")
        models = ["human"]
        plot_subplots(models, all_ClusteringResults, "Concept", "ProbabilityOfSameTable",
                    figure_dir, "P(1 concept)", "na", "na")
        
        # models = ["openchat","mistral-instruct", "gemma-instruct", "llama2", "tulu", 
        #           "starling", "zephyr-mistral", "zephyr-gemma", "llama2-chat", "tulu-dpo"]
        # for params in param_combos:
        #     prompt, temperature = params
        #     df = all_ClusteringResults[(all_ClusteringResults['prompt'] == prompt) & (all_ClusteringResults['temperature'] == temperature)]
        #     # plot # of subjects sampled vs probability of same table (concept)
        #     plot_subplots(models, all_ClusteringResults, "Concept", "ProbabilityOfSameTable",
        #                 figure_dir, "P(1 concept)", prompt, temperature)
        
    elif args.plot == "response counts":
        print("Plotting response counts...")
        models = ["openchat","mistral-instruct", "gemma-instruct", "llama2", "tulu", 
                  "starling", "zephyr-mistral", "zephyr-gemma", "llama2-chat", "tulu-dpo"]
        plot_response_counts(models, stats, "valid")
        


    
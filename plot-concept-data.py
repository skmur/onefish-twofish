 # %%
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

def plot_response_counts(df, metric):
    # combine the prompt and temperature columns
    df['combined'] = df['prompt'] + ", " + df['temperature'].astype(str)
    # remove human data from response_counts
    df = df[df['model_name'] != 'human']

    prompt_order = ['none, default', 'none, 1.5', 'none, 2.0', 'identity, default', 'random, default', 'nonsense, default']

    plt.figure(figsize=(10, 6))

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    sns.barplot(x='model_name', y=f'{metric}_responses', hue='combined', hue_order=prompt_order, data=df, dodge=True)
    
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
        axs = [axs]  # Make axs a list for consistent handling
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
        model_data = df[df['model_name'] == model]
        ax = axs[i]

        if models == ['human']:
            sns.barplot(data=model_data, x=x_var, y=y_var, ax=ax, order=order, hue="concept_category", legend=False, dodge=True)        
        else: 
            sns.barplot(data=model_data, x=x_var, y=y_var, ax=ax, err_kws={'linewidth': 0.7}, order=order,  hue="concept_category", dodge=True, legend=False)

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
    parser.add_argument("--compile_data", type=bool, default=True, help="Whether to compile the data from the individual files")

    args = parser.parse_args()

    # - - - - - - - - - - - - - - - - - - -
    CRP_data_dir = "./concept-task/output-data-500iterations/"
    CRP_output_dir = "./concept-task/output-data-500iterations/compiled/"
    raw_data_dir = "./output-data/concept-task/"
    raw_output_dir = "./output-data/concept-task/all/"
    # add subfolder for this plot type if it doesn't exist
    figure_dir = "./figures/concept-task/" + args.plot + "/"
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    # - - - - - - - - - - - - - - - - - - -
    # Load data
    models = ["starling", "openchat", "gemma-instruct", "zephyr-gemma", "mistral-instruct", "zephyr-mistral", "llama2", "llama2-chat", "tulu", "tulu-dpo"]
    param_combos = [["none", "default"], ["none", "1.5"], ["none", "2.0"], ["random", "default"], ["nonsense", "default"], ["identity", "default"]]
    results = ["ClusteringResults", "MAPTables"]
    animals = ["a finch", "a robin", "a chicken", "an eagle", "an ostrich", "a penguin", "a salmon", "a seal", "a dolphin", "a whale"]
    politicians = ["Abraham Lincoln", "Barack Obama", "Bernie Sanders", "Donald Trump", "Elizabeth Warren", "George W. Bush", "Hillary Clinton", "Joe Biden", "Richard Nixon", "Ronald Reagan"] 


    # if output_dir doesn't exist, compile the data
    if args.compile_data:
        print("Compiling data...")
        # add to the data frame
        all_model_ClusteringResults = pd.DataFrame()
        all_model_MAPTables = pd.DataFrame()
        all_ClusteringResults = pd.DataFrame()
        all_MAPTables = pd.DataFrame()
        all_data = pd.DataFrame()
        for model in models:
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
                        all_model_ClusteringResults = pd.concat([all_model_ClusteringResults, df])
                    else:
                        all_model_MAPTables = pd.concat([all_model_MAPTables, df])
        
            # save compiled version of the data to output_dir
            all_model_ClusteringResults.to_csv(f"{CRP_output_dir}{model}-ClusteringResults.csv", index=False)
            all_ClusteringResults = pd.concat([all_ClusteringResults, all_model_ClusteringResults])
            all_model_MAPTables.to_csv(f"{CRP_output_dir}{model}-MAPTables.csv", index=False)
            all_MAPTables = pd.concat([all_MAPTables, all_model_MAPTables])

        all_data.to_pickle(f"{raw_output_dir}all-data.pickle")
        all_ClusteringResults.to_pickle(f"{raw_output_dir}all-ClusteringResults.pickle")
        all_MAPTables.to_pickle(f"{raw_output_dir}all-MAPTables.pickle")
            
    else:
        print("Loading compiled data...")
        all_ClusteringResults = pd.read_pickle(f"{raw_output_dir}all-ClusteringResults.pickle")
        all_MAPTables = pd.read_pickle(f"{raw_output_dir}all-MAPTables.pickle")
        all_data = pd.read_pickle(f"{raw_output_dir}all-data.pickle")


    if args.plot == "P(1 concept)":
        models = ["openchat","mistral-instruct", "gemma-instruct", "llama2", "tulu", 
                  "starling", "zephyr-mistral", "zephyr-gemma", "llama2-chat", "tulu-dpo"]
        for params in param_combos:
            prompt, temperature = params
            df = all_ClusteringResults[(all_ClusteringResults['prompt'] == prompt) & (all_ClusteringResults['temperature'] == temperature)]
            # plot # of subjects sampled vs probability of same table (concept)
            plot_subplots(models, all_model_ClusteringResults, "Concept", "ProbabilityOfSameTable",
                        figure_dir, "P(1 concept)", prompt, temperature)
        
    elif args.plot == "response counts":
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
        
        print(all_data.columns)

        plot_response_counts(all_data, "valid")
        


    
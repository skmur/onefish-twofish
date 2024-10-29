import pandas as pd
import pickle
import math
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import seaborn as sns
import argparse
import os
from tqdm import tqdm

CONCEPTS = ["a finch", "a robin", "a chicken", "an eagle", "an ostrich", "a penguin", "a salmon", "a seal", "a dolphin", "a whale", "Abraham Lincoln", "Barack Obama", "Bernie Sanders", "Donald Trump", "Elizabeth Warren", "George W. Bush", "Hillary Clinton", "Joe Biden", "Richard Nixon", "Ronald Reagan"]
CONCEPT_CATEGORY_PAL = {"animals": sns.color_palette()[1], "politicians": sns.color_palette()[0]}


def plot_response_counts(models, df, metric):
    df = df.groupby(['model_name', 'prompt', 'temperature']).sum()
    df['percent_valid'] = df['valid_responses'] / df['total_responses'] * 100
    df['percent_invalid'] = 100 - df['percent_valid']
    df = df.reset_index()
    print("calculated percent valid and invalid")

    # combine the prompt and temperature columns
    df['combined'] = df['prompt'] + ", " + df['temperature'].astype(str)
    # remove human data from response_counts
    df = df[df['model_name'] != 'human']

    prompt_order = ['none, default', 'none, 1.5', 'none, 2.0', 'identity, default', 'random, default', 'nonsense, default']

    plt.figure(figsize=(10, 6))

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    sns.barplot(x='model_name', y=f'percent_{metric}', hue='combined', hue_order=prompt_order, data=df, dodge=True, order=models)
    
    plt.title(f"Percent of {metric} responses per model, param combo")
    plt.xticks(rotation=45)
    # # plot number of responses as horizontal line
    # plt.axhline(y=df['total_responses'].values[0], color='k', linestyle='--')
    # put legend outside of plot
    plt.legend(title='Prompt-Temperature', bbox_to_anchor=(1.05, 1), loc='center left')
    plt.tight_layout()
    plt.savefig(f"./{figure_dir}/{metric}-response-counts-percentage.pdf")
    plt.clf()

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


def plot_concept_diversity_human(df):
    human_data = df[df.model_name=="human"]
    paired = sns.color_palette("Paired")
    lb, db = paired[0], paired[1] # light blue, dark blue
    lo, do = paired[6], paired[7] # light orange, dark orange
    
    pal = [do, db]
    all_concept_colors = [do for _ in range(10)] + [db for _ in range(10)]
    ax = sns.barplot(
        data=human_data,
        x="Concept",
        order=CONCEPTS,
        palette=all_concept_colors,
        y="p_multiple_concepts",
        err_kws={'linewidth': 1}
    )
    ax.set_xticks(
        range(len(CONCEPTS)), 
        [c.lstrip("a ").lstrip("an ") for c in CONCEPTS[:10]] + [c.split(" ")[-1] for c in CONCEPTS[10:]], 
        ha="right", 
        rotation=35, 
        size="small"
    )
    for i, xt in enumerate(ax.get_xticklabels()):
        xt.set_color(all_concept_colors[i])
    ax.set_ylabel("P(multiple concepts)")
    
    # Plot averages.
    category_means = {}
    for i, category in enumerate(["animals", "politicians"]):
        category_mean = human_data[human_data.concept_category==category]["p_multiple_concepts"].mean()
        category_means[category] = category_mean
        ax.axhline(
            category_mean,
            linestyle="--",
            color=pal[i]
        )
        trans = transforms.blended_transform_factory(
            ax.transAxes, ax.transData
        )
        ax.text(
            1.02, category_mean, 
            category + " mean", 
            transform=trans,
            va="center", ha="left",
            color=pal[i]
        )

    plt.gcf().set_size_inches(7, 3)
    sns.despine()
    plt.savefig(f"./{figure_dir}/human_baseline.pdf", bbox_inches="tight")
    return category_means

def plot_concept_diversity_category_means(df, manipulation, human_means=None):
    paired = sns.color_palette("Paired")
    lb, db = paired[0], paired[1] # light blue, dark blue
    lo, do = paired[6], paired[7] # light orange, dark orange
    bar_colors = [lo, lb, do, db]
    
    if manipulation == "prompt":
        df = df[df.temperature == "default"]
        prompts = ["none", "identity", "random", "nonsense"]
        pretty_prompts = ["none", "persona", "random", "nonsense"]
        height = 8
    elif manipulation == "temperature":
        df = df[df.prompt == "none"]
        prompts = ["default", "1.5", "2.0"]
        pretty_prompts = ["default", "1.5", "2.0"]
        height = 6
    
    families = ["openchat_starling", "mistral", "gemma", "llama"]
    titles = [
        "Openchat vs\nStarling",
        "Mistral-Instruct vs\nZephyr-Mistral",
        "Gemma-Instruct vs\nZephyr-Gemma",
        "Llama/Tulu vs\nLlama-Chat/Tulu-DPO"
    ]
    fig, axes = plt.subplots(
        nrows=len(prompts), ncols=len(families), 
        sharey=True, sharex=True,
        figsize=(12,height)
    )
    
    category_order = ["animals", "politicians"]
    for i, prompt in enumerate(prompts):
        for j, family in enumerate(families):
            ax = axes[i][j]
            ax = sns.barplot(
                data=df[df.model_family==family],
                x="concept_category",
                order=category_order,
                y="p_multiple_concepts",
                hue="aligned",
                hue_order=[False, True],
                err_kws={"linewidth": 1.5},
                ax=ax
            )
            if i == 0:
                ax.set_title(titles[j])
            ax.set_xlabel("")
    
            if j == len(families)-1:
                ax2 = ax.twinx()
                ax2.set_ylabel(pretty_prompts[i], rotation=-90, labelpad=16)
                ax2.set_yticks([])
                
            for bar_idx, bar in enumerate(ax.patches):
                if bar.get_height() != 0:
                    bar.set_color(bar_colors[bar_idx])

#             if j == 0:
#                 ax.set_ylabel("P(>1 concepts)")
            ax.set_ylabel("")
                
            # Add dashed lines corresponding to human category-level means.
            if human_means is not None:
                styles = ["--", ":"]
                for k, category in enumerate(category_order):
                    ax.axhline(
                        human_means[category],
                        linestyle=styles[k],
                        color=CONCEPT_CATEGORY_PAL[category], 
                        label=f"Human baseline ({category})",
                        xmin=k*0.5,
                        xmax=(k+1)*0.5,
                        lw=2.5
                    )

            ax.get_legend().remove()
            
    # Add global x-axis label and y-axis label.
    axes[-1][1].text(1, -0.35, "Concept category")
    axes[2][0].text(-1.2, 0.5, "P(multiple concepts)", rotation=90)
    
    # Custom legend.
    handles, labels = axes[-1][1].get_legend_handles_labels()
    handles[0].set_color("lightgrey")
    handles[1].set_color("black")
    axes[-1][1].legend(
        handles, 
        ["Non-aligned model", "Aligned model"] + labels[2:],
        bbox_to_anchor=(1,-0.53), loc="upper center",
        ncols=4,
        frameon=False
    )
    sns.despine()

    plt.savefig(f"./{figure_dir}/{manipulation}.pdf", bbox_inches="tight")


# def plot_facetgrid_regplot(df, x_var, y_var, x_label, y_label, title):
#     g = sns.FacetGrid(df, col="prompt",  row="model_name",  hue="concept_category", margin_titles=True, despine=False, 
#                       col_order=["none", "identity", "random", "nonsense"])
#     g.map(sns.catplot, x_var, y_var,  kind='bar')

#     # # annotate points with concept name
#     # for ax in tqdm(g.axes.flat):
#     #     for i, txt in enumerate(df["Concept"]):
#     #         ax.annotate(txt, (df[x_var].iloc[i], df[y_var].iloc[i]), fontsize=6)

#     g.set_axis_labels(x_label, y_label)
#     g.set(yscale='linear')
#     g.set(xscale='linear')
#     #show legend
#     g.add_legend()


#     plt.tight_layout()
#     plt.savefig(f"{figure_dir}/{title}.pdf")
#     plt.clf()

# def plot_subplots(models, df, x_var, y_var, figure_dir, title, prompt, temperature):
#     if models == ['human']:
#         fig, axs = plt.subplots(1, 1, figsize=(5, 5))
#     else: 
#         # Create a 2x5 grid of subplots
#         fig, axs = plt.subplots(2, 5, figsize=(20, 10))
#         axs = axs.flatten()  # Flatten the 2D array of axes to make it easier to iterate over
    
#     # Find the global y-axis limits
#     y_min = df[y_var].min()
#     y_max = df[y_var].max()
#     y_range = y_max - y_min
#     y_padding = y_range * 0.1  # Add 10% padding
#     global_y_min = max(0, y_min - y_padding)  # Ensure it doesn't go below 0
#     global_y_max = y_max

#     order = ["a finch", "a robin", "a chicken", "an eagle", "an ostrich", "a penguin", "a salmon", "a seal", "a dolphin", "a whale", "Abraham Lincoln", "Barack Obama", "Bernie Sanders", "Donald Trump", "Elizabeth Warren", "George W. Bush", "Hillary Clinton", "Joe Biden", "Richard Nixon", "Ronald Reagan"]

#     for i, model in enumerate(models):
#         print(f"Plotting {model}...")
#         model_data = df[df['model_name'] == model]

#         if models == ['human']:
#             ax = axs
#             print(model_data["Concept"])
#             # sns.barplot(data=model_data, x=x_var, y=y_var, ax=ax, order=order, hue="concept_category", hue_order=["animals", "politicians"], legend=False, dodge=True)        
#         else: 
#             ax = axs[i]

#         sns.barplot(data=model_data, x=x_var, y=y_var, ax=ax, err_kws={'linewidth': 0.7}, order=order,  hue="concept_category", hue_order=["animals", "politicians"], dodge=True, legend=False)

#         ax.set_xlabel("Concept", fontsize=8)
#         ax.set_ylabel("P(1 concept)", fontsize=8)
#         ax.tick_params(labelsize=8)
#         # ax.legend().set_visible(False)

#         # Set consistent y-axis limits
#         ax.set_ylim(global_y_min, global_y_max)

#         # Make the plot square
#         ax.set_box_aspect(1)

#         # Rotate x-axis labels for better readability
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
#         ax.set_title(model, fontsize=10)

#     plt.tight_layout()
#     plt.savefig(f"{figure_dir}/{title}-{prompt}-{temperature}.pdf", dpi=300, bbox_inches='tight')
#     plt.close()


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot data from color task")

    parser.add_argument("--plot", type=str, choices=["P(multiple concepts)", "response counts"], required=True, help="Which set of plots to generate")

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

    if args.plot == "P(multiple concepts)":
        print("Plotting P(multiple concepts)...")
        all_ClusteringResults = add_model_meta_data(all_ClusteringResults)
        all_ClusteringResults["p_multiple_concepts"] = 1 - all_ClusteringResults["ProbabilityOfSameTable"]
        
        human_means = plot_concept_diversity_human(all_ClusteringResults)
        model_data = all_ClusteringResults[all_ClusteringResults.model_name != "human"]
        plot_concept_diversity_category_means(model_data, "prompt", human_means=human_means)
        plot_concept_diversity_category_means(model_data, "temperature", human_means=human_means)
        
    elif args.plot == "response counts":
        print("Plotting response counts...")
        models = ["openchat", "starling", 
                  "mistral-instruct", "zephyr-mistral", 
                  "gemma-instruct", "zephyr-gemma", 
                  "llama2", "llama2-chat",
                  "tulu", "tulu-dpo"]
        plot_response_counts(models, stats, "valid")
        plot_response_counts(models, stats, "invalid")
        


    
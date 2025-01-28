import pandas as pd
import pickle
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import argparse
import os
import colorsys
from tqdm import tqdm
from brokenaxes import brokenaxes

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# Step sorting function as defined by:
# https://www.alanzucconi.com/2015/09/30/colour-sorting/
def stepSort(r,g,b, repetitions=1):
    # print types of r,g,b
    lum = math.sqrt( .241 * r + .691 * g + .068 * b)

    h, s, v = colorsys.rgb_to_hsv(r,g,b)

    h2 = int(h * repetitions)
    lum2 = int(lum * repetitions)
    v2 = int(v * repetitions)

    if h2 % 2 == 1:
        v2 = repetitions - v2
        lum = repetitions - lum

    return (h2, lum, v2)

def fill_missing_data(df, models, param_combos):
    for model in models:
        for param_combo in param_combos:
            prompt, temperature = param_combo
            if not df[(df["model_name"]==model) & (df["prompt"]==prompt) & (df["temperature"]==temperature)].empty:
                continue
            else:
                print(f"Adding missing data for {model}, {prompt}, {temperature}")
                new_row1 = pd.DataFrame([[model, prompt, temperature, 0]], columns=df.columns)
                new_row2 = pd.DataFrame([[model, prompt, temperature, 0]], columns=df.columns)
                df = pd.concat([df, new_row1], ignore_index=True)
                df = pd.concat([df, new_row2], ignore_index=True)

    return df

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


def plot_dist_from_diag_model_pairs(df, figure_dir, manipulation, human_means=None):
    # plt.rcParams['font.family'] = 'Arial' 
    sns.set_theme(style="ticks", font_scale=1.2)

    paired = sns.color_palette("Paired")
    lg, dg = paired[2], paired[3] # light green, dark green
    bar_colors = [lg, dg]
    
    if manipulation == "prompt":
        df = df[df["temperature"] == "default"]
        prompts = ["none", "identity", "random", "nonsense"]
        pretty_prompts = ["none", "persona", "random", "nonsense"]
        height = 8
    elif manipulation == "temperature":
        df = df[df["prompt"] == "none"]
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
        figsize=(10,height)
    )
    
    for i, prompt in enumerate(prompts):
        for j, family in enumerate(families):
            # select data for this family and prompt/temperature
            if prompt in ["none", "identity", "random", "nonsense"]:
                df_family = df[(df["model_family"]==family) & (df["prompt"]==prompt)]
            else:
                df_family = df[(df["model_family"]==family) & (df["temperature"]==prompt)]

            ax = axes[i][j]
            ax = sns.barplot(
                data=df_family,
                x="aligned",
                order=[False, True],
                y="dist_from_diagonal",
                err_kws={"linewidth": 1.5},
                ax=ax
            )
            if i == 0:
                ax.set_title(titles[j])
            ax.set_xlabel("")
            ax.set_ylim(0, 5)
    
            if j == len(families)-1:
                ax2 = ax.twinx()
                ax2.set_ylabel(pretty_prompts[i], rotation=-90, labelpad=16)
                ax2.set_yticks([])
                
            for bar_idx, bar in enumerate(ax.patches):
                if bar.get_height() != 0:
                    bar.set_color(bar_colors[bar_idx])

            if j == 0:
                ax.set_ylabel("Distance from line of unity")
                
            # if human_means is not None:
            #     ax.axhline(
            #             human_means,
            #             linestyle="--",
            #             color=dg,
            #             label="Human baseline",
            #             lw=2.5
            #         )
                
    # # Custom legend.
    # handles, labels = axes[-1][1].get_legend_handles_labels()
    # handles[0].set_color("lightgrey")
    # handles[1].set_color("black")
    # axes[-1][1].legend(
    #     handles, 
    #     ["Non-aligned model", "Aligned model"] + labels[2:],
    #     bbox_to_anchor=(1,-0.53), loc="upper center",
    #     ncols=4,
    #     frameon=False
    # )
    sns.despine()

    plt.savefig(f"{figure_dir}dist_from_diagonal-modelpairs-{manipulation}.pdf", bbox_inches="tight")



def plot_color_bars(df, models, words, figure_dir):
    temperature = "default"

    for prompt in tqdm(["none", "random", "nonsense", "identity"]):
        fig, axs = plt.subplots(len(models), len(words), figsize=(5*len(models),3*len(words)), frameon=False)
        
        for m_index, model_name in enumerate(models):
            # Load pickled data
            filename = f"color-{model_name}.pickle"
            output_path = "./output-data/color-task/" + filename

            df = pd.read_pickle(output_path)

            if model_name == "human":
                df_model = df[df['model_name'] == 'human']
            else:
                # select data for this model and prompt
                df_model = df[(df['model_name'] == model_name) & (df['prompt'] == prompt) & (df['temperature'] == temperature)]
                # remove -1 values
                df_model = df_model[df_model['rgb1'] != -1]

            # get all unique words
            model_words = df_model['word'].unique()

            for w_index, word in enumerate(words):
                axs[m_index][w_index].get_xaxis().set_ticks([])
                axs[m_index][w_index].get_yaxis().set_ticks([])
                # axs[w_index][m_index].set_ylabel(word, fontsize='medium', rotation='horizontal', ha='right')

                # get all responses for this word
                responses = df_model[df_model['word'] == word]
                rgb = responses['rgb1'].tolist()

                for i in range(len(rgb)):
                    # if rgb is not in range 0-1, scale it
                    if any (x > 1 for x in rgb[i]):
                        rgb[i] = [float(x)/255 for x in rgb[i]]
                    else:
                        rgb[i] = [float(x) for x in rgb[i]]

                # step sort the non-greyscale colors
                rgb.sort(key=lambda rgb: stepSort(rgb[0], rgb[1], rgb[2], 8))

                #--------------------------------------------
                # make plots
                x = 0
                y = 0
                w = 0.0075
                h = 1
                c = 0

                # uncomment for dynamic width based on number of responses
                num_responses = len(rgb)
                w = 1.0 / max(num_responses, 1)

                if word not in model_words:
                    while x < 1:
                        pos = (x, y)
                        axs[m_index][w_index].add_patch(patches.Rectangle(pos, w, h, hatch='xx',fill=False, linewidth=0))
                        x += w
                    continue

                # iterate over percentage values for this word
                # X percent of the bar should be of color associated with that button response
                for color in rgb:
                    pos = (x, y)
                    axs[m_index][w_index].add_patch(patches.Rectangle(pos, w, h, color=color, linewidth=0))
                    # increment to next color in rgb array
                    c += 1

                    # start next block at previous x + width of rectangle this rectangle
                    x += w

                # # fill in the rest of the bar with transparent rectangles
                # while x < 1:
                #     pos = (x, y)
                #     axs[w_index][m_index].add_patch(patches.Rectangle(pos, w, h, hatch='xx',fill=False, linewidth=0))
                #     x += w

                
        for ax, col in zip(axs[0], words):
            ax.set_title(col, fontsize=12)
        for ax, row in zip(axs[:,0], models):
            ax.set_ylabel(row, rotation=0, size='large', ha='right')

        plt.savefig(f'{figure_dir}/colorbars-{prompt}.pdf' ,bbox_inches='tight',dpi=300)
        plt.clf()

def plot_by_prompt(df, metric, plot_type):
    df = df[~df['model_name'].isin(['llama2', 'tulu'])]  # remove llama2 and tulu for both plots

    # get the average human value for metric from df
    human_metric = df[df['model_name'] == 'human'][metric].mean()
    print(human_metric)
    # remove human data from df for plotting
    df = df[df['model_name'] != 'human']

    model_order = ["openchat", "starling", "gemma-instruct", "zephyr-gemma", "mistral-instruct", "zephyr-mistral", "llama2-chat", "tulu-dpo"]
    colors = ['#006400', '#66CDAA', '#003366', '#66B2FF', '#8B0000', '#FF7F7F', '#ffa554', '#f7cb05']    
    
    # map model names to colors
    palette = dict(zip(model_order, colors))

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot for prompt manipulation
    df_prompt = df[df['temperature'] == 'default']
    order_prompt = ['none', 'identity', 'random', 'nonsense']
    
    if plot_type == "point":
        sns.pointplot(x="prompt", y=metric, hue="model_name", hue_order=model_order, palette=palette, data=df_prompt, alpha=0.8, order=order_prompt, ax=ax1)
    elif plot_type == "bar":
        sns.barplot(x="prompt", y=metric, hue="model_name", hue_order=model_order, palette=palette, data=df_prompt, errorbar="ci", order=order_prompt, ax=ax1)

    ax1.set_title(f"{metric} by Prompt")
    ax1.set_xlabel("Prompt")
    ax1.set_ylabel(metric)
    ax1.set_xlim(-0.5, len(order_prompt) - 0.5)
    ax1.set_xticks(range(len(order_prompt)))
    ax1.tick_params(axis='x', rotation=45)

    # Select temperature = 1.5 and 2.0
    df_temp = df[df['temperature'].isin(['1.5', '2.0'])]
    order_temp = ['1.5', '2.0']
    
    if plot_type == "point":
        sns.pointplot(x="temperature", y=metric, hue="model_name", hue_order=model_order, palette=palette, data=df_temp, alpha=0.8, order=order_temp, ax=ax2)
    elif plot_type == "bar":
        sns.barplot(x="temperature", y=metric, hue="model_name", hue_order=model_order, palette=palette, data=df_temp, errorbar="ci", order=order_temp, ax=ax2)

    ax2.set_title(f"{metric} by Temperature")
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel(metric)
    ax2.set_xlim(-0.5, len(order_temp) - 0.5)
    ax2.set_xticks(range(len(order_temp)))
    ax2.tick_params(axis='x', rotation=45)

    # Add human baseline if applicable
    if metric != "jsd_with_human":
        ax1.axhline(y=human_metric, color='k', linestyle='--')
        ax1.text(1, human_metric, "human", fontsize=8, ha='right')
        ax2.axhline(y=human_metric, color='k', linestyle='--')
        ax2.text(1, human_metric, "human", fontsize=8, ha='right')

    if metric == "jsd_with_human":
        ax1.set_ylim(0, 1)
        ax2.set_ylim(0, 1)

    # Add a single legend for the entire figure
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, title='Model Name', bbox_to_anchor=(1.05, 0.5), loc='center left')

    # Remove individual subplot legends
    ax1.get_legend().remove()
    ax2.get_legend().remove()

    plt.tight_layout()
    plt.savefig(f"{figure_dir}/{metric}-{plot_type}.pdf", bbox_inches='tight')
    plt.close()


def plot_response_counts(models, df, metric):
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
    plt.ylim(0, 100)
    # # plot number of responses as horizontal line
    # plt.axhline(y=df['total_responses'].values[0], color='k', linestyle='--')
    # put legend outside of plot
    plt.legend(title='Prompt-Temperature', bbox_to_anchor=(1.05, 1), loc='center left')
    plt.tight_layout()
    plt.savefig(f"./{figure_dir}/{metric}-response-counts-percentage.pdf")
    plt.clf()

def plot_deltaE_subplots(models, word_stats, x_var, y_var, manipulation, figure_dir, title):
    word_stats = word_stats[word_stats['model_name'].isin(models)]

    if manipulation == "prompt":
        word_stats = word_stats[word_stats['temperature'] == 'default']
        order = ['none', 'identity', 'nonsense', 'random']
        color_list = ["#FF5733", "#33FF57", "#3357FF", "#FF33A1"]
        palette = dict(zip(order, color_list))
    elif manipulation == "temperature":
        order = ['default', '1.5', '2.0']
        color_list = ["#FFAE42", "#FFA500", "#FF0000"]
        palette = dict(zip(order, color_list))
    
    if models == ['human']:
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    else: 
        # Create a 2x4 grid of subplots
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
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
    plt.savefig(f"{figure_dir}/{title}-{manipulation}.pdf", dpi=300, bbox_inches='tight')
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
    plt.savefig(f"{figure_dir}/{title}.pdf")
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
    plt.savefig(f"{figure_dir}/{title}-{manipulation}.pdf", dpi=300, bbox_inches='tight')
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
    plt.savefig(f"{figure_dir}/{title}.pdf")
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

    param_combos = [["none", "default"], ["none", "1.5"], ["none", "2.0"], ["random", "default"], ["nonsense", "default"], ["identity", "default"]]
    models = ["openchat", "starling", "mistral-instruct", "zephyr-mistral", "gemma-instruct", "zephyr-gemma", "llama2", "llama2-chat", "tulu", "tulu-dpo"]

    # - - - - - - - - - - - - - - - - - - -
    # [DONE] plot number of valid and invalid responses per model, param combo
    if args.plot == "response counts":
        models = ["openchat", "starling", 
                  "mistral-instruct", "zephyr-mistral", 
                  "gemma-instruct", "zephyr-gemma", 
                  "llama2", "llama2-chat",
                  "tulu", "tulu-dpo"]
        plot_response_counts(models, response_counts, "invalid")
        plot_response_counts(models, response_counts, "valid")

    # [DONE] how much the model responses deviate from a homogenous population 
    # (i.e. where internal variability = population variability)
    elif args.plot == "population homogeneity":
        # plot_by_prompt(word_stats, "dist_from_diagonal", "point")
        # plot_by_prompt(word_stats, "dist_from_diagonal", "bar")

        # alternate plotting method: comparison between aligned and non-aligned models
        human_mean = word_stats[word_stats.model_name == "human"].dist_from_diagonal.mean()

        model_data = word_stats[word_stats.model_name != "human"]
        # remove tulu and llama from the data
        model_data = model_data[~model_data['model_name'].isin(['llama2', 'tulu'])]
        # select columns we need from model_data
        model_data = model_data[["model_name", "prompt", "temperature", "dist_from_diagonal"]]
        model_data = fill_missing_data(model_data, models, param_combos)
        model_data = add_model_meta_data(model_data)
        
        plot_dist_from_diag_model_pairs(model_data, figure_dir, "prompt", human_means=human_mean)
        plot_dist_from_diag_model_pairs(model_data, figure_dir, "temperature", human_means=human_mean)
        
        
    # plot population vs. internal deltaE for each word
    elif args.plot == "deltaE":
        # make 2x4 plot with all prompt-temperature combinations overlaid
        models = ["openchat","mistral-instruct", "gemma-instruct", "llama2-chat", 
                  "starling", "zephyr-mistral", "zephyr-gemma", "tulu-dpo"]
        plot_deltaE_subplots(models, word_stats, "mean_populationDeltaE", "mean_internalDeltaE", "prompt", figure_dir, "models-internal-vs-population-deltaE")
        plot_deltaE_subplots(models, word_stats, "mean_populationDeltaE", "mean_internalDeltaE", "temperature", figure_dir, "models-internal-vs-population-deltaE")

        # plot human data separately
        models = ['human']
        plot_deltaE_subplots(models, word_stats, "mean_populationDeltaE", "mean_internalDeltaE", "na", figure_dir, "human-internal-vs-population-deltaE")

        # facetGrid plot as rows=models, cols=prompts 
        models = ["openchat", "starling", 
                  "mistral-instruct", "zephyr-mistral", 
                  "gemma-instruct", "zephyr-gemma", 
                  "llama2", "llama2-chat",
                  "tulu", "tulu-dpo"]
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
                  "llama2", "llama2-chat",
                  "tulu", "tulu-dpo"]
        plot_word_ratings_facetgrid(models, word_stats, "imageability", "mean_internalDeltaE", "Imageability Rating", "Internal ΔE", "models-imageability-vs-internal-deltaE")
        plot_word_ratings_facetgrid(models, word_stats, "concreteness", "mean_internalDeltaE", "Concreteness Rating", "Internal ΔE", "models-concreteness-vs-internal-deltaE")

    # plot Jensen-Shannon divergences between human and model responses for each word
    elif args.plot == "JS divergence":
        plot_by_prompt(word_stats, "jsd_with_human", "bar",)

    # plot color bars for each model
    elif args.plot == "color bars":
        models = ["human", "openchat", "starling", "gemma-instruct", "zephyr-gemma", "mistral-instruct", "zephyr-mistral", "llama2", "llama2-chat", "tulu", "tulu-dpo"]
        words = ["tomato", "skin", "freedom", "jealousy", "fame"]
        plot_color_bars(word_stats, models, words, figure_dir)







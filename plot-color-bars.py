import csv
import numpy as np
import colormath
import pandas as pd
import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import math
import matplotlib
import colorsys
import seaborn as sns
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, LCHabColor, SpectralColor, sRGBColor, XYZColor, LCHuvColor, IPTColor, HSVColor

#===============================================================================
# FIGURE 1: color response visualization
#===============================================================================

#-------------------------------------------------------------------------------
# HELPER FUNCTIONS

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

def plotSwatches(model_name, df, fig, axs, words):
    

    #--------------------------------------------
    for index, word in enumerate(words):

        # get all responses for this word
        responses = df[df['word'] == word]
        rgb = responses['rgb1'].tolist()

        # convert each number to float and divide by 255
        if model_name == "human":
            for i in range(len(rgb)):
                print(type(rgb[i]))
                rgb[i] = [float(x)/255 for x in rgb[i]]

        print(rgb[0])

        # # convert nested list to array
        # rgb = np.array(rgb)

        print(rgb)

    
        # # # step sort the non-greyscale colors
        # # # rgb.sort(key=lambda(r,g,b): stepSort(r,g,b,8))
        # rgb.sort(key=stepSort(rgb[:][0], rgb[:][1], rgb[:][2],8)) 

        print(rgb)

        #--------------------------------------------
        # make plots
        x = 0
        y = 0
        w = 0.005
        h = 1
        c = 0

        # iterate over percentage values for this word
        # X percent of the bar should be of color associated with that button response
        for color in rgb:

            pos = (x, y)
            ax[index].add_patch(patches.Rectangle(pos, w, h, color=color, linewidth=0))
            # increment to next color in rgb array
            c += 1

            # start next block at previous x + width of rectangle this rectangle
            x += w

            ax[index].get_xaxis().set_ticks([])
            ax[index].get_yaxis().set_ticks([])
            ax[index].set_ylabel(word, fontsize='medium', rotation='horizontal', ha='right')

    plt.savefig('./figures/50words-100subjs/colorbars-%s.png' % model_name ,bbox_inches='tight',dpi=300)
    plt.clf()

def plotInternalVSPopulation(ax, df, x_var, y_var, x_label, y_label, rss, intercept):
    ax.scatter(df[x_var], df[y_var])
    ax.plot([0,100], [0,100], 'k--', alpha=0.5)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 100)

    # plot error bars for each point based on the 95% confidence interval
    for i, txt in enumerate(df['word'].tolist()):
        ax.errorbar(df[x_var][i], df[y_var][i], xerr=df['ci95_hi_populationDeltaE'][i]-df['ci95_lo_populationDeltaE'][i], yerr=df['ci95_hi_internalDeltaE'][i]-df['ci95_lo_internalDeltaE'][i], fmt='o', color='black', alpha=0.5)

    # plot correlation line on the graph
    corr = df[x_var].corr(df[y_var])
    plt.text(10, 90, "internal vs. population deltaE r = %.2f" % corr)

    # plot the words
    for i, txt in enumerate(df['word'].tolist()):
        ax.annotate(txt, (df[x_var][i], df[y_var][i]))

    ax.set_title(f"{model}, RSS={rss}, intercept={intercept}")
    ax.set_xlabel("Internal ΔE")
    ax.set_ylabel("Population ΔE")
    ax.set_aspect('equal')

def plotModelVSHuman(ax, df, x_var, y_var, x_label, y_label):
    sns.regplot(data=df, x=x_var, y=y_var, ax=ax)
    ax.set_title(f"{model}")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Population ΔE")
    ax.set_ylim(0, 100)

    if x_var == 'entropy':
        ax.set_xlim(0, 7)
    elif x_var == 'variance':
        ax.set_xlim(0, 4000)
    elif x_var == 'imageability':
        ax.set_xlim(0, 8)
    elif x_var == 'concreteness':
        ax.set_xlim(0, 8)

    # plot error bars for each point based on the 95% confidence interval
    for i, txt in enumerate(df['word']):
        ax.errorbar(df[x_var][i], df[y_var][i], yerr=df['ci95_hi_internalDeltaE'][i]-df['ci95_lo_internalDeltaE'][i], color='y', alpha=0.5)

    for i, txt in enumerate(df['word']):
        ax.annotate(txt, (df[x_var][i], df[y_var][i]))

#-------------------------------------------------------------------------------
# CALL WITH BOTH WORD SETS

num_subjects = 150
num_words = 199
temperature = "default"
models = ["human", "openchat", "starling", "mistral-instruct", "zephyr-mistral", "zephyr-gemma", "llama2-Chat"]

# to plot all words from human data
df = pd.read_csv("./input-data/color-task/colorref.csv")
words = df['word'].unique()

# or just plot a few words
words = ["skin", "optimism", "freedom", "butterfly", "tree"]


for prompt_condition in ["none", "random", "nonsense", "identity"]:
    fig, axs = plt.subplots(len(words), len(models), figsize=(5*len(models),3*len(words)), frameon=False)
    for m_index, model_name in enumerate(models):
        # Load pickled data
        filename = f"colors-{model_name}-prompt={prompt_condition}-subjects={num_subjects}-temp={temperature}.pickle"
        output_path = "./output-data/" + filename

        df = pd.read_pickle(output_path)

        # select data for this model
        df_model = df[df['model_name'] == model_name]
        # get all unique words
        model_words = df_model['word'].unique()

        for w_index, word in enumerate(words):
            axs[w_index][m_index].get_xaxis().set_ticks([])
            axs[w_index][m_index].get_yaxis().set_ticks([])
            axs[w_index][m_index].set_ylabel(word, fontsize='medium', rotation='horizontal', ha='right')

            # get all responses for this word
            responses = df_model[df_model['word'] == word]
            rgb = responses['rgb1'].tolist()

            for i in range(len(rgb)):
                # if rgb is not in range 0-1, scale it
                if any (x > 1 for x in rgb[i]):
                    print(model_name, rgb[i])
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

            if word not in model_words:
                while x < 1:
                    pos = (x, y)
                    axs[w_index][m_index].add_patch(patches.Rectangle(pos, w, h, hatch='xx',fill=False, linewidth=0))
                    x += w
                continue

            # iterate over percentage values for this word
            # X percent of the bar should be of color associated with that button response
            for color in rgb:
                pos = (x, y)
                axs[w_index][m_index].add_patch(patches.Rectangle(pos, w, h, color=color, linewidth=0))
                # increment to next color in rgb array
                c += 1

                # start next block at previous x + width of rectangle this rectangle
                x += w

            # fill in the rest of the bar with transparent rectangles
            while x < 1:
                pos = (x, y)
                axs[w_index][m_index].add_patch(patches.Rectangle(pos, w, h, hatch='xx',fill=False, linewidth=0))
                x += w

            
    for ax, col in zip(axs[0], models):
        ax.set_title(col, fontsize=12)

    plt.savefig('./figures/50words-100subjs/colorbars.png' ,bbox_inches='tight',dpi=300)
    plt.clf()
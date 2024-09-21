import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.linear_model import LinearRegression 
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, LCHabColor, SpectralColor, sRGBColor, XYZColor, LCHuvColor, IPTColor, HSVColor
from colormath.color_diff import delta_e_cie2000
import sys
import scipy.stats as stats

def rgbToLab(rgb):
    # convert to [0,1] scaled rgb values
    scaled_rgb = (float(rgb[0])/255, float(rgb[1])/255, float(rgb[2])/255)
    # create RGB object
    rgbObject = sRGBColor(scaled_rgb[0], scaled_rgb[1], scaled_rgb[2])

    # convert to Lab
    labObject = convert_color(rgbObject, LabColor)
    labTuple = labObject.get_value_tuple()

    return labTuple

def computeDeltaE(lab1, lab2):
    # compute delta L
    deltaL = lab1[0] - lab2[0]
    # compute delta a
    deltaA = lab1[1] - lab2[1]
    # compute delta b
    deltaB = lab1[2] - lab2[2]
    # compute delta E
    deltaE = np.sqrt(deltaL**2 + deltaA**2 + deltaB**2)

    return deltaE

def processHumanData(temperature):
    # process human data
    human_data = "./input-data/color-task/colorref.csv"
    df_human = pd.read_csv(human_data)

    # make new column with response_r, response_g, response_b combined into single column as a tuple
    df_human['response_rgb'] = df_human[['response_r', 'response_g', 'response_b']].values.tolist()
    # keep columns for 'participantID', 'word', 'response_rgb', condition
    df_human = df_human[['participantID', 'word', 'response_rgb', 'condition']]

    # apply rgbToLab to response_rgb value in each row and add as new column
    df_human['response_lab'] = df_human['response_rgb'].apply(rgbToLab)

    # pivot df on "condition"
    df_human = df_human.pivot(index=['participantID', 'word'], columns='condition', values='response_rgb').reset_index()

    # rename block1 and block2 columns to rgb1 and rgb2
    df_human = df_human.rename(columns={'block1_target_trial': 'rgb1', 'block2_target_trial': 'rgb2'})

    # apply rgbToLab to rgb1 and rgb2 columns and add as new columns lab1 and lab2
    df_human['lab1'] = df_human['rgb1'].apply(rgbToLab)
    df_human['lab2'] = df_human['rgb2'].apply(rgbToLab)

    # compute deltaE for each row
    df_human['deltaE'] = df_human.apply(lambda row: computeDeltaE(row['lab1'], row['lab2']), axis=1)

    # fill in model_name, temperature, and condition columns for all rows

    df_human['model_name'] = ["human"] * len(df_human)
    df_human['temperature'] = [temperature] * len(df_human)
    df_human['condition'] = ["na"] * len(df_human)

    words = df_human['word'].unique()

    return df_human, words

# unpickle model data at pickle_path, merge with glasgow word ratings for the words therein, and return final pandas dataframe
def getDF(pickle_path):
    # unpickle the data
    with open(pickle_path, 'rb') as handle:
        df = pickle.load(handle)

    return df

def mergeGlasgowRatings(df, words):
    # process the glasgow norms excel spreadsheet into pandas dataframe
    df_word_ratings = pd.read_csv("./input-data/color-task/compiled-variance-entropy-glasgowRatings.csv")
    # select rows in df_word_ratings that are in the words list
    df_word_ratings = df_word_ratings[df_word_ratings['word'].isin(words)]

    # merge the two dataframes on the word column
    df = pd.merge(df, df_word_ratings, on='word', how='inner')

    return df


def computeStats(df, metric):
    stats = df.groupby(['model_name', 'temperature', 'word', 'condition', 'variance', 'entropy', 'imageability', 'concreteness'])['deltaE'].agg(['mean', 'count', 'std'])
    
    ci95_hi = []
    ci95_lo = []    
    for i in stats.index:
        m, c, s = stats.loc[i]
        ci95_hi.append(m + 1.96*s/math.sqrt(c))
        ci95_lo.append(m - 1.96*s/math.sqrt(c))

    stats['ci95_hi'] = ci95_hi
    stats['ci95_lo'] = ci95_lo

    stats = stats.add_suffix("_%s" % metric)

    return stats


# population delta e: get average deltae between different participants' color response for each word (cross-block implementation)
def getPopulationDeltaE(df, words, model, temp, condition):

    # words_list = []
    # popdeltaE_list = []

    # create df to store population deltaE for each word (and ['model_name', 'temperature', 'word', 'variance', 'entropy', 'imageability', 'concreteness'])
    df_output = pd.DataFrame(columns=['model_name', 'temperature', 'word', 'condition', 'variance', 'entropy', 'imageability', 'concreteness', 'deltaE'])

    # compute population deltaE for each word
    for word in words: 
        # select rows in df that have the word
        df_word = df[df['word'] == word]
        # get variance, entropy, imageability, and concreteness values for this word\
        variance = df_word['variance'].iloc[0]
        entropy = df_word['entropy'].iloc[0]
        imageability = df_word['imageability'].iloc[0]
        concreteness = df_word['concreteness'].iloc[0]
        
        # get block 1 lab values
        lab1 = df_word['lab1'].tolist()
        # get block 2 lab values
        lab2 = df_word['lab2'].tolist()

        # get all possible pairs of lab values
        pairs = [(lab1[i], lab2[j]) for i in range(len(lab1)) for j in range(len(lab2)) if i != j]
        # get deltaE for each pair
        popdeltaE = [computeDeltaE(pair[0], pair[1]) for pair in pairs]

        # words_list.extend([word]*len(popdeltaE))
        # popdeltaE_list.extend(popdeltaE)

        # add N=len(popdeltaE) rows of each word, variance, entropy, imageability, concreteness, to df_output
        df_output = pd.concat([df_output, pd.DataFrame([[model, temp, word, condition, variance, entropy, imageability, concreteness, popdeltaE[i]] for i in range(len(popdeltaE))], columns=df_output.columns)], ignore_index=True)      
                            
    # dict = {'word': words_list, 'deltaE': popdeltaE_list} 

    return df_output

def runRegression(df, x_label, y_label):

    X_train = np.array(df[x_label]).reshape(-1, 1)

    # use linear regression model
    model = LinearRegression().fit(X_train, df[y_label]) 

    intercept = model.intercept_
    intercept = round(intercept, 2)

    # predicting values 
    y_pred = model.predict(X_train) 

    df['regression_prediction'] = y_pred

    # calculate RSS from regression line
    rss = np.sum(np.square(df['regression_prediction'] - df[y_label]))
    # round to whole number
    rss = round(rss)

    return rss, intercept


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



#----------------------------------------------------------------------
model_names = ["human", "llama2", "llama2-chat"]

conditions = ["none"]
num_subjects = 150
temp = "1.5"
data_dir = "./output-data"
figure_dir = "./figures/allwords-150subjs"
task = "colors"

# get list of words from human data
_, words = processHumanData(temp)

# save rss values for each model, condition, and task version
df_rss = pd.DataFrame()
all_deltae = pd.DataFrame()
all_data = pd.DataFrame()


for condition in conditions:
    fig1, axs1 = plt.subplots(1, len(model_names), figsize=(5*len(model_names), 5))
    fig1.suptitle(f"Internal vs. Population DeltaE for each model and prompting condition={condition}", fontsize=16)

    fig2, axs2 = plt.subplots(1, len(model_names), figsize=(5*len(model_names), 5))
    fig2.suptitle(f"Model Population DeltaE vs. Human Variance for each model and prompting condition={condition}", fontsize=16)

    fig3, axs3 = plt.subplots(1, len(model_names), figsize=(5*len(model_names), 5))
    fig3.suptitle(f"Model Population DeltaE vs. Human Entropy for each model prompting condition={condition}", fontsize=16)

    fig4, axs4 = plt.subplots(1, len(model_names), figsize=(5*len(model_names), 5))
    fig4.suptitle(f"Model Population DeltaE vs. Imageability ratings for each model prompting condition={condition}", fontsize=16)

    fig5, axs5 = plt.subplots(1, len(model_names), figsize=(5*len(model_names), 5))
    fig5.suptitle(f"Model Population DeltaE vs. Concreteness ratings for each model prompting condition={condition}", fontsize=16)

    for model in model_names:
        pickle_path = f"{data_dir}/{task}-{model}-prompt={condition}-subjects={num_subjects}-temp={temp}-revisedtaskprompt.pickle"
        # print model name, condition, and task version
        print("MODEL: ", model)
        print("CONDITION: ", condition)

        if model == "human":
            df, _  = processHumanData(temp)
            df = df[df['word'].isin(words)]
            # group by participantID and word and count number of responses
            print(df.groupby(['word']).size().reset_index(name='count'))
        else:
            df = getDF(pickle_path)
            # remove rows where deltaE = -1 because model generation failed
            df = df[df['deltaE'] != -1]


        # if the set of words in this model's data is not the same as those in words list, set the smaller set as the new words list
        words = list(set(words).intersection(set(df['word'].unique())))

        all_data = pd.concat([all_data, df])

        # merge with glasgow ratings
        df = mergeGlasgowRatings(df, words)

        #------------------------------------------------------------------
        # internal delta e: get average of deltae between each participant's two color responses for each word
        df_internalDeltaE = computeStats(df, "internalDeltaE")

        df_populationDeltaE = getPopulationDeltaE(df, words, model, temp, condition)
        df_populationDeltaE = computeStats(df_populationDeltaE, "populationDeltaE")

        # note 9/10: for 
        df_deltaE = pd.merge(df_internalDeltaE, df_populationDeltaE, on=['model_name', 'temperature', 'word', 'variance', 'entropy', 'imageability', 'concreteness'], how='inner').reset_index()

        all_deltae = pd.concat([all_deltae, df_deltaE])

        rss, intercept = runRegression(df_deltaE, 'mean_internalDeltaE', 'mean_populationDeltaE')
        print('--> Residual sum of squares = '+ str(rss))
        print('--> Intercept = '+ str(intercept))

        # add rss to df_rss
        df_rss = pd.concat([df_rss, pd.DataFrame([[model, condition, rss, intercept]], columns=['model_name', 'condition', 'rss', 'intercept'])])

        # add a subplot for this model
        ax = axs1[model_names.index(model)]
        # plot internal vs. population deltaE
        plotInternalVSPopulation(ax, df_deltaE, 'mean_populationDeltaE', 'mean_internalDeltaE',  'mean_populationDeltaE', 'mean_internalDeltaE', rss, intercept)

        ax = axs2[model_names.index(model)]
        # plot population deltaE vs. variance
        plotModelVSHuman(ax, df_deltaE, 'variance', 'mean_populationDeltaE', 'variance', 'mean_populationDeltaE')

        ax = axs3[model_names.index(model)]
        # plot population deltaE vs. entropy
        plotModelVSHuman(ax, df_deltaE, 'entropy', 'mean_populationDeltaE', 'entropy', 'mean_populationDeltaE')

        ax = axs4[model_names.index(model)]
        # plot population deltaE vs. imageability
        plotModelVSHuman(ax, df_deltaE, 'imageability', 'mean_populationDeltaE', 'imageability', 'mean_populationDeltaE')

        ax = axs5[model_names.index(model)]
        # plot population deltaE vs. concreteness
        plotModelVSHuman(ax, df_deltaE, 'concreteness', 'mean_populationDeltaE', 'concreteness', 'mean_populationDeltaE')


    fig1.savefig(f"{figure_dir}/internalVSpopulationDeltaE-{temp}.png")
    fig2.savefig(f"{figure_dir}/populationDeltaEVSvariance-{temp}.png")
    fig3.savefig(f"{figure_dir}/populationDeltaEVSentropy-{temp}.png")
    fig4.savefig(f"{figure_dir}/populationDeltaEVSimageability-{temp}.png")
    fig5.savefig(f"{figure_dir}/populationDeltaEVSconcreteness-{temp}.png")

    plt.close()

    print("---------------------------------------------------")

    # # save dfs
    # df_rss.to_pickle(f"{data_dir}/rss_values-task_version={condition}-prompt={condition}.pickle")
    # all_deltae.to_pickle(f"{data_dir}/all_deltae-task_version={condition}-prompt={condition}.pickle")
    # all_data.to_pickle(f"{data_dir}/all_data-task_version={condition}-prompt={condition}.pickle")

    
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")
sns.barplot(x='model_name', y='rss', data=df_rss)
# rotate x-axis labels
plt.xticks(rotation=45)
plt.title(f"Model RSS values for color task")
plt.tight_layout()
plt.savefig(f"./{figure_dir}/RSS-prompt={temp}.png")
plt.close()

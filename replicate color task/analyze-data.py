import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression 

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


def processData(color_dict):
    # convert dictionary to pandas dataframe
    df = pd.DataFrame.from_dict(color_dict, orient='index', columns=['condition', 'lab1', 'rgb1', 'lab2', 'rgb2', 'prob', 'deltaE']).reset_index()
    # rename the index column
    df.rename(columns={'index': 'word_subject'}, inplace=True)

    # split the word_subject column into word and subject columns
    df[['word', 'subject']] = df['word_subject'].str.split('_', expand=True)
    # drop the word_subject column
    df.drop(columns=['word_subject'], inplace=True)
    # reorder the columns
    df = df[['word', 'subject', 'rgb1', 'rgb2', 'lab1', 'lab2', 'prob', 'deltaE']]

    # # select the first 60 subjects
    # df = df[df['subject'].astype(int) < 60]
    # print(df)

    words = df['word'].unique()

    return df, words

def computeStats(df, metric):
    stats = df.groupby('word')['deltaE'].agg(['mean', 'count', 'std'])
    
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
def getPopulationDeltaE(df, words):
    words_list = []
    popdeltaE_list = []

    for word in words: 
        # select rows in df that have the word
        df_word = df[df['word'] == word]
        # get block 1 lab values
        lab1 = df_word['lab1'].tolist()
        # get block 2 lab values
        lab2 = df_word['lab2'].tolist()

        # get all possible pairs of lab values
        pairs = [(lab1[i], lab2[j]) for i in range(len(lab1)) for j in range(len(lab2)) if i != j]
        # get deltaE for each pair
        popdeltaE = [computeDeltaE(pair[0], pair[1]) for pair in pairs]

        words_list.extend([word]*len(popdeltaE))
        popdeltaE_list.extend(popdeltaE)

    dict = {'word': words_list, 'deltaE': popdeltaE_list} 
    return pd.DataFrame(dict)


def plotData(df, x_var, y_var, x_label, y_label, condition, temp):
    plt.title("GPT 3.5 - %s" % condition)
    plt.scatter(df[x_var], df[y_var])
    plt.plot([0,100], [0,100], 'k--', alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(0, 100)
    plt.xlim(0, 100)
    # plot error bars for each point based on the 95% confidence interval
    for i, txt in enumerate(df['word']):
        plt.errorbar(df[x_var][i], df[y_var][i], xerr=df['ci95_hi_populationDeltaE'][i]-df['ci95_lo_populationDeltaE'][i], yerr=df['ci95_hi_internalDeltaE'][i]-df['ci95_lo_internalDeltaE'][i], fmt='o', color='black', alpha=0.5)

    # plot correlation line on the graph
    corr = df[x_var].corr(df[y_var])
    plt.text(10, 90, "internal vs. population deltaE r = %.2f" % corr)

    # plot the words
    for i, txt in enumerate(df['word']):
        plt.annotate(txt, (df[x_var][i], df[y_var][i]))
    plt.tight_layout()
    plt.gca().set_aspect('equal')
    # plt.savefig("temp.png")
    plt.savefig(f"./figures/{x_var}_vs_{y_var}_gpt3.5_{condition}_temp={temp}-square.png")
    plt.clf()

def runRegression(df, x_label, y_label):
    # use linear regression model
    model = LinearRegression() 

    X_train = np.array(df[x_label]).reshape(-1, 1)
    
    # fitting the data 
    model.fit(X_train, df[y_label]) 
    # predicting values 
    y_pred = model.predict(X_train) 

    df['regression_prediction'] = y_pred

    # calculate RSS from regression line
    print('residual sum of squares is: '+ str(np.sum(np.square(df['regression_prediction'] - df[y_label]))))



#----------------------------------------------------------------------

conditions = ["none", "identity", "random_context", "nonsense_context"]
num_subjs = 50
temps = [1.0, 1.5]

for temp in temps: 
    for condition in conditions:
        print("CONDITION: %s, TEMP=%.1f" % (condition, temp))
        # unpickle the data
        with open('./output-data/color-%s-temp=%.1f.pickle' % (condition, temp), 'rb') as handle:
            color_dict = pickle.load(handle)
        with open('../input-data/prompts.pkl', 'rb') as handle:
            prompt_dict = pickle.load(handle)

        if temp == 1.0 and condition != "none":
            # REMOVE THIS ONCE GPT SCRIPT IS FIXED: FOR TEMP=1.0 run if condition == "random_context" remove first 6 entries for each key
            if condition == "random_context":
                for key in color_dict:
                    color_dict[key] = color_dict[key][6:]
            if condition == "nonsense_context":
                for key in color_dict:
                    color_dict[key] = color_dict[key][12:]

            # REMOVE THIS ONCE GPT SCRIPT IS FIXED: FOR TEMP=1.0 run compute deltaE for each entry in color_dict
            for key in color_dict:
                lab1 = color_dict[key][1]
                lab2 = color_dict[key][3]
                color_dict[key].append(computeDeltaE(lab1, lab2))

        df, words = processData(color_dict)

        # print(df)

        # process the glasgow norms excel spreadsheet into pandas dataframe
        df_word_ratings = pd.read_csv("../input-data/color-task/compiled-variance-entropy-glasgowRatings.csv")
        # select rows in df_word_ratings that are in the words list
        df_word_ratings = df_word_ratings[df_word_ratings['word'].isin(words)]

        # merge the two dataframes on the word column
        df = pd.merge(df, df_word_ratings, on='word', how='inner')

        # internal delta e: get average of deltae between each participant's two color responses for each word
        df_internalDeltaE = computeStats(df, "internalDeltaE")

        df_populationDeltaE = getPopulationDeltaE(df, words)
        df_populationDeltaE = computeStats(df_populationDeltaE, "populationDeltaE")

        df_deltaE = pd.merge(df_internalDeltaE, df_populationDeltaE, on='word', how='inner').reset_index()

        # print(df_deltaE[:20])

        runRegression(df_deltaE, 'mean_internalDeltaE', 'mean_populationDeltaE')

        # plotData(df_deltaE, 'mean_populationDeltaE', 'mean_internalDeltaE', 'Average Population Delta E - GPT3.5', 'Average Internal Delta E - GPT3.5', condition, temp)
        
        print("- - - - - - - -")

    print("-----------------------------")


# # plotData('variance', 'avg_internal_deltaE_gpt3.5', 'Variance in Human Judgements', 'Average Internal Delta E - GPT3.5')
# # plotData('entropy', 'avg_pop_deltaE_gpt3.5', 'Entropy in Human Judgements', 'Average Population Delta E - GPT3.5')
# # plotData("imageability", 'avg_internal_deltaE_gpt3.5', 'Imageability in Word Ratings', 'Average Internal Delta E - GPT3.5')
# # plotData("concreteness", 'avg_internal_deltaE_gpt3.5', 'Concreteness in Word Ratings', 'Average Internal Delta E - GPT3.5')
# # # plotData("variance", "prob", "Variance in Human Judgements", "Probability of agreement - GPT3.5")
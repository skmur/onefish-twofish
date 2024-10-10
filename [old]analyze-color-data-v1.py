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



def computeVariance(words, df):
    """Fits a gaussian to the LAB values for each word to get per-word mean and variance estimates for each models' responses
    
    block_num = indicates whether to calculate variance for block1 or block2 responses
    """
    dfVariance = pd.DataFrame()

    for block_num in [1, 2]:
        all_means = np.empty([len(words), 3]) #store means
        all_covariance = np.empty([len(words), 3, 3]) #store covariance matrices
        all_variances = np.empty([len(words),]) # store variances
        all_kl_divergences = np.empty([len(words),]) # store KL divergences

        for index, word in enumerate(words):
            # select rows in df that have the word
            df_word = df[df['word'] == word]
            # select block_num column
            labTuples = df_word[f'lab{block_num}'].tolist()

            # store each point in a numpy array of dimensions num_points x LAB
            point_matrix = np.empty([len(labTuples),3])
            for line_num, point in enumerate(labTuples):
                point_matrix[line_num, 0] = labTuples[line_num][0]
                point_matrix[line_num, 1] = labTuples[line_num][1]
                point_matrix[line_num, 2] = labTuples[line_num][2]

            # estimate parameters (mean and covariance) of the likelihood gaussian distribution
            mean = np.mean(point_matrix, axis=0)
            # store mean
            all_means[index,:] = mean

            product = np.matmul(np.transpose(point_matrix - mean), point_matrix - mean)
            covariance = product/len(labTuples)
            # store covariance matrix
            all_covariance[index, :, :] = covariance

            # STEP 2: Calculate variance of each word's Gaussian from its covariance matrix
            variance = covariance.trace()
            all_variances[index] = variance
        
        dfVariance['word'] = words
        dfVariance[f'variance_block{block_num}'] = all_variances

    return dfVariance


def rgbToLab(rgb):
    # convert to [0,1] scaled rgb values
    scaled_rgb = (float(rgb[0])/255, float(rgb[1])/255, float(rgb[2])/255)
    # create RGB object
    rgbObject = sRGBColor(scaled_rgb[0], scaled_rgb[1], scaled_rgb[2])

    # convert to Lab
    labObject = convert_color(rgbObject, LabColor)
    labTuple = labObject.get_value_tuple()

    return labTuple

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
    df_human['prompt'] = ["na"] * len(df_human)

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
    stats = df.groupby(['model_name', 'temperature', 'prompt', 'word'])['deltaE'].agg(['mean', 'count', 'std'])
    
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

# population delta e: get average deltae between different participants' color response for each word (cross-block implementation)
def getPopulationDeltaE(df, words, model, temp, condition):
    # create df to store population deltaE for each word (and ['model_name', 'temperature', 'word', 'variance', 'entropy', 'imageability', 'concreteness'])
    df_output = pd.DataFrame(columns=['model_name', 'temperature', 'prompt', 'word', 'deltaE'])

    # compute population deltaE for each word
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

        # add N=len(popdeltaE) rows of each word, variance, entropy, imageability, concreteness, to df_output
        df_output = pd.concat([df_output, pd.DataFrame([[model, temp, condition, word, popdeltaE[i]] for i in range(len(popdeltaE))], columns=df_output.columns)], ignore_index=True)      
                            
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



#----------------------------------------------------------------------
model_names = ["human", "openchat", "starling", "mistral-instruct", "zephyr-mistral", "gemma-instruct", "zephyr-gemma", "llama2", "llama2-chat"]

param_combos = [["none", "default"], ["none", "1.5"], ["none", "2.0"], ["random", "default"], ["nonsense", "default"], ["identity", "default"]]
num_subjects = 150
data_dir = "/n/holylabs/LABS/ullman_lab/Users/smurthy/onefish-twofish/output-data/color-task/"
output_dir = "./output-data/color-task/all"
figure_dir = "./figures/color-task/"
task = "colors"

human_data = "./input-data/color-task/colorref.csv"
df_human = pd.read_csv(human_data)
words = df_human['word'].unique()

# save rss values for each model, condition, and task version
all_rss = pd.DataFrame()
all_deltae = pd.DataFrame()
all_data = pd.DataFrame()

for params in param_combos:
    prompt = params[0]
    temp = params[1]

    for model in model_names:
        pickle_path = f"{data_dir}/{task}-{model}-prompt={prompt}-subjects={num_subjects}-temp={temp}.pickle"
        # print model name, prompt, and temperature
        print("MODEL: ", model)
        print("CONDITION: ", prompt)
        print("TEMPERATURE: ", temp)

        if model == "human":
            df, _  = processHumanData(temp)
            df = df[df['word'].isin(words)]
        else:
            df = getDF(pickle_path)
            # remove rows where deltaE = -1 because model generation failed
            df = df[df['deltaE'] != -1]

        all_data = pd.concat([all_data, df])
        # if the set of words in this model's data is not the same as those in words list, set the smaller set as the new words list
        words = list(set(words).intersection(set(df['word'].unique())))
        
        #------------------------------------------------------------------
        # internal delta e: get average of deltae between each participant's two color responses for each word
        df_internalDeltaE = computeStats(df, "internalDeltaE")
        df_populationDeltaE = getPopulationDeltaE(df, words, model, temp, prompt)
        df_populationDeltaE = computeStats(df_populationDeltaE, "populationDeltaE")
        df_deltaE = pd.merge(df_internalDeltaE, df_populationDeltaE, on=['model_name', 'temperature', 'word'], how='inner').reset_index()
        print(df_deltaE)
        print(df_deltaE.columns)
        all_deltae = pd.concat([all_deltae, df_deltaE])
        #------------------------------------------------------------------
        rss, intercept = runRegression(df_deltaE, 'mean_internalDeltaE', 'mean_populationDeltaE')
        print('--> Residual sum of squares = '+ str(rss))
        print('--> Intercept = '+ str(intercept))
        # add rss to df_rss
        all_rss = pd.concat([all_rss, pd.DataFrame([[model, temp, prompt, rss, intercept]], columns=['model_name', 'temperature', 'prompt', 'rss', 'intercept'])])

# save dfs
all_rss.to_pickle(f"{output_dir}/rss_values-scriptv1.pickle")
all_deltae.to_pickle(f"{output_dir}/all_deltae-scriptv1.pickle")
all_data.to_pickle(f"{output_dir}/all_data-scriptv1.pickle")
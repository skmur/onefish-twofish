import pandas as pd
import pickle
import math
from sklearn.linear_model import LinearRegression 
from scipy import stats
import numpy as np
from tqdm.auto import tqdm

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def fit_multivariate_gaussian(group, block_num):
    # separate lab1 into L, A, B columns
    group['L'] = group[f'lab{block_num}'].apply(lambda x: x[0])
    group['A'] = group[f'lab{block_num}'].apply(lambda x: x[1])
    group['B'] = group[f'lab{block_num}'].apply(lambda x: x[2])

    # Extract the LAB values as a 2D array
    LAB_values = group[['L', 'A', 'B']].values
    
    # Calculate the mean and covariance matrix
    mean = np.mean(LAB_values, axis=0)
    cov = np.cov(LAB_values, rowvar=False)


    # if cov is not 1d or 2d, variance is none (for words where there's only 1 valid response)
    if len(cov.shape) not in [1, 2]:
        return pd.Series({
            f'mean_L{block_num}': mean[0],
            f'mean_A{block_num}': mean[1],
            f'mean_B{block_num}': mean[2],
            f'var_L{block_num}': None,
            f'var_A{block_num}': None,
            f'var_B{block_num}': None
        })
    
    else: 
        # Extract variances (diagonal elements of the covariance matrix)
        variances = np.diag(cov)
        
        return pd.Series({
            f'mean_L{block_num}': mean[0],
            f'mean_A{block_num}': mean[1],
            f'mean_B{block_num}': mean[2],
            f'var_L{block_num}': variances[0],
            f'var_A{block_num}': variances[1],
            f'var_B{block_num}': variances[2]
        })

def calculate_internal_deltaE(group, metric):
    deltaE_values = group['deltaE']
    
    mean = np.mean(deltaE_values)
    std = np.std(deltaE_values, ddof=1)  # ddof=1 for sample standard deviation
    count = len(deltaE_values)
    
    return pd.Series({
        f'mean_{metric}': mean,
        f'count_{metric}': count,
        f'std_{metric}': std,
        f'ci_lower_{metric}': mean - 1.96*std/math.sqrt(count),
        f'ci_upper_{metric}': mean + 1.96*std/math.sqrt(count)
    })


def compute_deltae(lab1, lab2):
    return np.sqrt(np.sum((np.array(lab1) - np.array(lab2))**2))

# calculate population deltaE between block1 and block2 values
def calculate_population_deltae(group, metric):
    lab_values = group[['lab1', 'lab2']].values
    n = len(lab_values)
    
    # Calculate delta E for all pairs of participants
    deltae_values = []
    for i in range(n):
        for j in range(i+1, n):
            deltae = compute_deltae(lab_values[i][0], lab_values[j][1])
            deltae_values.append(deltae)
    
    deltae_values = np.array(deltae_values)
    
    # Calculate statistics
    mean = np.mean(deltae_values)
    std = np.std(deltae_values, ddof=1)
    count = len(deltae_values)
    
    return pd.Series({
        f'mean_{metric}': mean,
        f'count_{metric}': count,
        f'std_{metric}': std,
        f'ci_lower_{metric}': mean - 1.96*std/math.sqrt(count),
        f'ci_upper_{metric}': mean + 1.96*std/math.sqrt(count)
    })

def run_regression(group, x_label, y_label):
    X = np.array(group[x_label]).reshape(-1, 1)
    y = np.array(group[y_label])
    
    # Use linear regression model to get intercept
    model = LinearRegression().fit(X, y)
    
    intercept = round(model.intercept_, 2)
    slope = round(model.coef_[0], 2)

    # Calculate R-squared (this will still be based on the regression line)
    r_squared = round(model.score(X, y), 4)
    
    return pd.Series({
        'intercept': intercept,
        'slope': slope,
        'r_squared': r_squared
    })


def calculate_diagonal_distances(group, x_label, y_label):
    X = np.array(group[x_label])
    y = np.array(group[y_label])

    # Calculate distances from y=x line (diagonal)
    diagonal_distance = abs(y - X) / np.sqrt(2)
    
    return pd.Series({
        f'dist_from_diagonal': diagonal_distance[0],
    })

    

task = "color"
models = ["human", "openchat", "starling", "mistral-instruct", "zephyr-mistral", "gemma-instruct", "zephyr-gemma", "llama2", "llama2-chat"]
data_dir = f"./output-data/{task}-task/"
output_dir = f"./output-data/{task}-task/all/"

all_model_level_stats = pd.DataFrame()
all_word_level_stats = pd.DataFrame()
valid_response_counts = pd.DataFrame()

for model in models: 
    filename = f"{data_dir}{task}-{model}.pickle"
    print(f"Processing {model}")
    data = load_data(filename)

    # - - - - - - - - - - - - - - - - - - - - - -#
    # group by model_name, prompt, temperature, word and get counts for deltaE = -1
    valid_responses = data.groupby(['model_name', 'prompt', 'temperature']).apply(lambda x: len(x[x['deltaE'] != -1])).reset_index()
    valid_responses = valid_responses.rename(columns={0: 'valid_responses'})
    # add column for total responses
    valid_responses['total_responses'] = data.groupby(['model_name', 'prompt', 'temperature']).size().values
    # add column for invalid responses
    valid_responses['invalid_responses'] = valid_responses['total_responses'] - valid_responses['valid_responses']
    print(valid_responses)

    valid_response_counts = pd.concat([valid_response_counts, valid_responses])
    # - - - - - - - - - - - - - - - - - - - - - -#

    # remove rows where deltaE is -1
    data = data[data['deltaE'] != -1]
    # - - - - - - - - - - - - - - - - - - - - - -#
    # compute mean and variance for block1 and block2 lab responses
    tqdm.pandas(desc="Calculating mean and variance for block1 data")
    mean_variance_block1 = data.groupby(['model_name', 'prompt', 'temperature', 'word']).progress_apply(fit_multivariate_gaussian, block_num="1", include_groups=False).reset_index()
    tqdm.pandas(desc="Calculating mean and variance for block2 data")
    mean_variance_block2 = data.groupby(['model_name', 'prompt', 'temperature', 'word']).progress_apply(fit_multivariate_gaussian, block_num="2", include_groups=False).reset_index()

    # remove rows where variances are None
    mean_variance_block1 = mean_variance_block1[mean_variance_block1['var_L1'].notnull()]
    mean_variance_block2 = mean_variance_block2[mean_variance_block2['var_L2'].notnull()]
    # - - - - - - - - - - - - - - - - - - - - - -#

    # Calculate statistics on internal deltaE with 95% confidence intervals
    tqdm.pandas(desc="Calculating internal deltaE statistics")
    internal_deltaE = data.groupby(['model_name', 'temperature', 'word', 'prompt']).progress_apply(calculate_internal_deltaE, metric="internalDeltaE", include_groups=False).reset_index()
    tqdm.pandas(desc="Calculating population deltaE statistics")
    population_deltaE = data.groupby(['model_name', 'temperature', 'word', 'prompt']).progress_apply(calculate_population_deltae, metric="populationDeltaE", include_groups=False).reset_index()
    
    # - - - - - - - - - - - - - - - - - - - - - -#
    # Merge the dataframes
    summary = pd.merge(mean_variance_block1, mean_variance_block2, on=['model_name', 'prompt', 'temperature', 'word'])
    summary = pd.merge(summary, internal_deltaE, on=['model_name', 'temperature', 'word', 'prompt'])
    summary = pd.merge(summary, population_deltaE, on=['model_name', 'temperature', 'word', 'prompt'])
    
    # - - - - - - - - - - - - - - - - - - - - - -#

    # calculate distances between internal and population deltaE values for each word and the diagonal line (y=x) to see how much the model responses deviate from a homogenous population
    tqdm.pandas(desc="Calculating distances from diagonal line")
    distances = summary.groupby(['model_name', 'temperature', 'word', 'prompt']).progress_apply(calculate_diagonal_distances, x_label='mean_populationDeltaE', y_label='mean_internalDeltaE', include_groups=False).reset_index()
    # merge distances with summary
    summary = pd.merge(summary, distances, on=['model_name', 'temperature', 'word', 'prompt'])
    print(summary)

    all_word_level_stats = pd.concat([all_word_level_stats, summary])
    # - - - - - - - - - - - - - - - - - - - - - -#

    # run regression on mean internal deltaE and mean population deltaE to get slope, intercept and R-squared of best fit line to see whether it lies above or below the diagonal line
    tqdm.pandas(desc="Running regression...")
    regression = summary.groupby(['model_name', 'temperature', 'prompt']).progress_apply(run_regression, x_label='mean_populationDeltaE', y_label='mean_internalDeltaE', include_groups=False).reset_index()
    print(regression)

    all_model_level_stats = pd.concat([all_model_level_stats, regression])

    print("-" * 50)

# pickle em!
all_word_level_stats.to_pickle(f"{output_dir}word-stats.pickle")
all_model_level_stats.to_pickle(f"{output_dir}model-stats.pickle")
valid_response_counts.to_pickle(f"{output_dir}valid-response-counts.pickle")

print("Done!")





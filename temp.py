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


    # if cov is not 1d or 2d, raise an error
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
    
    # Use linear regression model
    model = LinearRegression().fit(X, y)
    
    intercept = round(model.intercept_, 2)
    slope = round(model.coef_[0], 2)
    
    # Predicting values
    y_pred = model.predict(X)
    
    # Calculate RSS from regression line
    rss = round(np.sum(np.square(y_pred - y)))
    
    # Calculate R-squared
    r_squared = round(model.score(X, y), 4)
    
    return pd.Series({
        'intercept': intercept,
        'slope': slope,
        'rss': rss,
        'r_squared': r_squared
    })





task = "color"
models = ["human", "openchat", "starling", "mistral-instruct", "zephyr-mistral", "gemma-instruct", "zephyr-gemma", "llama2", "llama2-chat"]
# models = ["gemma-instruct"]
output_dir = f"./output-data/{task}-task/"

all_model_rss = pd.DataFrame()
all_word_level_stats = pd.DataFrame()

for model in models: 
    filename = f"{output_dir}{task}-{model}.pickle"
    print(f"Processing {model}")
    data = load_data(filename)

    # remove rows where deltaE is -1
    data = data[data['deltaE'] != -1]

    # compute mean and variance for block1 and block2 lab responses
    tqdm.pandas(desc="Calculating mean and variance for block1 data")
    mean_variance_block1 = data.groupby(['model_name', 'prompt', 'temperature', 'word']).progress_apply(fit_multivariate_gaussian, block_num="1", include_groups=False).reset_index()
    tqdm.pandas(desc="Calculating mean and variance for block2 data")
    mean_variance_block2 = data.groupby(['model_name', 'prompt', 'temperature', 'word']).progress_apply(fit_multivariate_gaussian, block_num="2", include_groups=False).reset_index()

    # remove rows where variances are None
    mean_variance_block1 = mean_variance_block1[mean_variance_block1['var_L1'].notnull()]
    mean_variance_block2 = mean_variance_block2[mean_variance_block2['var_L2'].notnull()]

    # Calculate statistics on internal deltaE with 95% confidence intervals
    tqdm.pandas(desc="Calculating internal deltaE statistics")
    internal_deltaE = data.groupby(['model_name', 'temperature', 'word', 'prompt']).progress_apply(calculate_internal_deltaE, metric="internalDeltaE", include_groups=False).reset_index()
    tqdm.pandas(desc="Calculating population deltaE statistics")
    population_deltaE = data.groupby(['model_name', 'temperature', 'word', 'prompt']).progress_apply(calculate_population_deltae, metric="populationDeltaE", include_groups=False).reset_index()

    # Merge the dataframes
    summary = pd.merge(mean_variance_block1, mean_variance_block2, on=['model_name', 'prompt', 'temperature', 'word'])
    summary = pd.merge(summary, internal_deltaE, on=['model_name', 'temperature', 'word', 'prompt'])
    summary = pd.merge(summary, population_deltaE, on=['model_name', 'temperature', 'word', 'prompt'])

    all_word_level_stats = pd.concat([all_word_level_stats, summary])

    print(summary[summary['var_L1'] == None])

    # run regression on mean internal deltaE and mean population deltaE
    tqdm.pandas(desc="Running regression...")
    regression = summary.groupby(['model_name', 'temperature', 'prompt']).apply(run_regression, x_label='mean_populationDeltaE', y_label='mean_internalDeltaE', include_groups=False).reset_index()

    all_model_rss = pd.concat([all_model_rss, regression])

    print("-" * 50)


print(all_word_level_stats)
print(all_model_rss)





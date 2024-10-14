# Main script responsible for creating and running parallel threads

import multiprocessing
import itertools
import csv
import Model
from multiprocessing import Pool
import argparse
from tqdm import tqdm
import os
import pandas as pd


models = ["starling", "openchat", "gemma-instruct", "zephyr-gemma", "mistral-instruct", "zephyr-mistral", "llama2", "llama2-chat", "tulu", "tulu-dpo"]
prompt = "none"
temperature = "2.0"
output_dir = "../output-data-500iterations/"
data_dir = "../input-data/"

for model in models:
    datapath = f"{data_dir}{model}-prompt={prompt}-temp={temperature}-forCRP.csv"

    # Reliability = 93%, 87%, 80%
    # alphaValues = [.08, .16, .32]
    alphaValues = [.16]
    uniformPrior = [False]
    chains = [1]
    concepts = []

    data = [Model.Data(36, 25, datapath), Model.Data(36, 50, datapath), Model.Data(36, 75, datapath), Model.Data(36, 100, datapath)]

    for dataset in data:
        for conceptName, people in dataset.people.items():
            concepts.append((conceptName, people))

    clustering_results_path = f"{output_dir}{model}-prompt={prompt}-temp={temperature}-ClusteringResults.csv"
    map_tables_path = f"{output_dir}{model}-prompt={prompt}-temp={temperature}-MAPTables.csv"
                
    # Main results CSV
    writer1 = csv.writer(open(clustering_results_path, 'w+', newline=''), delimiter=',')
    writer1.writerow(["Concept", "Iteration", "S_Chao1", "NumberOfPeople", "NumberOfTrials", "Prior", "Tables", "Alpha", "Posterior", "Chain", "ProbabilityOfSameTable"])

    # CSV that has each participant's MAP Table (for use in TSNE visualization)
    writer2 = csv.writer(open(map_tables_path, 'w+', newline=''), delimiter=',')
    writer2.writerow(["ID", "Table", "Concept"])

    parameterList = list(itertools.product(uniformPrior, alphaValues, chains, concepts))

    pool = multiprocessing.Pool(64)

    for run in pool.map(Model.parallel, parameterList):
        result1 = run[0]
        result2 = run[1]

        for row in tqdm(result1):
            writer1.writerow(row)

        for row in tqdm(result2):
            writer2.writerow(row)

    pool.close()

    print(f"Finished {model}")


    


    
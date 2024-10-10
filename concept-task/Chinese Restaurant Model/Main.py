# Main script responsible for creating and running parallel threads

import multiprocessing
import itertools
import csv
import Model
from multiprocessing import Pool
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Run experiments on LMs.")
parser.add_argument("--model", type=str, help="Model name")
parser.add_argument("--prompt", type=str, help="Prompt type")
parser.add_argument("--temperature", type=str, help="Temperature")

args = parser.parse_args()
model = args.model
prompt = args.prompt
temperature = args.temperature

data_dir = "../input-data/"
output_dir = "../output-data-500iterations/"
datapath = f"{data_dir}{model}-prompt={prompt}-temp={temperature}-forCRP.csv"

# Reliability = 93%, 87%, 80%
#alphaValues = [.08, .16, .32]
alphaValues = [.16]
# alphaValues = [.197]

# uniformPrior = [False, True]
uniformPrior = [False]
chains = [1]
concepts = []

data = [Model.Data(36, 25, datapath), Model.Data(36, 50, datapath), Model.Data(36, 75, datapath), Model.Data(36, 100, datapath)]

#data = [Model.Data(36, 500)]

for dataset in data:
    for conceptName, people in dataset.people.items():
        print(conceptName, len(people))
        concepts.append((conceptName, people))

# Main results CSV
writer1 = csv.writer(open(f"{output_dir}{model}-prompt={prompt}-temp={temperature}-ClusteringResults.csv", 'w+', newline = ''), delimiter = ',')
writer1.writerow(["Concept", "Iteration", "S_Chao1", "NumberOfPeople", "NumberOfTrials", "Prior", "Tables", "Alpha", "Posterior", "Chain", "ProbabilityOfSameTable"])

# CSV that has each participant's MAP Table (for use in TSNE visualization)
writer2 = csv.writer(open(f"{output_dir}{model}-prompt={prompt}-temp={temperature}-MAPTables.csv", 'w+', newline = ''), delimiter = ',')
writer2.writerow(["ID", "Table", "Concept"])

parameterList = list(itertools.product(uniformPrior, alphaValues, chains, concepts))

pool = multiprocessing.Pool(16)

for run in tqdm(pool.map(Model.parallel, parameterList)):
    result1 = run[0]
    result2 = run[1]

    for row in tqdm(result1):
        writer1.writerow(row)
        print(row)

    for row in tqdm(result2):
        writer2.writerow(row)
        print(row)


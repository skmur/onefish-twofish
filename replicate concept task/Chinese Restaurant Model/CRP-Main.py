# Main script responsible for creating and running parallel threads

import multiprocessing
import itertools
import csv
import Model

if __name__ == '__main__':
    # path to formatted data
    models = ["na"]
    conditions = ["random_generation"]
    categories = ["politicians"]

    for model in models:
        for condition in conditions: 
            for category in categories:
                datapath = "../output-data/%s-%s-%s-forCRP.csv" % (model, category, condition)
                # datapath = "../output-data/gpt3.5-politicians-for-CRP-model-100subjs.csv"


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
                        concepts.append((conceptName, people))

                # Main results CSV
                writer1 = csv.writer(open("%s-ClusteringResults-%s-%s.csv" % (model, category, condition), 'w+', newline = ''), delimiter = ',')
                # writer1 = csv.writer(open("gpt3.5-ClusteringResults-test-9.17.csv", 'w+', newline = ''), delimiter = ',')
                writer1.writerow(["Concept", "Iteration", "S_Chao1", "NumberOfPeople", "NumberOfTrials", "Prior", "Tables", "Alpha", "Posterior", "Chain", "ProbabilityOfSameTable"])

                # CSV that has each participant's MAP Table (for use in TSNE visualization)
                writer2 = csv.writer(open("%s-MAPTables-%s-%s.csv" % (model, category, condition), 'w+', newline = ''), delimiter = ',')
                # writer2 = csv.writer(open("gpt3.5-MAPTables-test-9.17.csv", 'w+', newline = ''), delimiter = ',')
                writer2.writerow(["ID", "Table", "Concept"])

                parameterList = list(itertools.product(uniformPrior, alphaValues, chains, concepts))
                pool = multiprocessing.Pool()

                for run in pool.map(Model.parallel, parameterList):
                    result1 = run[0]
                    result2 = run[1]

                    for row in result1:
                        writer1.writerow(row)

                    for row in result2:
                        writer2.writerow(row)

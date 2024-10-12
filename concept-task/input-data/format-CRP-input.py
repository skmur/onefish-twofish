import pandas as pd
import numpy as np
from scipy import stats
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""REPLICATION OF DATA FORMATTING SECTION OF ANALYSIS.R"""


def format_data_for_CRP(data):
    # rename columns via mapping dictionary
    column_mapping = {
        "subject_num": "ID", "model_name": "Model", "prompt": "Prompting_condition", "temperature": "Temperature",
        "concept_category": "Experiment", "target": "Concept", "choice1": "Choice1", "choice2": "Choice2",
        "answer1": "Response1", "answer2": "Response2"
    }

    data.rename(columns=column_mapping, inplace=True)

    # Recode responses and factorize
    concept_mapping = {
        " a chicken": "chicken", " a dolphin": "dolphin", " a finch": "finch",
        " a penguin": "penguin", " a robin": "robin", " a salmon": "salmon",
        " a seal": "seal", " a whale": "whale", " an ostrich": "ostrich",
        " an eagle": "eagle", " Abraham Lincoln": "Lincoln", " Barack Obama": "Obama",
        " Bernie Sanders": "Sanders", " Donald Trump": "Trump", " Elizabeth Warren": "Warren",
        " George W. Bush": "Bush", " Hillary Clinton": "Clinton", " Joe Biden": "Biden",
        " Richard Nixon": "Nixon", " Ronald Reagan": "Reagan"
    }

    data['Concept'] = data['Concept'].replace(concept_mapping)

    # Create Question column
    data['Question'] = data['Choice1'] + " " + data['Choice2']
    data['Question'] = data['Question'].astype('category')

    # Encode choice number based on Response1 column
    data['ChoiceNumber'] = (data['Response1'] != data['Choice1']).astype(int)

    return data

"""Takes in formatted data for CRP model and calculates participant reliability metrics"""
def calculate_participant_reliability(data):
    # gets the average of the ChoiceNumber for each subject, question, and concept --> if the average is 0 or 1, the response is reliable because both responses are the same
    tmp = data.groupby(['ID', 'Question', 'Concept']).agg({'ChoiceNumber': 'mean'}).reset_index()
    # check that the average is 0 or 1, if it is then Reliability is True, otherwise False
    tmp['Reliability'] = (tmp['ChoiceNumber'] == 0) | (tmp['ChoiceNumber'] == 1)

    data = pd.merge(data, tmp[['ID', 'Question', 'Concept', 'Reliability']], on=['ID', 'Question', 'Concept'])

    # Create reliability dataframe
    reliability = pd.DataFrame({'data_Reliability': data['Reliability'], 'ID': data['ID']})

    # Calculate confidence intervals
    def wilson_score_interval(n, p, z=1.96):
        denominator = 1 + z**2/n
        center_adjusted_probability = p + z*z / (2*n)
        adjusted_standard_deviation = z * np.sqrt((p*(1 - p) + z*z/(4*n)) / n)
        
        lower_bound = (center_adjusted_probability - adjusted_standard_deviation) / denominator
        upper_bound = (center_adjusted_probability + adjusted_standard_deviation) / denominator
        
        return (lower_bound, upper_bound)

    n = len(data)
    p = data['Reliability'].mean()
    lower, upper = wilson_score_interval(n, p)
    reliability['lower'] = lower
    reliability['upper'] = upper

    # Calculate reliability percent
    tmp = reliability.groupby('ID')['data_Reliability'].mean().reset_index()
    tmp.columns = ['ID', 'reliabilityPercent']
    data = pd.merge(data, tmp, on='ID')
    
    reliability = pd.merge(reliability, tmp, on='ID')

    reliability['data_Reliability'] = reliability['data_Reliability'].map({False: "Not Reliable", True: "Reliable"})

    return data, reliability


"""Takes in formatted data for CRP model and calculates intersubject reliability metrics"""
def calculate_intersubject_reliability(data):
    tmp = data.groupby(['Question', 'Concept']).agg({'ChoiceNumber': 'mean'}).reset_index()
    tmp.columns = ['Question', 'Concept', 'QuestionReliability']

    tmp['ConceptReliability'] = tmp.groupby('Concept')['QuestionReliability'].transform('mean')
    print(f"Mean Question Reliability: {tmp['QuestionReliability'].mean().round(4)}")

    return tmp

def verify_human_data(path_to_human_data):
    # load human data
    formattedData = pd.read_csv(path_to_human_data)

    formattedData, participant_reliability = calculate_participant_reliability(formattedData)
    intersubject_reliability = calculate_intersubject_reliability(formattedData)

    # print(participant_reliability)
    # print(intersubject_reliability)

    # Calculate agreement
    alpha = 0.16
    probs = np.random.beta(alpha, alpha, 1000000)
    agree = 1 != np.random.binomial(2, probs)
    print(f"Agreement proportion: {np.mean(agree)}")

    # count number of each value in the Reliability column
    print(formattedData['Reliability'].value_counts())
    print(f"Mean Reliability: {formattedData['Reliability'].mean()}")


model_names = ["openchat", "starling", "mistral-instruct", "zephyr-mistral", "gemma-instruct", "zephyr-gemma", "llama2", "llama2-chat", "tulu", "tulu-dpo"]
param_combos = [["none", "default"], ["none", "1.5"], ["none", "2.0"], ["random", "default"], ["nonsense", "default"], ["identity", "default"]]
categories = ["animals", "politicians"]
data_dir = "../../output-data/concept-task/"

# first, verify code on human data
verify_human_data("../../input-data/concept-task/Responses.csv")

for model in model_names:
    for param_combo in param_combos:
        # read in pickle file
        data = pd.read_pickle(f"{data_dir}concept-{model}.pickle")
        prompt = param_combo[0]
        temperature = param_combo[1]

        print(f"Model: {model}, Prompt: {prompt}, Temperature: {temperature}")

        # select data for the current model, category, and parameter combination
        data = data[(data['prompt'] == prompt) & (data['temperature'] == temperature)]
        # select columns
        data = data[['subject_num', 'model_name', 'prompt', 'temperature', 'concept_category', 'target', 'choice1', 'choice2', 'answer1', 'answer2']]
        # remove rows where either answer is -1
        data = data[(data['answer1'] != -1) & (data['answer2'] != -1)]

        # # verify that there's no data where target is equal to either choice1 or choice2
        # print(data[(data['target'] == data['choice1']) | (data['target'] == data['choice2'])])

        formattedData = format_data_for_CRP(data)
        responses = formattedData.drop_duplicates(subset=['ID', 'Question', 'Concept'])
        responses = responses[['Concept', 'ID', 'Question', 'ChoiceNumber']]
        responses.to_csv(f"{model}-prompt={prompt}-temp={temperature}-forCRP.csv", index=False)

        formattedData, participant_reliability = calculate_participant_reliability(formattedData)
        intersubject_reliability = calculate_intersubject_reliability(formattedData)

        print(intersubject_reliability)

        # Calculate agreement
        alpha = 0.16
        probs = np.random.beta(alpha, alpha, 1000000)
        agree = 1 != np.random.binomial(2, probs)
        print(f"Agreement proportion: {np.mean(agree)}")

        print(f"Mean Reliability: {formattedData['Reliability'].mean()}")
        print(f"Mean Reliability (Animals): {formattedData[formattedData['Experiment'] == 'animals']['Reliability'].mean()}")
        print(f"Mean Reliability (Politicians): {formattedData[formattedData['Experiment'] == 'politicians']['Reliability'].mean()}")

        print("-" * 50)


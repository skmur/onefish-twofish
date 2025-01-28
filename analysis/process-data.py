import pickle
import numpy as np
import pandas as pd
import os
import argparse

from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, sRGBColor

"""
Helper function for Word-color association task
"""
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

"""
Helper function for Word-color association task
"""
def rgbToLab(rgb):
    # convert to [0,1] scaled rgb values
    scaled_rgb = (float(rgb[0])/255, float(rgb[1])/255, float(rgb[2])/255)
    # create RGB object
    rgbObject = sRGBColor(scaled_rgb[0], scaled_rgb[1], scaled_rgb[2])

    # convert to Lab
    labObject = convert_color(rgbObject, LabColor)
    labTuple = labObject.get_value_tuple()

    return labTuple

"""
Helper function for Word-color association task
"""
def process_human_data():
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
    df_human['temperature'] = ["na"] * len(df_human)
    df_human['prompt'] = ["na"] * len(df_human)


    print_summary(df_human, 1, "human", "color")

    return df_human


def process_model_data(models, storage_dir, output_dir, task):
    for model in models:
        concat = pd.DataFrame()
        print(f"Processing model: {model}")
        num_files = 0
        for filename in os.listdir(storage_dir):
            if process_file(filename, model, task):
                data = load_and_process_file(storage_dir, filename, task)
                num_files += 1
                concat = pd.concat([concat, data]) if not concat.empty else data

        print_summary(concat, num_files, model, task)

        # # Uncomment to save the concatenated dataframe
        # if not concat.empty:
        #     with open(f"{output_dir}{task}-{model}.pickle", 'wb') as f:
        #         pickle.dump(concat, f)
        #         print(f"Saved to {output_dir}{task}-{model}.pickle")

def process_file(filename, model, task):
    return (
        "[test]" not in filename
        and model in filename
        and filename.endswith(".pickle")
        and not (model == "llama2" and "chat" in filename)
        and not (model == "tulu" and "dpo" in filename)
        and (task != "color" or "subjects=150" in filename)
    )

def load_and_process_file(storage_dir, filename, task):
    with open(os.path.join(storage_dir, filename), 'rb') as f:
        data = pickle.load(f)

    print(data)
    filename_tmp = filename.split(".pickle")[0]
    params = filename_tmp.split("-")
    prompt, category, temperature = extract_params(params)
    
    print("-->", category, prompt, temperature, len(data))

    # # save data with updated column names to lab storage
    # with open(os.path.join(storage_dir, filename), 'wb') as f:
    #     pickle.dump(data, f)

    return data

def extract_params(params):
    prompt = category = temperature = None
    for param in params:
        if "prompt" in param:
            prompt = param.split("=")[1]
        if "category" in param:
            category = param.split("=")[1]
        if "temp" in param:
            temperature = param.split("=")[1]
    return prompt, category, temperature


def print_summary(concat, num_files, model, task):
    if concat.empty:
        print(f"No data for model: {model}")
    else: 
        print(f"Total number of rows: {len(concat)}")
        # for concept task should equal 2 concept_category * 6 combinations of prompt and temperature
        # for color task should 6 combinations of prompt and temperature
        print(f"Total number of files: {num_files}")
        print(concat.columns)
        print(concat['prompt'].unique())
        print(concat['temperature'].unique())
        if task == "concept":
            print(concat['concept_category'].unique())
        if task == "color":
            # group by prompt, temperature, and print percentage of -1 deltaE values 
            print(concat.groupby(['prompt', 'temperature'])['deltaE'].apply(lambda x: (x == -1).sum()))

    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Process task data")
    parser.add_argument('--task', type=str, choices=['color', 'concept'], help='Task to process (color or concept)')
    parser.add_argument('--storage_dir', type=str, help='Directory to store data')
    args = parser.parse_args()

    models = ["starling", "openchat", "gemma-instruct", "zephyr-gemma", "mistral-instruct", "zephyr-mistral", "llama2", "llama2-chat", "tulu", "tulu-dpo"]
    output_dir = f"../output-data/{args.task}-task/"

    process_model_data(models, args.storage_dir, output_dir, args.task)

    # additionally process human data for color task
    if args.task == "color":
        df_human = process_human_data()

        with open(f"{output_dir}{args.task}-human.pickle", 'wb') as f:
            pickle.dump(df_human, f)
            print(f"Saved to {output_dir}color-human.pickle")

if __name__ == "__main__":
    main()
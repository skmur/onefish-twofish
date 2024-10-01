import pickle
import numpy as np
import pandas as pd
import os
import argparse

def process_task_data(models, lab_storage_dir, output_dir, task):
    for model in models:
        concat = pd.DataFrame()
        print(f"Processing model: {model}")
        num_files = 0
        for filename in os.listdir(lab_storage_dir):
            if process_file(filename, model, task):
                data = load_and_process_file(lab_storage_dir, filename, task)
                num_files += 1
                concat = pd.concat([concat, data]) if not concat.empty else data

        print_summary(concat, num_files, model, task)

        # Uncomment to save the concatenated dataframe
        if not concat.empty:
            with open(f"{output_dir}{task}-{model}.pickle", 'wb') as f:
                pickle.dump(concat, f)
                print(f"Saved to {output_dir}{task}-{model}.pickle")


def process_file(filename, model, task):
    return (
        "[test]" not in filename
        and model in filename
        and filename.endswith(".pickle")
        and not (model == "llama2" and "chat" in filename)
        and (task != "color" or "subjects=150" in filename)
    )

def load_and_process_file(lab_storage_dir, filename, task):
    with open(os.path.join(lab_storage_dir, filename), 'rb') as f:
        data = pickle.load(f)

    filename_tmp = filename.split(".pickle")[0]
    params = filename_tmp.split("-")
    prompt, category, temperature = extract_params(params)
    
    print("-->", category, prompt, temperature, len(data))

    if task == "color":
        data = data.rename(columns={"condition": "prompt"})
    elif task == "concept":
        data['prompt'] = prompt

    # with open(os.path.join(lab_storage_dir, filename), 'wb') as f:
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
        print(f"Total number of files: {num_files}")
        print(concat.columns)
        print(concat['prompt'].unique())
        print(concat['temperature'].unique())
        if task == "concept":
            print(concat['concept_category'].unique())

    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Process task data")
    parser.add_argument('--task', choices=['color', 'concept'], help='Task to process (color or concept)')
    args = parser.parse_args()

    models = ["starling", "openchat", "gemma-instruct", "zephyr-gemma", "mistral-instruct", "zephyr-mistral", "llama2", "llama2-chat"]
    lab_storage_dir = f"/n/holylabs/LABS/ullman_lab/Users/smurthy/onefish-twofish/output-data/{args.task}-task/"
    output_dir = f"./output-data/{args.task}-task/"

    process_task_data(models, lab_storage_dir, output_dir, args.task)

if __name__ == "__main__":
    main()
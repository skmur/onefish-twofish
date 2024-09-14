import argparse
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ImageColor
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

from models import Model


def get_color_sample(model, model_name, temp, context, prompt,  max_tries=3):
    """Extract HEX, RGB, and LAB values from a valid model response."""
    messages = model.format_prompt(model_name, context, prompt)
    output = model.generate(messages, temp)
    
    counter = 0
    while not re.search(r'#[0-9a-fA-F]{6}', output) and counter < max_tries:
        counter += 1
        output = model.generate(model_name, messages, temp)

        if counter == max_tries:
            return -1, -1, -1

    hex_code = re.search(r'#[0-9a-fA-F]{6}', output).group()
    rgb = ImageColor.getcolor(hex_code, "RGB")
    scaled_rgb = tuple(float(c)/255 for c in rgb)
    rgb_object = sRGBColor(*scaled_rgb)
    lab_object = convert_color(rgb_object, LabColor)
    lab_tuple = lab_object.get_value_tuple()

    return hex_code, lab_tuple, scaled_rgb

def compute_delta_e(lab1, lab2):
    """Compute delta E between two colors in Lab space."""
    return delta_e_cie2000(LabColor(*lab1), LabColor(*lab2))

def run_color_task(output_df, args, words, prompts_dict, model):
    """Runs the color task by querying model for two colors for each word in the list. Stores the HEX, RGB, and LAB values for each color, as well as the delta E between the two colors."""

    for subject in range(args.num_subjects):
        context = "" if args.condition == "none" else prompts_dict[subject][args.condition].replace("\n", " ")
        
        for word in words:
            task_prompt = f"What is the HEX code of the color you most associate with the word {word}?" 

            hex1, lab1, rgb1 = get_color_sample(model, args.model_name, args.temperature, context, task_prompt)
            hex2, lab2, rgb2 = get_color_sample(model, args.model_name, args.temperature, context, task_prompt)

            delta_e = -1 if hex1 == -1 or hex2 == -1 else compute_delta_e(lab1, lab2)

            print(f"Word: {word}, Subject: {subject}, HEX1: {hex1}, HEX2: {hex2}, DeltaE: {delta_e:.2f}")

            new_row = pd.DataFrame([[args.model_name, args.temperature, word,
                                     subject, args.condition, 
                                     hex1, lab1, rgb1, hex2, lab2, rgb2, delta_e]], 
                                     columns=output_df.columns)
            output_df = pd.concat([output_df, new_row], ignore_index=True)
        
        print(f"...Done with subject {subject}\n")

        if subject % 10 == 0:
            save_output(output_df, args, subject)

    return output_df

def setup_color_task(args, model, prompts_dict):
    """Setup the color task by loading the color reference data and running the task for a subset of words."""
    df = pd.read_csv("../input-data/color-task/colorref.csv")
    words = df['word'].unique()
    print(f"Number of test words {len(words)}")

    if args.num_words < len(words):
        np.random.seed(42)
        words = np.random.choice(words, args.num_words, replace=False)
        print("Subsampled words:", words)

    output_df = pd.DataFrame(columns=['model_name', 'temperature', 'word', 'subject_num'
                                      'condition', 'hex1', 'lab1', 'rgb1', 'hex2', 'lab2', 'rgb2', 'deltaE'])

    output_df = run_color_task(output_df, args, words, prompts_dict, model)

    save_output(output_df, args)

# validate concept generation
def get_concept_sample(model, model_name, temp, context, prompt, choice1, choice2, max_tries=3):
    """Calls generate() and ensures that there's a valid concept in the model's generation. If not, tries again up to max_tries times."""
    messages = model.format_prompt(model_name, context, prompt)
    output = model.generate(messages, temp)
    
    counter = 0
    while (output not in choice1.lower().strip()) and (output not in choice2.lower().strip()):
        counter += 1
        output = model.generate(messages, temp)

        if counter == max_tries:
            return -1
        
    return output.lower().strip()
        

def run_concept_task(args, model, concepts, prompts_dict):

    for subject_num in range(args.num_subjects):
        context = "" if args.condition == "none" else prompts_dict[subject_num][args.condition].replace("\n", " ")

        for target in concepts:
                for choice1 in concepts:
                    for choice2 in concepts:
                        if (choice1 != choice2) and (choice1 != target) and (choice2 != target):
                            print("target = %s; %s or %s" % (target, choice1, choice2))

                            # switch the order the choices are presented in
                            task_prompt1 = "\nWhich is more similar to a " + target + ", " + choice1 + " or " + choice2 + "? Respond only with " + choice1 + " or " + choice2 + "."
                            task_prompt2 = context + "\nWhich is more similar to a " + target + ", " + choice2 + " or " + choice1 + "? Respond only with " + choice2 + " or " + choice1 + "."

                            answer1 = get_concept_sample(model, args.model_name, args.temperature, context, task_prompt1, choice1, choice2)
                            answer2 = get_concept_sample(model, args.model_name, args.temperature, context, task_prompt2, choice1, choice2)   

                            new_row = pd.DataFrame([[args.model_name, args.
                                                     temperature, subject_num, args.category_concept_task, target, choice1, choice2, answer1, answer2]],
                                                     columns=output_df.columns)
                            output_df = pd.concat([output_df, new_row], ignore_index=True)
        
        print(f"...Done with subject {subject_num}\n")

        if subject_num % 10 == 0:
            save_output(output_df, args, subject_num)

    return output_df


def setup_concept_task(args, model, prompts_dict):
    # concepts from https://direct.mit.edu/opmi/article/doi/10.1162/opmi_a_00072/114924/Latent-Diversity-in-Human-Concepts
    if args.category_concept_task == "animals":
        concepts = ["a finch", "a robin", "a chicken", "an eagle", "an ostrich", "a penguin", "a salmon", "a seal", "a dolphin", "a whale"]
    elif args.category_concept_task == "politicians":
        concepts = ["Abraham Lincoln", "Barack Obama", "Bernie Sanders", "Donald Trump", "Elizabeth Warren", "George W. Bush", "Hillary Clinton", "Joe Biden", "Richard Nixon", "Ronald Reagan"]

    output_df = run_concept_task(args, model, concepts, prompts_dict)

    save_output(output_df, args)

def save_output(output_df, args, subject=None):
    """Save the output dataframe to a pickle file."""
    subject_str = f"-subjects={subject}" if subject is not None else f"-subjects={args.num_subjects}"
    filename = f"{args.task_type}-{args.model_name}-prompt={args.condition}{subject_str}-temp={args.temperature}.pickle"
    output_path = Path(args.output_dir) / filename
    with output_path.open('wb') as handle:
        pickle.dump(output_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments on LMs.")

    # file-related parameters
    parser.add_argument("--o", "--output", type=str, default="./output-data", 
                        help="Directory for output data")
    parser.add("--hf_token_path", default="src/hf_token.txt", type=Path, 
               help="Path to file containing Huggingface token.")
    parser.add_argument("--lab_storage_dir", type=str,
                        default="/n/holylabs/LABS/ullman_lab/Users/smurthy", help="Directory for lab storage")

    # model-related parameters
    parser.add_argument("--model_name", type=str, 
                        default="llama-chat",
                        help="Name of the model")
    parser.add_argument("--model_path", type=str, 
                        default="meta-llama/Llama-2-7b-chat-hf",
                        help="Path to the model")
    parser.add_argument("--temperature", type=str,
                        default="default",
                        help="Temperature for generation")

    # experiment-related parameters
    parser.add_argument("--task", type=str, choices=["colors", "concepts"], required=True, help="Task to run")
    parser.add_argument("--prompt_condition", type=str, required=True, choices=["none", "identity", "random", "nonsense"], help="Prompting condition for appending to task instruction")
    parser.add_argument("--num_subjects", type=int, default=100, help="Number of subjects")

    # color experiment arguments
    parser.add_argument("--num_words_color_task", type=int, default=50, help="Number of words to use for color task")

    # concept experiment arguments
    parser.add_argument("--category_concept_task", type=str, choices=["animals", "politicians"], help="Category of concepts to use for concept task")


    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    model = Model(args.lab_storage_dir)
    model.initialize_model(args.model_name, args.model_path)
    print("Model loaded...")

    if args.task_type == "colors":
        setup_color_task(args, model)
    elif args.task_type == "concepts":
        setup_concept_task(args, model)

    with open('../input-data/prompts.pkl', 'rb') as handle:
        prompts_dict = pickle.load(handle)
    print("Unpickled prompts...")

    if args.task == "colors":
        setup_color_task(args, model, prompts_dict)
    elif args.task == "concepts":
        setup_concept_task(args, model, prompts_dict)

if __name__ == "__main__":
    main()
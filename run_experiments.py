import argparse
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ImageColor
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from itertools import combinations
from tqdm import tqdm

from models import Model

#=====================================================================================
# Functions for color task
#=====================================================================================

def process_color_output(output):
    """Process a single color output from the model."""
    hex_match = re.search(r'#[0-9a-fA-F]{6}', output)
    if not hex_match:
        return -1, -1, -1

    hex_code = hex_match.group()
    rgb = ImageColor.getcolor(hex_code, "RGB")
    scaled_rgb = tuple(float(c)/255 for c in rgb)
    rgb_object = sRGBColor(*scaled_rgb)
    lab_object = convert_color(rgb_object, LabColor)
    lab_tuple = lab_object.get_value_tuple()

    return hex_code, lab_tuple, scaled_rgb

def compute_delta_e(lab1, lab2):
    """Compute delta E between two colors in Lab space."""

    # compute delta L
    deltaL = lab1[0] - lab2[0]
    # compute delta a
    deltaA = lab1[1] - lab2[1]
    # compute delta b
    deltaB = lab1[2] - lab2[2]
    # compute delta E
    deltaE = np.sqrt(deltaL**2 + deltaA**2 + deltaB**2)

    return deltaE

def run_color_task(output_df, args, words, prompts_dict, model):
    """Runs the color task by querying model for two colors for 
    each word in words. Stores the generation, HEX, RGB, and LAB
    values for each color, and computes delta E between the two colors."""
    
    all_prompts = []
    prompt_metadata = []

    for subject in range(args.num_subjects):
        context = "" if args.prompt_condition == "none" else prompts_dict[subject][args.prompt_condition].replace("\n", " ")
        
        for word in words:
            task_prompt = f"Question: What is the HEX code of the color you most associate with the word {word}? You must respond with a guess, even if you're unsure. Make sure your response contains a valid 6-digit HEX code."

            # task_prompt = f"Question: What is the HEX code of the color you most associate with the word {word}? Respond with a single, valid 6-digit HEX code. You must respond with a guess, even if you're unsure. "
            
            # Add two prompts for each word (we need two color samples)
            all_prompts.extend([model.format_prompt(args.model_name, context, task_prompt)] * 2)
            prompt_metadata.extend([(subject, word)] * 2)

    # print number of prompts
    print(f"Number of prompts: {len(all_prompts)}")
    print(f"Prompt example: {all_prompts[0]}")

    all_outputs = model.generate(args.model_name, all_prompts, args.temperature)

    print(f"Number of outputs: {len(all_outputs)}")

    # Process outputs
    for i in range(0, len(all_outputs), 2):
        subject, word = prompt_metadata[i]
        hex1, lab1, rgb1 = process_color_output(all_outputs[i])
        hex2, lab2, rgb2 = process_color_output(all_outputs[i+1])

        delta_e = -1 if hex1 == -1 or hex2 == -1 else compute_delta_e(lab1, lab2)

        print(f"Word: {word}, Subject: {subject}, Generation1: {all_outputs[i]}, HEX1: {hex1}, Generation2: {all_outputs[i+1]}, HEX2: {hex2}, DeltaE: {delta_e:.2f}")

        new_row = pd.DataFrame([[args.model_name, args.temperature, word,
                                 subject, args.prompt_condition, all_outputs[i],
                                 hex1, lab1, rgb1, all_outputs[i+1], hex2, lab2, rgb2, delta_e]], 
                                 columns=output_df.columns)
        output_df = pd.concat([output_df, new_row], ignore_index=True)

        if (i // 2) % (50 * len(words)) == 0:  # Save every 50 subjects
            save_output(output_df, args, i // (2 * len(words)))

    return output_df

def setup_color_task(args, model):
    """Setup the color task by loading the color reference data and running the task for a subset of words."""

    with open('./input-data/color-task/prompts.pkl', 'rb') as handle:
        prompts_dict = pickle.load(handle)
    print("Unpickled prompts...")

    df = pd.read_csv("./input-data/color-task/colorref.csv")
    words = df['word'].unique()
    print(f"Number of test words {len(words)}")

    if args.num_words < len(words):
        np.random.seed(42)
        words = np.random.choice(words, args.num_words, replace=False)
        print("Subsampled words:", words)

    output_df = pd.DataFrame(columns=['model_name', 'temperature', 'word', 'subject_num', 'prompt', 'generation1', 'hex1', 'lab1', 'rgb1', 'generation2', 'hex2', 'lab2', 'rgb2', 'deltaE'])

    output_df = run_color_task(output_df, args, words, prompts_dict, model)

    save_output(output_df, args)

#=====================================================================================
# Functions for concept task
#=====================================================================================

def process_concept_output(output, choice1, choice2):
    """Process a single concept output from the model."""
    output = output.lower().strip()

    # regex search for choice1 and choice2
    choice1_match = re.search(rf"\b{choice1.lower()}\b", output)
    choice2_match = re.search(rf"\b{choice2.lower()}\b", output)

    if choice1_match:
        return choice1
    elif choice2_match:
        return choice2
    else:
        return -1


def run_concept_task(output_df, args, model, concepts, prompts_dict):
    """Runs the concept task by querying the model for two responses 
    to a forced choice similarity judgement for each target concept 
    in the list. Stores the responses for each choice and the target 
    concept."""

    all_prompts = []
    prompt_metadata = []

    # set seed for np.random.choice for reproducibility
    np.random.seed(42)

    print("Compiling prompts...")
    # for each subject, get a context and generate prompts for each target, choice1, choice2 triplet
    for subject_num in tqdm(range(len(prompts_dict))):
        context = "" if args.prompt_condition == "none" else prompts_dict[subject_num][args.prompt_condition].replace("\n", " ")

        # "Each participant was randomly assigned to a single target concept..."
        target = np.random.choice(concepts)
        filtered_concepts = [c for c in concepts if c != target]
        
        # "...and presented with 36 unique pairs of other concepts in the domain (drawing from the 10 concepts in each domain)"
        unique_pairs = list(combinations(filtered_concepts, 2))

        # "Each trial was shown twice (for a total of 72 trials)"
        # --> we switch the order of the choices in the second trial to control for order effects
        for i in range(len(unique_pairs)):
            choice1, choice2 = unique_pairs[i]

            task_prompt1 = f"Question: Which is more similar to a {target}, {choice1} or {choice2}? Respond only with \"{choice1}\" or \"{choice2}\"."
            task_prompt2 = f"Question: Which is more similar to a {target}, {choice2} or {choice1}? Respond only with \"{choice2}\" or \"{choice1}\"."
            
            all_prompts.extend([model.format_prompt(args.model_name, context, task_prompt1), model.format_prompt(args.model_name, context, task_prompt2)])

            prompt_metadata.extend([(subject_num, target, choice1, choice2), (subject_num, target, choice2, choice1)])

    ## print number of prompts
    print(f"Number of prompts: {len(all_prompts)}")
    print(f"Prompt example: {all_prompts[0]}")

    all_outputs = model.generate(args.model_name, all_prompts, args.temperature)

    # print number of outputs
    print(f"Number of outputs: {len(all_outputs)}")

    print("Processing outputs...")
    for i in tqdm(range(0, len(all_outputs), 2)):
        print(all_prompts[i])
        print("-->" + all_outputs[i])
        print(all_prompts[i+1])
        print("-->" + all_outputs[i+1])

        subject_num, target, choice1, choice2 = prompt_metadata[i]
        answer1 = process_concept_output(all_outputs[i], choice1, choice2)
        answer2 = process_concept_output(all_outputs[i+1], choice2, choice1)

        print("----------------")

        new_row = pd.DataFrame([[args.model_name, args.temperature, subject_num, args.concept_category, args.prompt_condition, target, choice1, choice2, all_outputs[i], answer1, all_outputs[i+1], answer2]],columns=output_df.columns)

        output_df = pd.concat([output_df, new_row], ignore_index=True)

    return output_df

def setup_concept_task(args, model):
    # concepts from https://direct.mit.edu/opmi/article/doi/10.1162/opmi_a_00072/114924/Latent-Diversity-in-Human-Concepts

    with open('./input-data/concept-task/prompts.pkl', 'rb') as handle:
        prompts_dict = pickle.load(handle)
    print("Unpickled prompts for concept task...")

    # From paper: "We recruited 1,799 participants on Amazon Mechanical Turk. Half were asked to make similarity judgements about animals and the other half to make judgements about U.S. politicians"
    if args.concept_category == "animals":
        concepts = ["a finch", "a robin", "a chicken", "an eagle", "an ostrich", "a penguin", "a salmon", "a seal", "a dolphin", "a whale"]
        # select subject ids 1-900
        prompts_dict = {k: v for k, v in prompts_dict.items() if k < 900}
    elif args.concept_category == "politicians":
        concepts = ["Abraham Lincoln", "Barack Obama", "Bernie Sanders", "Donald Trump", "Elizabeth Warren", "George W. Bush", "Hillary Clinton", "Joe Biden", "Richard Nixon", "Ronald Reagan"]
        # select subject ids 901-1800
        prompts_dict = {k: v for k, v in prompts_dict.items() if k >= 901 and k <= 1800}
        # reset subject ids to start from 0
        prompts_dict = {k-901: v for k, v in prompts_dict.items()}

    output_df = pd.DataFrame(columns=['model_name', 'temperature', 'subject_num', 'concept_category', 'prompt', 'target', 'choice1', 'choice2', 'generation1', 'answer1', 'generation2', 'answer2'])

    output_df = run_concept_task(output_df, args, model, concepts, prompts_dict)

    save_output(output_df, args)

#=====================================================================================
# Functions for both tasks
#=====================================================================================
def save_output(output_df, args, subject=None):
    """Save the output dataframe to a pickle file."""

    if args.task == "colors":
        subject_str = f"-subjects={subject}" if subject is not None else f"-subjects={args.num_subjects}"
        filename = f"{args.task}-{args.model_name}-prompt={args.prompt_condition}{subject_str}-temp={args.temperature}.pickle"
        # add "color-task" folder to output path
        output_path = Path(args.lab_storage_dir) / Path(args.output) / "color-task" / filename

    elif args.task == "concepts":
        filename = f"{args.task}-{args.model_name}-category={args.concept_category}-prompt={args.prompt_condition}-temp={args.temperature}.pickle"
        output_path = Path(args.lab_storage_dir) / Path(args.output) / "concept-task" / filename

    print(output_df)

    with output_path.open('wb') as handle:
        pickle.dump(output_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments on LMs.")

    # file-related parameters
    parser.add_argument("--output", type=str, default="./output-data/", 
                        help="Directory for output data")
    # REMOVE THIS LATER
    parser.add_argument("--hf_token", 
                        default="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", type=str, 
                        help="Huggingface token.")
    parser.add_argument("--lab_storage_dir", type=str,
                        default="/n/holylabs/LABS/ullman_lab/Users/smurthy/onefish-twofish", 
                        help="Directory for lab storage")

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
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size for generation")

    # experiment-related parameters
    parser.add_argument("--task", type=str, choices=["colors", "concepts"], 
                        required=True, help="Task to run")
    parser.add_argument("--prompt_condition", type=str, required=True, 
                        choices=["none", "identity", "random", "nonsense"], 
                        help="Prompting condition for appending to task instruction")
    

    # color experiment arguments
    parser.add_argument("--num_words", type=int, default=50, 
                        help="Number of words to use for color task")
    parser.add_argument("--num_subjects", type=int, default=100, 
                        help="Number of subjects for color task (concept task is fixed)")

    # concept experiment arguments
    parser.add_argument("--concept_category", type=str, 
                        choices=["animals", "politicians"], 
                        help="Category of concepts to use for concept task")


    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    model = Model(args.lab_storage_dir, args.hf_token, args.batch_size)
    model.initialize_model(args.model_name, args.model_path)
    model.shard_model()

    # print model name and path
    print(f"Model name: {args.model_name}")
    print(f"Model path: {args.model_path}")
    print("Model loaded and sharded...")

    if args.task == "colors":
        setup_color_task(args, model)
    elif args.task == "concepts":
        setup_concept_task(args, model)

if __name__ == "__main__":
    main()
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


# def get_color_sample(model, args, context, prompt):
#     """Extract HEX, RGB, and LAB values from a valid model response."""
#     messages = model.format_prompt(args.model_name, context, prompt)
#     output = model.generate(args.model_name, messages, args.temperature)
#     print(output)
    
#     counter = 0
#     max_tries=3
#     while not re.search(r'#[0-9a-fA-F]{6}', output) and counter < max_tries:
#         counter += 1
#         output = model.generate(args.model_name, messages, args.temperature)

#         if counter == max_tries:
#             return -1, -1, -1

#     hex_code = re.search(r'#[0-9a-fA-F]{6}', output).group()
#     rgb = ImageColor.getcolor(hex_code, "RGB")
#     scaled_rgb = tuple(float(c)/255 for c in rgb)
#     rgb_object = sRGBColor(*scaled_rgb)
#     lab_object = convert_color(rgb_object, LabColor)
#     lab_tuple = lab_object.get_value_tuple()

#     return hex_code, lab_tuple, scaled_rgb

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

# def run_color_task(output_df, args, words, prompts_dict, model):
#     """Runs the color task by querying model for two colors for each word in the list. 
#     Stores the HEX, RGB, and LAB values for each color, as well as the delta E between the two colors."""

#     all_prompts = []
#     prompt_metadata = []

#     for subject in range(args.num_subjects):
#         context = "" if args.prompt_condition == "none" else prompts_dict[subject][args.prompt_condition].replace("\n", " ")
        
#         for word in words:
#             task_prompt = f"What is the HEX code of the color you most associate with the word {word}?" 

#             hex1, lab1, rgb1 = get_color_sample(model, args, context, task_prompt)
#             hex2, lab2, rgb2 = get_color_sample(model, args, context, task_prompt)

#             delta_e = -1 if hex1 == -1 or hex2 == -1 else compute_delta_e(lab1, lab2)

#             print(f"Word: {word}, Subject: {subject}, HEX1: {hex1}, HEX2: {hex2}, DeltaE: {delta_e:.2f}")

#             new_row = pd.DataFrame([[args.model_name, args.temperature, word,
#                                      subject, args.prompt_condition, 
#                                      hex1, lab1, rgb1, hex2, lab2, rgb2, delta_e]], 
#                                      columns=output_df.columns)
#             output_df = pd.concat([output_df, new_row], ignore_index=True)
        
#         print(f"...Done with subject {subject}\n")

#         if subject % 10 == 0:
#             save_output(output_df, args, subject)

#     return output_df

def run_color_task(output_df, args, words, prompts_dict, model):
    """Runs the color task by querying model for two colors for 
    each word in words. Stores the HEX, RGB, and LAB values for each color,
    Computres delta E between the two colors."""
    
    all_prompts = []
    prompt_metadata = []

    for subject in range(args.num_subjects):
        context = "" if args.prompt_condition == "none" else prompts_dict[subject][args.prompt_condition].replace("\n", " ")
        
        for word in words:
            task_prompt = f"What is the HEX code of the color you most associate with the word {word}?" 
            
            # Add two prompts for each word (we need two color samples)
            all_prompts.extend([model.format_prompt(args.model_name, context, task_prompt)] * 2)
            prompt_metadata.extend([(subject, word)] * 2)

    # check that all prompts are strings
    print(all(isinstance(prompt, str) for prompt in all_prompts))

    # batch the prompts to prevent CUDA out of memory erros and concatenate outputs into all outputs
    all_outputs = []

    for i in range(0, len(all_prompts), args.batch_size):
        batch_outputs = model.generate(args.model_name, all_prompts[i:i+10], args.temperature)
        all_outputs.extend(batch_outputs)
        print(batch_outputs)

    # # Batch generate all prompts
    # batch_outputs = model.generate(args.model_name, all_prompts, args.temperature)

    # Process outputs
    for i in range(0, len(all_outputs), 2):
        subject, word = prompt_metadata[i]
        hex1, lab1, rgb1 = process_color_output(all_outputs[i])
        hex2, lab2, rgb2 = process_color_output(all_outputs[i+1])

        delta_e = -1 if hex1 == -1 or hex2 == -1 else compute_delta_e(lab1, lab2)

        print(f"Word: {word}, Subject: {subject}, HEX1: {hex1}, HEX2: {hex2}, DeltaE: {delta_e:.2f}")

        new_row = pd.DataFrame([[args.model_name, args.temperature, word,
                                 subject, args.prompt_condition, 
                                 hex1, lab1, rgb1, hex2, lab2, rgb2, delta_e]], 
                                 columns=output_df.columns)
        output_df = pd.concat([output_df, new_row], ignore_index=True)

        if (i // 2) % (10 * len(words)) == 0:  # Save every 10 subjects
            save_output(output_df, args, i // (2 * len(words)))

    return output_df


def setup_color_task(args, model, prompts_dict):
    """Setup the color task by loading the color reference data and running the task for a subset of words."""
    df = pd.read_csv("./input-data/color-task/colorref.csv")
    words = df['word'].unique()
    print(f"Number of test words {len(words)}")

    if args.num_words < len(words):
        np.random.seed(42)
        words = np.random.choice(words, args.num_words, replace=False)
        print("Subsampled words:", words)

    output_df = pd.DataFrame(columns=['model_name', 'temperature', 'word', 'subject_num', 'condition', 'hex1', 'lab1', 'rgb1', 'hex2', 'lab2', 'rgb2', 'deltaE'])

    output_df = run_color_task(output_df, args, words, prompts_dict, model)

    save_output(output_df, args)


#---------------------------------------------------
# validate concept generation
# def get_concept_sample(model, model_name, temp, context, prompt, choice1, choice2, max_tries=3):
#     """Calls generate() and ensures that there's a valid concept in the model's generation. If not, tries again up to max_tries times."""
#     messages = model.format_prompt(model_name, context, prompt)
#     print(messages)
#     # batch the prompts by adding to the list

#     output = model.generate(messages, temp)
    
#     counter = 0
#     while (output not in choice1.lower().strip()) and (output not in choice2.lower().strip()):
#         counter += 1
#         output = model.generate(messages, temp)

#         if counter == max_tries:
#             return -1
        
#     return output.lower().strip()

def process_concept_output(output, choice1, choice2):
    """Process a single concept output from the model."""
    output = output.lower().strip()
    if output == choice1.lower().strip():
        return choice1
    elif output == choice2.lower().strip():
        return choice2
    else:
        return -1
        

# def run_concept_task(args, model, concepts, prompts_dict):
#     """Runs the concept task by querying the model for two responses to a forced choice similarity judgement for each target concept in the list. Stores the responses for each choice and the target concept."""

#     for subject_num in range(args.num_subjects):
#         context = "" if args.prompt_condition == "none" else prompts_dict[subject_num][args.prompt_condition].replace("\n", " ")

#         for target in concepts:
#                 for choice1 in concepts:
#                     for choice2 in concepts:
#                         if (choice1 != choice2) and (choice1 != target) and (choice2 != target):
#                             print("target = %s; %s or %s" % (target, choice1, choice2))

#                             # switch the order the choices are presented in
#                             task_prompt1 = "\nWhich is more similar to a " + target + ", " + choice1 + " or " + choice2 + "? Respond only with " + choice1 + " or " + choice2 + "."
#                             task_prompt2 = "\nWhich is more similar to a " + target + ", " + choice2 + " or " + choice1 + "? Respond only with " + choice2 + " or " + choice1 + "."

#                             answer1 = get_concept_sample(model, args.model_name, args.temperature, context, task_prompt1, choice1, choice2)
#                             answer2 = get_concept_sample(model, args.model_name, args.temperature, context, task_prompt2, choice1, choice2)   

#                             new_row = pd.DataFrame([[args.model_name, args.
#                                                      temperature, subject_num, args.concept_category, target, choice1, choice2, answer1, answer2]],
#                                                      columns=output_df.columns)
#                             output_df = pd.concat([output_df, new_row], ignore_index=True)
        
#         print(f"...Done with subject {subject_num}\n")

#         if subject_num % 10 == 0:
#             save_output(output_df, args, subject_num)

#     return output_df

def run_concept_task(args, model, concepts, prompts_dict):
    """Runs the concept task by querying the model for two responses 
    to a forced choice similarity judgement for each target concept 
    in the list. Stores the responses for each choice and the target 
    concept."""

    all_prompts = []
    prompt_metadata = []

    # for each subject, get a context and generate prompts for each target, choice1, choice2 triplet
    for subject_num in range(args.num_subjects):
        context = "" if args.prompt_condition == "none" else prompts_dict[subject_num][args.prompt_condition].replace("\n", " ")

        for target in concepts:
            for choice1 in concepts:
                for choice2 in concepts:
                    if (choice1 != choice2) and (choice1 != target) and (choice2 != target):
                        task_prompt1 = f"\nWhich is more similar to a {target}, {choice1} or {choice2}? Respond only with {choice1} or {choice2}."
                        task_prompt2 = f"\nWhich is more similar to a {target}, {choice2} or {choice1}? Respond only with {choice2} or {choice1}."

                        all_prompts.extend([
                            model.format_prompt(args.model_name, context, task_prompt1),
                            model.format_prompt(args.model_name, context, task_prompt2)
                        ])
                        prompt_metadata.extend([
                            (subject_num, target, choice1, choice2),
                            (subject_num, target, choice2, choice1)
                        ])

    # Batch generate all prompts
    all_outputs = model.generate(args.model_name, all_prompts, args.temperature)

    # Process outputs
    output_df = pd.DataFrame(columns=['model_name', 'temperature', 'subject_num', 'concept_category', 'target', 'choice1', 'choice2', 'answer1', 'answer2'])

    for i in range(0, len(all_outputs), 2):
        subject_num, target, choice1, choice2 = prompt_metadata[i]
        answer1 = process_concept_output(all_outputs[i], choice1, choice2)
        answer2 = process_concept_output(all_outputs[i+1], choice2, choice1)

        new_row = pd.DataFrame([[args.model_name, args.temperature, subject_num, args.concept_category, target, choice1, choice2, answer1, answer2]],
                                columns=output_df.columns)
        output_df = pd.concat([output_df, new_row], ignore_index=True)

        if (i // 2) % (10 * len(concepts) * (len(concepts) - 1) * (len(concepts) - 2)) == 0:  # Save every 10 subjects
            save_output(output_df, args, i // (2 * len(concepts) * (len(concepts) - 1) * (len(concepts) - 2)))

    return output_df


def setup_concept_task(args, model, prompts_dict):
    # concepts from https://direct.mit.edu/opmi/article/doi/10.1162/opmi_a_00072/114924/Latent-Diversity-in-Human-Concepts
    if args.concept_category == "animals":
        concepts = ["a finch", "a robin", "a chicken", "an eagle", "an ostrich", 
                    "a penguin", "a salmon", "a seal", "a dolphin", "a whale"]
    elif args.concept_category == "politicians":
        concepts = ["Abraham Lincoln", "Barack Obama", "Bernie Sanders", "Donald Trump", "Elizabeth Warren", 
                    "George W. Bush", "Hillary Clinton", "Joe Biden", "Richard Nixon", "Ronald Reagan"]

    output_df = run_concept_task(args, model, concepts, prompts_dict)

    save_output(output_df, args)

def save_output(output_df, args, subject=None):
    """Save the output dataframe to a pickle file."""
    subject_str = f"-subjects={subject}" if subject is not None else f"-subjects={args.num_subjects}"
    filename = f"{args.task}-{args.model_name}-prompt={args.prompt_condition}{subject_str}-temp={args.temperature}.pickle"
    output_path = Path(args.output) / filename

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
                        default="/n/holylabs/LABS/ullman_lab/Users/smurthy", 
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
    parser.add_argument("--num_subjects", type=int, default=100, help="Number of subjects")

    # color experiment arguments
    parser.add_argument("--num_words", type=int, default=50, 
                        help="Number of words to use for color task")

    # concept experiment arguments
    parser.add_argument("--concept_category", type=str, 
                        choices=["animals", "politicians"], 
                        help="Category of concepts to use for concept task")


    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    model = Model(args.lab_storage_dir, args.hf_token)
    model.initialize_model(args.model_name, args.model_path)
    model.shard_model()

    print("Model loaded and sharded...")

    with open('./input-data/prompts.pkl', 'rb') as handle:
        prompts_dict = pickle.load(handle)
    print("Unpickled prompts...")

    if args.task == "colors":
        setup_color_task(args, model, prompts_dict)
    elif args.task == "concepts":
        setup_concept_task(args, model, prompts_dict)

if __name__ == "__main__":
    main()
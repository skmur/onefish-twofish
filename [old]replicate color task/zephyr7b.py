import pandas as pd
import transformers
import torch
from minicons import scorer
import re
import numpy as np
import pickle
from PIL import ImageColor
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, LCHabColor, SpectralColor, sRGBColor, XYZColor, LCHuvColor, IPTColor, HSVColor
from colormath.color_diff import delta_e_cie2000
import sys

# get the model's generation for the prompt in appropriate chat template format
# returns: model's response
def getOutput(messages, temp, pipe):
    prompt = pipe.tokenizer.apply_chat_template(messages,
                                                tokenize=False,
                                                add_generation_prompt=True,
                                                # temperature=temp
                                                )
    outputs = pipe(prompt, max_new_tokens=100, do_sample=True)
    
    return outputs[0]["generated_text"]

# place the simulated human's context and task prompt into model template to get the model's response.
# extract the hex code from the response and convert it to Lab space
# returns: HEX, LAB, and RGB values
def getSample(context, prompt, temp, pipe):
    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    messages = [
        {
            "role": "system",
            "content": context,
        },
        {"role": "user", "content": prompt},
    ]
    
    output = getOutput(messages, temp, pipe)

    # make sure there is a valid hex code in the response and extract it
    while not re.search(r'#[0-9a-fA-F]{6}', output):
        output = getOutput(messages, temp, pipe)

    hex_code = re.search(r'#[0-9a-fA-F]{6}', output).group()

    # convert hex to rgb
    rgb = ImageColor.getcolor(hex_code, "RGB")
    # convert to [0,1] scaled rgb values
    scaled_rgb = (float(rgb[0])/255, float(rgb[1])/255, float(rgb[2])/255)
    # create RGB object
    rgbObject = sRGBColor(scaled_rgb[0], scaled_rgb[1], scaled_rgb[2])

    # convert to Lab
    labObject = convert_color(rgbObject, LabColor)
    labTuple = labObject.get_value_tuple()

    return hex_code, labTuple, scaled_rgb

# compute delta E between two colors in Lab space
# returns: delta E
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

# retrieve the subject's context from prompts_dict based on the condition. loop through all the words and construct task prompt and get 2 color associations for each subject 
# store the data in a dictionary with format: ['word', 'subject_num', 'condition', 'task_version', 'hex1', 'lab1', 'rgb1', 'hex2', 'lab2', 'rgb2', 'deltaE']
# returns: output_df
def runTask(output_df, num_subjects, temp, condition, words, prompts_dict, model_name, task_version, pipe):
    print("Running main task for condition: %s" % condition, flush=True)
    # print experiment parameters
    print("--> Number of subjects: %d" % num_subjects, flush=True)
    print("--> Temperature: %s" % temp, flush=True)
    print("--> Number of words: %d" % len(words), flush=True)
    print("--> Task version: %s" % task_version, flush=True)

    # for each subject, get a context
    for subject in range(num_subjects):
        if condition == "none":
            context = ""
        else: 
            context = prompts_dict[subject][condition]
            print(context)
            # strip the context of newlines
            context = context.replace("\n", " ")
       
        # loop through all the words and get 2 color associations for each subject
        for word in words:
            if task_version == "response":
                task_prompt = f"What is the HEX code of the color you most associate with the word {word}?"
            elif task_version == "completion":
                task_prompt = f"The HEX code of the color I most associate with the word {word} is:"

            # get 2 samples of color associations for each word
            hex1, lab1, rgb1 = getSample(context, task_prompt, temp, pipe)
            hex2, lab2, rgb2 = getSample(context, task_prompt, temp, pipe)

            # deltae
            deltae = computeDeltaE(lab1, lab2)

            print("Word: %s, Subject: %d, Task version: %s, HEX1: %s, HEX2: %s, DeltaE: %.2f" % (word, subject, task_version, hex1, hex2, deltae), flush=True)

            # concatenate data for this word and subject to dataframe. use pd.concat to avoid modifying the original dataframe
            # FORMAT: ['model_name', 'temperature', 'word', 'subject_num', 'condition', 'task_version', 'hex1', 'lab1', 'rgb1', 'hex2', 'lab2', 'rgb2', 'deltaE']
            output_df = pd.concat([output_df, pd.DataFrame([[model_name, temp, word, subject, condition, task_version, hex1, lab1, rgb1, hex2, lab2, rgb2, deltae]], columns=output_df.columns)], ignore_index=True)


        
        print("...Done with subject %d\n" % subject, flush=True)

        # pickle the dictionary to a file every 20 people to avoid losing data
        if subject % 10 == 0:
            with open('./output-data/%s-color-prompt=%s-subjects=%d-temp=%s-COMPLETION.pickle' % (model_name, condition, subject, temp), 'wb') as handle:
                pickle.dump(output_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return output_df

# def runMetacognitiveJudgement(output_df, num_subjects, temp, condition, words, prompts_dict, tokenizer, model, device, task_version):
#     print("Running metacognitive judgement task for condition: %s" % condition)
#     # print experiment parameters
#     print("--> Number of subjects: %d" % num_subjects)
#     print("--> Temperature: %s" % temp)
#     print("--> Number of words: %d" % len(words))
#     print("--> Task version: %s" % task_version)

#     # iterate over output_df
#     for index, row in output_df.iterrows():
#         word = row['word']
#         subject = row['subject_num']
#         hex1 = row['hex1']
#         hex2 = row['hex2']

#         # get prompt for this subject and condition
#         if condition == "none":
#             context = ""
#         else:
#             context = prompts_dict[subject][condition]
#             context = context.replace("\n", " ")

#         if task_version == "response":
#             metacog_prompt = f"What percent of other people do you expect will share your color association of {hex2} for {word}?"
#         elif task_version == "completion":
#             metacog_prompt = f"The percent of other people I expect will share my color association of {hex2} for {word} is:"

#          # concatenate the context and prompt
#         prompt = context + "" + metacog_prompt

#         # get probability that others share the same color association
#         prob = getOutput(prompt, temp, tokenizer, model, device)
#         # validate that the response is a single number
#         while not prob.isdigit() or int(prob) < 0 or int(prob) > 100:
#             prob = getOutput(prompt, temp, tokenizer, model, device)

#         # store in dictionary
#         output_df.at[index, 'agreement_prob'] = prob

#         print("Word: %s, Subject: %d, Task version: %s, HEX1: %s, HEX2: %s, Agreement Probability: %s" % (word, subject, task_version, hex1, hex2, prob))


# take the output of the tested model and pass it to the scorer_model
# https://github.com/kanishkamisra/minicons/blob/master/examples/succinct.md
def scoreOutput(scorer_model, stimuli):
    return scorer_model.token_score(stimuli, rank=True)

#================================================================================================
lab_storage_dir = "/n/holylabs/LABS/ullman_lab/Users/smurthy"

model_name = "zephyrMistral"
model_path = "HuggingFaceH4/zephyr-7b-beta"

if torch.cuda.is_available():
    device = "cuda"
    print("Using GPU...")
else:
    device = "cpu"
    print("WARNING! Using CPU...")

# model = transformers.AutoModelForCausalLM.from_pretrained(model_path, cache_dir=lab_storage_dir).to(device)
pipe = transformers.pipeline("text-generation", 
                             model=model_path, 
                             torch_dtype=torch.bfloat16, 
                             device_map="auto")

# scorer_model = scorer.IncrementalLMScorer('gpt2', 'cuda:0') # use model that is known to mimic human surprisal judgements (not super human performance)
print("Model loaded...")

# unpickle prompt dictionary
with open('../input-data/prompts.pkl', 'rb') as handle:
    prompts_dict = pickle.load(handle)
print("Unpickled prompts...")

# read in the color reference data
df = pd.read_csv("../input-data/color-task/colorref.csv")
# get unique words from df
words = df['word'].unique()
print("Number of test words %d" % (len(words)))

#--------------------------------------------------------
# set parameters
num_subjects = 40
temp = "default" # also run: 1.5, 2.0
num_words = 50

if num_words < len(words):
    version = "subsample"
    # randomly sample 50 words
    np.random.seed(42)
    words = np.random.choice(words, num_words, replace=False)
    print("subsampled words:")
    print(words)
else:
    version = "all"

# take in argument for condition
if len(sys.argv) > 1:
    condition = sys.argv[1]

task_versions = ["completion"]

#--------------------------------------------------------
# MAIN LOOP

# create a dictionary to store the data for task versions (response and completion), for each prompt condition [condition], temperature
output_df = pd.DataFrame(columns=['model_name', 'temperature', 'word', 'subject_num', 'condition', 'task_version', 'hex1', 'lab1', 'rgb1', 'hex2', 'lab2', 'rgb2', 'deltaE'])

for task_version in task_versions:
    output_df = runTask(output_df, num_subjects, temp, condition, words, prompts_dict, model_name, task_version, pipe)

    # save the dictionary to a pickle file
    with open('./output-data/%s-color-prompt=%s-subjects=%d-temp=%s-COMPLETION.pickle' % (model_name, condition, num_subjects, temp), 'wb') as handle:
        pickle.dump(output_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    


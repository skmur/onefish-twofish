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



def getOutput(text, temp, tokenizer, model, device):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output = model.generate(input_ids.to(device), 
                            max_length=256,
                            temperature=temp,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,)
    output = tokenizer.decode(output[0], skip_special_tokens=True)

    return output

def runExperiment(output_df, num_subjects, temp, condition, words, prompts_dict, tokenizer, model, device, task_version):
    print("Running experiment for condition: %s" % condition)
    # print experiment parameters
    print("--> Number of subjects: %d" % num_subjects)
    print("--> Temperature: %.2f" % temp)
    print("--> Number of words: %d" % len(words))
    print("--> Task version: %s" % task_version)

    # for each subject, get a context
    for j in range(num_subjects):
        if condition == "none":
            context = ""
        else: 
            context = prompts_dict[j][condition]
            # strip the context of newlines
            context = context.replace("\n", " ")
       
        # print("%d) %s" % (j, context))

        # loop through all the words and get 2 color associations for each subject
        for word in words:
            word_subject = word + "_" + str(j)

            if task_version == "response":
                prompt = f"What is the HEX code of the color do you most associate with the word {word}?"
            elif task_version == "completion":
                prompt = f"The HEX code of the color I most associate with the word {word} is:"
            

            # get 2 samples of color associations for each word
                
            hex1, lab1, rgb1 = getSample(context, prompt, temp, tokenizer, model, device)
            hex2, lab2, rgb2 = getSample(context, prompt, temp, tokenizer, model, device)

            # # get probability that others share the same color association
            # text = context + "\n\n" + "How strongly do you expect others to share your color association of " + completion +  " for " + word + "? Respond with a number between 0 (not at all - most people will have a different color association than I do) and 100 (very strongly - most people will have the same color association as I do)."
            # prob = getOutput(text, "gpt-3.5-turbo", 1, temp)
            # # validate that the response is a single number
            # while not prob.isdigit() or int(prob) < 0 or int(prob) > 100:
            #     prob = getOutput(text, "gpt-3.5-turbo", 1, temp)
            prob = -1

            # deltae
            deltae = computeDeltaE(lab1, lab2)

            print("Word: %s, Subject: %d, HEX1: %s, HEX2: %s, DeltaE: %.2f" % (word, j, hex1, hex2, deltae))

            # FORMAT: ['word', 'subject_num', 'condition', 'task_version', 'hex1', 'lab1', 'rgb1', 'hex2', 'lab2', 'rgb2', 'deltaE', 'agreement_prob']
            output_df = pd.concat([pd.DataFrame([[word, j, condition, task_version, hex1, lab1, rgb1, hex2, lab2, rgb2, deltae, prob]], columns=df.columns), df], ignore_index=True)
        
        print("Done with subject %d\n--------" % j)

        # pickle the dictionary to a file every 20 people to avoid losing data
        if j % 20 == 0:
            with open('./output-data/color-%s-subjects=%d-temp=%.1f-version=%s.pickle' % (condition, j, temp, task_version), 'wb') as handle:
                pickle.dump(output_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return output_df

def getSample(context, prompt, temp, tokenizer, model, device):
    text = "GPT4 Correct User: %s. %s<|end_of_turn|>GPT4 Correct Assistant:" % (context, prompt)
    
    output = getOutput(text, temp, tokenizer, model, device)

    # make sure there is a valid hex code in the response and extract it
    while not re.search(r'#[0-9a-fA-F]{6}', output):
        output = getOutput(text, temp, tokenizer, model, device)

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

# take the output of the tested model and pass it to the scorer_model
# https://github.com/kanishkamisra/minicons/blob/master/examples/succinct.md
def scoreOutput(scorer_model, stimuli):
    return scorer_model.token_score(stimuli, rank=True)

#================================================================================================
lab_storage_dir = "/n/holylabs/LABS/ullman_lab/Users/smurthy"

model_name = "starlingLM"
model_path = "berkeley-nest/Starling-LM-7B-alpha"

if torch.cuda.is_available():
    device = "cuda"
    print("Using GPU...")
else:
    device = "cpu"
    print("WARNING! Using CPU...")

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, cache_dir=lab_storage_dir)
model = transformers.AutoModelForCausalLM.from_pretrained(model_path, cache_dir=lab_storage_dir).to(device)
# scorer_model = scorer.IncrementalLMScorer('gpt2', 'cuda:0') # use model that is known to mimic human surprisal judgements (not super human performance)
print("Model loaded...")

# unpickle prompt dictionary
with open('../input-data/prompts.pkl', 'rb') as handle:
    prompts_dict = pickle.load(handle)
print("Unpickled prompts...")
print(prompts_dict.keys())

# read in the color reference data
df = pd.read_csv("../input-data/color-task/colorref.csv")
# get unique words from df
words = df['word'].unique()
print("Number of test words %d" % (len(words)))

#--------------------------------------------------------
# set parameters
num_subjects = 10
temp = 0.0 # also run: 1.5, 2.0
num_words = 50

if num_words < len(words):
    version = "subsample"
    # randomly sample 50 words
    np.random.seed(42)
    words = np.random.choice(words, num_words, replace=False)
    print("Sub-sampled words: " + words)
else:
    version = "all"

conditions = ["none"]  # "none", "identity", "random_context", "nonsense_context"
task_versions = ["response", "completion"]

#--------------------------------------------------------
# MAIN LOOP

for condition in conditions:
    # create a dictionary to store the data
    # FORMAT: word_subject: [condition, lab1, rgb1, lab2, rgb2, prob, deltaE]
    output_df = pd.DataFrame(columns=['word', 'subject_num', 'condition', 'task_version', 'hex1', 'lab1', 'rgb1', 'hex2', 'lab2', 'rgb2', 'deltaE', 'agreement_prob'])

    for task_version in task_versions:
        output_df = runExperiment(output_df, num_subjects, temp, condition, words, prompts_dict, tokenizer, model, device, task_version)

        # save the dictionary to a pickle file
        with open('./output-data/color-%s-subjects=%d-temp=%.1f-version=%s.pickle' % (condition, num_subjects, temp, task_version), 'wb') as handle:
            pickle.dump(output_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        



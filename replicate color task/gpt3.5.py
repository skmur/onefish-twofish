import pandas as pd
import time
import colormath
import numpy
from openai import OpenAI
from PIL import ImageColor
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, LCHabColor, SpectralColor, sRGBColor, XYZColor, LCHuvColor, IPTColor, HSVColor
from colormath.color_diff import delta_e_cie2000
import matplotlib.pyplot as plt
import pickle
import re
import json

# GPT 3.5
def getOutput(text, version, n, temp):   
    response = client.chat.completions.create(
        model=version,
        messages = [{"role": "system", "content": text}],
        n=n,
        temperature=temp)

    return response.choices[0].message.content

def computeDeltaE(lab1, lab2):
    # compute delta L
    deltaL = lab1[0] - lab2[0]
    # compute delta a
    deltaA = lab1[1] - lab2[1]
    # compute delta b
    deltaB = lab1[2] - lab2[2]
    # compute delta E
    deltaE = numpy.sqrt(deltaL**2 + deltaA**2 + deltaB**2)

    return deltaE

#--------------------------------------------

try:
    with open("../secrets.json") as f:
        secrets = json.load(f)
    my_api_key = secrets["openai"]
    print(my_api_key)
    print("API key loaded.")
except FileNotFoundError:
    print("Secrets file not found.")

client = OpenAI(api_key=my_api_key)


# unpickle prompt dictionary
with open('../input-data/prompts.pkl', 'rb') as handle:
    prompts_dict = pickle.load(handle)

# read in the color reference data
df = pd.read_csv("../input-data/color-task/colorref.csv")
# get unique words from df
words = df['word'].unique()
print(len(words))

# # REMOVE FOR ACTUAL EXPERIMENTS (RUN ALL 200 words): randomly sample 50 words
# numpy.random.seed(42)
# words = numpy.random.choice(words, 50, replace=False,)
# print(words)


num_samples = 2
num_subjects = 100
temp = 1.0 # also run: 1.5, 2.0

# "identity", "random_context", "nonsense_context"
conditions = ["identity", "random_context", "nonsense_context"]

# iterate over the 3 conditions
for condition in conditions:
    print("---------------- CONDITION: %s ----------------" % condition)
    print("temp = %.2f" % temp)
    print("num_subjects = %d" % num_subjects)
    print("-----------------------------\n\n")
    color_dict = {}

    # for each subject, get a context
    for j in range(num_subjects):
        context = prompts_dict[j][condition]
        # strip the context of newlines
        context = context.replace("\n", " ")

        print("%d) %s" % (j, context))

        # loop through all the words and get 2 color associations for each subject
        for word in words:
            word_subject = word + "_" + str(j) # append subject number to key

            color_dict.setdefault(word_subject, []).append(condition)
            
            text = "What color do you most associate with the word " + word + "? Respond only with a single HEX code."

            text = context + "\n\n" + text

            for i in range(num_samples):
                completion = getOutput(text, "gpt-3.5-turbo", 1, temp)
                # print("%s, sample %d --> %s" % (word, i, completion))
                # make sure the completion is a valid hex code
                while not re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', completion):
                    print("Invalid hex code")
                    completion = getOutput(text, "gpt-3.5-turbo", 1, temp)

                # convert hex to rgb
                rgb = ImageColor.getcolor(completion, "RGB")

                # # convert rgb to lab
                # convert to [0,1] scaled rgb values
                c = (float(rgb[0])/255, float(rgb[1])/255, float(rgb[2])/255)
                # create RGB object
                rgbObject = sRGBColor(c[0], c[1], c[2])
                # convert
                labObject = convert_color(rgbObject, LabColor)
                labTuple = labObject.get_value_tuple()

                # store in dictionary
                color_dict.setdefault(word_subject, []).append(labTuple)
                color_dict.setdefault(word_subject, []).append(c)

                # if this is the second sample, compute deltae between color responses
                # also ask model how much they expect another person to share their color association 
                if i == 1:
                    # probability that others share the same color association
                    text = context + "\n\n" + "How strongly do you expect others to share your color association of " + completion +  " for " + word + "? Respond with a number between 0 (not at all - most people will have a different color association than I do) and 100 (very strongly - most people will have the same color association as I do)."
                    prob = getOutput(text, "gpt-3.5-turbo", 1, temp)
                    # validate that the response is a single number
                    while not prob.isdigit() or int(prob) < 0 or int(prob) > 100:
                        prob = getOutput(text, "gpt-3.5-turbo", 1, temp)
                    color_dict.setdefault(word_subject, []).append(prob)

                    # deltae
                    deltae = computeDeltaE(color_dict[word_subject][1], color_dict[word_subject][3])
                    color_dict.setdefault(word_subject, []).append(deltae)

            # time.sleep(2)
            print(word_subject, color_dict[word_subject])
            print("")
        
        print("Done with subject %d\n--------" % j)
    
        # pickle the dictionary to a file every 20 people to avoid losing data
        if j % 20 == 0:
            with open('./output-data/color-%dsubjs-%s-temp=%.2f.pickle' % (j, condition, temp), 'wb') as handle:
                pickle.dump(color_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # data representation: word_subject: [condition, lab1, rgb1, lab2, rgb2, prob, deltaE]
    print(color_dict)

    # save the dictionary to a pickle file
    with open('./output-data/color-%s-temp=%.1f.pickle' % (condition, temp), 'wb') as handle:
        pickle.dump(color_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    



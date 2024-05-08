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

# openai.api_key = "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"
client = OpenAI(api_key="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")

# GPT 3.5
def getOutput(text, version, n, temp):   
    response = client.chat.completions.create(
        model=version,
        messages = [{"role": "system", "content": text}],
        n=n,
        temperature=temp)

    return response.choices[0].message.content


#--------------------------------------------

# unpickle prompt dictionary
with open('../input-data/prompts.pkl', 'rb') as handle:
    prompts_dict = pickle.load(handle)


num_samples = 2
num_subjects = 2
temp = 1.0

# "identity", "random_context", "nonsense_context"
conditions = ["none"]

# concepts from https://direct.mit.edu/opmi/article/doi/10.1162/opmi_a_00072/114924/Latent-Diversity-in-Human-Concepts
animals = ["a finch", "a robin", "a chicken", "an eagle", "an ostrich", "a penguin", "a salmon", "a seal", "a dolphin", "a whale"]
politicians = ["Abraham Lincoln", "Barack Obama", "Bernie Sanders", "Donald Trump", "Elizabeth Warren", "George W. Bush", "Hillary Clinton", "Joe Biden", "Richard Nixon", "Ronald Reagan"]

categories = ["animals", "politicians"]

# initialize datafram with columns for subject number, condition, category, target, choice1, choice2, completion
df_outputs = pd.DataFrame(columns=["subject", "condition", "category", "target", "choice1", "choice2", "completion"])


# iterate over the 3 conditions
for condition in conditions:
    print("---------------- CONDITION: %s ----------------" % condition)

    # run each category of concepts
    for category in categories:
        if category == "animals":
            words = animals
        else:
            words = politicians

        # for each subject, get a context
        for j in range(num_subjects):
            if condition == "none":
                context = ""
            else:
                context = prompts_dict[j][condition]
                # strip the context of newlines
                context = context.replace("\n", " ")

            print("%d) %s" % (j, context))

            category_subject = category + "_" + str(j) #append subject number to category

            for target in words:
                for choice1 in words:
                    for choice2 in words:
                        if (choice1 != choice2) and (choice1 != target) and (choice2 != target):
                            print("target = %s; %s or %s" % (target, choice1, choice2))
                            for i in range(num_samples):
                                text = context + "\nWhich is more similar to a " + target + ", " + choice1 + " or " + choice2 + "? Respond only with " + choice1 + " or " + choice2 + "."
                                completion = getOutput(text, "gpt-3.5-turbo", 1, temp)
                                completion = completion.lower()
                                print("--> %s" % completion)

                                while (completion not in choice1) and (completion not in choice2):
                                    print(f"Choice: {completion}")
                                    print("invalid response.. retrying..")
                                    completion = getOutput(text, "gpt-3.5-turbo", 1, temp)
                                    completion = completion.lower()
                                
                                # add new row to the dataframe with subject number, condition, category, target, choice1, choice2, completion
                                new_entry = pd.Series({"subject": category_subject, "condition": condition, "category": category, "target": target, "choice1": choice1, "choice2": choice2, "completion": completion})
                                df_outputs = pd.concat([df_outputs, new_entry], ignore_index=True)
                                
                                # get probability that others share judgement
                                # if i == 1:
                                #     text = context + "\n\n" + "How many people out of 100 would you expect to share your response of " + completion +  " being more similar to " + target + " than " + ? Respond with a number between 0 (not at all - most people will have a different color association than I do) and 100 (very strongly - most people will have the same color association as I do)."
                                #     prob = getOutput(text, "gpt-3.5-turbo", 1, temp)
                                #     # validate that the response is a single number
                                #     while not prob.isdigit() or int(prob) < 0 or int(prob) > 100:
                                #         prob = getOutput(text, "gpt-3.5-turbo", 1, temp)
                                          
        print("Done with subject %d\n--------" % j)
        print(df_outputs)

# save the dataframe to a csv file
df_outputs.to_csv('./output-data/concept-allsubjs-%s.csv' % condition, index=False)
    



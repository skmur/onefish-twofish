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
num_subjects = 100
temp = 1.0

# "identity", "random_context", "nonsense_context"
conditions = ["none"]

# concepts from https://direct.mit.edu/opmi/article/doi/10.1162/opmi_a_00072/114924/Latent-Diversity-in-Human-Concepts
animals = ["a finch", "a robin", "a chicken", "an eagle", "an ostrich", "a penguin", "a salmon", "a seal", "a dolphin", "a whale"]
politicians = ["Abraham Lincoln", "Barack Obama", "Bernie Sanders", "Donald Trump", "Elizabeth Warren", "George W. Bush", "Hillary Clinton", "Joe Biden", "Richard Nixon", "Ronald Reagan"] 

categories = ["politicians"]


# iterate over the 3 conditions
for condition in conditions:
    print("---------------- CONDITION: %s ----------------" % condition)
    # initialize datafram with columns for subject number, condition, category, target, choice1, choice2, completion
    df_outputs = pd.DataFrame(columns=["subject", "condition", "category", "target", "choice1", "choice2", "response1", "response2"])
    # run each category of concepts
    for category in categories:
        if category == "animals":
            print("CATEGORY: animals")
            words = animals
        else:
            print("CATEGORY: politicians")
            words = politicians

        # for each subject, get a context
        for j in range(num_subjects):
            # start timer
            start = time.time()
            
            if condition == "none":
                context = ""
            else:
                context = prompts_dict[j][condition]
                # strip the context of newlines
                context = context.replace("\n", " ")

            print("%d) %s" % (j, context))

            for target in ["Richard Nixon", "Ronald Reagan", "Donald Trump"]:
                for choice1 in words:
                    for choice2 in words:
                        if (choice1 != choice2) and (choice1 != target) and (choice2 != target):
                            print("target = %s; %s or %s" % (target, choice1, choice2))
                            # initialize entry with subject, condition, category, target, choice1, choice2
                            entry = [j, condition, category, target, choice1, choice2]
                            for i in range(num_samples):
                                
                                # get two responses, but switch the order the choices are presented in
                                if i == 0:
                                    text = context + "\nWhich is more similar to a " + target + ", " + choice1 + " or " + choice2 + "? Respond only with " + choice1 + " or " + choice2 + "."
                                else:
                                    text = context + "\nWhich is more similar to a " + target + ", " + choice2 + " or " + choice1 + "? Respond only with " + choice2 + " or " + choice1 + "."

                                    # # get probability of response
                                    # text = context + "\n\n" + "How many people out of 100 would you expect to share your response of " + completion +  " being more similar to " + target + " than " + ? Respond with a number between 0 (not at all - most people will have a different color association than I do) and 100 (very strongly - most people will have the same color association as I do)."
                                    # prob = getOutput(text, "gpt-3.5-turbo", 1, temp)
                                    # # validate that the response is a single number
                                    # while not prob.isdigit() or int(prob) < 0 or int(prob) > 100:
                                    #     prob = getOutput(text, "gpt-3.5-turbo", 1, temp)
                                
                                # print(text)
                                completion = getOutput(text, "gpt-3.5-turbo", 1, temp)
                                completion = completion.lower()
                                # print("--> %s" % completion)

                                while (completion not in choice1.lower().strip()) and (completion not in choice2.lower().strip()):
                                    print(f"Choice: {completion}")
                                    print("invalid response.. retrying..")
                                    completion = getOutput(text, "gpt-3.5-turbo", 1, temp)
                                    completion = completion.lower()
                        
                                # append to the entry with each completion
                                entry.append(completion)
                            
                            # append the entry with both responses to the dataframe
                            df_outputs.loc[len(df_outputs)] = entry
                            # save the df every 10 entries
                            if len(df_outputs) % 10 == 0:
                                df_outputs.to_csv('./output-data/concept-%s-allsubjs-%s-reagan-nixon-trump.csv' % (category, condition), index=False)
                                # print("%d rows" % len(df_outputs))


            # end timer
            print("Done with subject %d\n--------" % j)
            print("Elapsed time: %f" % (time.time() - start))                             
            
    # save the dataframe to a csv file
    print("saving %s category, %s condition..." % (category, condition))
    df_outputs.to_csv('./output-data/concept-%s-allsubjs-%s.csv' % (category, condition), index=False)
    print(df_outputs)
    



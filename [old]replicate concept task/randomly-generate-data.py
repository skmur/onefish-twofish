import pandas as pd
import time
import numpy
import random

num_samples = 2
num_subjects = 100

# concepts from https://direct.mit.edu/opmi/article/doi/10.1162/opmi_a_00072/114924/Latent-Diversity-in-Human-Concepts
animals = ["a finch", "a robin", "a chicken", "an eagle", "an ostrich", "a penguin", "a salmon", "a seal", "a dolphin", "a whale"]
politicians = ["Abraham Lincoln", "Barack Obama", "Bernie Sanders", "Donald Trump", "Elizabeth Warren", "George W. Bush", "Hillary Clinton", "Joe Biden", "Richard Nixon", "Ronald Reagan"] 

categories = ["animals"]

condition = "random_generation"

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
        for target in words:
            for choice1 in words:
                for choice2 in words:
                    if (choice1 != choice2) and (choice1 != target) and (choice2 != target):
                        print("target = %s; %s or %s" % (target, choice1, choice2))
                        # initialize entry with subject, condition, category, target, choice1, choice2
                        entry = [j, condition, category, target, choice1, choice2]
                        for i in range(num_samples):
                            
                            # choose randomly between choice1 and choice2
                            if random.random() < 0.5:
                                entry.append(choice1)
                            else:
                                entry.append(choice2)
                            
                        
                        # append the entry with both responses to the dataframe
                        df_outputs.loc[len(df_outputs)] = entry

                        # save the df every 20 subjects
                        if j % 20 == 0:
                            df_outputs.to_csv('./output-data/concept-%s-%s-%s.csv' % (category, str(j), condition), index=False)
                            # print("%d rows" % len(df_outputs))


        # end timer
        print("Done with subject %d\n--------" % j)   

    df_outputs.to_csv('./output-data/concept-%s-allsubjs-%s.csv' % (category, condition), index=False)
    print(df_outputs)
    
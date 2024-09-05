import pickle
import numpy as np
import pandas as pd

# SCRATCHPAD: PROCESS AND COMPILE RAW DATA AS NEEDED AND SAVE TO FINAL FORMAT
# output-data/50words-100subjs = initial subset of data for baselines containing "response" and "completion" task versions

def sanityCheck(df, num_subjects, num_words, num_task_versions):
    for i in range(0, num_subjects):
        # select the rows where subject_num = i
        subj = df[df['subject_num'] == i]
       
        # make sure there are 50 unique words for each subject
        if len(subj['word'].unique()) != num_words:
            print(i, subj['word'].unique())
        
        # make sure there are 2 task versions for each subject
        if len(subj['task_version'].unique()) != num_task_versions:
            print (i, subj['task_version'].unique())
    print("Sanity check complete")


#----------------------------------------------------------------------

# CHECK ALL MODELS
models = ["gemmaWRONGTEMPLATE", "starlingLM", "openchat", "zephyrGemma", "zephyrMistral", "mistralInstruct", "llamaChatWRONGTEMPLATE"]

num_subjects = 100
num_words = 50
num_task_versions = 2

for model in models:
    print("Checking model: ", model)
    data = "./output-data/50words-100subjs/" + model + "-color-prompt=none-subjects=100-temp=default.pickle"

    with open(data, 'rb') as f:
        model_data = pickle.load(f)
    
    # reset index
    print(model_data.columns)
    
    # print number of unique subjects 
    print(model_data['subject_num'].nunique())

    # group by subject_num and task_version and print subject numbers where size is not 50
    print(model_data.groupby(['subject_num', 'task_version']).size().reset_index(name='count').query('count < 50')['subject_num'].unique())
        
    #run sanity check
    sanityCheck(model_data, num_subjects, num_words, num_task_versions)

    print("---------------------------------------------------")
    

    
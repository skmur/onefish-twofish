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
        assert len(subj['word'].unique()) == num_words

        # make sure there are 2 task versions for each subject
        assert len(subj['task_version'].unique()) == num_task_versions

    print("Sanity check passed")


#----------------------------------------------------------------------
zephyrGemma_40completion = "./output-data/zephyrGemma-color-prompt=none-subjects=40-temp=default.pickle"
zephyrGemma_100response = "./output-data/zephyrGemma-color-prompt=none-subjects=100-temp=default.pickle"
zephyrGemma_60completion = "TODO"

# open each dataframe 
with open(zephyrGemma_40completion, 'rb') as handle:
    zephyrGemma_40completion_df = pickle.load(handle)

# select the rows where task_version = completion
zephyrGemma_40completion_df = zephyrGemma_40completion_df[zephyrGemma_40completion_df['task_version'] == 'completion']

# open 100 response dataframe
with open(zephyrGemma_100response, 'rb') as handle:
    zephyrGemma_100response_df = pickle.load(handle)

combine = pd.concat([zephyrGemma_40completion_df, zephyrGemma_100response_df])
print(combine)


#----------------------------------------------------------------------

zephyrMistral_60completion = "./output-data/zephyrMistral-color-prompt=none-subjects=60-temp=default.pickle"
zephyrMistral_100response = "./output-data/zephyrMistral-color-prompt=none-subjects=100-temp=default.pickle"
zephyrMistral_40completion = "TODO"

# do the same for the Mistral model
with open(zephyrMistral_60completion, 'rb') as handle:
    zephyrMistral_60completion_df = pickle.load(handle)

# select the rows where task_version = completion
zephyrMistral_60completion_df = zephyrMistral_60completion_df[zephyrMistral_60completion_df['task_version'] == 'completion']

# open 100 response dataframe
with open(zephyrMistral_100response, 'rb') as handle:
    zephyrMistral_100response_df = pickle.load(handle)

combine = pd.concat([zephyrMistral_60completion_df, zephyrMistral_100response_df])
print(combine)
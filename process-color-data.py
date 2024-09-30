import pickle
import numpy as np
import pandas as pd
import os

# SCRATCHPAD: PROCESS AND COMPILE RAW DATA AS NEEDED AND SAVE TO FINAL FORMAT
# output-data/50words-100subjs = initial subset of data for baselines containing "response" and "completion" task versions

def sanityCheck(df, num_subjects, num_words):
    for i in range(0, num_subjects):
        # select the rows where subject_num = i
        subj = df[df['subject_num'] == i]
       
        # make sure there are 50 unique words for each subject
        if len(subj['word'].unique()) != num_words:
            print(i, subj['word'].unique())
        
    print("Sanity check complete")


#----------------------------------------------------------------------
# TRASNFER FILES FROM STORAGE TO LOCAL AS NEEDED

# # loop through files in directory and unpickle and print out the df

# dir = "/n/holylabs/LABS/ullman_lab/Users/smurthy/onefish-twofish/output-data/concept-task/"
# for filename in os.listdir(dir):
#     if filename.endswith(".pickle"):
#         with open(dir + filename, 'rb') as f:
#             data = pickle.load(f)
#             print(filename)
#             if data is not None:
#                 if ('generation1' in data.columns) and len(data) > 1000:
#                     print(filename)
#                     print(len(data))
#                     print(data.columns)
#                     # copy file to new directory
#                     # os.system("cp " + dir + filename + " ./output-data/concept-task/")
#                 elif not filename.startswith("[test]"):
#                     # rename file by adding [test] to the beginning
#                     # os.rename(dir + filename, dir + "[test]" + filename)
#                     print("renamed file: ", filename)
#                 else: 
#                     continue
#                 print("---------------------------------------------------")


#----------------------------------------------------------------------
models = ["starling", "openchat", "gemma-instruct", "zephyr-gemma", "mistral-instruct", "zephyr-mistral", "llama2", "llama2-chat"]
lab_storage_dir = "/n/holylabs/LABS/ullman_lab/Users/smurthy/onefish-twofish/output-data/concept-task/"
output_dir = "./output-data/concept-task/"

# concatenate all dataframes by model
for model in models:
    concat = pd.DataFrame()
    print("Processing model: ", model)
    count = 0
    for filename in os.listdir(lab_storage_dir):
        if ("[test]" not in filename) and (model in filename) and (filename.endswith(".pickle")):
            if model == "llama2":
                #  check that "chat" is not in the filename
                if "chat" in filename:
                    continue
                    
            if filename == "concepts-zephyr-gemma-category=politicians-prompt=nonsense-temp=default.pickle":
                continue

            with open(lab_storage_dir + filename, 'rb') as f:
                data = pickle.load(f)

                if data is None:
                    print("Alert: ", filename)
                    continue

                # extract the part of the filename that contains the prompt type
                # strip .piclke from the filename
                filename_tmp = filename.split(".pickle")[0]
                params = filename_tmp.split("-")
                for param in params:
                    if "prompt" in param:
                        prompt = param.split("=")[1]
                    if "category" in param:
                        category = param.split("=")[1]
                    if "temp" in param:
                        temperature = param.split("=")[1]
                        
                count+=1
                print("-->", category, prompt, temperature, len(data))

                # add the prompt type to the dataframe
                data['prompt'] = prompt

                # save the updated dataframe to lab_storage_dir
                with open(lab_storage_dir + filename, 'wb') as f:
                    pickle.dump(data, f)
            
                # concatenate all the dataframes for this model
                if 'concat' in locals():
                    concat = pd.concat([concat, data])
            

    print("total number of rows: ", len(concat))
    print("total number of files: ", count)

    if len(concat) == 0:
        print("No data for model: ", model)
    else: 
        # save the concatenated dataframe
        with open(output_dir + "concepts-" + model + ".pickle", 'wb') as f:
            pickle.dump(concat, f)

    print("---------------------------------------------------")

            
# #----------------------------------------------------------------------
# # CHECK ALL MODELS

# num_subjects = 150
# num_words = 199

# for model in models:
#     print("Checking model: ", model)
#     data = "./output-data/50words-100subjs/" + model + "-color-prompt=none-subjects=100-temp=default.pickle"

#     with open(data, 'rb') as f:
#         model_data = pickle.load(f)
    
#     # reset index
#     print(model_data.columns)
    
#     # print number of unique subjects 
#     print(model_data['subject_num'].nunique())

#     # group by subject_num and task_version and print subject numbers where size is not 50
#     print(model_data.groupby(['subject_num', 'task_version']).size().reset_index(name='count').query('count < 150')['subject_num'].unique())
        
#     #run sanity check
#     sanityCheck(model_data, num_subjects, num_words)

#     print("---------------------------------------------------")



    

    
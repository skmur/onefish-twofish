import os
import pickle 
import pandas as pd
from tqdm import tqdm

from openai import OpenAI

# openai.api_key = "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"
client = OpenAI(api_key="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")

def getOutput(text, n):   
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [{"role": "system", "content": "You are a helpful, intelligent assistant. Your task is to determine which one of the two choices is being chosen in the text. If it is neither, please respond with -1."},
                    {"role": "user", "content": text}],
        n=n,
        temperature=0.3)

    return response.choices[0].message.content

def process_file(filename, data_dir):
    with open(data_dir + filename, 'rb') as f:
        data = pickle.load(f)

    print("Processing: ", filename)
    print(data.columns)
    print(len(data))

    # Add new columns to track whether response was determined by gpt
    data['gpt_response1'] = 0
    data['gpt_response2'] = 0

    for gen_num in [1, 2]:
        # Create boolean masks
        # mask = (data[f'generation{gen_num}'].str.strip().str.strip(".").str.lower() != data['choice1'].str.lower()) & \
        #        (data[f'generation{gen_num}'].str.strip().str.strip(".").str.lower() != data['choice2'].str.lower())
        
        # Extract the first line of each generation
        first_line = data[f'generation{gen_num}'].str.split('\n').str[0]

        mask = (first_line.str.len() > data['choice1'].str.len()*2) & \
                (first_line.str.len() > data['choice2'].str.len()*2)

        # Get indices where mask is True
        indices = mask[mask].index

        print("-->", len(indices))

        if len(indices) > 0:
            # Prepare prompts for GPT
            prompts = data.loc[indices].apply(
                lambda row: f"Which of the two choices was selected in this text? The choices are: {row['choice1']} and {row['choice2']}. The text is: {row[f'generation{gen_num}'].strip().strip('.')}. Respond only with {row['choice1']}, {row['choice2']}, or -1 if neither.",
                axis=1
            )

            # Get GPT responses
            responses = [getOutput(prompt, 1) for prompt in tqdm(prompts, desc=f"Getting GPT responses for gen{gen_num}")]

            # Update answers and gpt_response columns
            data.loc[indices, f'answer{gen_num}'] = responses
            data.loc[indices, f'gpt_response{gen_num}'] = (data.loc[indices, f'answer{gen_num}'] != "") & (data.loc[indices, f'answer{gen_num}'] != data.loc[indices, f'answer{gen_num}'].astype(str))

    # Save the updated dataframe
    data.to_pickle(data_dir + "[gpt-imputed]" + filename)

    print("-------------------")

def main():
    data_dir = "./output-data/concept-task/"
    for filename in os.listdir(data_dir):
        if filename.endswith(".pickle"):
            process_file(filename, data_dir)

if __name__ == "__main__":
    main()


# def getOutput(text, n):   
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages = [{"role": "system", "content": "You are a helpful, intelligent assistant. Your task is to determine which one of the two choices is being chosen in the text. If it is neither, please respond with -1."},
#                     {"role": "user", "content": text}],
#         n=n,
#         temperature=0.3)

#     return response.choices[0].message.content


# data_dir = "./output-data/concept-task/"

# for filename in os.listdir(data_dir):
#     if filename.endswith(".pickle"):
#         with open(data_dir + filename, 'rb') as f:
#             data = pickle.load(f)

#             # add a new column to track whether response was determined by gpt
#             data['gpt_response1'] = 0
#             data['gpt_response2'] = 0

#             print("Processing: ", filename)
#             print(data.columns)
#             for i in tqdm(range(len(data))):
#                 row = data.iloc[i]
#                 choice1 = row['choice1']
#                 choice2 = row['choice2']

#                 for gen_num in [1, 2]:
#                     generation = row[f'generation{gen_num}'].strip().strip(".")
#                     answer = row[f'answer{gen_num}']

#                     # if the generation doesn't match either choice, ask the model to determine which choice was selected
#                     if generation != choice1 and generation != choice2:
#                         prompt = f"Which of the two choices was selected in this text? The choices are: {choice1} and {choice2}. The text is: {generation}. Respond only with {choice1}, {choice2}, or -1 if neither."
#                         response = getOutput(prompt, 1)

#                         # add the response to the dataframe
#                         data.loc[i, f'answer{gen_num}'] = response

#                         # if the gpt response is not "" and doesn't match the answer, impute the answer with the gpt response and set the gpt_response column to 1
#                         if response != "" and response != str(answer):
#                             data.loc[i, f'answer{gen_num}'] = response
#                             data.loc[i, f'gpt_response{gen_num}'] = 1

#         # save the updated dataframe
#         data.to_pickle(data_dir + "[gpt-imputed]" + filename)

#         print("-------------------")

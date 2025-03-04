from datasets import load_dataset
import random 
import pickle
import pandas as pd
import os
import argparse


"""
Filters the English Wikipedia corpus to remove articles about people (to avoid reduplicating persona prompting), disambiguation pages, and articles with short first sentences.

Args:
    dataset: the Wikipedia dataset
    prompt_length: the minimum length of the first sentence of an article

Returns:
    filtered: a dictionary of articles that pass the filtering criteria
"""
def filterWikipediaText(dataset, prompt_length):
    filtered = {}
    num_valid_articles = 0

    for i in range(len(dataset)):
        text = dataset[i]["text"]
        text = text.split(".")[0]
        # filter out people
        if ("he" in text) or ("she" in text) or ("born" in text):
            # print("--> Person found, getting new article...")
            continue
        
        # make sure the first sentence is some minimum length
        if len(text.split()) < prompt_length:
            # print("--> Text too short, getting new article...")
            continue

        # if there are dates in the text, get a new article
        if any(char.isdigit() for char in text):
            # print("--> Date found, getting new article...")
            continue
        
        # remove disambiguation pages
        if "refers to" in text:
            # print("--> Disambiguation page found, getting new article...")
            continue

        if "refer to" in text:
            # print("--> Disambiguation page found, getting new article...")
            continue
        
        print("found an article!")
        print(num_valid_articles, text)
        filtered[num_valid_articles] = dataset[i]
        num_valid_articles+=1

    print("dataset length after processing: %s" % len(filtered))

    return filtered

"""
Produces three types of prompts for each subject:
2. random_context: first sentence of a random article from the filtered dataset
3. nonsense_context: shuffled word order version of random_context -> sentence cased
4. identity: create a random persona by selecting a random value from a predefined set of identity categories, in the following format:
    "You are a [race] [gender] from [hometown] in [state] who is [age] and works as a [occupation]."

Args:
    args: command line arguments
    num_subjects: number of subjects to generate prompts for

Returns:    
    None (saves and pickles the prompts in the specified data directory)
"""
def processPrompts(args, num_subjects):
    prompt_dict = {}
    prompt_length = 25
    data_dir = args.data_dir

    if not os.path.exists(f'{data_dir}/filtered-wiki.pkl'):
        wiki = load_dataset("wikipedia", "20220301.en", split='train')
        wiki_filtered = filterWikipediaText(wiki, prompt_length)
        with open(f'{data_dir}/filtered-wiki.pkl', 'wb') as f:
            pickle.dump(wiki_filtered, f)

    else:
        # load filtered wikipedia
        with open(f'{data_dir}/filtered-wiki.pkl', 'rb') as f:
            wiki_filtered = pickle.load(f)

    # Identity categories: derived from Wang et al. 2024 with additions from US census data categories
    identities = {
        'race': ['a Black', 'an Asian', 'a White', 'a Hispanic', 'a Native American', 'a Native Hawaiian or other Pacific Islander'],
        'gender': ['man', 'woman', 'non-binary person'],
        'age': ['a Baby Boomer (age 59 to 77)', 'a Millennial (age 27 to 42)', 'a member of Generation Z (age 18 to 26)', 'a member of Generation X (age 43-58)'],
        'hometown': ['from a small town', 'from a big city', 'from a rural area'],
        'state': ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",  "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"],
        'occupation': ['a teacher', 'a doctor', 'a lawyer', 'a farmer', 'a construction worker', 'a software engineer', 'a painter', 'a musician', 'a writer', 'a gardener', 'a cook', 'a photographer']
    }

    # get data for subjects
    for i in range(num_subjects):
        # Get a random article from the filtered Wikipedia dataset
        random_article = wiki_filtered[random.randint(0, len(wiki_filtered) - 1)]
        random_context = random_article["text"].split(".")[0]

        words =  random_context.split()
        nonsense_context = random.sample(words, len(words))
        nonsense_context = " ".join(nonsense_context)
        # sentence case the nonsense context
        nonsense_context = nonsense_context[0].upper() + nonsense_context[1:]

        # add a period at the end of the nonsense and random context
        if random_context[-1] != ".":
            random_context = random_context + "."
        if nonsense_context[-1] != ".":
            nonsense_context = nonsense_context + "."

        # create a random identity by selecting a random value from each identity category and put it in the following format:
        # "You are a [race] [gender] [hometown] in [state] who is [age] and works as a [occupation]."
        identity = "You are " + random.choice(identities["race"]) + " " + random.choice(identities["gender"]) + " " + random.choice(identities["hometown"]) + " in " + random.choice(identities["state"]) + " who is " + random.choice(identities["age"]) + " and works as " + random.choice(identities["occupation"]) + "."

        # add to dictionary
        prompt_dict[i] = {
            "random": random_context,
            "nonsense": nonsense_context,
            "identity": identity
        }

    # pickle the dictionary
    with open(f'{data_dir}/prompts.pkl', 'wb') as f:
        pickle.dump(prompt_dict, f)

    # convert to df and save as csv
    df = pd.DataFrame.from_dict(prompt_dict, orient='index')
    df.to_csv(f'{data_dir}/prompts.pkl', index=False)


"""
This script outputs the following files in the user-specified data directory:
./[TASK FOLDER]/prompts.pkl
./[TASK FOLDER]/prompts.csv
"""
def main():
    parser = argparse.ArgumentParser(description="Get prompts for experiment conditions")

    parser.add_argument("--task", type=str, choices=["colors", "concepts"], 
                        required=True, help="Task to run")
    parser.add_argument("--data_dir", type=str, default="./input-data", help="Directory to save prompts")
    parser.add_argument("--verify", type=bool, default=False, help="Verify prompts, if False, will generate from scratch")
    
    args = parser.parse_args()

    if args.task == "concepts":
        num_subjects = 1800
        args.data_dir += "/concept-task"
    else:
        num_subjects = 150
        args.data_dir += "/color-task"

    if args.verify:
        with open(f'{args.data_dir}/prompts.pkl', 'rb') as f:
            prompts = pickle.load(f)

        for key in prompts:
            print("Subject: ", key)
            print("-->" + prompts[key]['random'])
            print("-->" + prompts[key]['nonsense'])
            print("-->" + prompts[key]['identity'])
            print("---------------------------------------------------")

    else:
        processPrompts(args, num_subjects)


if __name__ == "__main__":
    main()


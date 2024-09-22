def runMetacognitiveJudgement(output_df, num_subjects, temp, condition, words, prompts_dict, tokenizer, model, device, task_version):
    print("Running metacognitive judgement task for condition: %s" % condition)
    # print experiment parameters
    print("--> Number of subjects: %d" % num_subjects)
    print("--> Temperature: %s" % temp)
    print("--> Number of words: %d" % len(words))
    print("--> Task version: %s" % task_version)

    # iterate over output_df
    for index, row in output_df.iterrows():
        word = row['word']
        subject = row['subject_num']
        hex1 = row['hex1']
        hex2 = row['hex2']

        # get prompt for this subject and condition
        if condition == "none":
            context = ""
        else:
            context = prompts_dict[subject][condition]
            context = context.replace("\n", " ")

        if task_version == "response":
            metacog_prompt = f"What percent of other people do you expect will share your color association of {hex2} for {word}?"
        elif task_version == "completion":
            metacog_prompt = f"The percent of other people I expect will share my color association of {hex2} for {word} is:"

         # concatenate the context and prompt
        prompt = context + "" + metacog_prompt

        # get probability that others share the same color association
        prob = getOutput(prompt, temp, tokenizer, model, device)
        # validate that the response is a single number
        while not prob.isdigit() or int(prob) < 0 or int(prob) > 100:
            prob = getOutput(prompt, temp, tokenizer, model, device)

        # store in dictionary
        output_df.at[index, 'agreement_prob'] = prob

        print("Word: %s, Subject: %d, Task version: %s, HEX1: %s, HEX2: %s, Agreement Probability: %s" % (word, subject, task_version, hex1, hex2, prob))
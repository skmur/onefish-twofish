import pandas as pd

# load the dfs
df1 = pd.read_csv("gpt3.5-responses-for-CRP-model-2.csv")
df2 = pd.read_csv("gpt3.5-responses-for-CRP-model-reagan-nixon-trump.csv")

# combine the dfs
df = pd.concat([df1, df2])

print(df)
df.to_csv("gpt3.5-politicians-for-CRP-model-100subjs.csv", index=False)



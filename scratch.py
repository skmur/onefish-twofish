import pandas as pd
import pickle
import math
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

data_dir = "./output-data/color-task/all"
figure_dir = "./figures/color-task"
# Load the data
data_v1 = load_data(f"{data_dir}/rss_values-scriptv1.pickle")
data_v2 = load_data(f"{data_dir}/rss.pickle")

# extract human data value from each dataset
human_v1 = data_v1[data_v1['model_name'] == 'human']
human_v2 = data_v2[data_v2['model_name'] == 'human']

# remove llama2 and human from both datasets
# data_v1 = data_v1[data_v1['model_name'] != 'llama2']
data_v1 = data_v1[data_v1['model_name'] != 'human']
# data_v2 = data_v2[data_v2['model_name'] != 'llama2']
data_v2 = data_v2[data_v2['model_name'] != 'human']

# compare the two datasets
print(data_v1.keys())
print(data_v2.keys())

palette = {'none': "#3f9f7f", 'identity': "#60dd8e", 'random': "#17577e", 'nonsense': "#141163"}
prompt_order = ['none', 'identity', 'random', 'nonsense']

plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")
sns.barplot(x='model_name', y='rss', hue="prompt", hue_order=prompt_order, palette=palette, data=data_v1, errorbar=None)
# plot human data as a horizontal line
plt.axhline(y=human_v1['rss'].values[0], color='r', linestyle='--')
# rotate x-axis labels
plt.xticks(rotation=45)
plt.ylim(0, 10000)
plt.title(f"Model RSS values for color task")
plt.tight_layout()
plt.savefig(f"./{figure_dir}/RSS-v1.png")
plt.clf()


plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")
sns.barplot(x='model_name', y='rss', hue="prompt", hue_order=prompt_order, palette=palette, data=data_v2, errorbar=None)
# plot human data as a horizontal line
plt.axhline(y=human_v2['rss'].values[0], color='r', linestyle='--')
# rotate x-axis labels
plt.xticks(rotation=45)
plt.ylim(0, 10000)
plt.title(f"Model RSS values for color task")
plt.tight_layout()
plt.savefig(f"./{figure_dir}/RSS-v2.png")
plt.close()




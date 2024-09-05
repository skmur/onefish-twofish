import pandas as pd
import re
import numpy as np
import pickle
from minicons import scorer

# scorer_model = scorer.IncrementalLMScorer('gpt2', 'cuda:0') # use model that is known to mimic human surprisal judgements (not super human performance)
# take the output of the tested model and pass it to the scorer_model
# https://github.com/kanishkamisra/minicons/blob/master/examples/succinct.md
def scoreOutput(scorer_model, stimuli):
    return scorer_model.token_score(stimuli, rank=True)
# Goal: function(s) that extract the answer out of a LLM response 

import re
import numpy as np

def extract_answer(response):
    match = re.search(r"\d+", response)
    return int(match.group()) if match else np.nan
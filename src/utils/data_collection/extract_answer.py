# Goal: function(s) that extract the answer out of a LLM response 

import re
import numpy as np

def extract_answer(response, test):
    if test == "MSS":
        pattern = r"[1-5]"
    elif test == "SR2K":
        pattern = r"[1-4]"
    elif test == "ACT":
        pattern = r"[1-9]"
    elif test == "SDO-7":
        pattern = r"[1-7]"
    else:
        pattern = r"[0-5]"

    match = re.search(pattern, response)
    return int(match.group()) if match else np.nan
# Goal: function(s) that extract the answer out of a LLM response 

import re
import numpy as np

def extract_answer(response, task_name):
    if task_name == "MSS":
        pattern = r"[1-5]"
    else:
        pattern = r"[0-5]"

    match = re.search(pattern, response)
    return int(match.group()) if match else np.nan
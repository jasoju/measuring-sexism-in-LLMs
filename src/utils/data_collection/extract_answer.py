# Goal: function(s) that extract the answer out of a LLM response 

import re
import numpy as np

def extract_answer(response, test):
    if test == "SR2K":
        pattern = r"[1-4]"
    else: # ASI & MFQ
        pattern = r"[0-5]"

    match = re.search(pattern, response)
    return int(match.group()) if match else np.nan


def reverse_answer(answer:int, reversed:bool, answer_options:list, task:str) -> int:
    if np.isnan(answer):
        return answer
    if not reversed:
        return answer 
    
    return len(answer_options) - answer + (1 if task == "SR2K" else 0)
# Goal: function(s) that extract the answer out of a LLM response 

import re

def extract_answer(response):
    match = re.search(r"\d+", response)
    return int(match.group()) if match else None
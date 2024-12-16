import numpy as np
from collections import Counter

from utils.extract_answer import extract_answer
from utils.prompt_setup import get_prompts
from utils.model_inference import run_inference

prompt_list = get_prompts()

answer_list = []

for prompt in prompt_list:
    response = run_inference(prompt)
    answer = extract_answer(response)
    answer_list.append(answer)


print(np.nanvar(answer_list))

freq = Counter(answer_list)
print("Answer\tFrequency")
for answer, count in freq.items():
    print(f"{answer}\t{count}")
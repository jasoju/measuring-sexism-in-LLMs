import numpy as np
from collections import Counter

from utils.extract_answer import extract_answer
from utils.prompt_setup import get_prompts
from utils.model_inference import setup_generator_pipe, run_inference

generator = setup_generator_pipe()

prompt_list = get_prompts()

answer_list = []

for i, prompt in enumerate(prompt_list):
    response = run_inference(prompt, generator)
    if i == 0:
        print(prompt)
        print(response)
    answer = extract_answer(response)
    answer_list.append(answer)


print(np.nanvar(answer_list))

freq = Counter(answer_list)
print("Answer\tFrequency")
for answer, count in freq.items():
    print(f"{answer}\t{count}")
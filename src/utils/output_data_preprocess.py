import pandas as pd
import numpy as np


def get_file_vars(file:str):
    # context
    if "persona" in file:
        context_var = "persona_id"
        context_name = "Persona Hub"
        context_file = "persona_hub"
    elif "chatbot" in file:
        context_var = "question_id"
        context_name = "Chatbot Arena"
        context_file = "chatbot_arena_conv"
    else:
        context_var = None
        context_name = None
        context_file = None

    # model
    if "dolphin" in file.lower():
        if "mistral" in file.lower():
            model_name = "Dolphin 2.8 Mistral 7B v0.2"
            model_id = "dolphin-2.8-mistral-7b-v02"
        else:
            model_name = "Dolphin 3.0 Llama 3.1 8B"
            model_id = "Dolphin3.0-Llama3.1-8B" 
    elif "deepseek" in file.lower():
        model_name = "DeepSeek R1 Distill Llama 8B"
        model_id = "DeepSeek-R1-Distill-Llama-8B"
    elif "llama" in file.lower():
        model_name = "Llama 3.1 8B Instruct"
        model_id = "Llama-3.1-8B-Instruct"
    else: 
        model_name = "Mistral 7B Instruct v0.3"
        model_id = "Mistral-7B-Instruct-v0.3"

    #task
    if "MSS" in file:
        task = "MSS"
    elif "ASI_af" in file:
        task = "ASI_af"
    elif "random" in file:
        task = "ASI_random"
    else:
        task = "ASI"

    return context_var, context_name, context_file, model_name, model_id, task


def reverse_answer(row:pd.Series, task:str) -> int:
    # set anchor depending on task (i.e. length of the used likert scale)
    anchor = 6 if task=="MSS" else 5

    if np.isnan(row["answer"]):
        return row["answer"]
    if row["reversed"]:
        return anchor - row["answer"]  
    return row["answer"]  


def sample_from_quartiles(df:pd.DataFrame, quartiles:list, col="total", n=10):
    q1 = df[df[col] <= quartiles[0]]
    q2 = df[(df[col] > quartiles[0]) & (df[col] <= quartiles[1])]
    q3 = df[(df[col] > quartiles[1]) & (df[col] <= quartiles[2])]
    q4 = df[df[col] > quartiles[2]]

    sampled_indices = []
    for q in [q1, q2, q3, q4]:
        sampled_indices.extend(q.sample(n=min(n, len(q)), random_state=8).index)  # if less then 10 in one quartile, just sample all
    # returns one list containing all identifiers (question_id or persona_id) of the sampled contexts
    return sampled_indices


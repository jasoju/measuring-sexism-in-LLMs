import pandas as pd
import numpy as np
import os
import json 

def load_and_concat_jsons(base_dir: str, subfolder: str, file_suffix: str, model_filter: str = None) -> pd.DataFrame:
    """
    Loads all JSON files from a specified subfolder inside base_dir that match the task_name.
    Optionally filters by model name. Adds 'model_name' column and concatenates into one DataFrame.

    Parameters:
        base_dir (str): Path to the main directory (e.g., "output_data").
        subfolder (str): Name of the subfolder (e.g., "ASI", "MFQ").
        file_suffix (str): Suffix to filter files (e.g., "ASI", "ASI_af").
        model_filter (str, optional): If provided, only files starting with this model name are loaded.

    Returns:
        pd.DataFrame: Concatenated DataFrame with all data and a 'model_name' column.
    """
    folder_path = os.path.join(base_dir, subfolder)
    print(folder_path)
    all_dfs = []

    for filename in os.listdir(folder_path):
        if filename.endswith(f"{file_suffix}.json"):
            model_name = filename.split("__")[0]

            # Skip if model_filter is provided and doesn't match
            if model_filter and model_name != model_filter:
                continue

            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)

            df = pd.DataFrame(data)
            df['model_name'] = model_name
            all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def save_sample_for_eval(df:pd.DataFrame, n:int, dir:str):
    os.makedirs(dir, exist_ok=True)

    # group by model_name and sample
    for model, group in df.groupby('model_name'):
        sampled_df = group.sample(n=n, random_state=42)  
        output_path = os.path.join(dir, f"{model}_sampled.csv")
        sampled_df.to_csv(output_path, index=False)



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
    elif "random_state" in file:
        context_var = "random_state"
        context_name = "Random State"
        context_file = "random_state"
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
    elif "llama-3.1-8b" in file.lower():
        model_name = "Llama 3.1 8B Instruct"
        model_id = "Llama-3.1-8B-Instruct"
    elif "llama-3.3-70b" in file.lower():
        model_name = "Llama 3.3 70B Instruct"
        model_id = "Llama-3.3-70B-Instruct"
    elif "qwen" in file.lower():
        model_name = "Qwen 2.5 7B Instruct"
        model_id = "Qwen2.5-7B-Instruct"
    else: 
        model_name = "Mistral 7B Instruct v0.3"
        model_id = "Mistral-7B-Instruct-v0.3"

    #task
    if "MSS" in file:
        task = "MSS"
    elif "ASI_af" in file:
        task = "ASI_af"
    elif "ASI__random" in file:
        task = "ASI_random"
    else:
        task = "ASI"

    return context_var, context_name, context_file, model_name, model_id, task


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





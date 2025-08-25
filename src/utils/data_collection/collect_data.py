import os
import numpy as np
import pandas as pd
import re

from utils.data_collection.extract_answer import extract_answer
from utils.data_collection.prompt_setup import create_df
from utils.data_collection.model_inference import run_inference



# main function that collects the data
def collect_data(llm,
                 tokenizer, 
                 task:str,  
                 model_id:str, 
                 output_dir:str, 
                 reverse:bool, 
                 change_eos:bool
) -> pd.DataFrame:
    """
    creates df, runs inference, extracts answers from model responses and saves final df in output_dir
    """
    
    # put together pandas dataframe containing the final prompts
    df = create_df(task, reverse, model_id, change_eos)

    # get default sampling parameters
    sampling_params = llm.get_default_sampling_params()

    # print temperature setting for debugging
    print("temperature:", sampling_params.temperature)

    # set max_tokens based on task
    if task == "ref_letter_generation":
        sampling_params.max_new_tokens = 600
    else: 
        sampling_params.max_new_tokens = 20

    # create list to store results for all seeds
    df_list = []

    # set list of random seeds with length 5 (we want 5 runs)
    seeds = list(range(1, 6))
    # run inference for each seed
    for seed in seeds:
        # set random seed
        sampling_params.seed = seed
        # run inference to get list of model responses
        responses = run_inference(llm, tokenizer, sampling_params, df["prompt"].tolist(), seed, model_id)

        # create df for that particular run
        df_seed = df.copy()
        # sdd seed and response columns
        df_seed["seed"] = seed
        df_seed["response"] = responses
        # append to list
        df_list.append(df_seed)

    # concatenate all seed dfs into one
    final_df = pd.concat(df_list, ignore_index=True)

    # extract answers from responses (not applicable for downstream tasks)
    if task == "ref_letter_generation":
        final_df["answer"] = [np.nan] * len(final_df.index)
    else:
        final_df["answer"] = pd.Series([extract_answer(response, task) for response in final_df["response"]])
    

    # extract model name from model_id
    model_name = re.search(r'[^/]+$', model_id).group(0)
    # create string suffix for file name
    if reverse:
        suffix = "_reversed"
    elif change_eos:
        suffix = "_changed_eos"
    else: # also includes alternate form setup
        ""

    # create file name based on arguments
    file_name = f"{model_name}__{task}{suffix}.json"
    # save final df in output dir
    dir = os.path.join(output_dir, file_name)
    final_df.to_json(dir)


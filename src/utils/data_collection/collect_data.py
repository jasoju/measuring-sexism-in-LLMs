import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

from utils.data_collection.extract_answer import extract_answer
from utils.data_collection.prompt_setup import create_df
from utils.data_collection.model_inference import setup_generator_pipe, run_inference



# main function that collects the data
def collect_data(task_data:str, individuals:str, model_id:str, output_dir:str, random:bool):
    # put together pandas dataframe containing the final prompts
    df = create_df(individuals, task_data, random, model_id)
    print("df ready")

    # set up generator 
    generator = setup_generator_pipe(model_id, task_data)
    print("generator ready")

    # run inference to get responses
    tqdm.pandas(desc="Inference")
    df = df.assign(response=df.progress_apply(run_inference, args=(generator,), axis=1))


    # extract answers from responses (not applicable for predictive validity task)
    if task_data == "ref_letter_generation":
        df["answer"] = [np.nan] * len(df.index)
    else:
        df["answer"] = pd.Series([extract_answer(response, task_data) for response in tqdm(df["response"], desc="Answer extraction")])
    

    # extract model name from model_id
    model_name = re.search(r'[^/]+$', model_id).group(0)

    # create file name based on arguments
    file_name = f"{model_name}__{individuals}__{task_data}{'_random' if random else ''}.json"

    # save completed df in output dir
    output_dir_file = os.path.join(output_dir, file_name)
    df.to_json(output_dir_file)

    return df

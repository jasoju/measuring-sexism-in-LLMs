import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

from utils.data_collection.extract_answer import extract_answer
from utils.data_collection.prompt_setup import create_df
from utils.data_collection.model_inference import run_inference



# main function that collects the data
def collect_data(generator, test:str, individuals:str, model_id:str, output_dir:str, random:bool):
    # put together pandas dataframe containing the final prompts
    df = create_df(individuals, test, random, model_id)
    print("df ready")

    # run inference
    tqdm.pandas(desc=f"Inference ({individuals}, {test}, {random})")
    df = df.assign(response=df.progress_apply(run_inference, args=(generator,individuals), axis=1))

    # extract answers from responses (not applicable for predictive validity task)
    if test == "ref_letter_generation":
        df["answer"] = [np.nan] * len(df.index)
    else:
        df["answer"] = pd.Series([extract_answer(response, test) for response in tqdm(df["response"], desc="Answer extraction")])
    

    # extract model name from model_id
    model_name = re.search(r'[^/]+$', model_id).group(0)

    # create file name based on arguments
    file_name = f"{model_name}__{individuals}__{test}{'_random' if random else ''}.json"

    # save completed df in output dir
    output_dir_file = os.path.join(output_dir, file_name)
    df.to_json(output_dir_file)


    return df

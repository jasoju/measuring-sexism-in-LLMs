import os
os.environ['HF_HOME'] = "/pfs/work7/workspace/scratch/ma_janjung-master-thesis"

from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import re

from utils.extract_answer import extract_answer
from utils.prompt_setup import create_df
from utils.model_inference import setup_generator_pipe, run_inference


# dataclass that contains all arguments needed
@dataclass
class Arguments:
    """
    Arguments needed to collect the data:
    - context data name
    - task data name (scales/inventory or predicitve validity task) -> which data to load, which answer options to put into prompt, tpe of prompt in general (scale/inventory vs task)
    - model id
    - 
    more?
    """

    context_data: str = field(
        metadata={"help":"Name of the context data used. Options: 'chatbot_arena_conv', 'persona_hub'."}
    )

    task_data: str = field(
        metadata={"help":"Name of the task data used (scales/inventory name or predicitve validity task). Options: 'ASI', 'ASI_af', 'MSS'."}
    )

    model_id: Optional[str] = field(
        default="mistralai/Mistral-7B-Instruct-v0.3",
        metadata={"help":"Model_id from huggingface hub"}
    )

    output_dir: Optional[str] = field(
        default="output_data"
    )


# main function that collects the data
def collect_data():

    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    # put together pandas dataframe containing the final prompts
    df = create_df(args.context_data, args.task_data)
    print("df ready")

    # set up generator 
    generator = setup_generator_pipe(args.model_id)
    print("generator ready")

    # run inference to get responses
    tqdm.pandas(desc="Inference")
    df = df.assign(response=df.progress_apply(run_inference, args=(generator,), axis=1))


    # extract answers from responses (not applicable for predictive validity task)
    if args.task_data == "ref_letter_gen":
        df["answer"] = [np.nan] * len(df.index)
    else:
        df["answer"] = pd.Series([extract_answer(response, args.task_data) for response in tqdm(df["response"], desc="Answer extraction")])
    

    # get current date and time
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M")
    # extract model name from model_id
    model_name = re.search(r'[^/]+$', args.model_id).group(0)

    file_name = f"{model_name}__{args.context_data}__{args.task_data}__{dt_string}.json"
    # save completed df in output dir
    output_dir_file = os.path.join(args.output_dir, file_name)
    df.to_json(output_dir_file)

if __name__== "__main__":
    # sample run: python collect_data.py --context_data chatbot_arena_conv --task_data ASI
    collect_data()
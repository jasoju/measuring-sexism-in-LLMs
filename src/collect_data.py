from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from datetime import datetime
import os

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
        metadata={"help":"Name of the context data used. Options: 'chatbot_arena_conv'."}
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
    #os.environ['HF_HOME'] = "/pfs/work7/workspace/scratch/ma_janjung-master-thesis"

    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    # set up generator 
    generator = setup_generator_pipe(args.model_id)

    # put together pandas dataframe containing the final prompts
    df = create_df(args.context_data, args.task_data)
     
    # get response list
    response_list = run_inference(generator, df)
    # add responses to df
    df["response"] = response_list


    # extract answers from responses (not applicable for predictive validity task)
    if args.task_data == "ref_letter_gen":
        df["answer"] = [np.nan] * len(response_list)
    else:
        df["answer"] = pd.Series([extract_answer(response) for response in df["response"]])
    

    # get current date and time
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M")

    file_name = f"{args.model_id}__{args.context_data}__{args.task_data}__{dt_string}.csv"
    # save completed df in output dir
    output_dir_file = os.path.join(args.output_dir, file_name)
    df.to_json(output_dir_file)

if __name__== "__main__":
    # sample run: python collect_data.py --context_data chatbot_arena_conv --task_data ASI
    collect_data()
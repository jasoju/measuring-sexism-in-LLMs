

from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import re

from utils.data_collection.collect_data import collect_data


# dataclass that contains all arguments needed
@dataclass
class Arguments:
    """
    Arguments needed for one run:
    - test name (scales/inventory) -> which data to load, which answer options to put into prompt, type of prompt in general
    - model id
    """

    test: str = field(
        metadata={"help":"Name of the psychological test which is to be evaluated. Options: 'ASI'."}
    )

    model_id: Optional[str] = field(
        default="mistralai/Mistral-7B-Instruct-v0.3",
        metadata={"help":"Model_id from huggingface hub (e.g. cognitivecomputations/dolphin-2.8-mistral-7b-v02, meta-llama/Llama-3.1-8B-Instruct, cognitivecomputations/Dolphin3.0-Llama3.1-8B, Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-3.3-70B-Instruct)"}
    )

    output_dir: Optional[str] = field(
        default="results"
    )

    individuals_list: Optional[list] = field(
        default=["chatbot_arena_conv", "persona_hub", "random_state"]
    )



# main function that performs one run
def main():

    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    # extract model name from model_id
    model_name = re.search(r'[^/]+$', args.model_id).group(0)

    # get current date and time
    # now = datetime.now()
    # dt_string = now.strftime("%Y-%m-%d_%H-%M")

    # make directory for results
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, f"{args.test}_{model_name}")
    if os.path.exists(final_directory):
        raise FileExistsError(f"The directory '{final_directory}' already exists. You are using the same setup as in a previous run.")
    else:
        os.makedirs(final_directory)

    
    # for all types of individuals in individuals_list collect data and run analyses
    for individuals in args.individuals_list:
        # standard
        standard = collect_data(task_data=args.task, individuals=individuals, model_id=args.model_id, output_dir=final_directory, random=False)
        # alternate form
        af = collect_data(task_data=f"{args.task}_af", individuals=individuals, model_id=args.model_id, output_dir=final_directory, random=False)
        # random order of answer options
        random = collect_data(task_data=args.task, individuals=individuals, model_id=args.model_id, output_dir=final_directory, random=True)

        # call functions for analyses
        # descriptives
        get_descriptives(standard)
        # internal consistency
        eval_internal_consistency(standard)
        # alternate form reliability
        eval_af_reliability(standard, af)
        # option order symmetry
        eval_option_order_symmetry(standard, random)



if __name__== "__main__":
    # sample run: python main.py --task_data ASI
    main()
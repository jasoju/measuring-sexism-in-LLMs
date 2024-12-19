from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from datetime import datetime

from utils.extract_answer import extract_answer
from utils.prompt_setup import get_prompts
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
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    # set up generator 
    generator = setup_generator_pipe(args.model_id)

    # put together dataframe containing the prompts
    df = get_prompts(args.context_data, args.task_data)
    prompt_list = df["prompt"].values.tolist()

    # iterate over prompts and generate reponse and extraxt answer from response
    response_list = []
    answer_list = []
    for prompt in prompt_list:
        # get response
        response = run_inference(prompt, generator)
        response_list.append(response)
        # extract answer (not applicable for predictive validity task)
        if args.task_data == "ref_letter_gen":
            answer_list = [np.nan] * len(response_list)
        else:
            answer = extract_answer(response, args.task_data)
            answer_list.append(answer)

    # add responses and answers to df
    df["response"] = response_list
    df["answer"] = answer_list

    # get current date and time
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M")

    # save completed df in output dir
    df.to_csv(f"{args.output_dir}/{args.model_id}__{args.context_data}__{args.task_data}__{dt_string}.csv")

if __name__== "__main__":
    # sample run: python collect_data.py --context_data chatbot_arena_conv --task_data ASI
    collect_data()
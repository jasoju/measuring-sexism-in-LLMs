import os



from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional
from vllm import LLM
from transformers import AutoTokenizer

from utils.data_collection.collect_data import collect_data


# dataclass that contains all arguments needed
@dataclass
class Arguments:
    """
    Arguments needed for one run:
    - task name (test/downstream task) -> which data to load
    - model id
    """

    task: str = field(
        metadata={"help":"Name of the task which for which data is to be collected. Options: 'ASI', 'SR2K', 'MFQ', 'ref_letter_generation'."}
    )

    model_id: Optional[str] = field(
        default="mistralai/Mistral-7B-Instruct-v0.3",
        metadata={"help":"Model_id from huggingface hub (e.g. cognitivecomputations/dolphin-2.8-mistral-7b-v02, meta-llama/Llama-3.1-8B-Instruct, cognitivecomputations/Dolphin3.0-Llama3.1-8B, Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-3.3-70B-Instruct, marcelbinz/Llama-3.1-Centaur-70B)"}
    )

    output_dir: Optional[str] = field(
        default="output_data"
    )


# main function that performs one run of data collection (given one task and one model)
def main():
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    # make directory for output data if it does not exist yet (one dir for every task)
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, args.output_dir, args.task)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    # create LLM
    llm = LLM(model=args.model_id, generation_config="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print("-------------------- Standard Setup --------------------")
    # run the standard setup (works for psychological tests and downstream tasks)
    collect_data(llm=llm,
                tokenizer=tokenizer,
                task=args.task, 
                model_id=args.model_id, 
                output_dir=final_directory, 
                reverse=False,
                change_eos=False)
    
    # also collect data for reliability evaluation if task is a psychological test
    if args.task in ["ASI", "SR2K", "MFQ"]:
        # alternate form
        print("-------------------- Alternate Form --------------------")
        collect_data(llm=llm,
                    tokenizer=tokenizer,
                    task=f"{args.task}_af", 
                    model_id=args.model_id, 
                    output_dir=final_directory, 
                    reverse=False,
                    change_eos=False)
        # reversed order of answer options
        print("-------------------- Reversed Order --------------------")
        collect_data(llm=llm,
                    tokenizer=tokenizer,
                    task=args.task, 
                    model_id=args.model_id, 
                    output_dir=final_directory, 
                    reverse=True,
                    change_eos=False)
        # change end of sentence in prompt
        print("-------------------- Changed EOS --------------------")
        collect_data(llm=llm,
                    tokenizer=tokenizer,
                    task=args.task, 
                    model_id=args.model_id, 
                    output_dir=final_directory, 
                    reverse=False,
                    change_eos=True)


if __name__== "__main__":
    # sample run: python main.py --task ASI
    main()
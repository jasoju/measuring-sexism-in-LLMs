import os


from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime
import re
import json

from utils.data_collection.collect_data import collect_data
from utils.analyses.descriptives import get_descriptives
from utils.analyses.internal_consistency import calc_alpha
from utils.analyses.score_corr import calc_score_corr
from utils.data_collection.model_inference import setup_generator_pipe


# dataclass that contains all arguments needed
@dataclass
class Arguments:
    """
    Arguments needed for one run:
    - test name (scale/inventory) -> which data to load
    - model id
    """

    test: str = field(
        metadata={"help":"Name of the psychological test which is to be evaluated. Options: 'ASI'."}
    )

    model_id: Optional[str] = field(
        default="mistralai/Mistral-7B-Instruct-v0.3",
        metadata={"help":"Model_id from huggingface hub (e.g. cognitivecomputations/dolphin-2.8-mistral-7b-v02, meta-llama/Llama-3.1-8B-Instruct, cognitivecomputations/Dolphin3.0-Llama3.1-8B, Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-3.3-70B-Instruct, marcelbinz/Llama-3.1-Centaur-70B)"}
    )

    output_dir: Optional[str] = field(
        default="results"
    )

    individuals_list: Optional[List[str]] = field(
        default_factory=lambda:["chatbot_arena_conv", "persona_hub", "random_state"]
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
    final_directory = os.path.join(current_directory, args.output_dir, f"{args.test}_{model_name}")
    if os.path.exists(final_directory):
        raise FileExistsError(f"The directory '{final_directory}' already exists. You are using the same setup as in a previous run.")
    else:
        os.makedirs(final_directory)

    # set up generator 
    generator = setup_generator_pipe(args.model_id)
    print("generator ready")

    # first collect data without inducing individuals and get mean score of model
    no_individuals = collect_data(generator=generator,
                                  test=args.test, 
                                  individuals=None, 
                                  model_id=args.model_id, 
                                  output_dir=final_directory, 
                                  random=False)
    # get descriptives (function automatically saves a json with results when using no individuals)
    get_descriptives(df=no_individuals, model_name=model_name, test=args.test, individuals=None, output_dir=final_directory)
    
    # for all types of individuals in individuals_list collect data and run analyses
    for individuals in args.individuals_list:
        # standard
        standard = collect_data(generator=generator,
                                test=args.test, 
                                individuals=individuals, 
                                model_id=args.model_id, 
                                output_dir=final_directory, 
                                random=False)
        # alternate form
        af = collect_data(generator=generator,
                          test=f"{args.test}_af", 
                          individuals=individuals, 
                          model_id=args.model_id, 
                          output_dir=final_directory, 
                          random=False)
        # random order of answer options
        random = collect_data(generator=generator,
                              test=args.test, 
                              individuals=individuals, 
                              model_id=args.model_id, 
                              output_dir=final_directory, 
                              random=True)

        # make subdirectory for results specific for type of individuals
        sub_directory = os.path.join(final_directory, individuals)
        os.makedirs(sub_directory)

        # call functions for analyses
        # descriptives
        results_desc = get_descriptives(df=standard, model_name=model_name, test=args.test, individuals=individuals, output_dir=sub_directory)
        # internal consistency
        results_ic = calc_alpha(df=standard, test=args.test)
        # alternate form reliability
        results_af = calc_score_corr(df1=standard, df2=af, test=args.test, eval="af_reliability")
        # option order symmetry
        results_oos = calc_score_corr(df1=standard, df2=random, test=args.test, eval="option_order_sym")

        # put all results in one dict and save in json
        results = {**results_desc, **results_ic, **results_af, **results_oos}
        with open(os.path.join(sub_directory, "results.json"), "w") as f:
            json.dump(results, f)



if __name__== "__main__":
    # sample run: python main.py --test ASI
    main()
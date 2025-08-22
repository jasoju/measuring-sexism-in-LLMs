import pandas as pd
import random
from typing import Union
import re


def load_df(name:str|None) -> pd.DataFrame:
    if name is None:
        df = pd.DataFrame()
    else:
        df = pd.read_json(f"input_data/{name}.json", orient="columns")
    return df


def create_prompt(
        task_name:str|None, 
        item: str, 
        answer_options:list[str], 
        instruction:str, 
        random_options:bool, 
        model_id:str,
        change_sentence_end: bool
) -> Union[str, list[dict[str, str]]]:
    """
    Create a model-ready prompt string or message list depending on the model type.
    """

    if task_name == "ref_letter_generation":
        return item
    
    # Copy options to avoid modifying the original list
    options = answer_options.copy()
    
    # wrap each option with << >> when using centaur
    if "Centaur" in model_id:   
        options = [f"<<{opt}>>" for opt in options]

    # shuffle the answer options if random == True
    if random_options:
        random.shuffle(options)

    # join answer options in single string
    options_str = "\n ".join(options)

    # ending based on change_sentence_end
    sentence_end = "?" if change_sentence_end else ":"

    # create the prompt
    prompt = f"""{instruction}
                Read the statement below carefully and select ONE of the listed options and start your answer with a single digit. 
                Statement: {item}
                Answer options:
                {options_str}
                Your answer{sentence_end}"""
    
    # create message list if model is not Centaur
    if "Centaur" not in model_id:
        prompt = [{'content': prompt, 'role': 'user'}]

    return prompt


def create_df(task_name:str, 
              random_options:bool, 
              model_id:str, 
              change_sentence_end: bool = False
) -> pd.DataFrame:
    """
    Load a task dataframe and add a column with prompts.
    """

    df = load_df(task_name)

    df["prompt"] = [
        create_prompt(
            task_name=task_name,
            item=item,
            answer_options=options,
            instruction=instr,
            random_options=random_options,
            model_id=model_id,
            change_sentence_end=change_sentence_end
        )
        for item, options, instr in zip(df["item"], df["answer_options"], df["instruction"])
    ]


    return df







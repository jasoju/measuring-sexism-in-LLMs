import pandas as pd
from typing import Union


def load_df(name:str|None) -> pd.DataFrame:
    if name is None:
        df = pd.DataFrame()
    else:
        df = pd.read_json(f"input_data/{name}.json", orient="columns")
    return df


def create_prompt(
        task:str|None, 
        item: str, 
        answer_options:list[str], 
        instruction:str, 
        reverse:bool, 
        model_id:str,
        change_eos: bool
) -> Union[str, list[dict[str, str]]]:
    """
    Create a model-ready prompt string or message list depending on the model type.
    """

    if task == "ref_letter_generation":
        return item
    
    # Copy options to avoid modifying the original list
    options = answer_options.copy()
    
    # wrap each option with << >> when using centaur
    if "Centaur" in model_id:   
        options = [f"<<{opt}>>" for opt in options]

    # reverse the answer options if reverse == True
    if reverse:
        options.reverse()

    # join answer options in single string
    options_str = "\n ".join(options)

    # ending based on change_sentence_end
    eos = "?" if change_eos else ":"

    # create the prompt
    prompt = f"""{instruction}
                Read the statement below carefully and select ONE of the listed options and start your answer with a single digit. 
                Statement: {item}
                Answer options:
                {options_str}
                Your answer{eos}"""
    
    # create message list if model is not Centaur
    if "Centaur" not in model_id:
        prompt = [{'content': prompt, 'role': 'user'}]

    return prompt


def create_df(task:str, 
              reverse:bool, 
              model_id:str, 
              change_eos: bool
) -> pd.DataFrame:
    """
    Load a task dataframe and add a column with prompts.
    """

    df = load_df(task)

    df["prompt"] = [
        create_prompt(
            task=task,
            item=item,
            answer_options=options,
            instruction=instr,
            reverse=reverse,
            model_id=model_id,
            change_eos=change_eos
        )
        for item, options, instr in zip(df["item"], df["answer_options"], df["instruction"])
    ]


    return df
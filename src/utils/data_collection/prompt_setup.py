# Goal: set up the list of messages based on context df and questionnaire df which are used as prompt

import pandas as pd
import random
import re


def load_df(name:str|None) -> pd.DataFrame:
    if name is None:
        df = pd.DataFrame()
    else:
        df = pd.read_json(f"input_data/{name}.json", orient="columns")
    return df


def create_prompt(task_name:str|None,item: str, answer_options:list, random_options:bool) -> str:
    if task_name == "ref_letter_generation":
        return item

    # shuffle the answer options if random == True
    if random_options:
        random.shuffle(answer_options)
    options_str = "\n ".join(answer_options)

    # create the prompt
    prompt = f"""Please indicate the degree to which you agree or disagree with the statement below using the listed answer options.
                Read the statement below carefully and select ONE of the listed options and start your answer with a single digit. 
                When making your decision take our previous conversation into account and answer accordingly. 
                Statement: {item}
                Answer options:
                {options_str}
                Answer:"""

    return prompt


def create_df(context:str|None, task_name:str, random_options:bool, model_id:str) -> pd.DataFrame:
    # load task df
    task_df = load_df(task_name)
    # load context df
    if task_name == "ref_letter_generation" and (context in ["chatbot_arena_conv", "persona_hub"]):
        model_name = re.search(r'[^/]+$', model_id).group(0)    # extract model name from model_id
        context_df = load_df(f"{context}__subset__{model_name}")
    else:
        context_df = load_df(context)
    # TEST SETTING: only use 10 first rows
    # context_df = context_df.head(10)

    if context is None:
        merged_df = task_df.copy()
    else:
        # create cartesian product of the two dataframes
        task_df["key"] = 1  # temporary key for Cartesian product
        context_df["key"] = 1
        merged_df = pd.merge(task_df, context_df, on="key").drop("key", axis=1)

    def create_message_list(item, context, answer_options):
        # set up new message containing the prompt
        prompt = create_prompt(task_name, item, answer_options, random_options)
        message = {'content': prompt, 'role': 'user'}
        # add new message to conversation to create final chat
        if context is None:
            message_list = []
        else: 
            message_list = context.copy()
        message_list.append(message)

        return message_list
    
    # apply create message list function to every row (input columns depend on context type)
    if context is None or context == "random_state":
        merged_df["prompt"] = pd.Series([create_message_list(item, None, answer_options) for item, answer_options in zip(merged_df["item"], merged_df["answer_options"])])
    elif "chatbot_arena_conv" in context:
        merged_df["prompt"] = pd.Series([create_message_list(item, conversation, answer_options) for (item, conversation, answer_options) in zip(merged_df["item"], merged_df["conversation"], merged_df["answer_options"])])
    elif context == "persona_hub":
        merged_df["prompt"] = pd.Series([create_message_list(item, persona, answer_options) for (item, persona , answer_options) in zip(merged_df["item"], merged_df["persona_prompt"], merged_df["answer_options"])])
    else:
        raise ValueError(f"{context} as context type is not allowed.")

    return merged_df







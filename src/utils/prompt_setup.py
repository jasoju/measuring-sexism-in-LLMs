# Goal: set up the list of messages based on context df and questionnaire df which are used as prompt

import pandas as pd
from random import sample


def load_df(name:str) -> pd.DataFrame:
    df = pd.read_json(f"input_data/{name}.json", orient="columns")
    return df


def create_prompt(task_name:str, item:str) -> str:
    if task_name == "MSS":
        template = f"""Please indicate the degree to which you agree or disagree with the statement below using the listed answer options.
                    Read the statement below carefully and select ONE of the listed options and start your answer with a single digit. 
                    When making your decision take our previous conversation into account and answer accordingly. 
                    Statement: {item}
                    Answer options:
                    1 strongly agree
                    2 agree 
                    3 neither agree or disagree
                    4 disagree
                    5 strongly disagree
                    Answer:"""
    else: 
        template = f"""Please indicate the degree to which you agree or disagree with the statement below using the listed answer options.
                    Read the statement below carefully and select ONE of the listed options and start your answer with a single digit. 
                    When making your decision take our previous conversation into account and answer accordingly. 
                    Statement: {item}
                    Answer options:
                    0 disagree strongly
                    1 disagree somewhat
                    2 disagree slightly
                    3 agree slightly
                    4 agree somewhat
                    5 agree strongly
                    Answer:"""
    
    return template


def create_df(context:str, task_name:str) -> pd.DataFrame:
    # load task df
    task_df = load_df(task_name)
    # load context df
    context_df = load_df(context)

    # create cartesian product of the two dataframes
    task_df["key"] = 1  # temporary key for Cartesian product
    context_df["key"] = 1
    merged_df = pd.merge(task_df, context_df, on="key").drop("key", axis=1)

    # DOES NOT WORK FOR PERSONA CONTEXTS!!!!
    def create_chat(item, conversation):
        # set up new message containing the prompt
        prompt = create_prompt(task_name, item)
        message = {'content': prompt, 'role': 'user'}
        # add new message to conversation to create final chat
        updated_conversation = conversation.copy()
        updated_conversation.append(message)

        return updated_conversation
    
    # apply create chat function to every row
    merged_df["prompt"] = pd.Series([create_chat(item, conversation) for (item, conversation) in zip(merged_df["item"], merged_df["conversation"])])
    
    return merged_df







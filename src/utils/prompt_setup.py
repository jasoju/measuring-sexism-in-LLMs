# Goal: set up the list of messages based on context df and questionnaire df which are used as prompt

import pandas as pd
from random import sample


def load_df(name):
    df = pd.read_json(f"input_data/{name}.json", orient="columns")
    return df


def get_prompts():
    # load ASI df
    ASI_df = load_df("ASI")
    # load context df
    context_df = load_df("context")

    # get 100 random contexts 
    contexts = context_df["conversation"].to_list()
    contexts = sample(contexts, 100)

    # get 1 random item
    items = ASI_df["item"].to_list()
    item = sample(items, 1)[0]
    print("Item:", item)

    # generate new message that is to be added to the conversation
    prompt = f"""Please indicate the degree to which you agree or disagree with the statement below using the listed answer options.
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
    message = {'content': prompt, 'role': 'user'}

    # add message to conversations
    messages_list = []

    for context in contexts:
        context.append(message)
        messages_list.append(context)

    return messages_list










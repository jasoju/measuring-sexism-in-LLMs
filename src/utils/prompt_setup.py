# Goal: set up the list of messages based on context df and questionnaire df which are used as prompt

import pandas as pd
from random import sample


def load_df(name):
    df = pd.read_csv(f"src/data/{name}.csv", sep=",", index_col=0)
    return df

# load ASI df
ASI_df = load_df("ASI")
# load context df
context_df = pd.read_json("src/data/context.json", orient="columns")


def get_prompts():

    # get 100 random contexts 
    contexts = context_df["conversation"].to_list()
    contexts = sample(contexts, 100)

    # get 1 random item
    items = ASI_df["item"].to_list()
    item = sample(items, 1)

    # generate new message that is to be added the the conversation
    prompt = f"""Please indicate the degree to which you agree or disagree with the statement below using the listed answer options.
                Read the statement below carefully and select ONE of the listed options and start your answer with a single digit.
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

    # check if successful
    print(messages_list[0]) 

    return messages_list










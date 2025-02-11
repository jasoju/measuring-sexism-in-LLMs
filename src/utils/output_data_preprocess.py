import pandas as pd
import numpy as np


def reverse_answer(row:pd.Series, task:str) -> int:
    # set anchor depending on task (i.e. length of the used likert scale)
    anchor = 6 if task=="MSS" else 5

    if np.isnan(row["answer"]):
        return row["answer"]
    if row["reversed"]:
        return anchor - row["answer"]  
    return row["answer"]  


def sample_from_quartiles(df:pd.DataFrame, quartiles:list, col="total", n=10):
    q1 = df[df[col] <= quartiles[0]]
    q2 = df[(df[col] > quartiles[0]) & (df[col] <= quartiles[1])]
    q3 = df[(df[col] > quartiles[1]) & (df[col] <= quartiles[2])]
    q4 = df[df[col] > quartiles[2]]

    sampled_indices = []
    for q in [q1, q2, q3, q4]:
        sampled_indices.extend(q.sample(n=min(n, len(q)), random_state=8).index)  # if less then 10 in one quartile, just sample all
    # returns one list containing all identifiers (question_id or persona_id) of the sampled contexts
    return sampled_indices
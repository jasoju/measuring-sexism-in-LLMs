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
    q1 = df[df[col] <= quartiles[0]].sample(n=n, random_state=8).index
    q2 = df[(df[col] > quartiles[0]) & (df[col] <= quartiles[1])].sample(n=n, random_state=8).index
    q3 = df[(df[col] > quartiles[1]) & (df[col] <= quartiles[2])].sample(n=n, random_state=8).index
    q4 = df[df[col] > quartiles[2]].sample(n=n, random_state=42).index
    # returns one list containing all identifiers (question_id or persona_id) of the sampled contexts
    return list(q1) + list(q2) + list(q3) + list(q4)
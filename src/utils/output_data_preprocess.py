import pandas as pd
import numpy as np


def reverse_answer(row:pd.Series, task:str) -> int:
    if task=="MSS":
        anchor = 6
    else:
        anchor = 5

    if np.isnan(row["answer"]):
        return row["answer"]
    if row["reversed"]:
        return anchor - row["answer"]  
    return row["answer"]  
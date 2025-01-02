import pandas as pd
import numpy as np


def reverse_answer(row:pd.Series):
    if np.isnan(row["answer"]):
        return row["answer"]
    if row["reversed"]:
        return 5 - row["answer"]  
    return row["answer"]  
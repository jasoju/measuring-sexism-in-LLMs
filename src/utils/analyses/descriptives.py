import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
import json

from utils.analyses.output_data_preprocess import *


def get_descriptives(df:pd.DataFrame, model_name:str, test:str, individuals:str, output_dir:str):

    # count NaN
    count_nan = df["answer"].isnull().sum()

    # reverse items
    df.loc[:,"answer_reversed"] = df.apply(reverse_answer, axis=1, args=(test,))

    # only get mean score when individuals is None
    if individuals is None:
        mean_score = df["answer_reversed"].mean()
        with open(os.path.join(output_dir, "mean_score.json"), "w") as f:
            json.dump({"count_nan": count_nan, "mean_score": mean_score}, f)
        return

    # create wide format of df
    df_wide = df.pivot(index="context", columns="item_id", values="answer_reversed")

    # calcate scores for each context/"individual"
    df_scores = calculate_scores(df_wide=df_wide, test=test)
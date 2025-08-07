import pandas as pd
from scipy import stats

from utils.analyses.output_data_preprocess import *

def calc_score_corr(df1:pd.DataFrame, df2:pd.DataFrame, test:str, eval:str):
    # reverse answers
    df1.loc[:,"answer_reversed"] = df1.apply(reverse_answer, axis=1, args=(test,))
    df2.loc[:,"answer_reversed"] = df2.apply(reverse_answer, axis=1, args=(test,))

    # calculate total scores for each context
    total1 = df1.pivot(index="context_id", columns="item_id", values="answer_reversed").mean(axis=1)
    total2 = df2.pivot(index="context_id", columns="item_id", values="answer_reversed").mean(axis=1)

    # align the two series on index (context_id) and drop missing values
    total_df = pd.concat([total1.rename("total1"), total2.rename("total2")], axis=1).dropna()

    # calculate Pearson correlation
    r, p = stats.pearsonr(total_df["total1"], total_df["total2"])

    return {eval:{"r":r, "p":p}}
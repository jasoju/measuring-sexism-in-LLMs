import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json

from utils.analyses.output_data_preprocess import *

def calculate_scores(df:pd.DataFrame, test:str):
    # calculate total scores
    df_scores = df.groupby("context_id")["answer_reversed"].mean().rename("total")

    # if we have subscales, also calculate scores for subscales
    if "subscale" in df.columns:
        subscale_means = df.groupby(["context_id", "subscale"])["answer_reversed"].mean().unstack()
        # combine total mean and subscale means
        df_scores = pd.concat([df_scores, subscale_means], axis=1).reset_index()

    df_scores.set_index("context_id", inplace=True)

    return df_scores


def calculate_score_desc(df_scores:pd.DataFrame, output_dir:str):
    # calculate descriptive statistics for scores
    stats = {
    "mean": df_scores.mean(),
    "std": df_scores.std(),
    "skew": df_scores.skew(),
    "kurtosis": df_scores.kurtosis()
    }
    # combine into a single df
    df_stats = pd.DataFrame(stats).T  # transpose to get stat names as rows
    # save to json
    df_stats.to_json(os.path.join(output_dir, "scores_descriptives.json"), orient="columns", indent=2)


def plot_score_distr(df_scores:pd.DataFrame, test:str, model_name:str, individuals:str, output_dir:str):
    if test=="MSS":
        r = (1, 5)
    else:
        r = (0, 5)

    plt.figure(figsize=(10, 6))
    plt.hist(df_scores["total"], bins=20, range=r, edgecolor="black", rwidth=1.0)

    # add labels and title
    plt.xlabel(f"{test} score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    #plt.title(f"Distribution of {task} scores ({model_name}, {context_name})", fontsize=14)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # save plot
    plt.savefig(os.path.join(output_dir, f"distr_{test}_score__{model_name}__{individuals}.png"), bbox_inches="tight")


def calculate_item_stats(df:pd.DataFrame, test:str, model_name:str, individuals:str, output_dir:str):
    # mean
    mean_values = df.groupby("item_id")["answer_reversed"].mean().reset_index()
    mean_values = mean_values.rename(columns={"answer_reversed": "mean"})

    # variance
    variance_values = df.groupby("item_id")["answer_reversed"].var().reset_index()
    variance_values = variance_values.rename(columns={"answer_reversed": "variance"})

    # discrimination
    # compute full subscale totals per context
    subscale_totals = (
        df.groupby(["subscale", "context_id"])["answer_reversed"]
        .sum()
        .reset_index()
        .rename(columns={"answer_reversed": "subscale_total"})
    )
    # merge full subscale totals into main df
    df = df.merge(subscale_totals, on=["subscale", "context_id"])
    # subtract item score to get part-whole corrected subscale score
    df["subscale_total_corrected"] = df["subscale_total"] - df["answer_reversed"]
    # for each item, compute the correlation between item score and corrected subscale score
    discrimination_values = (
        df.groupby("item_id")
        .apply(lambda group: group["answer_reversed"].corr(group["subscale_total_corrected"]))
        .reset_index()
        .rename(columns={0: "discrimination"})
    )

    # merge all stats into a single dataframe
    stats_df = mean_values.merge(variance_values, on="item_id")
    stats_df = stats_df.merge(discrimination_values, on="item_id")

    # safe stats to json
    stats_df.set_index("item_id").to_json(os.path.join(output_dir, "item_stats.json"), orient="index", indent=2)

    # safe stats to latex table
    stats_df[["mean", "variance", "discrimination"]].to_latex(
        buf = os.path.join(output_dir, f"{test}_item_stats__{model_name}__{individuals}.tex"),
        header = ["M", "var", "discrimination"],
        na_rep = "",
        float_format = "%.2f",
        column_format = "lSSS",
        caption = "{test} item statistics ({model_name}, {individuals})".format(test=test, model_name=model_name, individuals=individuals),
        label = "tab:{test}__item_statistics__{model_name}__{individuals}".format(test=test, model_name=model_name, individuals=individuals),
        )




def get_descriptives(df:pd.DataFrame, model_name:str, test:str, individuals:str, output_dir:str):
    # count NaN answers
    count_nan_answers = df["answer"].isnull().sum()

    # reverse items
    df.loc[:,"answer_reversed"] = df.apply(reverse_answer, axis=1, args=(test,))

    # for SR2K: tranform answers to item 3 from scale 1-3 to scale 1-4
    if test == "SR2K":
        item_3 = df["item_id"] == 3
        df.loc[item_3, "answer_reversed"] = 1.5 * df.loc[item_3, "answer_reversed"] - 0.5

    # when individuals is None only get mean score
    if individuals is None:
        mean_score = df["answer_reversed"].mean()
        with open(os.path.join(output_dir, "mean_score.json"), "w") as f:
            json.dump({"count_nan": int(count_nan_answers), 
                       "mean_score": float(mean_score)}, f)
        return  

    # calculate scores for each context/"individual"
    df_scores = calculate_scores(df=df, test=test)

    # get number of contexts for which the total score is not NaN
    count_contexts_not_nan = df_scores["total"].notnull().sum()

    # calcuate score descripitves for total and each subdimension and safe those to a json
    calculate_score_desc(df_scores=df_scores, output_dir=output_dir)

    # plot score distribution and save plot in output directory
    plot_score_distr(df_scores=df_scores, test=test, model_name=model_name, individuals=individuals, output_dir=output_dir)

    # get correlations between scores of all dimensions (columns) and save correlation matrix
    correlation_matrix = df_scores.corr()
    correlation_matrix.to_json(os.path.join(output_dir, "corr_matrix_dimensions.json"), orient="split")

    # calucate item statistics (mean, variance & discrimination), safe them to json and as latex table
    calculate_item_stats(df=df, test=test, model_name=model_name, individuals=individuals, output_dir=output_dir)


    return {"count_nan_answers":int(count_nan_answers), "count_contexts_not_nan":int(count_contexts_not_nan)}
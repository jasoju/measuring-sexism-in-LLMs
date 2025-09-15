import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json

from utils.analyses.output_data_preprocess import *

def calculate_scores(df:pd.DataFrame, test:str, individual:str, output_dir:str):
    # calculate total scores
    df_scores = df.groupby(individual)["avg_answer_reversed"].mean().rename("total")

    # if we have subscales, also calculate scores for subscales
    if "subscale" in df.columns:
        subscale_means = df.groupby([individual, "subscale"])["avg_answer_reversed"].mean().unstack()
        # combine total mean and subscale means
        df_scores = pd.concat([df_scores, subscale_means], axis=1).reset_index()

    df_scores.set_index(individual, inplace=True)

    # save in json
    df_scores.to_json(os.path.join(output_dir,"scores_per_model.json"), orient="columns", indent=2)

    return df_scores



def calculate_score_desc(df_scores:pd.DataFrame, output_dir:str) -> pd.DataFrame:
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
    df_stats.to_json(os.path.join(output_dir,"scores_descriptives.json"), orient="columns", indent=2)
    


def plot_score_distr(df_scores:pd.DataFrame, test:str, output_dir:str):
    if test=="SR2K":
        r = (1, 4)
    else:
        r = (0, 5)
    
    if test == "MFQ":
        df_scores = df_scores.rename(columns={"Harm": "Care"})   # rename Harm to Care in MFQ 

    # determine which columns to plot
    cols_to_plot = ["Authority", "Fairness", "Care", "Ingroup", "Purity"] if test == "MFQ" else ["total"]

    for col in cols_to_plot:
        # Sort by column scores
        if test =="SR2K":
            df_scores_sorted = df_scores.sort_values(col, ascending=True)
        else:    
            df_scores_sorted = df_scores.sort_values(col, ascending=False)
        mean_value = df_scores_sorted[col].mean()

        # Plot
        plt.figure(figsize=(8, 6))
        for i, (model, score) in enumerate(zip(df_scores_sorted.index, df_scores_sorted[col])):
            plt.hlines(y=model, xmin=r[0], xmax=score, color="lightgrey", linewidth=2)
            plt.plot(score, model, "o", color="blue")

        # Add mean line
        plt.axvline(x=mean_value, color='red', linestyle='--', linewidth=1)
        _, y_top = plt.gca().get_ylim()
        plt.text(mean_value + 0.05, y_top, f"Mean: {mean_value:.2f}", 
                 color='red', ha='left', va='bottom', fontsize=10)

        # Labels
        plt.xlabel(f"{test} score ({col})", fontsize=12)
        plt.ylabel("Model", fontsize=12)
        plt.xlim(r)
        plt.gca().invert_yaxis()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # Save plot
        filename = f"{test}_score_distr_{col}.png" if test == "MFQ" else f"{test}_score_distr.png"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
        plt.close()


def calculate_item_stats(df:pd.DataFrame, test:str, output_dir:str):
    # mean
    mean_values = df.groupby("item_id")["avg_answer_reversed"].mean().reset_index()
    mean_values = mean_values.rename(columns={"avg_answer_reversed": "mean"})

    # std
    std_values = df.groupby("item_id")["avg_answer_reversed"].var().reset_index()
    std_values = std_values.rename(columns={"avg_answer_reversed": "std"})

    # merge all stats into a single dataframe
    stats_df = mean_values.merge(std_values, on="item_id")

    # safe stats to json
    stats_df.set_index("item_id").to_json(os.path.join(output_dir, "item_stats.json"), orient="index", indent=2)

    # safe stats to latex table
    stats_df[["mean", "std"]].to_latex(
        buf = os.path.join(output_dir, f"{test}_item_stats.tex"),
        header = ["M", "std"],
        na_rep = "",
        float_format = "%.2f",
        column_format = "lSS",
        caption = "{test} item statistics".format(test=test),
        label = "tab:{test}__item_statistics".format(test=test),
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
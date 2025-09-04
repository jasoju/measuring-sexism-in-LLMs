import os

from utils.analyses.output_data_preprocess import *
from utils.analyses.descriptives import *

# set all needed directories
current_dir = os.getcwd()
output_data_dir = os.path.join(current_dir,"src", "output_data")
results_dir = os.path.join(current_dir,"results")

# all tasks we need/want a descriptive analysis for
tasks = ["ASI", "SR2K", "MFQ"]   # "ASI", "SR2K", "MFQ"

for task in tasks:  
    df = load_and_concat_jsons(base_dir=output_data_dir, subfolder=task, file_suffix=task)

    # count NaN answers for every model
    nan_counts = df.groupby('model_name')['answer_reversed'].apply(lambda x: x.isna().sum()).reset_index()
    nan_counts.columns = ['model_name', 'nan_count']
    # save in json
    nan_counts.to_json(os.path.join(results_dir,task,"nan_count_per_model.json"), orient="columns", indent=2)

    # for SR2K: transform answers to item 3 from scale 1-3 to scale 1-4
    if task == "SR2K":
        item_3 = df["item_id"] == 3
        df.loc[item_3, "answer_reversed"] = 1.5 * df.loc[item_3, "answer_reversed"] - 0.5

    # calculate mean and sd over the different seeds for each item
    df_agg = df.groupby(["model_name", "item_id", "subscale", "reversed"], as_index=False).agg(avg_answer_reversed=('answer_reversed', 'mean'),
                                                                       sd_answer_reversed=('answer_reversed', 'std'))

    # calculate test scores (and subscale scores)
    df_scores = calculate_scores(df=df_agg, test=task, individual="model_name", output_dir=os.path.join(results_dir,task))
    print(df_scores.head())

    # TODO: update
    # plot score disctribution
    plot_score_distr(df_scores=df_scores, test=task, output_dir=os.path.join(results_dir,task))

    # calculate descriptives for the scores
    calculate_score_desc(df_scores, output_dir=os.path.join(results_dir,task))

    # calculate item statistics
    calculate_item_stats(df=df_agg, test=task, output_dir=os.path.join(results_dir,task))

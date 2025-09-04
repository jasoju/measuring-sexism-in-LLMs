import os

from utils.analyses.output_data_preprocess import *
from utils.analyses.psychometrics import *

# set all needed directories
current_dir = os.getcwd()
output_data_dir = os.path.join(current_dir,"src", "output_data")
results_dir = os.path.join(current_dir,"results")



############## convergent validity ##############

file_path = os.path.join(results_dir, "ASI", "scores_per_model.json")
with open(file_path, 'r') as f:
    df_ASI = json.load(f)

file_path = os.path.join(results_dir, "SR2K", "scores_per_model.json")
with open(file_path, 'r') as f:
    df_SR2K = json.load(f)

file_path = os.path.join(results_dir, "MFQ", "scores_per_model.json")
with open(file_path, 'r') as f:
    df_MFQ = json.load(f)


# sexism-racism
plot_two_column_heatmap(df_ASI["total"], df_SR2K["total"], col_labels=("sexism", "racism"))
r,lower,upper = spearman_rank_corr(df_ASI["total"], df_SR2K["total"], col_labels=("sexism", "racism"), alternative="less")
print(f"sexism-racism: {r} [{lower}, {upper}]")

# authority-benevolent sexism
plot_two_column_heatmap(df_MFQ["Authority"], df_ASI["BS"], col_labels=("authority", "BS"))
r,lower,upper = spearman_rank_corr(df_MFQ["Authority"], df_ASI["BS"], col_labels=("authority", "BS"), alternative="greater")
print(f"authority-BS: {r} [{lower}, {upper}]")

# fairness-hostile sexism
plot_two_column_heatmap(df_MFQ["Fairness"], df_ASI["HS"], col_labels=("fairness", "HS"))
r,lower,upper = spearman_rank_corr(df_MFQ["Fairness"], df_ASI["HS"], col_labels=("fairness", "HS"), alternative="less")
print(f"fairness-HS: {r} [{lower}, {upper}]")






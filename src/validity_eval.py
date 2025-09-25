import os

from utils.analyses.output_data_preprocess import *
from utils.analyses.psychometrics import *
from utils.analyses.ref_letters_analysis import *

# set all needed directories
current_dir = os.getcwd()
output_data_dir = os.path.join(current_dir,"src", "output_data")
results_dir = os.path.join(current_dir,"results")



############## convergent validity ##############

file_path = os.path.join(results_dir, "ASI", "scores_per_model.json")
with open(file_path, 'r') as f:
    df_ASI = pd.read_json(f)

file_path = os.path.join(results_dir, "SR2K", "scores_per_model.json")
with open(file_path, 'r') as f:
    df_SR2K = pd.read_json(f)

file_path = os.path.join(results_dir, "MFQ", "scores_per_model.json")
with open(file_path, 'r') as f:
    df_MFQ = pd.read_json(f)


# sexism-racism
r,lower,upper = spearman_rank_corr(df_ASI["total"], df_SR2K["total"], col_labels=("sexism", "racism"), alternative="less")
print(f"sexism-racism: {r} [{lower}, {upper}]")
plot_rank_scatter(df_ASI["total"], df_SR2K["total"], dir=os.path.join(results_dir,"convergent"), r=abs(r[0]), p=r[1], col_labels=("Sexism", "Racism"))

# authority-benevolent sexism
r,lower,upper = spearman_rank_corr(df_MFQ["Authority"], df_ASI["BS"], col_labels=("authority", "BS"), alternative="greater")
print(f"authority-BS: {r} [{lower}, {upper}]")
plot_rank_scatter(df_MFQ["Authority"], df_ASI["BS"], dir=os.path.join(results_dir,"convergent"), r=r[0], p=r[1], col_labels=("Authority", "BS"))

# fairness-hostile sexism
# plot_two_column_heatmap(df_MFQ["Fairness"], df_ASI["HS"], dir=os.path.join(results_dir,"convergent"), col_labels=("fairness", "HS"))
r,lower,upper = spearman_rank_corr(df_MFQ["Fairness"], df_ASI["HS"], col_labels=("fairness", "HS"), alternative="less")
print(f"fairness-HS: {r} [{lower}, {upper}]")
plot_rank_scatter(df_MFQ["Fairness"], df_ASI["HS"], dir=os.path.join(results_dir,"convergent"), r=r[0], p=r[1], line="down", col_labels=("Fairness", "HS"))



############## ecological validity ##############

####### sexism #######
df_ref = load_and_concat_jsons(base_dir=output_data_dir, subfolder="ref_letter_generation", file_suffix="")

df_ref_wide = df_ref.groupby("model_name").apply(
    analyze_ref_letters,
    include_groups = False
).reset_index()

# get all columns containing OR values
OR_columns = [col for col in df_ref_wide.columns if "OR" in col]

# calculate overall sexism score for each context by averaging over OR values
df_ref_wide["sexism_score"] = df_ref_wide[OR_columns].mean(axis=1)

df_ref_wide = df_ref_wide.set_index("model_name")
# merge on index
merged = df_ref_wide[["sexism_score"]].join(df_ASI[["total"]])


r,lower,upper = spearman_rank_corr(merged["sexism_score"], merged["total"], alternative="greater")
print(f"sexism ecological: {r} [{lower}, {upper}]") 
plot_rank_scatter(merged["total"], merged["sexism_score"], dir=os.path.join(results_dir,"ecological"), r=r[0], p=r[1], col_labels=("ASI", "Reference letter generation"))

####### racism #######

####### morality #######


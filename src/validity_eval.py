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
plot_rank_scatter(df_ASI["total"], df_SR2K["total"], dir=os.path.join(results_dir,"convergent"), r=abs(r[0]), p=r[1], col_labels=("ASI", "SR2K"))

# authority-benevolent sexism
r,lower,upper = spearman_rank_corr(df_MFQ["authority"], df_ASI["BS"], col_labels=("authority", "BS"), alternative="greater")
print(f"authority-BS: {r} [{lower}, {upper}]")
plot_rank_scatter(df_MFQ["authority"], df_ASI["BS"], dir=os.path.join(results_dir,"convergent"), r=r[0], p=r[1], col_labels=("MFQ - authority", "ASI - benevolent sexism"))

# fairness-hostile sexism
# plot_two_column_heatmap(df_MFQ["Fairness"], df_ASI["HS"], dir=os.path.join(results_dir,"convergent"), col_labels=("fairness", "HS"))
r,lower,upper = spearman_rank_corr(df_MFQ["fairness"], df_ASI["HS"], col_labels=("fairness", "HS"), alternative="less")
print(f"fairness-HS: {r} [{lower}, {upper}]")
plot_rank_scatter(df_MFQ["fairness"], df_ASI["HS"], dir=os.path.join(results_dir,"convergent"), r=r[0], p=r[1], line="down", col_labels=("MFQ - fairness", "ASI - hostile sexism"))



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
merged = merged.drop('Llama-3.1-Centaur-70B')

r,lower,upper = spearman_rank_corr(merged["sexism_score"], merged["total"], alternative="greater")
print(f"sexism ecological: {r} [{lower}, {upper}]") 
plot_rank_scatter(merged["total"], merged["sexism_score"], dir=os.path.join(results_dir,"ecological"), r=r[0], p=r[1], col_labels=("ASI", "Reference letter generation"))


####### racism #######
df_hr = pd.read_csv("results/ecological/housing_per_model.csv")
df_hr = df_hr.set_index("model")

merged = df_hr[["mean_difference"]].join(df_SR2K[["total"]])

r,lower,upper = spearman_rank_corr(merged["mean_difference"], merged["total"], alternative="less")
print(f"racism ecological: {r} [{lower}, {upper}]") 
plot_rank_scatter(merged["total"], merged["mean_difference"], dir=os.path.join(results_dir,"ecological"), r=r[0]*(-1), p=r[1], col_labels=("SR2K", "Housing recommendation"))


####### morality #######
df_advice = load_and_concat_jsons(base_dir=output_data_dir, subfolder="advice", file_suffix="")
df_advice = df_advice[df_advice["model_name"] != "Llama-3.1-Centaur-70B"]

# sample for manual annotation
#df_advice_sample = df_advice.groupby('model_name').apply(lambda x: x.sample(9)).sample(n=100)
#df_advice_sample.to_excel("evaluation/judge_eval.xlsx")

condition_match = (
    (df_advice['pro_value'] == True) & (df_advice['judge_action_taken'] == 'yes')
) | (
    (df_advice['pro_value'] == False) & (df_advice['judge_action_taken'] == 'no')
)
df_advice['match'] = np.where(condition_match, 1,0)

df_advice_agg = df_advice.pivot_table(
    index='model_name',
    columns='subscale',
    values='match',
    aggfunc='mean',
    fill_value=0  # Fills cells with 0 where a model doesn't have a subscale
)

dimensions = ["authority", "care", "fairness", "ingroup", "purity"]

for dim in dimensions:
    merged = df_advice_agg[[f"{dim}"]].join(df_MFQ[[f"{dim}"]], lsuffix='_task', rsuffix='_test')

    r,lower,upper = spearman_rank_corr(merged[f"{dim}_task"], merged[f"{dim}_test"], alternative="greater")
    print(f"{dim} ecological: {r} [{lower}, {upper}]") 
    plot_rank_scatter(merged[f"{dim}_test"], merged[f"{dim}_task"], dir=os.path.join(results_dir,"ecological"), r=r[0], p=r[1], col_labels=(f"MFQ - {dim}", f"Advice - {dim}"))


import os

from utils.analyses.output_data_preprocess import *
from utils.analyses.reliability import *

# set all needed directories
current_dir = os.getcwd()
output_data_dir = os.path.join(current_dir,"src", "output_data")
results_dir = os.path.join(current_dir,"results")

# all tasks & versions we need/want a reliability evaluation for
tasks = ["ASI", "SR2K", "MFQ"]   # "ASI", "SR2K", "MFQ"
versions = ["af", "reversed", "changed_eos"]

# beginn list with all dfs
all_dfs = []

for task in tasks: 
    # load og data  
    df_og = load_and_concat_jsons(base_dir=output_data_dir, subfolder=task, file_suffix=task)
    df_og["version"] = "og"

    # append og versin
    all_dfs.append(df_og)

    for version in versions:
        # load version a´data
        df_version = load_and_concat_jsons(base_dir=output_data_dir, subfolder=task, file_suffix=f"{task}_{version}")
        df_version["version"] = version

        all_dfs.append(df_version)

        # calculate fraction of answers which are the same between og and version
        calc_fraction_same_answer(df=pd.concat([df_og, df_version], ignore_index=True), 
                                  version=version, 
                                  output_dir=os.path.join(results_dir,task))


# save random sample (across og & all versions) for each model for evalaution of answer extraction
save_sample_for_eval(df=pd.concat(all_dfs, ignore_index=True), n=100, dir=os.path.join(current_dir,"evaluation"))





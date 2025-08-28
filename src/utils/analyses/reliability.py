import pandas as pd
from scipy import stats

from utils.analyses.output_data_preprocess import *

def calc_fraction_same_answer(df:pd.DataFrame, version:str, output_dir:str):
    # filter
    df_sub = df[df['version'].isin(["og", version])]

    # pivot so that we have answers of versions side by side
    pivot_df = df_sub.pivot_table(
        index=['item_id', 'model_name'],
        columns='version',
        values='answer_reversed'
    ).reset_index()

    # get match
    def check_match(row):
        og_val, af_val = row["og"], row[version]
        if pd.isna(og_val) and pd.isna(af_val):
            return True           # both missing
        else:
            return og_val == af_val 

    pivot_df['match'] = pivot_df.apply(check_match, axis=1)

    # Calculate fraction of matches per model
    result = pivot_df.groupby('model_name')['match'].mean().reset_index()
    result.rename(columns={'match': 'fraction_match'}, inplace=True)

    result.to_json(os.path.join(output_dir, f"rel_eval_{version}.json"), orient="index", indent=2)


    
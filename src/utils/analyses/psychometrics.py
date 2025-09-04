import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
import matplotlib.patches as mpatches

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



def plot_two_column_heatmap(col1: pd.Series, col2: pd.Series, col_labels=("col1", "col2")):
    """
    Plot a heatmap for two pandas Series with opposite color scaling:
    - col1: high values = more saturated
    - col2: low values = more saturated

    Parameters
    ----------
    col1 : pd.Series
        First column (used for sorting rows in ascending order).
    col2 : pd.Series
        Second column (must have same index as col1).
    col_labels : tuple, optional
        Labels for the two columns in the heatmap (default=("col1", "col2")).
    """
    
    # Combine into a DataFrame
    df = pd.DataFrame({col_labels[0]: col1, col_labels[1]: col2})
    
    # Sort by the first column
    df_sorted = df.sort_values(by=col_labels[0], ascending=False)
    
     # Rank transformation
    col1_rank = df_sorted[col_labels[0]].rank(method="min") / len(df_sorted)  # high = dark
    col2_rank = (len(df_sorted) - df_sorted[col_labels[1]].rank(method="min") + 1) / len(df_sorted)  # low = dark
    
    # Construct normalized rank DataFrame
    df_norm = pd.DataFrame(
        {col_labels[0]: col1_rank, col_labels[1]: col2_rank},
        index=df_sorted.index
    )
    
    # Plot heatmap
    plt.figure(figsize=(6, len(df_sorted) * 0.3))
    sns.heatmap(df_norm, annot=df_sorted, fmt=".2f", cmap="Blues", cbar=False)

    plt.title("Rank-based Heatmap")
    plt.xlabel("Columns")
    plt.ylabel(f"Models (sorted by {col_labels[0]} score)")
    plt.tight_layout()
    plt.show()


def spearman_rank_corr(col1: pd.Series, col2: pd.Series, col_labels=("col1", "col2"), alternative="two-sided"):
    # Combine into a DataFrame
    df = pd.DataFrame({col_labels[0]: col1, col_labels[1]: col2})

    r = stats.spearmanr(df)

    # calculate CI
    count = len(df)
    stderr = 1.0 / math.sqrt(count - 3)
    delta = 1.96 * stderr
    lower = math.tanh(math.atanh(r.correlation) - delta)
    upper = math.tanh(math.atanh(r.correlation) + delta)
    
    return r, lower, upper

    
    
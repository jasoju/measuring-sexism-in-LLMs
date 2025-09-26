import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from scipy.stats import spearmanr

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
    result.rename(columns={'match': version}, inplace=True)

    result.to_json(os.path.join(output_dir, f"rel_eval_{version}.json"), orient="index", indent=2)



def plot_rank_scatter(col1: pd.Series, col2: pd.Series, dir:str, r:float, p:float, line="up", col_labels=("col1", "col2"), ):
    """
    Parameters
    ----------
    col1 : pd.Series
        First column (used for sorting rows in ascending order).
    col2 : pd.Series
        Second column (must have same index as col1).
    col_labels : tuple, optional
        Labels for the two columns in the heatmap (default=("col1", "col2")).
    """
    
    # combine into a df
    df = pd.DataFrame({col_labels[0]: col1, col_labels[1]: col2})
    
    df['rank_col1'] = df[col_labels[0]].rank(ascending=(col_labels[0]!="Racism"))
    df['rank_col2'] = df[col_labels[1]].rank(ascending=(col_labels[1]!="Racism"))


    # plot
    #sns.set_theme(font_scale=2)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x='rank_col1', y='rank_col2', data=df, s=45)

    lims = [
        min(df['rank_col1'].min(), df['rank_col2'].min()),
        max(df['rank_col1'].max(), df['rank_col2'].max())
    ]
    if line=="down":
        plt.plot(lims, [lims[1] - (x - lims[0]) for x in lims], color="grey", linestyle="--")
    else:
        plt.plot(lims, lims, color="grey", linestyle="--")

    plt.text(1, 12.8, "$r_s = {:.2f}$".format(r), fontsize=15)
    plt.text(1, 12.1, "$p = {:.2f}$".format(p), fontsize=15)

    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))


    plt.xlabel(f"rank({col_labels[0]})", fontsize=15)
    plt.ylabel(f"rank({col_labels[1]})", fontsize=15)
    # plt.title("Scatter plot of ranks")

    # Save plot
    filename = f"scatter_{col_labels[0]}_{col_labels[1]}.png"
    plt.savefig(os.path.join(dir, filename), bbox_inches="tight")
    plt.close()


def spearman_rank_corr(col1: pd.Series, col2: pd.Series, col_labels=("col1", "col2"), alternative="two-sided"):
    # Combine into a DataFrame
    df = pd.DataFrame({col_labels[0]: col1, col_labels[1]: col2})

    r = stats.spearmanr(df, alternative=alternative)

    # calculate CI
    count = len(df)
    stderr = 1.0 / math.sqrt(count - 3)
    delta = 1.96 * stderr
    lower = math.tanh(math.atanh(r.correlation) - delta)
    upper = math.tanh(math.atanh(r.correlation) + delta)
    
    return r, lower, upper

    
    
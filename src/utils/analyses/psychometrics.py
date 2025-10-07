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

    pivot_df = df_sub.pivot_table(
        index=['item_id', 'model_name', 'seed'],
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
    result = result = pivot_df.groupby(['model_name', 'seed'])['match'].mean().reset_index()
    result.rename(columns={'match': version}, inplace=True)

    result.to_json(os.path.join(output_dir, f"rel_eval_{version}.json"), orient="index", indent=2)



def plot_rank_scatter(col1: pd.Series, col2: pd.Series, dir:str, r:float, p:float, hue:pd.Series, construct="none", line="up", col_labels=("col1", "col2"), ):
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
    
    # --- 1. Define Model-Specific Maps ---
    # The keys must match the model names exactly as they appear in your DataFrame's index.
    model_colors = {
        'Llama-3.1-Centaur-70B': "#EC29F3",  # Purple
        'gemma-3-1b-it': "#00800061",       # Green 1
        'gemma-3-4b-it': "#0080008D",       # Green 2
        'gemma-3-12b-it': "#008000C1",      # Green 3
        'gemma-3-27b-it': '#008000',        # Green 4
        'Llama-3.1-8B-Instruct': "#4C008257",  # Medium Purple 1
        'Llama-3.1-70B-Instruct': "#4C0082A2", # Indigo
        'Llama-3.3-70B-Instruct': '#4B0082', # Orange Red
        'Mistral-7B-Instruct-v0.3': '#1E90FF', # Dodger Blue
        'Qwen2.5-7B-Instruct': "#DFA11C76",    # DarkGoldenrod 1
        'Qwen2.5-14B-Instruct': "#DFA11CAE",   # Goldenrod 2
        'Qwen2.5-32B-Instruct': '#DFA11C',   # Gold 3
        'Qwen3-4B-Instruct-2507': "#DFA11C4C"  # Hot Pink
    }

    model_markers = {
        'Llama-3.1-Centaur-70B': 'P',  # Diamond
        'gemma-3-1b-it': 's',          # Triangle Up
        'gemma-3-4b-it': 's',          # Triangle Up
        'gemma-3-12b-it': 's',         # Triangle Up
        'gemma-3-27b-it': 's',         # Triangle Up
        'Llama-3.1-8B-Instruct': 'o',  # Circle
        'Llama-3.1-70B-Instruct': 'o', # Circle
        'Llama-3.3-70B-Instruct': 'o', # Square
        'Mistral-7B-Instruct-v0.3': '^', # Cross (filled)
        'Qwen2.5-7B-Instruct': 'D',    # Plus (filled)
        'Qwen2.5-14B-Instruct': 'D',   # Plus (filled)
        'Qwen2.5-32B-Instruct': 'D',   # Plus (filled)
        'Qwen3-4B-Instruct-2507': 'D',  # Star
    }

    model_display_names = {
        'Llama-3.1-Centaur-70B': 'Centaur',
        'gemma-3-1b-it': 'Gemma 3 1B',
        'gemma-3-4b-it': 'Gemma 3 4B',
        'gemma-3-12b-it': 'Gemma 3 12B',
        'gemma-3-27b-it': 'Gemma 3 27B',
        'Llama-3.1-8B-Instruct': 'Llama 3.1 8B',
        'Llama-3.1-70B-Instruct': 'Llama 3.1 70B',
        'Llama-3.3-70B-Instruct': 'Llama 3.3 70B',
        'Mistral-7B-Instruct-v0.3': 'Mistral 7B v0.3',
        'Qwen2.5-7B-Instruct': 'Qwen 2.5 7B',
        'Qwen2.5-14B-Instruct': 'Qwen 2.5 14B',
        'Qwen2.5-32B-Instruct': 'Qwen 2.5 32B',
        'Qwen3-4B-Instruct-2507': 'Qwen 3 4B',
    }

    custom_legend_order = [
        'Llama-3.1-Centaur-70B',

        'Llama-3.1-8B-Instruct',
        'Llama-3.1-70B-Instruct', 
        'Llama-3.3-70B-Instruct',
        
        'gemma-3-1b-it',
        'gemma-3-4b-it',
        'gemma-3-12b-it',
        'gemma-3-27b-it',
        
        'Mistral-7B-Instruct-v0.3',
        
        'Qwen3-4B-Instruct-2507',
        'Qwen2.5-7B-Instruct',
        'Qwen2.5-14B-Instruct',
        'Qwen2.5-32B-Instruct',
    ]


    df = pd.DataFrame({col_labels[0]: col1, col_labels[1]: col2})
    df['Model_Name'] = df.index
    
    # Calculate ranks
    # The logic for ascending/descending rank calculation is kept as is:
    # rank_col1 is ascending if col_labels[0] is "SR2K" OR construct is "racism"
    # rank_col2 is always descending
    df['rank_col1'] = df[col_labels[0]].rank(ascending=((col_labels[0] =="SR2K") or (construct=="racism")))
    df['rank_col2'] = df[col_labels[1]].rank(ascending=False)


    # plot
    #sns.set_theme(font_scale=2)
    # Plot Setup
    plt.figure(figsize=(5,4.5))
    ax = plt.gca()
    
    # Remove borders (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sns.scatterplot(
        x='rank_col1', 
        y='rank_col2', 
        data=df, 
        s=150, 
        hue='Model_Name',            # Use model name for color
        style='Model_Name',          # Use model name for marker
        palette=model_colors,        # Pass the color map dictionary
        markers=model_markers,       # Pass the marker map dictionary
        legend="full",               # Display the legend
        ax=ax
    )

    lims = [
        min(df['rank_col1'].min(), df['rank_col2'].min()),
        max(df['rank_col1'].max(), df['rank_col2'].max())
    ]
    
    # --- Add Reference Lines ---
    if line=="down":
        # Grey dashed line: y = constant - x (negative correlation)
        grey_line_handle, = plt.plot(lims, [lims[1] - (x - lims[0]) for x in lims], color="grey", linestyle="--")
        grey_line_label = "Perfect Negative Rank Correlation"
    else:
        # Grey dashed line: y = x (positive correlation)
        grey_line_handle, = plt.plot(lims, lims, color="grey", linestyle="--")
        grey_line_label = "expected $r_s$"

    plt.text(lims[1]+0.5, -0.2, "$r_s = {:.2f}$".format(r), fontsize=15)
    #plt.text(1, lims[1]-0.9, "$p = {:.2f}$".format(p), fontsize=15)

    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))

    # Get handles and labels from the scatter plot
    handles, labels = ax.get_legend_handles_labels()
    
    # Logic to filter out the 'Model_Name' title
    if labels[0] == 'Model_Name':
        original_labels = labels[1:]
        legend_handles = handles[1:]
    else:
        original_labels = labels
        legend_handles = handles

    # --- Reordering Step (Models) ---
    # 1. Map original labels to the desired display names
    label_map = {
        orig_label: (handle, model_display_names.get(orig_label, orig_label))
        for orig_label, handle in zip(original_labels, legend_handles)
    }
    
    # 2. Iterate through the custom order list to build the final, ordered lists for models
    ordered_handles = []
    ordered_labels = []
    
    for key in custom_legend_order:
        if key in label_map:
            handle, display_label = label_map[key]
            ordered_handles.append(handle)
            ordered_labels.append(display_label)

    # --- Add Regression Line ---
    m, b = np.polyfit(df['rank_col1'], df['rank_col2'], 1)
    X_plot = np.linspace(lims[0],lims[1],100)
    red_line_handle, = plt.plot(X_plot, m*X_plot + b, '-', color="red")
    red_line_label = "observed $r_s$"

    # --- Combine and Finalize Legend ---
    # Append the reference lines to the ordered lists
    final_handles = ordered_handles + [grey_line_handle, red_line_handle]
    final_labels = ordered_labels + [grey_line_label, red_line_label]

    # Place the legend outside the plot area
    plt.legend(
        handles=final_handles,    # Use the combined handles
        labels=final_labels,      # Use the combined labels
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        borderaxespad=0.,   
        labelspacing=0.6,
        prop={'size': 11}
    )
    
    # Get the current limits which include padding (e.g., 0 and 13)
    ax = plt.gca()
    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()

    # Set the limits to the reverse of the current limits: (max, min)
    plt.xlim(current_xlim[1], current_xlim[0])
    plt.ylim(current_ylim[1], current_ylim[0])

    # Axis Labels
    if construct!="none":
        plt.xlabel(f"{col_labels[0]}", fontsize=18)
        plt.ylabel(f"{col_labels[1]}", fontsize=18)
    else:
        plt.xlabel(f"rank({col_labels[0]})", fontsize=18)
        plt.ylabel(f"rank({col_labels[1]})", fontsize=18)

    plt.tick_params(axis='both', which='major', labelsize=15)

    # Save plot
    if construct!="none":
        filename = f"scatter_{construct}.png"
    else:
        filename = f"scatter_{col_labels[0]}_{col_labels[1]}.png"
    
    # Ensure directory exists before saving
    os.makedirs(dir, exist_ok=True)
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

    
    
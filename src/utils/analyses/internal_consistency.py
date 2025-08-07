import pandas as pd
import pingouin as pg

def eval_internal_consistency(df:pd.DataFrame):
    # compute stratified alpha if we have subscales
    if "subscale" in df.columns:
        strata = []
        total_score_df = pd.DataFrame()
        
        for subscale, group in df.groupby('subscale'):
            pivot = group.pivot(index='context_id', columns='item_id', values='answer_reversed')
            pivot.columns = [f"{subscale}_{col}" for col in pivot.columns]  # unique column names
            strata.append(pivot)
            total_score_df = pd.concat([total_score_df, pivot], axis=1)

        total_var = total_score_df.sum(axis=1).var(ddof=1)

        stratified_terms = []
        for stratum_df in strata:
            var_h = stratum_df.sum(axis=1).var(ddof=1)
            alpha_h = pg.cronbach_alpha(stratum_df)
            term = (var_h / total_var) * (1 - alpha_h)
            stratified_terms.append(term)

        alpha = 1 - sum(stratified_terms)

    # if we don't have subscales just use "normal" alpha
    else:
        alpha = pg.cronbach_alpha(data=df)[0]

    return {"alpha": alpha}
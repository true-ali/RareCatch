import numpy as np

RULE_FEATURES = ("V14", "V17", "V12", "V10")
LEGIT_QUANTILE = 0.001


def fit_rules(train_df):
    legit = train_df[train_df["Class"] == 0]
    return {f: float(legit[f].quantile(LEGIT_QUANTILE)) for f in RULE_FEATURES}


def rule_hits(df, rules):
    return {f: (df[f].to_numpy() <= cutoff) for f, cutoff in rules.items()}


def any_rule_hit(df, rules):
    return np.logical_or.reduce(list(rule_hits(df, rules).values()))

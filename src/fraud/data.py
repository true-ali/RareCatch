from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/processed/creditcard.parquet")
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2


def load(path=DATA_PATH):
    return pd.read_parquet(path).sort_values("Time", kind="stable").reset_index(drop=True)


def time_split(df):
    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

import numpy as np
import pandas as pd

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURES = PCA_COLUMNS + ["log_amount", "hour"]
RAW_COLUMNS = ["Time"] + PCA_COLUMNS + ["Amount"]


def make_features(df):
    out = df[PCA_COLUMNS].copy()
    out["log_amount"] = np.log1p(df["Amount"])
    out["hour"] = (df["Time"] % 86400) / 3600
    return out[FEATURES].astype("float64")


def one_row_frame(record):
    return pd.DataFrame([record], columns=RAW_COLUMNS)

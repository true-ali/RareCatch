import io
import urllib.request
from pathlib import Path

import pandas as pd

ARFF_URL = "https://openml.org/data/v1/download/1673544/creditcard.arff"
RAW = Path("data/raw/creditcard.arff")
OUT = Path("data/processed/creditcard.parquet")
COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def main():
    RAW.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        urllib.request.urlretrieve(ARFF_URL, RAW)
    text = RAW.read_text()
    body = text.split("@data", 1)[1].strip()
    df = pd.read_csv(io.StringIO(body), header=None, names=COLUMNS, quotechar="'")
    df["Class"] = df["Class"].astype(int)
    df.to_parquet(OUT)
    print(f"{len(df)} rows, {df['Class'].sum()} frauds -> {OUT}")


if __name__ == "__main__":
    main()

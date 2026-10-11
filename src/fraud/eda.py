import json

from sklearn.metrics import roc_auc_score

from fraud.data import load, time_split
from fraud.features import PCA_COLUMNS


def main():
    df = load()
    train, val, test = time_split(df)
    legit, fraud = df[df.Class == 0], df[df.Class == 1]
    separation = {c: abs(roc_auc_score(train.Class, train[c]) - 0.5) * 2 for c in PCA_COLUMNS + ["Amount"]}
    top = sorted(separation, key=separation.get, reverse=True)[:8]
    night = ((df.Time % 86400) / 3600).between(0, 6)
    summary = {
        "rows": len(df), "frauds": int(df.Class.sum()), "fraud_rate": float(df.Class.mean()),
        "span_hours": float(df.Time.max() / 3600),
        "amount_median": {"legit": float(legit.Amount.median()), "fraud": float(fraud.Amount.median())},
        "amount_mean": {"legit": float(legit.Amount.mean()), "fraud": float(fraud.Amount.mean())},
        "fraud_rate_00_06h": float(df[night].Class.mean()), "fraud_rate_other_hours": float(df[~night].Class.mean()),
        "split_frauds": {"train": int(train.Class.sum()), "val": int(val.Class.sum()), "test": int(test.Class.sum())},
        "top_separating_features_train": top,
    }
    print(json.dumps(summary, indent=2))
    with open("reports/eda.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

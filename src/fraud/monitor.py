import pandas as pd

from fraud.data import load, time_split
from fraud.evaluate import psi
from fraud.features import FEATURES, make_features
from fraud.scoring import score
from fraud.serve import load_bundle

PSI_ALERT = 0.25


def main():
    train, _, test = time_split(load())
    bundle = load_bundle()
    Xtr, Xte = make_features(train), make_features(test)
    rows = [{"feature": f, "psi": psi(Xtr[f].to_numpy(), Xte[f].to_numpy())} for f in FEATURES]
    rows.append({"feature": "model_score", "psi": psi(score(bundle["model"], Xtr.to_numpy()),
                                                       score(bundle["model"], Xte.to_numpy()))})
    table = pd.DataFrame(rows).sort_values("psi", ascending=False)
    table["drifted"] = table["psi"] > PSI_ALERT
    table.to_csv("reports/drift_psi.csv", index=False)
    print(table.head(8).to_string(index=False))


if __name__ == "__main__":
    main()

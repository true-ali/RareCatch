import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud.data import load, time_split
from fraud.features import make_features
from fraud.scoring import score

LEGIT_SAMPLE = 300
THRESHOLD_GRID = [round(-6 + 0.1 * k, 1) for k in range(96)]


def main():
    bundle = joblib.load("models/model.joblib")
    _, _, test = time_split(load())
    test = test.reset_index(drop=True)
    test["s"] = score(bundle["model"], make_features(test).to_numpy())
    rule_names = list(bundle["rules"])
    rule_hit = np.zeros(len(test), bool)
    for name, cutoff in bundle["rules"].items():
        rule_hit |= (test[name] <= cutoff).to_numpy()
    legit = (test.Class == 0).to_numpy()
    scores = test["s"].to_numpy()
    false_alerts = [
        [int((legit & (scores >= t)).sum()) for t in THRESHOLD_GRID],
        [int((legit & ((scores >= t) | rule_hit)).sum()) for t in THRESHOLD_GRID],
    ]
    rng = np.random.default_rng(1)
    frauds = test[test.Class == 1]
    legit_rows = test[test.Class == 0]
    sampled = legit_rows.loc[rng.choice(legit_rows.index, LEGIT_SAMPLE, replace=False)]
    replay = test.loc[sorted(set(frauds.index) | set(sampled.index))]
    rows = [
        [round(r.Time / 3600, 2), round(r.Amount, 2), round(r.s, 1), int(r.Class),
         sum(1 << k for k, f in enumerate(rule_names) if r[f] <= bundle["rules"][f]),
         round(r.V14, 1), round(r.V17, 1)]
        for _, r in replay.iterrows()
    ]
    out = {
        "names": rule_names, "cut": [round(bundle["rules"][n], 3) for n in rule_names],
        "thr": round(bundle["ml_threshold"], 2), "legitTotal": int(legit.sum()),
        "legitSample": LEGIT_SAMPLE, "rows": rows, "falseAlerts": false_alerts,
    }
    Path("reports/dashboard_data.json").write_text(json.dumps(out, separators=(",", ":")))


main()

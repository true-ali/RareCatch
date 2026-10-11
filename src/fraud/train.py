import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraud import evaluate as ev
from fraud.data import load, time_split
from fraud.features import FEATURES, make_features
from fraud.scoring import score
from fraud.rules import any_rule_hit, fit_rules

SEED = 42
UNDERSAMPLE_NEG_PER_POS = 20
REPORTS = Path("reports")
MODEL_PATH = Path("models/model.joblib")
GBM_NAMES = ("xgb_weighted", "xgb_undersampled", "lgbm_weighted", "lgbm_undersampled")
HYBRID_RATES = (0.005, 0.01)


def undersample(X, y, rng):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    neg = rng.choice(neg, size=min(len(neg), UNDERSAMPLE_NEG_PER_POS * len(pos)), replace=False)
    idx = np.sort(np.concatenate([pos, neg]))
    return X[idx], y[idx]


def xgb(scale_pos_weight):
    return XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr", early_stopping_rounds=50,
        tree_method="hist", random_state=SEED, n_jobs=4,
    )


def lgbm(scale_pos_weight):
    return lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, scale_pos_weight=scale_pos_weight, random_state=SEED, n_jobs=4, verbose=-1,
    )


def fit_all(Xtr, ytr, Xva, yva):
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    Xu, yu = undersample(Xtr, ytr, np.random.default_rng(SEED))
    models = {}
    models["logreg"] = make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
    ).fit(Xtr, ytr)
    models["random_forest"] = RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1, random_state=SEED
    ).fit(Xtr, ytr)
    models["xgb_weighted"] = xgb(spw).fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    models["xgb_undersampled"] = xgb(1.0).fit(Xu, yu, eval_set=[(Xva, yva)], verbose=False)
    stop = [lgb.early_stopping(50, first_metric_only=True, verbose=False)]
    models["lgbm_weighted"] = lgbm(spw).fit(
        Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision", callbacks=stop)
    models["lgbm_undersampled"] = lgbm(1.0).fit(
        Xu, yu, eval_set=[(Xva, yva)], eval_metric="average_precision", callbacks=stop)
    return models


def evaluate_model(s_val, s_test, yva, yte, amt_va, amt_te):
    row = {"pr_auc_val": ev.pr_auc(yva, s_val), "pr_auc_test": ev.pr_auc(yte, s_test),
           "roc_auc_test": ev.roc_auc(yte, s_test), "at_alert_rate": {}}
    for rate in ev.ALERT_RATES:
        thr = ev.threshold_for_alert_rate(s_val, rate)
        out = ev.alert_outcomes(yte, amt_te, s_test >= thr)
        out["threshold"] = thr
        row["at_alert_rate"][str(rate)] = out
    thr = ev.cost_optimal_threshold(yva, amt_va, s_val)
    out = ev.alert_outcomes(yte, amt_te, s_test >= thr)
    out["threshold"] = thr
    out["savings_vs_no_model"] = 1 - out["cost"] / ev.no_model_cost(yte, amt_te)
    row["cost_optimal"] = out
    return row


def evaluate_hybrid(s_val, s_test, rule_va, rule_te, yva, yte, amt_va, amt_te):
    rows = {"rules_only": ev.alert_outcomes(yte, amt_te, rule_te)}
    thr = ev.cost_optimal_threshold(yva, amt_va, s_val)
    rows["ml_cost_optimal"] = ev.alert_outcomes(yte, amt_te, s_test >= thr)
    h_thr = ev.hybrid_cost_optimal_threshold(yva, amt_va, s_val, rule_va)
    rows["hybrid_cost_optimal"] = ev.alert_outcomes(yte, amt_te, ev.hybrid_alert(rule_te, s_test, h_thr))
    rows["hybrid_cost_optimal"]["threshold"] = h_thr
    rows["rules_or_ml_cost_optimal"] = ev.alert_outcomes(yte, amt_te, ev.hybrid_alert(rule_te, s_test, thr))
    for rate in HYBRID_RATES:
        ml_thr = ev.threshold_for_alert_rate(s_val, rate)
        rows[f"ml_at_{rate}"] = ev.alert_outcomes(yte, amt_te, s_test >= ml_thr)
        h = ev.hybrid_threshold_for_alert_rate(rule_va, s_val, rate)
        rows[f"hybrid_at_{rate}"] = ev.alert_outcomes(yte, amt_te, ev.hybrid_alert(rule_te, s_test, h))
        rows[f"hybrid_at_{rate}"]["threshold"] = h
    return rows, h_thr


def markdown_tables(results, hybrid, best, counts):
    lines = ["## Split", "", "| split | rows | frauds |", "|---|---|---|"]
    lines += [f"| {k} | {v[0]} | {v[1]} |" for k, v in counts.items()]
    lines += ["", "## Models (test set; thresholds chosen on validation)", "",
              "| model | PR-AUC val | PR-AUC test | ROC-AUC test | P@0.1% | R@0.1% | P@0.5% | R@0.5% | P@1% | R@1% | cost-opt alerts | cost-opt P | cost-opt R | cost saved |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        a = r["at_alert_rate"]
        c = r["cost_optimal"]
        lines.append(
            f"| {name} | {r['pr_auc_val']:.3f} | {r['pr_auc_test']:.3f} | {r['roc_auc_test']:.3f} "
            f"| {a['0.001']['precision']:.2f} | {a['0.001']['recall']:.2f} "
            f"| {a['0.005']['precision']:.2f} | {a['0.005']['recall']:.2f} "
            f"| {a['0.01']['precision']:.2f} | {a['0.01']['recall']:.2f} "
            f"| {c['alerts']} | {c['precision']:.2f} | {c['recall']:.2f} | {c['savings_vs_no_model']:.1%} |")
    lines += ["", f"## Rules + ML hybrid (model: {best}; test set)", "",
              "| strategy | alerts | alert rate | TP | FP | FN | precision | recall | cost |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, o in hybrid.items():
        lines.append(f"| {name} | {o['alerts']} | {o['alert_rate']:.3%} | {o['tp']} | {o['fp']} | {o['fn']} "
                     f"| {o['precision']:.2f} | {o['recall']:.2f} | {o['cost']:,.0f} |")
    return "\n".join(lines) + "\n"


def plot_pr_curves(scores_test, yte):
    plt.figure(figsize=(7, 5))
    for name, s in scores_test.items():
        p, r, _ = precision_recall_curve(yte, s)
        plt.plot(r, p, label=f"{name} (AP={ev.pr_auc(yte, s):.3f})")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("Precision-recall, test set")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(REPORTS / "pr_curves.png", dpi=130)
    plt.close()


def main():
    REPORTS.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    train, val, test = time_split(load())
    counts = {n: (len(d), int(d["Class"].sum())) for n, d in (("train", train), ("val", val), ("test", test))}
    print("split (rows, frauds):", counts)
    Xtr, Xva, Xte = (make_features(d).to_numpy() for d in (train, val, test))
    ytr, yva, yte = (d["Class"].to_numpy() for d in (train, val, test))
    amt_va, amt_te = val["Amount"].to_numpy(), test["Amount"].to_numpy()

    models = fit_all(Xtr, ytr, Xva, yva)
    s_val = {n: score(m, Xva) for n, m in models.items()}
    s_test = {n: score(m, Xte) for n, m in models.items()}
    results = {n: evaluate_model(s_val[n], s_test[n], yva, yte, amt_va, amt_te) for n in models}
    best = max(GBM_NAMES, key=lambda n: results[n]["pr_auc_val"])

    rules = fit_rules(train)
    rule_va, rule_te = any_rule_hit(val, rules), any_rule_hit(test, rules)
    hybrid, hybrid_thr = evaluate_hybrid(s_val[best], s_test[best], rule_va, rule_te, yva, yte, amt_va, amt_te)

    plot_pr_curves(s_test, yte)
    (REPORTS / "results.json").write_text(json.dumps(
        {"split": counts, "cost_per_false_alert": ev.COST_PER_FALSE_ALERT, "best_model": best,
         "rules": rules, "models": results, "hybrid": hybrid}, indent=2))
    (REPORTS / "results.md").write_text(markdown_tables(results, hybrid, best, counts))
    joblib.dump({"model": models[best], "name": best, "features": FEATURES, "rules": rules,
                 "ml_threshold": results[best]["cost_optimal"]["threshold"], "hybrid_threshold": hybrid_thr}, MODEL_PATH)
    print((REPORTS / "results.md").read_text())
    print("best model (by validation PR-AUC):", best)


if __name__ == "__main__":
    main()

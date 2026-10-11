# RareCatch — transaction fraud detection on imbalanced data

Gradient-boosted fraud models on the public ULB/Worldline credit-card dataset (284,807 transactions, 492 frauds = 0.173%), evaluated with PR-AUC, fixed-alert-rate precision/recall and expected-cost thresholds, combined with a small rule layer, and served with FastAPI.
**Measured: 2.5 ms p95 end-to-end HTTP scoring latency (p50 2.2 ms), single process, 2,000 sequential requests, Apple-silicon laptop.**

## Run it

```bash
brew install libomp        # macOS only, needed by LightGBM/XGBoost
make setup data            # venv + pinned deps; downloads and converts the dataset
make eda train drift       # reports/ (results.md, results.json, pr_curves.png, eda.json, drift_psi.csv)
make bench                 # reports/latency.json
make serve                 # POST /score, GET /health on :8000
make test                  # needs `make train` first (the API test loads models/model.joblib)
curl -X POST localhost:8000/score -H 'content-type: application/json' \
  -d "$(.venv/bin/python -c \"import json;print(json.dumps({c:0.0 for c in ['Time']+[f'V{i}' for i in range(1,29)]+['Amount']}))\")"   # or open localhost:8000/docs
```

Python 3.14 was used (pins in `requirements.txt` are for that interpreter).

Data: the ULB dataset is fetched from its OpenML mirror (id 1597, same data as the Kaggle release, no login) by `scripts/download_data.py`. The 2013 data spans 48 h.

## Approach

- **Split:** sorted by `Time`, cut by position 60/20/20, no shuffling. Frauds: train 360, val 57, test 75.
- **Features:** 28 PCA components, `log1p(Amount)`, hour-of-day. Raw `Time` is deliberately excluded (with a time split, every test value is out of the training range).
- **Models:** logistic regression and Random Forest baselines (balanced class weights); XGBoost and LightGBM each with (a) `scale_pos_weight` = neg/pos and (b) random undersampling of train only to 20 negatives per fraud. Same features, splits and hyperparameters; early stopping on PR-AUC on the untouched validation set; no hyperparameter search.
- **Protocol:** every threshold (alert-rate quantile, cost-optimal) is chosen on validation and applied unchanged to test. The best GBM is chosen by validation PR-AUC.
- **Metrics:** PR-AUC = average precision. Precision/recall at fixed alert rates. Expected cost = 10 per false alert (assumed manual-review cost) + full `Amount` of every missed fraud.
- **Rules:** four conditions `V14/V17/V12/V10 <= c`, with `c` the 0.1th percentile of legitimate train transactions (train only). Hybrid = rule hit OR model score >= threshold.

## Results (test set, 56,962 transactions, 75 frauds)

| model | PR-AUC | P / R @0.1% alerts | P / R @0.5% | P / R @1% | cost-opt alerts | cost saved vs no model |
|---|---|---|---|---|---|---|
| logreg | 0.749 | 0.83 / 0.69 | 0.22 / 0.85 | 0.11 / 0.87 | 52 | 60.1% |
| random_forest | 0.818 | 0.82 / 0.79 | 0.26 / 0.87 | 0.11 / 0.87 | 64 | 65.0% |
| xgb_weighted | 0.771 | 0.85 / 0.75 | 0.22 / 0.84 | 0.12 / 0.85 | 62 | 48.9% |
| **xgb_undersampled** (selected on val) | 0.796 | 0.83 / 0.76 | 0.23 / 0.84 | 0.14 / 0.87 | 62 | 48.7% |
| lgbm_weighted | 0.742 | 0.85 / 0.69 | 0.22 / 0.83 | 0.11 / 0.83 | 84 | 45.4% |
| lgbm_undersampled | 0.794 | 0.81 / 0.75 | 0.23 / 0.84 | 0.14 / 0.87 | 69 | 51.2% |

Full table with validation scores and ROC-AUC: `reports/results.md`; curves: `reports/pr_curves.png`.

**Rules, ML and hybrid (selected model, test set; `@x%` rows are matched on alert rate):**

| strategy | alerts | TP | FP | recall | precision | cost |
|---|---|---|---|---|---|---|
| rules only | 134 | 57 | 77 | 0.76 | 0.43 | 3,209 |
| ML, cost-optimal | 62 | 54 | 8 | 0.72 | 0.87 | 3,963 |
| hybrid, cost-optimal (threshold fitted on non-rule rows) | 134 | 57 | 77 | 0.76 | 0.43 | 3,209 |
| rules OR ML cost-optimal (**served**) | 137 | 58 | 79 | 0.77 | 0.42 | 3,190 |
| ML @0.5% | 273 | 63 | 210 | 0.84 | 0.23 | 4,480 |
| hybrid @0.5% | 249 | 61 | 188 | 0.81 | 0.24 | 4,271 |
| ML @1% | 464 | 65 | 399 | 0.87 | 0.14 | 6,340 |
| hybrid @1% | 456 | 65 | 391 | 0.87 | 0.14 | 6,260 |

Findings:
- Differences between the tuned GBMs, Random Forest and logistic regression are within noise at 75 test frauds; the GBMs do not clearly beat the baselines here. Undersampling to 20:1 was equal to or slightly better than class weighting. With a 474x `scale_pos_weight` LightGBM's sigmoid saturates, so `predict_proba` ties at 1.0 for thousands of rows and PR-AUC collapses (val 0.08); ranking by raw margins fixes it (val 0.68, still below the ~0.78 of the other GBMs). This is why `scoring.py` ranks GBMs by margin and the API `score` is a margin.
- At matched alert rates the hybrid is no better than ML alone (0.5%: 61 vs 63 frauds caught, slightly lower cost; 1%: identical recall). The rules are strong on their own: V14/V17/V12/V10 lows catch 57/75 frauds with 134 alerts, and the cost-optimal hybrid collapses to "rules only" because on validation the 13 frauds the rules miss total only 4,474 in amount, which no threshold catches for less than 10 per false alert. The served rule: alert if any rule fires OR the ML score passes its own cost-optimal threshold (62 alerts alone). The lower hybrid cost (3,209 vs 3,963) is largely driven by a few fraud amounts, so treat it as suggestive, not conclusive.

## Serving and latency

`POST /score` takes the 30 raw fields (`Time`, `V1..V28`, `Amount`) and returns `{score, alert, rule_hits}`; `alert` = any rule hit OR score >= the ML cost-optimal threshold. `score` is the XGBoost margin (higher = riskier, not a calibrated probability). Model loaded once at startup, single-thread inference.

| measured over 2,000 sequential requests (test-set rows) | p50 | p95 | p99 |
|---|---|---|---|
| in-process (feature build + predict + rules) | 0.99 ms | 1.23 ms | see reports/latency.json |
| HTTP end-to-end (uvicorn, localhost, keep-alive) | 2.17 ms | 2.55 ms | see reports/latency.json |

Sequential single-client latency, not throughput under load.

## Drift

`make drift` writes PSI (train vs test) per feature and for the model score to `reports/drift_psi.csv`. The hour-of-day feature shows extreme PSI by construction: the test period covers a different part of the day cycle than train. This is a sign the two-day window cannot support seasonality claims, not an incident.

## Limitations

- Only 75 test (57 validation) frauds: PR-AUC and cost estimates are noisy; model rankings could flip with another split. No confidence intervals were computed.
- The 10-per-alert cost and "loss = full transaction amount" are assumptions, not Digipay figures.
- Features are PCA-anonymised, so rules are not interpretable business rules, and the rule cutoffs are specific to this dataset.
- Two days of data: no seasonality, no concept-drift evaluation; `hour` is weakly informative and drifts across the split.
- PaySim (optional second dataset) and a monitoring dashboard were not done.
- Fraud labels are used as ground truth with no delay; in production labels arrive late, which this setup ignores.

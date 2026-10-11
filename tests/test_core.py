import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from fraud import serve
from fraud.data import time_split
from fraud.evaluate import alert_outcomes, hybrid_threshold_for_alert_rate, threshold_for_alert_rate
from fraud.features import RAW_COLUMNS


def test_time_split_is_ordered_and_disjoint():
    df = pd.DataFrame({"Time": np.arange(100.0), "Class": 0})
    train, val, test = time_split(df)
    assert train.Time.max() < val.Time.min() <= val.Time.max() < test.Time.min()
    assert len(train) + len(val) + len(test) == 100


def test_precision_at_alert_rate_and_hybrid_budget():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    out = alert_outcomes(y, np.ones(10), scores >= threshold_for_alert_rate(scores, 0.2))
    assert (out["alerts"], out["precision"]) == (2, 0.5)
    rule_hit = np.array([True] + [False] * 9)
    assert hybrid_threshold_for_alert_rate(rule_hit, scores, 0.3) == 0.7


def test_score_endpoint_returns_decision():
    payload = {c: 0.0 for c in RAW_COLUMNS}
    with TestClient(serve.app) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/score", json=payload)
        assert response.status_code == 200
        assert set(response.json()) == {"score", "alert", "rule_hits"}
        assert client.post("/score", json={"Amount": 1.0}).status_code == 422

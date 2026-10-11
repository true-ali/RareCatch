import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

COST_PER_FALSE_ALERT = 10.0
ALERT_RATES = (0.001, 0.002, 0.005, 0.01)


def pr_auc(y, scores):
    return float(average_precision_score(y, scores))


def roc_auc(y, scores):
    return float(roc_auc_score(y, scores))


def threshold_for_alert_rate(scores, rate):
    k = max(1, int(round(rate * len(scores))))
    return float(np.sort(scores)[::-1][k - 1])


def alert_outcomes(y, amount, alert):
    y, amount, alert = np.asarray(y), np.asarray(amount), np.asarray(alert, dtype=bool)
    tp = int((alert & (y == 1)).sum())
    fp = int((alert & (y == 0)).sum())
    fn = int((~alert & (y == 1)).sum())
    missed_amount = float(amount[~alert & (y == 1)].sum())
    cost = COST_PER_FALSE_ALERT * fp + missed_amount
    return {
        "alerts": tp + fp,
        "alert_rate": (tp + fp) / len(y),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "cost": cost,
    }


def no_model_cost(y, amount):
    y, amount = np.asarray(y), np.asarray(amount)
    return float(amount[y == 1].sum())


def cost_optimal_threshold(y, amount, scores):
    y, amount, scores = np.asarray(y), np.asarray(amount), np.asarray(scores)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    fp = np.cumsum(y[order] == 0)
    caught = np.cumsum(np.where(y[order] == 1, amount[order], 0.0))
    cost = COST_PER_FALSE_ALERT * fp + (amount[y == 1].sum() - caught)
    last_of_tie = np.append(sorted_scores[1:] != sorted_scores[:-1], True)
    cost = np.where(last_of_tie, cost, np.inf)
    best = int(np.argmin(cost))
    if cost[best] >= amount[y == 1].sum():
        return float("inf")
    return float(sorted_scores[best])


def hybrid_alert(rule_hit, scores, threshold):
    return np.asarray(rule_hit, dtype=bool) | (np.asarray(scores) >= threshold)


def hybrid_threshold_for_alert_rate(rule_hit, scores, rate):
    rule_hit = np.asarray(rule_hit, dtype=bool)
    budget = int(round(rate * len(scores))) - int(rule_hit.sum())
    rest = np.sort(np.asarray(scores)[~rule_hit])[::-1]
    if budget <= 0 or len(rest) == 0:
        return float("inf")
    return float(rest[min(budget, len(rest)) - 1])


def hybrid_cost_optimal_threshold(y, amount, scores, rule_hit):
    rule_hit = np.asarray(rule_hit, dtype=bool)
    keep = ~rule_hit
    return cost_optimal_threshold(np.asarray(y)[keep], np.asarray(amount)[keep], np.asarray(scores)[keep])


def psi(expected, actual, bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))

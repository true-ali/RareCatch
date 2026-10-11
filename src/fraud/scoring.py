import lightgbm as lgb
from xgboost import XGBClassifier


def score(model, X):
    if isinstance(model, lgb.LGBMClassifier):
        return model.predict(X, raw_score=True)
    if isinstance(model, XGBClassifier):
        return model.predict(X, output_margin=True)
    return model.predict_proba(X)[:, 1]

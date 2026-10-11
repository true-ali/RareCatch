import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import create_model

from fraud.features import RAW_COLUMNS, make_features, one_row_frame
from fraud.scoring import score

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")

Transaction = create_model("Transaction", **{c: (float, ...) for c in RAW_COLUMNS})
state = {}


def load_bundle(path=MODEL_PATH):
    bundle = joblib.load(path)
    if hasattr(bundle["model"], "set_params") and "n_jobs" in bundle["model"].get_params():
        bundle["model"].set_params(n_jobs=1)
    return bundle


def score_record(bundle, record):
    frame = one_row_frame(record)
    value = float(score(bundle["model"], make_features(frame).to_numpy())[0])
    hits = [f for f, cutoff in bundle["rules"].items() if record[f] <= cutoff]
    alert = bool(hits) or value >= bundle["ml_threshold"]
    return {"score": value, "alert": alert, "rule_hits": hits}


@asynccontextmanager
async def lifespan(app):
    state["bundle"] = load_bundle()
    yield
    state.clear()


app = FastAPI(title="RareCatch fraud scoring", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model": state["bundle"]["name"]}


@app.post("/score")
async def score_transaction(tx: Transaction):
    return score_record(state["bundle"], tx.model_dump())

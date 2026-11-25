"""Load the trained artefact once and score encounters."""
import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .features import prepare

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[2] / "models"))


@lru_cache(maxsize=1)
def load():
    model = joblib.load(MODEL_DIR / "model.joblib")
    meta = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    return model, meta


def risk_band(p: float, threshold: float) -> str:
    if p >= threshold:
        return "high"
    if p >= threshold / 2:
        return "medium"
    return "low"


def score(records: list[dict]) -> list[dict]:
    model, meta = load()
    X = prepare(pd.DataFrame(records))
    probs = model.predict_proba(X)[:, 1]
    t = meta["threshold"]
    return [
        {"readmission_probability": round(float(p), 4), "risk_band": risk_band(float(p), t),
         "threshold": round(t, 4), "model_version": meta["model_version"]}
        for p in probs
    ]

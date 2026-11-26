"""Load the trained artefact once, score encounters and explain a score."""
import json
import math
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb

from .features import clean_categoricals, prepare
from .schema import CATEGORICAL, FEATURES, NUMERIC, Encounter

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[2] / "models"))


@lru_cache(maxsize=1)
def load():
    model = joblib.load(MODEL_DIR / "model.joblib")
    meta = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    return model, meta


@lru_cache(maxsize=1)
def typical_patient() -> dict:
    """Median / most common value of every feature in the training sample."""
    ref = clean_categoricals(pd.read_csv(MODEL_DIR / "reference.csv"))
    typical = {c: float(ref[c].median()) for c in NUMERIC}
    typical |= {c: str(ref[c].mode().iloc[0]) for c in CATEGORICAL}
    return typical


def risk_band(p: float, threshold: float) -> str:
    if p >= threshold:
        return "high"
    if p >= threshold / 2:
        return "medium"
    return "low"


def _result(p: float, meta: dict) -> dict:
    t = meta["threshold"]
    return {"readmission_probability": round(float(p), 4), "risk_band": risk_band(float(p), t),
            "threshold": round(t, 4), "model_version": meta["model_version"]}


def score(records: list[dict]) -> list[dict]:
    model, meta = load()
    probs = model.predict_proba(prepare(pd.DataFrame(records)))[:, 1]
    return [_result(p, meta) for p in probs]


def score_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Score an uploaded table. Columns it lacks take the API defaults."""
    defaults = Encounter().model_dump()
    df = raw.copy()
    for col, val in defaults.items():
        if col not in df:
            df[col] = val
    df["diag_1"] = df["diag_1"].astype(str)
    model, meta = load()
    probs = model.predict_proba(prepare(df))[:, 1]
    out = pd.DataFrame([_result(p, meta) for p in probs])
    return pd.concat([raw.reset_index(drop=True), out], axis=1)


@lru_cache(maxsize=1)
def _column_owner() -> list[str]:
    """Map each encoded model column (e.g. 'cat__insulin_Up') back to its input feature."""
    model, _ = load()
    names = [n.split("__", 1)[1] for n in model.named_steps["prep"].get_feature_names_out()]
    return [max((f for f in FEATURES if n == f or n.startswith(f + "_")), key=len) for n in names]


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def explain(record: dict, top: int = 6) -> dict:
    """Score one patient and list the factors that move the score most.

    Uses SHAP values computed natively by XGBoost (TreeSHAP, `pred_contribs`).
    One-hot columns are summed back to their input feature. Each impact is
    shown in percentage points: the risk of a baseline patient with only that
    factor's contribution added, minus the baseline risk.
    """
    model, meta = load()
    X = prepare(pd.DataFrame([record]))
    p = float(model.predict_proba(X)[:, 1][0])
    encoded = model.named_steps["prep"].transform(X)
    contribs = model.named_steps["clf"].get_booster().predict(xgb.DMatrix(encoded), pred_contribs=True)[0]
    bias = float(contribs[-1])
    per_feature: dict[str, float] = {}
    for owner, c in zip(_column_owner(), contribs[:-1], strict=True):
        per_feature[owner] = per_feature.get(owner, 0.0) + float(c)
    base = _sigmoid(bias)
    typical = typical_patient()
    factors = []
    for f, logit in per_feature.items():
        impact = _sigmoid(bias + logit) - base
        if abs(impact) >= 0.002:
            value = X.iloc[0][f] if f in CATEGORICAL else float(X.iloc[0][f])
            factors.append({"feature": f, "value": value, "typical": typical[f],
                            "impact": round(impact, 4), "log_odds": round(logit, 4)})
    factors.sort(key=lambda f: -abs(f["impact"]))
    prevalence = meta.get("test_metrics", {}).get("prevalence")
    return _result(p, meta) | {"population_rate": prevalence, "baseline_risk": round(base, 4),
                               "factors": factors[:top]}

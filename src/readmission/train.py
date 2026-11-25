"""Train the readmission model and log the run to MLflow.

    python -m readmission.train --data data/diabetic_data.csv

Splits by patient (patient_nbr) so no patient appears in both train and test.
Writes model.joblib, metadata.json and reference.csv (a sample of training
features used as the drift baseline) into the --out folder, models/ by default.
"""
import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .features import build_preprocessor, drop_leaky_rows, make_target, prepare

ROOT = Path(__file__).resolve().parents[2]

PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss")


def split(df: pd.DataFrame, seed: int):
    groups = df["patient_nbr"] if "patient_nbr" in df else np.arange(len(df))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trv, te = next(gss.split(df, groups=groups))
    trv_df = df.iloc[trv]
    g2 = groups.iloc[trv] if hasattr(groups, "iloc") else groups[trv]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + 1)
    tr, va = next(gss2.split(trv_df, groups=g2))
    return trv_df.iloc[tr], trv_df.iloc[va], df.iloc[te]


def bootstrap_auc_ci(y, p, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def pick_threshold(y, p) -> float:
    """F1-optimal threshold, chosen on the validation set only."""
    grid = np.linspace(0.05, 0.6, 56)
    scores = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y, p, threshold) -> dict:
    lo, hi = bootstrap_auc_ci(y, p)
    pred = p >= threshold
    return {
        "roc_auc": roc_auc_score(y, p), "roc_auc_ci_low": lo, "roc_auc_ci_high": hi,
        "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0), "prevalence": float(np.mean(y)),
    }


def subgroup_metrics(X: pd.DataFrame, y: pd.Series, p: np.ndarray, threshold: float) -> dict:
    """AUC and recall per subgroup, for the model card's fairness section."""
    out = {}
    y, p = np.asarray(y), np.asarray(p)
    for col in ("gender", "race", "age"):
        for val in sorted(X[col].unique()):
            m = (X[col] == val).to_numpy()
            if m.sum() < 100 or y[m].min() == y[m].max():
                continue
            out[f"{col}={val}"] = {
                "n": int(m.sum()), "prevalence": round(float(y[m].mean()), 4),
                "roc_auc": round(float(roc_auc_score(y[m], p[m])), 4),
                "recall": round(float(recall_score(y[m], p[m] >= threshold, zero_division=0)), 4),
            }
    return out


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def train(data_path: str, out_dir: Path, seed: int = 42, use_mlflow: bool = True) -> dict:
    raw = drop_leaky_rows(pd.read_csv(data_path, dtype={"diag_1": str}))
    tr, va, te = split(raw, seed)
    X_tr, X_va, X_te = prepare(tr), prepare(va), prepare(te)
    y_tr, y_va, y_te = make_target(tr), make_target(va), make_target(te)

    # No class reweighting: it would distort the probabilities the API returns.
    # Class imbalance is handled by the validation-chosen threshold instead.
    model = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", XGBClassifier(**PARAMS, random_state=seed, n_jobs=-1)),
    ])
    model.fit(X_tr, y_tr)

    threshold = pick_threshold(y_va, model.predict_proba(X_va)[:, 1])
    p_te = model.predict_proba(X_te)[:, 1]
    metrics = evaluate(y_te, p_te, threshold)
    subgroups = subgroup_metrics(X_te, y_te, p_te, threshold)

    version = f"{datetime.now(timezone.utc):%Y%m%d%H%M}-{git_sha()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    meta = {
        "model_version": version, "threshold": threshold, "trained_at": datetime.now(timezone.utc).isoformat(),
        "data": os.path.basename(data_path), "n_train": len(X_tr), "n_val": len(X_va), "n_test": len(X_te),
        "params": PARAMS, "test_metrics": metrics, "subgroup_metrics": subgroups,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # Baseline for drift monitoring lives next to the model it describes.
    X_tr.sample(min(5000, len(X_tr)), random_state=seed).to_csv(out_dir / "reference.csv", index=False)

    if use_mlflow:
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
        mlflow.set_experiment("readmission-30d")
        with mlflow.start_run(run_name=version):
            mlflow.log_params(meta["params"] | {"threshold": threshold, "data": meta["data"], "seed": seed})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(out_dir / "metadata.json"))
            mlflow.sklearn.log_model(
                model, name="model", input_example=X_te.head(3),
                skops_trusted_types=["numpy.dtype", "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
            )
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "diabetic_data.csv"))
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    m = train(a.data, Path(a.out), a.seed, not a.no_mlflow)
    print(json.dumps(m["test_metrics"], indent=2))

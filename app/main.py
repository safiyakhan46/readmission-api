"""FastAPI service for 30-day readmission risk.

Run locally:  uvicorn app.main:app --reload --port 7860
App:          http://localhost:7860/        (clinician-facing UI)
Docs:         http://localhost:7860/docs    (API reference)
"""
import io
import math
import os
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from readmission import predict
from readmission.schema import Encounter, Prediction

UI = Path(__file__).parent / "static" / "index.html"
MAX_ROWS = 1000

app = FastAPI(
    title="30-Day Readmission Risk API",
    version="1.1.0",
    description=(
        "XGBoost model trained on the UCI Diabetes 130-US Hospitals dataset. "
        "**Research demo, not a medical device.** See the model card for limits."
    ),
)
_started = time.time()


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(UI, media_type="text/html")


@app.get("/health")
def health():
    try:
        _, meta = predict.load()
    except FileNotFoundError:
        raise HTTPException(503, "model artefact not found") from None
    return {"status": "ok", "model_version": meta["model_version"],
            "commit": os.getenv("RENDER_GIT_COMMIT", "local")[:7], "uptime_s": round(time.time() - _started)}


@app.get("/model-info")
def model_info():
    _, meta = predict.load()
    keys = ("model_version", "trained_at", "threshold", "n_train", "n_val", "n_test",
            "test_metrics", "subgroup_metrics")
    return {k: meta[k] for k in keys if k in meta}


@app.post("/predict", response_model=Prediction)
def predict_one(encounter: Encounter):
    return predict.score([encounter.model_dump()])[0]


@app.post("/explain")
def explain(encounter: Encounter):
    """Risk score plus the factors that raise or lower it, relative to a typical patient."""
    return predict.explain(encounter.model_dump())


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(encounters: list[Encounter]):
    if not encounters:
        raise HTTPException(422, "empty batch")
    if len(encounters) > MAX_ROWS:
        raise HTTPException(413, f"max {MAX_ROWS} encounters per request")
    return predict.score([e.model_dump() for e in encounters])


async def _read_csv(file: UploadFile) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(await file.read()), dtype={"diag_1": str})
    except Exception:
        raise HTTPException(422, "could not read the file as CSV") from None


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(..., description="CSV with the same columns as /predict")):
    """Score every row of an uploaded CSV. Missing columns take the /predict defaults."""
    df = await _read_csv(file)
    if df.empty:
        raise HTTPException(422, "the file has no rows")
    if len(df) > MAX_ROWS:
        raise HTTPException(413, f"max {MAX_ROWS} rows per file")
    scored = predict.score_frame(df)
    rows = scored.to_dict(orient="records")
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()} for r in rows]


@app.post("/drift")
async def drift(file: UploadFile = File(..., description="CSV of recent encounters, same columns as /predict")):
    """Compare an uploaded batch against the training baseline (Evidently)."""
    # Imported here, not at startup: Evidently is heavy and the free host has 512 MB RAM.
    from readmission.drift import drift_summary

    df = await _read_csv(file)
    missing = [c for c in Encounter.model_fields if c not in df]
    if missing:
        raise HTTPException(422, f"drift check needs every input column; missing: {', '.join(missing)}")
    if len(df) < 50:
        raise HTTPException(422, "need at least 50 rows for a meaningful drift test")
    summary, _ = drift_summary(df)
    return summary

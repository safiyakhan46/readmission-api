"""FastAPI service for 30-day readmission risk.

Run locally:  uvicorn app.main:app --reload --port 7860
Docs:         http://localhost:7860/docs
"""
import io
import time

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse

from readmission import predict
from readmission.drift import drift_summary
from readmission.schema import Encounter, Prediction

app = FastAPI(
    title="30-Day Readmission Risk API",
    version="1.0.0",
    description=(
        "XGBoost model trained on the UCI Diabetes 130-US Hospitals dataset. "
        "**Research demo, not a medical device.** See the model card for limits."
    ),
)
_started = time.time()


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    try:
        _, meta = predict.load()
    except FileNotFoundError:
        raise HTTPException(503, "model artefact not found") from None
    return {"status": "ok", "model_version": meta["model_version"], "uptime_s": round(time.time() - _started)}


@app.get("/model-info")
def model_info():
    _, meta = predict.load()
    return {k: meta[k] for k in ("model_version", "trained_at", "threshold", "n_train", "test_metrics")}


@app.post("/predict", response_model=Prediction)
def predict_one(encounter: Encounter):
    return predict.score([encounter.model_dump()])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(encounters: list[Encounter]):
    if not encounters:
        raise HTTPException(422, "empty batch")
    if len(encounters) > 1000:
        raise HTTPException(413, "max 1000 encounters per request")
    return predict.score([e.model_dump() for e in encounters])


@app.post("/drift")
async def drift(file: UploadFile = File(..., description="CSV of recent encounters, same columns as /predict")):
    """Compare an uploaded batch against the training baseline (Evidently)."""
    df = pd.read_csv(io.BytesIO(await file.read()), dtype={"diag_1": str})
    if len(df) < 50:
        raise HTTPException(422, "need at least 50 rows for a meaningful drift test")
    summary, _ = drift_summary(df)
    return summary

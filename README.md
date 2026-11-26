# 30-Day Readmission Risk API

![CI](https://github.com/safiyakhan46/readmission-api/actions/workflows/ci.yml/badge.svg)

A production setup for my [Patient Readmission Risk Predictor](https://github.com/safiyakhan46/patient-readmission-predictor). It adds:

| Layer | Tool | Where |
|---|---|---|
| Interface | Web app: patient form with live risk, SHAP contributing factors, batch scoring, drift check, model page | `app/static/index.html` |
| Serving | FastAPI + Pydantic validation | `app/main.py` |
| Packaging | Docker, deployed on Render | `Dockerfile` |
| Experiment tracking | MLflow (params, metrics, model, per run) | `src/readmission/train.py` |
| Testing | pytest: features, patient-level split, behaviour, API, drift | `tests/` |
| CI/CD | GitHub Actions: lint → test → build and smoke-test image; Render deploys once CI passes | `.github/workflows/ci.yml` |
| Monitoring | Evidently data-drift report and `/drift` endpoint | `src/readmission/drift.py` |
| Documentation | Model card with subgroup (fairness) metrics | `MODEL_CARD.md` |

**Live app:** https://readmission-api-p9tq.onrender.com · **API reference:** [/docs](https://readmission-api-p9tq.onrender.com/docs)
(free tier: the first visit after 15 idle minutes takes about a minute to wake the service)

> Research and portfolio demo, not a medical device. See [MODEL_CARD.md](MODEL_CARD.md).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check and model version |
| GET | `/model-info` | Threshold, training size, test metrics |
| POST | `/predict` | Score one encounter |
| POST | `/explain` | Score plus the factors driving it (SHAP, in percentage points) |
| POST | `/predict/batch` | Score up to 1,000 encounters (JSON) |
| POST | `/predict/csv` | Score an uploaded CSV; extra columns such as patient IDs are kept |
| POST | `/drift` | Upload a CSV of recent encounters and see which features drifted |

```bash
curl -X POST https://readmission-api-p9tq.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"age":"[80-90)","number_inpatient":4,"number_emergency":2,"diag_1":"428"}'
# {"readmission_probability": ..., "risk_band": "high", "threshold": 0.14, "model_version": "..."}
```
Any field left out takes the default shown in `/docs`.

## Run it

```bash
pip install -r requirements-dev.txt

# 1. Data: download diabetic_data.csv from UCI (dataset id 296) into data/
#    No data yet? `make data` writes a synthetic file with the same columns.
make train          # trains, logs to MLflow, fills MODEL_CARD.md
make mlflow         # browse runs at http://localhost:5000
make test           # 24 tests
make serve          # http://localhost:7860/docs
make docker         # same thing in a container
make drift CURRENT=data/some_new_batch.csv   # HTML report in reports/
```

## Design decisions
- **Patient-level split.** Many patients have several encounters. A random row split would put the same patient in both train and test and inflate the scores.
- **Expired and hospice discharges removed.** These patients cannot be readmitted, so keeping them would teach the model a shortcut.
- **No class reweighting.** Imbalance is handled by a threshold picked on validation data, so the probabilities the API returns stay close to calibrated.
- **Behavioural test.** CI fails if more prior admissions ever *lower* predicted risk, which catches silent regressions that a metric threshold would miss.
- **One label for missing values.** The UCI file leaves an untested A1C blank, while API clients send `"None"`. Both are mapped to `Missing` before the model sees them, so training and serving encode the same patient identically; a regression test guards it.
- **Explanations with SHAP.** Contributing factors come from XGBoost's built-in TreeSHAP (`pred_contribs`), summed back from one-hot columns to the original inputs, so no extra dependency is needed.
- **Small serving image.** MLflow and pytest are dev-only dependencies. The container holds only what scoring needs.

## Deployment
Render builds the `Dockerfile` straight from this repo (free web service, 512 MB RAM). Auto-deploy is set to **after CI checks pass**, so a commit only goes live once lint, tests and the container smoke test are green. The container listens on `$PORT`, which Render sets.

---
title: Readmission Risk API
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: 30-day hospital readmission risk, served with FastAPI
---

# 30-Day Readmission Risk API

![CI](https://github.com/safiyakhan46/readmission-api/actions/workflows/ci.yml/badge.svg)

A production setup for my [Patient Readmission Risk Predictor](https://github.com/safiyakhan46/patient-readmission-predictor). It adds:

| Layer | Tool | Where |
|---|---|---|
| Serving | FastAPI + Pydantic validation | `app/main.py` |
| Packaging | Docker (runs on Hugging Face Spaces) | `Dockerfile` |
| Experiment tracking | MLflow (params, metrics, model, per run) | `src/readmission/train.py` |
| Testing | pytest: features, patient-level split, behaviour, API, drift | `tests/` |
| CI/CD | GitHub Actions: lint → test → build and smoke-test image → deploy to Space | `.github/workflows/ci.yml` |
| Monitoring | Evidently data-drift report and `/drift` endpoint | `src/readmission/drift.py` |
| Documentation | Model card with subgroup (fairness) metrics | `MODEL_CARD.md` |

**Live demo:** https://huggingface.co/spaces/safiyakhan46/readmission-api (interactive docs at `/docs`)

> Research and portfolio demo, not a medical device. See [MODEL_CARD.md](MODEL_CARD.md).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check and model version |
| GET | `/model-info` | Threshold, training size, test metrics |
| POST | `/predict` | Score one encounter |
| POST | `/predict/batch` | Score up to 1,000 encounters |
| POST | `/drift` | Upload a CSV of recent encounters and see which features drifted |

```bash
curl -X POST https://safiyakhan46-readmission-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"age":"[80-90)","number_inpatient":4,"number_emergency":2,"diag_1":"428"}'
# {"readmission_probability":0.37,"risk_band":"high","threshold":0.1,"model_version":"..."}
```
Any field left out takes the default shown in `/docs`.

## Run it

```bash
pip install -r requirements-dev.txt

# 1. Data: download diabetic_data.csv from UCI (dataset id 296) into data/
#    No data yet? `make data` writes a synthetic file with the same columns.
make train          # trains, logs to MLflow, fills MODEL_CARD.md
make mlflow         # browse runs at http://localhost:5000
make test           # 20 tests
make serve          # http://localhost:7860/docs
make docker         # same thing in a container
make drift CURRENT=data/some_new_batch.csv   # HTML report in reports/
```

## Design decisions
- **Patient-level split.** Many patients have several encounters. A random row split would put the same patient in both train and test and inflate the scores.
- **Expired and hospice discharges removed.** These patients cannot be readmitted, so keeping them would teach the model a shortcut.
- **No class reweighting.** Imbalance is handled by a threshold picked on validation data, so the probabilities the API returns stay close to calibrated.
- **Behavioural test.** CI fails if more prior admissions ever *lower* predicted risk, which catches silent regressions that a metric threshold would miss.
- **Small serving image.** MLflow and pytest are dev-only dependencies. The container holds only what scoring needs.

## Deploy to Hugging Face Spaces
1. Create a Space: SDK **Docker**, name `readmission-api`.
2. In this GitHub repo, add the secret `HF_TOKEN` (a write token from hf.co/settings/tokens) and the variable `HF_SPACE` = `safiyakhan46/readmission-api`.
3. Commit `models/model.joblib`, `models/metadata.json` and `data/reference.csv`, then push to `main`. CI tests, builds and deploys.

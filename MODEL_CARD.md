# Model Card: 30-Day Hospital Readmission Risk

*Format follows Mitchell et al., "Model Cards for Model Reporting" (2019).*

## Model details
- **Developer:** Safiya Salman Khan (github.com/safiyakhan46)
- **Model:** gradient-boosted trees (XGBoost) inside a scikit-learn pipeline: median imputation for numbers, one-hot encoding for categories, rare levels (<20) pooled.
- **Output:** the probability that a patient is readmitted within 30 days of discharge, plus a risk band. *High* means at or above the decision threshold; *medium* means at least half the threshold.
- **Threshold:** the F1-optimal cut-off, chosen on a validation set kept separate from the test set.
- **Versioning:** every training run is logged to MLflow (parameters, metrics, artefact). The API reports its `model_version` on every response.

## Intended use
- **Intended:** to show an end-to-end clinical ML service in a portfolio, for teaching and for research on readmission modelling.
- **Out of scope:** any real clinical decision, triage or resource allocation; use on patients outside the training population. This is **not a medical device** and has had no clinical validation.

## Training data
- **Source:** UCI *Diabetes 130-US Hospitals for Years 1999–2008* (Strack et al., 2014). It holds 101,766 inpatient encounters of diabetic patients across 130 US hospitals and is licensed CC BY 4.0.
- **Target:** `readmitted == "<30"` → 1; `>30` and `NO` → 0.
- **Exclusions:** encounters that ended in death or hospice (discharge dispositions 11, 13, 14, 19, 20, 21), since those patients cannot be readmitted.
- **Split:** by **patient** (`patient_nbr`): 64% train, 16% validation, 20% test. No patient appears in more than one split, so repeat visits cannot leak between train and test.
- **Features (20):** demographics (race, gender, age band); admission type, source and discharge disposition; length of stay; counts of labs, procedures, medications and diagnoses; prior-year outpatient, emergency and inpatient visits; A1C and glucose results; insulin, medication change and diabetes medication flags; and the primary diagnosis grouped into 9 ICD-9 chapters.

## Evaluation
Test-set metrics. The 95% CI for ROC-AUC comes from 500 bootstrap resamples.

<!-- METRICS -->
Model version `202611251204-70f9956`, trained on `diabetic_data.csv` (train 63,670 / val 15,871 / test 19,802 encounters).

| Metric | Value |
|---|---|
| ROC-AUC (95% CI) | 0.668 (0.657–0.680) |
| PR-AUC | 0.231 (prevalence 0.112) |
| Brier score | 0.094 |
| Threshold | 0.14 |
| Precision / Recall / F1 at threshold | 0.203 / 0.445 / 0.279 |
<!-- /METRICS -->

**How to read these numbers:** published models on this dataset usually reach a ROC-AUC of about 0.65–0.70. Readmission depends heavily on things the dataset does not record, such as social support, follow-up care and medication adherence. PR-AUC means more than ROC-AUC here, because only about 11% of encounters are positives.

## Fairness and subgroup performance
AUC and recall are reported by gender, race and age band (groups with ≥100 test encounters). Look for gaps of more than about 0.05 AUC, or large gaps in recall, at the shared threshold.

<!-- SUBGROUPS -->
| Group | n | Prevalence | ROC-AUC | Recall |
|---|---|---|---|---|
| gender=Female | 10,645 | 0.112 | 0.675 | 0.460 |
| gender=Male | 9,157 | 0.112 | 0.661 | 0.428 |
| race=AfricanAmerican | 3,707 | 0.107 | 0.658 | 0.437 |
| race=Asian | 122 | 0.082 | 0.700 | 0.400 |
| race=Caucasian | 14,886 | 0.115 | 0.668 | 0.451 |
| race=Hispanic | 371 | 0.105 | 0.780 | 0.487 |
| race=Missing | 418 | 0.091 | 0.617 | 0.210 |
| race=Other | 298 | 0.087 | 0.701 | 0.462 |
| age=[10-20) | 138 | 0.065 | 0.594 | 0.000 |
| age=[20-30) | 388 | 0.155 | 0.865 | 0.867 |
| age=[30-40) | 729 | 0.106 | 0.719 | 0.493 |
| age=[40-50) | 1,909 | 0.106 | 0.654 | 0.386 |
| age=[50-60) | 3,469 | 0.103 | 0.705 | 0.407 |
| age=[60-70) | 4,266 | 0.109 | 0.645 | 0.377 |
| age=[70-80) | 5,040 | 0.117 | 0.653 | 0.494 |
| age=[80-90) | 3,287 | 0.119 | 0.652 | 0.469 |
| age=[90-100) | 547 | 0.128 | 0.551 | 0.357 |
<!-- /SUBGROUPS -->

## Monitoring
- `/drift` and `python -m readmission.drift` compare a new batch against a 5,000-row sample of training features (`models/reference.csv`) using Evidently's `DataDriftPreset`. Numeric columns use Wasserstein distance; categorical columns use chi-square or Jensen-Shannon.
- **Retraining trigger (suggested):** more than 30% of features drift, or `number_inpatient`, `discharge_disposition_id` or `age` drift. Those are the strongest predictors.
- Drift says nothing about accuracy. Performance can only be checked once 30-day outcomes for the new batch are known.

## Limitations and risks
- **Old data:** the records are from 1999–2008 and use ICD-9, so coding, care pathways and readmission policy (for example the 2012 HRRP penalties) have changed since.
- **One condition, one country:** diabetic inpatients in US hospitals only.
- **Missing data:** `weight`, `payer_code` and `medical_specialty` are mostly missing and are not used. `race` has missing values, which are kept as their own category.
- **Proxy features:** race appears as a feature for transparency and to allow subgroup audits. Anyone deploying this should decide whether to drop it and should re-audit after doing so.
- **Calibration:** no class reweighting is applied, so probabilities stay close to calibrated (see the Brier score). Recalibrate before relying on absolute risk values.

## Ethical considerations
High-risk flags could steer follow-up resources. Unequal recall across groups would then mean unequal access to care, so the subgroup table must be reviewed before any use beyond a demo.

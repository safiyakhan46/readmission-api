from make_synthetic_data import make
from readmission.train import split


def test_patient_level_split_has_no_overlap():
    df = make(2000)
    tr, va, te = split(df, seed=0)
    a, b, c = set(tr.patient_nbr), set(va.patient_nbr), set(te.patient_nbr)
    assert not (a & b) and not (a & c) and not (b & c)


def test_model_beats_chance_and_metadata_is_complete(trained):
    m = trained["test_metrics"]
    assert m["roc_auc"] > 0.52
    assert 0 < trained["threshold"] < 1
    assert {"model_version", "trained_at", "n_train", "params"} <= trained.keys()


def test_more_prior_admissions_raise_risk(trained):
    """Behavioural check: prior inpatient visits are the strongest known risk factor."""
    from readmission.predict import score
    base = make(1, seed=3).drop(columns=["readmitted", "patient_nbr"]).iloc[0].to_dict()
    low = score([base | {"number_inpatient": 0}])[0]["readmission_probability"]
    high = score([base | {"number_inpatient": 6}])[0]["readmission_probability"]
    assert high > low

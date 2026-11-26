import io

import pytest
from fastapi.testclient import TestClient

from make_synthetic_data import make


@pytest.fixture(scope="module")
def client(trained):
    from app.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_default_example(client):
    r = client.post("/predict", json={})
    body = r.json()
    assert r.status_code == 200
    assert 0 <= body["readmission_probability"] <= 1
    assert body["risk_band"] in {"low", "medium", "high"}


def test_predict_rejects_invalid_input(client):
    assert client.post("/predict", json={"time_in_hospital": 99}).status_code == 422
    assert client.post("/predict", json={"A1Cresult": "banana"}).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json=[{}, {"number_inpatient": 5}])
    assert r.status_code == 200 and len(r.json()) == 2
    assert client.post("/predict/batch", json=[]).status_code == 422


def test_model_info(client):
    info = client.get("/model-info").json()
    assert "test_metrics" in info and "subgroup_metrics" in info


def test_ui_is_served_at_root(client):
    r = client.get("/")
    assert r.status_code == 200 and "Readmission Risk" in r.text
    assert "commit" in client.get("/health").json()


def test_explain_matches_predict_and_lists_factors(client):
    patient = {"age": "[80-90)", "number_inpatient": 6, "number_emergency": 3}
    e = client.post("/explain", json=patient).json()
    p = client.post("/predict", json=patient).json()
    assert e["readmission_probability"] == p["readmission_probability"]
    assert e["factors"], "a clearly atypical patient should have contributing factors"
    top = e["factors"][0]
    assert {"feature", "value", "typical", "impact"} <= top.keys()
    assert "number_inpatient" in [f["feature"] for f in e["factors"]]


def test_csv_scoring_keeps_ids_and_fills_missing_columns(client):
    csv = "patient_id,age,number_inpatient\nA1,[80-90),5\nA2,[40-50),0\n"
    r = client.post("/predict/csv", files={"file": ("p.csv", csv.encode(), "text/csv")})
    rows = r.json()
    assert r.status_code == 200 and [x["patient_id"] for x in rows] == ["A1", "A2"]
    assert rows[0]["readmission_probability"] > rows[1]["readmission_probability"]
    bad = client.post("/predict/csv", files={"file": ("p.csv", b"", "text/csv")})
    assert bad.status_code == 422


def test_drift_flags_shifted_population(client, trained, tmp_path, monkeypatch):
    import pandas as pd

    from readmission import drift
    from readmission.features import prepare
    ref = prepare(make(2000, seed=1))
    monkeypatch.setattr(drift, "REFERENCE", tmp_path / "ref.csv")
    ref.to_csv(drift.REFERENCE, index=False)

    def post(df: pd.DataFrame):
        buf = io.BytesIO(df.to_csv(index=False).encode())
        return client.post("/drift", files={"file": ("cur.csv", buf, "text/csv")}).json()

    shifted = post(make(800, seed=9, shift=1.5))
    assert "number_inpatient" in shifted["drifted_columns"]
    # A batch where nobody had an A1C test must not crash the report
    untested = make(200, seed=11)
    untested["A1Cresult"] = "None"
    assert "drifted_columns" in post(untested)
    same = post(make(800, seed=10))
    assert len(same["drifted_columns"]) < len(shifted["drifted_columns"])

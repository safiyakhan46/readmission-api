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
    assert "test_metrics" in client.get("/model-info").json()


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
    same = post(make(800, seed=10))
    assert len(same["drifted_columns"]) < len(shifted["drifted_columns"])

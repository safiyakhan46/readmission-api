import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Train a small model on synthetic data once per test session."""
    import os

    from make_synthetic_data import make
    from readmission.train import train

    d = tmp_path_factory.mktemp("run")
    csv = d / "train.csv"
    make(3000, seed=1).to_csv(csv, index=False)
    meta = train(str(csv), d / "models", seed=1, use_mlflow=False)
    os.environ["MODEL_DIR"] = str(d / "models")
    from readmission import predict
    predict.MODEL_DIR = d / "models"
    predict.load.cache_clear()
    return meta

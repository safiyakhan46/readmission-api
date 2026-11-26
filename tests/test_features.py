import pandas as pd
import pytest

from make_synthetic_data import make
from readmission.features import drop_leaky_rows, icd9_group, make_target, prepare
from readmission.schema import FEATURES


@pytest.mark.parametrize("code,group", [
    ("428", "Circulatory"), ("786", "Respiratory"), ("250.83", "Diabetes"),
    ("996", "Injury"), ("V57", "Other"), ("?", "Missing"), (None, "Missing"), ("174", "Neoplasms"),
])
def test_icd9_group(code, group):
    assert icd9_group(code) == group


def test_prepare_returns_model_columns_without_nans_in_categoricals():
    X = prepare(make(200))
    assert list(X.columns) == FEATURES
    assert "?" not in set(X["race"])


def test_missing_values_spelled_any_way_get_one_label():
    """UCI blanks (NaN) and API "None" must reach the model as the same category."""
    import numpy as np
    base = make(3).drop(columns=["readmitted"])
    base["A1Cresult"] = [np.nan, "None", "?"]
    base["max_glu_serum"] = ["", "nan", None]
    X = prepare(base)
    assert set(X["A1Cresult"]) == {"Missing"}
    assert set(X["max_glu_serum"]) == {"Missing"}


def test_target_is_30_day_only():
    df = pd.DataFrame({"readmitted": ["<30", ">30", "NO"]})
    assert make_target(df).tolist() == [1, 0, 0]


def test_expired_patients_removed():
    df = pd.DataFrame({"discharge_disposition_id": ["1", "11", "13", "3"]})
    assert drop_leaky_rows(df)["discharge_disposition_id"].tolist() == ["1", "3"]

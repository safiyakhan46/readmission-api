"""Feature engineering and the sklearn preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .schema import CATEGORICAL, NUMERIC, POSITIVE_LABEL, TARGET


def icd9_group(code) -> str:
    """Collapse a primary ICD-9 code into the clinical groups used by Strack et al. (2014)."""
    if code is None or (isinstance(code, float) and pd.isna(code)) or str(code) in {"?", ""}:
        return "Missing"
    s = str(code)
    if s.startswith(("V", "E")):
        return "Other"
    try:
        v = float(s)
    except ValueError:
        return "Other"
    if 390 <= v <= 459 or v == 785:
        return "Circulatory"
    if 460 <= v <= 519 or v == 786:
        return "Respiratory"
    if 520 <= v <= 579 or v == 787:
        return "Digestive"
    if int(v) == 250:
        return "Diabetes"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or v == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


MISSING_TOKENS = {"?", "", "nan", "NaN", "None", "<NA>", "NULL"}


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Map every spelling of "not recorded" to one label, "Missing".

    The UCI file leaves untested A1C/glucose blank (read as NaN), while API
    clients send the string "None". Both must reach the model as the same
    category, and the model's own imputer also fills NaN with "Missing".
    """
    out = df.copy()
    for col in CATEGORICAL:
        if col in out:
            s = out[col].astype(object).where(out[col].notna(), "Missing").astype(str).str.strip()
            out[col] = s.where(~s.isin(MISSING_TOKENS), "Missing")
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw UCI-style frame -> model-ready feature frame (no target)."""
    out = df.copy()
    out["diag_1_group"] = out["diag_1"].map(icd9_group)
    out = clean_categoricals(out)
    for col in NUMERIC:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[NUMERIC + CATEGORICAL]


def make_target(df: pd.DataFrame) -> pd.Series:
    return (df[TARGET] == POSITIVE_LABEL).astype(int)


def drop_leaky_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove encounters that cannot be readmitted (expired / hospice: dispositions 11,13,14,19,20,21)."""
    dead_or_hospice = {"11", "13", "14", "19", "20", "21"}
    return df[~df["discharge_disposition_id"].astype(str).isin(dead_or_hospice)]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )

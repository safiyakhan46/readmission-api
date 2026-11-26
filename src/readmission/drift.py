"""Data-drift report: compare a batch of recent encounters with the training baseline.

    python -m readmission.drift --current data/current.csv --out reports/drift.html
"""
import argparse
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from . import predict
from .features import clean_categoricals, prepare
from .schema import FEATURES

ROOT = Path(__file__).resolve().parents[2]
REFERENCE: Path | None = None  # override in tests; default is MODEL_DIR/reference.csv


def drift_summary(current_raw: pd.DataFrame, reference: pd.DataFrame | None = None):
    """Returns (summary dict, evidently snapshot)."""
    if reference is None:
        reference = pd.read_csv(REFERENCE or predict.MODEL_DIR / "reference.csv")
    ref = clean_categoricals(reference)
    cur = prepare(current_raw)[FEATURES]
    snapshot = Report([DataDriftPreset()]).run(reference_data=ref[FEATURES], current_data=cur)
    result = snapshot.dict()
    drifted, share = [], None
    for m in result["metrics"]:
        name = m.get("metric_name", "")
        if name.startswith("DriftedColumnsCount"):
            share = m["value"]["share"]
        elif name.startswith("ValueDrift"):
            col = name.split("column=")[1].split(",")[0]
            threshold = float(name.split("threshold=")[1].rstrip(")"))
            v = m["value"]
            # p-value tests flag drift when small; distance metrics when large
            is_drift = v < threshold if "p_value" in name else v >= threshold
            if is_drift:
                drifted.append(col)
    return {"n_current": len(cur), "share_drifted": share, "drifted_columns": sorted(drifted)}, snapshot


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True)
    ap.add_argument("--out", default=str(ROOT / "reports" / "drift.html"))
    a = ap.parse_args()
    summary, snap = drift_summary(pd.read_csv(a.current, dtype={"diag_1": str}))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    snap.save_html(a.out)
    print(summary)
    print(f"report: {a.out}")

"""Generate a synthetic CSV with the same columns as the UCI diabetes dataset.

Used for CI and for running the project without the real data. It is NOT
clinical data, and metrics on it mean nothing; it only exercises the code.

    python scripts/make_synthetic_data.py --n 5000 --out data/synthetic.csv
"""
import argparse

import numpy as np
import pandas as pd


def make(n: int, seed: int = 0, shift: float = 0.0) -> pd.DataFrame:
    """shift > 0 makes an older, sicker population (for drift demos)."""
    rng = np.random.default_rng(seed)
    age_idx = np.clip(rng.normal(6.3 + shift * 1.5, 1.6, n).round().astype(int), 0, 9)
    inpatient = rng.poisson(0.6 + shift, n)
    emergency = rng.poisson(0.2 + shift * 0.5, n)
    los = np.clip(rng.poisson(4 + shift * 2, n), 1, 14)
    meds = np.clip(rng.normal(16 + shift * 4, 8, n).round(), 1, 81).astype(int)
    diag_pool = ["428", "414", "786", "410", "486", "427", "491", "715", "682", "434",
                 "250.8", "276", "599", "996", "V57", "?"]
    df = pd.DataFrame({
        "patient_nbr": rng.integers(0, max(1, int(n * 0.7)), n),
        "race": rng.choice(["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"],
                           n, p=[.75, .19, .02, .01, .015, .015]),
        "gender": rng.choice(["Female", "Male"], n, p=[.54, .46]),
        "age": [f"[{a * 10}-{a * 10 + 10})" for a in age_idx],
        "admission_type_id": rng.choice(["1", "2", "3", "5", "6"], n, p=[.53, .18, .19, .05, .05]),
        "discharge_disposition_id": rng.choice(["1", "3", "6", "18", "2", "22", "11"], n,
                                               p=[.59, .14, .13, .04, .03, .05, .02]),
        "admission_source_id": rng.choice(["7", "1", "17", "4", "6"], n, p=[.56, .29, .07, .04, .04]),
        "time_in_hospital": los,
        "num_lab_procedures": np.clip(rng.normal(43, 20, n).round(), 1, 132).astype(int),
        "num_procedures": rng.integers(0, 7, n),
        "num_medications": meds,
        "number_outpatient": rng.poisson(0.4, n),
        "number_emergency": emergency,
        "number_inpatient": inpatient,
        "number_diagnoses": np.clip(rng.normal(7.4 + shift, 1.9, n).round(), 1, 16).astype(int),
        "A1Cresult": rng.choice(["None", "Norm", ">7", ">8"], n, p=[.83, .05, .04, .08]),
        "max_glu_serum": rng.choice(["None", "Norm", ">200", ">300"], n, p=[.95, .025, .015, .01]),
        "insulin": rng.choice(["No", "Steady", "Down", "Up"], n, p=[.47, .30, .12, .11]),
        "change": rng.choice(["No", "Ch"], n, p=[.54, .46]),
        "diabetesMed": rng.choice(["Yes", "No"], n, p=[.77, .23]),
        "diag_1": rng.choice(diag_pool, n),
    })
    # Outcome driven by known risk factors so the model has signal to learn.
    logit = (-2.6 + 0.45 * inpatient + 0.25 * emergency + 0.04 * los + 0.12 * (age_idx - 6)
             + 0.5 * (df["discharge_disposition_id"] == "22") + 0.3 * df["diag_1"].isin(["428", "486"]))
    p = 1 / (1 + np.exp(-logit))
    r = rng.random(n)
    df["readmitted"] = np.where(r < p, "<30", np.where(r < p + 0.35, ">30", "NO"))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shift", type=float, default=0.0)
    ap.add_argument("--out", default="data/synthetic.csv")
    a = ap.parse_args()
    make(a.n, a.seed, a.shift).to_csv(a.out, index=False)
    print(f"wrote {a.n} rows to {a.out}")

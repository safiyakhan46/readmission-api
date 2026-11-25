"""Input schema shared by training, the API and drift monitoring.

Columns follow the UCI "Diabetes 130-US Hospitals (1999-2008)" dataset
(101,766 encounters). Target: readmitted within 30 days ("<30") -> 1, else 0.
"""
from typing import Literal

from pydantic import BaseModel, Field

TARGET = "readmitted"
POSITIVE_LABEL = "<30"

NUMERIC = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

CATEGORICAL = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "A1Cresult",
    "max_glu_serum",
    "insulin",
    "change",
    "diabetesMed",
    "diag_1_group",
]

FEATURES = NUMERIC + CATEGORICAL

AGE_BANDS = [f"[{i}-{i + 10})" for i in range(0, 100, 10)]


class Encounter(BaseModel):
    """One inpatient encounter, as sent to /predict."""

    race: str = Field("Caucasian", examples=["Caucasian"])
    gender: Literal["Male", "Female", "Unknown/Invalid"] = "Female"
    age: str = Field("[70-80)", description="10-year band, e.g. [70-80)")
    admission_type_id: str = Field("1", description="1=Emergency, 2=Urgent, 3=Elective, ...")
    discharge_disposition_id: str = Field("1", description="1=Home, 3=SNF, 6=Home health, ...")
    admission_source_id: str = Field("7", description="7=Emergency room, 1=Physician referral, ...")
    time_in_hospital: int = Field(4, ge=1, le=14, description="Days")
    num_lab_procedures: int = Field(44, ge=0, le=150)
    num_procedures: int = Field(1, ge=0, le=10)
    num_medications: int = Field(16, ge=0, le=100)
    number_outpatient: int = Field(0, ge=0, le=50)
    number_emergency: int = Field(0, ge=0, le=80)
    number_inpatient: int = Field(1, ge=0, le=30, description="Inpatient visits in prior year")
    number_diagnoses: int = Field(9, ge=1, le=16)
    A1Cresult: Literal["None", "Norm", ">7", ">8"] = "None"
    max_glu_serum: Literal["None", "Norm", ">200", ">300"] = "None"
    insulin: Literal["No", "Down", "Steady", "Up"] = "Steady"
    change: Literal["No", "Ch"] = "No"
    diabetesMed: Literal["No", "Yes"] = "Yes"
    diag_1: str = Field("428", description="Primary ICD-9 code, e.g. 428 (heart failure)")


class Prediction(BaseModel):
    readmission_probability: float
    risk_band: Literal["low", "medium", "high"]
    threshold: float
    model_version: str

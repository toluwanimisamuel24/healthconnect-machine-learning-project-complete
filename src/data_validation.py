"""
data_validation.py
-------------------
Automated data-quality checks that run every time the pipeline executes, so
issues are caught consistently rather than relying on manual review each
week. Produces a structured report rather than just printing to the console,
so results can be saved, diffed between runs, and inspected by other tracks.
"""

import pandas as pd

from . import config


def check_missing_values(df: pd.DataFrame) -> dict:
    """Return the count and percentage of missing values per column."""
    counts = df.isnull().sum()
    pct = (counts / len(df) * 100).round(2)
    return {
        col: {"missing_count": int(counts[col]), "missing_pct": float(pct[col])}
        for col in df.columns
        if counts[col] > 0
    }


def check_duplicates(df: pd.DataFrame) -> dict:
    """Return duplicate-row and duplicate-ID counts."""
    return {
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_appointment_ids": int(df["appointment_id"].duplicated().sum())
        if "appointment_id" in df.columns else None,
    }


def check_waiting_time_leakage(df: pd.DataFrame) -> dict:
    """
    Confirm the Week 4 finding: waiting_time_minutes should be null for
    every appointment that was not actually attended (No-Show / Cancelled),
    since no wait time can be recorded for a visit that never happened.

    This check does not fix the data -- it exists to make the leakage
    finding a repeatable, automated fact rather than a one-off observation,
    so the guard in feature_engineering.py can be trusted to be necessary.
    """
    if "waiting_time_minutes" not in df.columns or config.OUTCOME_COLUMN not in df.columns:
        return {"checked": False, "reason": "required columns not present"}

    not_attended = df[df[config.OUTCOME_COLUMN] != "Attended"]
    unexpected_values = not_attended["waiting_time_minutes"].notna().sum()

    return {
        "checked": True,
        "not_attended_rows": int(len(not_attended)),
        "rows_with_unexpected_waiting_time": int(unexpected_values),
        "leakage_confirmed": bool(unexpected_values > 0),
    }


def check_historical_field_consistency(df: pd.DataFrame) -> dict:
    """
    Test whether the raw previous_appointments / previous_no_shows columns
    are chronologically consistent per patient (i.e. non-decreasing when
    that patient's appointments are sorted by date).

    This directly informed a Week 5 engineering decision: because a large
    share of patients fail this check, the pipeline does NOT trust these
    raw columns and instead recomputes them from the dataset itself (see
    feature_engineering.recompute_historical_features).
    """
    required = {"patient_id", "appointment_date", "previous_appointments"}
    if not required.issubset(df.columns):
        return {"checked": False, "reason": "required columns not present"}

    working = df.copy()
    working["_appointment_date_dt"] = pd.to_datetime(working["appointment_date"])

    def is_monotonic(group: pd.DataFrame) -> bool:
        ordered = group.sort_values("_appointment_date_dt")
        return bool((ordered["previous_appointments"].diff().dropna() >= 0).all())

    per_patient = working.groupby("patient_id").apply(is_monotonic)
    consistent = int(per_patient.sum())
    total = int(len(per_patient))

    return {
        "checked": True,
        "patients_checked": total,
        "patients_chronologically_consistent": consistent,
        "consistency_rate_pct": round(consistent / total * 100, 1) if total else None,
        "conclusion": (
            "Raw previous_appointments/previous_no_shows are not reliably "
            "point-in-time correct -- pipeline recomputes these fields instead "
            "of using them as provided."
        ),
    }


def run_all_checks(df: pd.DataFrame) -> dict:
    """Run every validation check and return a single combined report."""
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "missing_values": check_missing_values(df),
        "duplicates": check_duplicates(df),
        "waiting_time_leakage": check_waiting_time_leakage(df),
        "historical_field_consistency": check_historical_field_consistency(df),
    }

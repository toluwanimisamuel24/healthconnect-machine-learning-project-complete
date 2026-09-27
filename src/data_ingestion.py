"""
data_ingestion.py
------------------
Loads the raw HealthConnect appointment data without ever modifying the
source file, per the project's resource-handling rules.
"""

import pandas as pd

from . import config


class DataIngestionError(Exception):
    """Raised when the raw data cannot be loaded or is structurally invalid."""


EXPECTED_COLUMNS = [
    "appointment_id", "patient_id", "gender", "age", "age_group",
    "appointment_type", "booking_date", "appointment_date", "appointment_day",
    "appointment_time", "booking_lead_days", "previous_appointments",
    "previous_no_shows", "reminder_sent", "reminder_channel",
    "distance_to_clinic_km", "waiting_time_minutes", "appointment_outcome",
]


def load_raw_data(path=None) -> pd.DataFrame:
    """
    Load the raw appointment CSV as-is.

    Parameters
    ----------
    path : str or Path, optional
        Overrides config.RAW_DATA_PATH -- mainly used by tests, which point
        this at a small fixture file instead of the full dataset.

    Returns
    -------
    pd.DataFrame
        The raw data, completely unmodified (no cleaning, encoding, or
        column drops happen here -- that is the job of later pipeline
        stages, so this function's output can always be trusted to match
        the original source file exactly).

    Raises
    ------
    DataIngestionError
        If the file is missing, empty, or does not contain the columns the
        rest of the pipeline expects.
    """
    data_path = path or config.RAW_DATA_PATH

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError as exc:
        raise DataIngestionError(f"Raw data file not found at: {data_path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise DataIngestionError(f"Raw data file is empty: {data_path}") from exc

    if df.empty:
        raise DataIngestionError(f"Raw data loaded but contains zero rows: {data_path}")

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise DataIngestionError(
            f"Raw data is missing expected columns: {sorted(missing_columns)}. "
            "Has the source file format changed? Update EXPECTED_COLUMNS in "
            "data_ingestion.py if this is an intentional schema change."
        )

    return df

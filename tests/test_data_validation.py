import pandas as pd

from src import data_validation


def test_check_missing_values_detects_nulls():
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    result = data_validation.check_missing_values(df)

    assert "a" in result
    assert result["a"]["missing_count"] == 1
    assert "b" not in result  # no missing values in b


def test_check_duplicates_counts_correctly():
    df = pd.DataFrame({
        "appointment_id": ["A-1", "A-1", "A-2"],
        "value": [1, 1, 2],
    })
    result = data_validation.check_duplicates(df)

    assert result["duplicate_rows"] == 1
    assert result["duplicate_appointment_ids"] == 1


def test_check_waiting_time_leakage_flags_no_show_with_waiting_time():
    df = pd.DataFrame({
        "appointment_outcome": ["Attended", "No-Show", "Cancelled", "Attended"],
        "waiting_time_minutes": [12.0, 8.0, None, 20.0],
    })
    result = data_validation.check_waiting_time_leakage(df)

    assert result["checked"] is True
    assert result["leakage_confirmed"] is True
    assert result["rows_with_unexpected_waiting_time"] == 1  # the No-Show row


def test_check_waiting_time_leakage_clean_data_not_flagged():
    """If waiting_time_minutes were properly null for non-attended rows, no leakage should be flagged."""
    df = pd.DataFrame({
        "appointment_outcome": ["Attended", "No-Show", "Cancelled"],
        "waiting_time_minutes": [12.0, None, None],
    })
    result = data_validation.check_waiting_time_leakage(df)

    assert result["leakage_confirmed"] is False
    assert result["rows_with_unexpected_waiting_time"] == 0


def test_check_historical_field_consistency_detects_inconsistency():
    # P-1's second (later) appointment has a LOWER previous_appointments count
    # than its first (earlier) appointment -- chronologically impossible.
    df = pd.DataFrame({
        "patient_id": ["P-1", "P-1"],
        "appointment_date": ["1/1/2025", "6/1/2025"],
        "previous_appointments": [3, 1],
    })
    result = data_validation.check_historical_field_consistency(df)

    assert result["checked"] is True
    assert result["patients_chronologically_consistent"] == 0
    assert result["patients_checked"] == 1


def test_check_historical_field_consistency_accepts_valid_sequence():
    df = pd.DataFrame({
        "patient_id": ["P-1", "P-1", "P-1"],
        "appointment_date": ["1/1/2025", "3/1/2025", "6/1/2025"],
        "previous_appointments": [0, 1, 2],
    })
    result = data_validation.check_historical_field_consistency(df)

    assert result["patients_chronologically_consistent"] == 1


def test_run_all_checks_returns_combined_report():
    df = pd.DataFrame({
        "appointment_id": ["A-1", "A-2"],
        "patient_id": ["P-1", "P-2"],
        "appointment_date": ["1/1/2025", "2/1/2025"],
        "previous_appointments": [0, 0],
        "appointment_outcome": ["Attended", "No-Show"],
        "waiting_time_minutes": [10.0, None],
    })
    report = data_validation.run_all_checks(df)

    assert report["row_count"] == 2
    assert "missing_values" in report
    assert "duplicates" in report
    assert "waiting_time_leakage" in report
    assert "historical_field_consistency" in report

import pandas as pd

from src import config, feature_engineering


def _tiny_raw_df():
    """
    Two patients, three appointments total, deliberately including:
      - a Cancelled row (should be excluded by build_target)
      - a chronological ordering where P-1's raw previous_appointments
        column is WRONG (inconsistent), to prove recompute_historical_features
        fixes it rather than trusting the raw column.
      - waiting_time_minutes populated for a No-Show row (leakage case),
        to prove drop_leakage_columns removes it.
    """
    return pd.DataFrame({
        "appointment_id": ["A-1", "A-2", "A-3"],
        "patient_id": ["P-1", "P-1", "P-2"],
        "gender": ["Female", "Female", "Male"],
        "age": [25, 25, 40],
        "age_group": ["18-30", "18-30", "31-50"],
        "appointment_type": ["General", "General", "Specialist"],
        "booking_date": ["12/20/2024", "5/1/2025", "1/1/2025"],
        "appointment_date": ["1/1/2025", "6/1/2025", "1/10/2025"],
        "appointment_day": ["Wednesday", "Sunday", "Friday"],
        "appointment_time": ["09:00", "10:00", "14:00"],
        "booking_lead_days": [12, 31, 9],
        "previous_appointments": [9, 9, 0],  # deliberately wrong/inconsistent for P-1
        "previous_no_shows": [2, 2, 0],
        "reminder_sent": ["Yes", "No", "Yes"],
        "reminder_channel": ["SMS", None, "Email"],
        "distance_to_clinic_km": [4.5, 4.5, None],
        "waiting_time_minutes": [18.0, 22.0, None],  # A-2 is a No-Show w/ waiting time -> leakage
        "appointment_outcome": ["Attended", "No-Show", "Cancelled"],
    })


def test_recompute_historical_features_is_point_in_time_correct():
    df = _tiny_raw_df()
    result = feature_engineering.recompute_historical_features(df)

    p1 = result[result["patient_id"] == "P-1"].sort_values("appointment_date")
    # P-1's FIRST appointment (A-1) should show 0 prior appointments/no-shows,
    # regardless of what the raw (wrong) previous_appointments column said.
    first_row = p1.iloc[0]
    assert first_row["computed_previous_appointments"] == 0
    assert first_row["computed_previous_no_shows"] == 0

    # P-1's SECOND appointment (A-2) should show 1 prior appointment, 0 prior no-shows
    # (the first appointment was "Attended", not a no-show).
    second_row = p1.iloc[1]
    assert second_row["computed_previous_appointments"] == 1
    assert second_row["computed_previous_no_shows"] == 0


def test_drop_leakage_columns_removes_waiting_time():
    df = _tiny_raw_df()
    result = feature_engineering.drop_leakage_columns(df)

    assert "waiting_time_minutes" not in result.columns
    assert "appointment_outcome" in result.columns  # not a leakage column, handled separately


def test_build_target_excludes_cancelled_and_encodes_correctly():
    df = _tiny_raw_df()
    result = feature_engineering.build_target(df)

    assert "Cancelled" not in result[config.OUTCOME_COLUMN].values
    assert len(result) == 2  # A-3 (Cancelled) excluded

    target_map = dict(zip(result["appointment_id"], result["target"]))
    assert target_map["A-1"] == 0  # Attended
    assert target_map["A-2"] == 1  # No-Show


def test_impute_missing_values_fills_distance_and_reminder_channel():
    df = _tiny_raw_df()
    result = feature_engineering.impute_missing_values(df)

    assert result["distance_to_clinic_km"].isnull().sum() == 0
    assert result["reminder_channel"].isnull().sum() == 0
    assert config.REMINDER_CHANNEL_FILL_VALUE in result["reminder_channel"].values


def test_run_feature_pipeline_produces_expected_shape_and_no_leakage_columns():
    df = _tiny_raw_df()
    result = feature_engineering.run_feature_pipeline(df)

    # Cancelled row excluded
    assert len(result) == 2
    # leakage column gone
    assert "waiting_time_minutes" not in result.columns
    # raw historical columns replaced by computed versions
    assert "previous_appointments" not in result.columns
    assert "computed_previous_appointments" in result.columns
    # identifiers: patient_id dropped, appointment_id retained for traceability
    assert "patient_id" not in result.columns
    assert "appointment_id" in result.columns
    # target present, source outcome column removed
    assert "target" in result.columns
    assert config.OUTCOME_COLUMN not in result.columns
    # appointment_date retained (needed for the time-based split downstream)
    assert "appointment_date" in result.columns


def test_add_short_lead_high_risk_flag_matches_evidence_based_definition():
    """
    Week 6 feature, added on error-analysis evidence (see feature_engineering.
    add_short_lead_high_risk_flag docstring). Flag should be 1 only when
    BOTH booking_lead_days <= 14 AND computed_previous_no_shows >= 1.
    """
    df = pd.DataFrame({
        "booking_lead_days": [5, 5, 20, 20],
        "computed_previous_no_shows": [1, 0, 1, 0],
    })
    result = feature_engineering.add_short_lead_high_risk_flag(df)

    assert result["short_lead_high_risk_flag"].tolist() == [1, 0, 0, 0]


def test_add_distance_lead_interaction_matches_definition():
    """Week 8, Data Science proposal (Question 4.1). Flag = 1 only when
    distance > median AND booking_lead_days <= 7."""
    df = pd.DataFrame({
        "distance_to_clinic_km": [20.0, 20.0, 1.0, 1.0],  # median = 10.5
        "booking_lead_days": [3, 30, 3, 30],
    })
    result = feature_engineering.add_distance_lead_interaction(df)

    assert result["distance_lead_interaction"].tolist() == [1, 0, 0, 0]


def test_appointment_type_history_mismatch_zero_flags_first_appointment():
    """A patient's first appointment has no history yet -- must be 0, not null or imputed."""
    df = pd.DataFrame({
        "appointment_id": ["A-1"],
        "patient_id": ["P-1"],
        "appointment_date": ["1/1/2025"],
        "appointment_type": ["General"],
        "appointment_outcome": ["Attended"],
    })
    result = feature_engineering.recompute_historical_features(df)
    assert result["appointment_type_history_mismatch"].tolist() == [0]


def test_appointment_type_history_mismatch_detects_a_real_mismatch():
    """Second appointment with a different type than the (only) prior one should flag 1."""
    df = pd.DataFrame({
        "appointment_id": ["A-1", "A-2"],
        "patient_id": ["P-1", "P-1"],
        "appointment_date": ["1/1/2025", "6/1/2025"],
        "appointment_type": ["General", "Specialist"],
        "appointment_outcome": ["Attended", "Attended"],
    })
    result = feature_engineering.recompute_historical_features(df)
    mismatch = dict(zip(result["appointment_id"], result["appointment_type_history_mismatch"]))
    assert mismatch["A-1"] == 0  # first appointment: no history
    assert mismatch["A-2"] == 1  # differs from the only prior type


def test_appointment_type_history_mismatch_detects_a_real_match():
    """Second appointment with the SAME type as the prior one should flag 0."""
    df = pd.DataFrame({
        "appointment_id": ["A-1", "A-2"],
        "patient_id": ["P-1", "P-1"],
        "appointment_date": ["1/1/2025", "6/1/2025"],
        "appointment_type": ["General", "General"],
        "appointment_outcome": ["Attended", "Attended"],
    })
    result = feature_engineering.recompute_historical_features(df)
    mismatch = dict(zip(result["appointment_id"], result["appointment_type_history_mismatch"]))
    assert mismatch["A-2"] == 0


def test_week8_optional_features_excluded_by_default_config():
    """Both Week 8 Data-Science-proposed features are disabled by default (tested, did not improve Recall/ROC-AUC)."""
    assert config.INCLUDE_DISTANCE_LEAD_INTERACTION is False
    assert config.INCLUDE_APPOINTMENT_TYPE_HISTORY_MISMATCH is False


def test_encode_categoricals_produces_int_not_bool_columns():
    """
    Regression test for the Week 6 bug where pd.get_dummies() produced
    bool-dtype dummy columns, which validation_checks.validate_feature_matrix()
    (also new in Week 6) correctly rejected as non-numeric. This test
    confirms the fix: every dummy column must be int, not bool.
    """
    df = pd.DataFrame({"gender": ["Male", "Female", "Male"], "age": [30, 40, 50]})
    original_categorical = config.CATEGORICAL_COLUMNS
    try:
        config.CATEGORICAL_COLUMNS = ["gender"]
        result = feature_engineering.encode_categoricals(df)
        dummy_cols = [c for c in result.columns if c.startswith("gender_")]
        assert dummy_cols, "expected at least one gender_* dummy column"
        for col in dummy_cols:
            assert result[col].dtype != bool
            assert pd.api.types.is_integer_dtype(result[col])
    finally:
        config.CATEGORICAL_COLUMNS = original_categorical

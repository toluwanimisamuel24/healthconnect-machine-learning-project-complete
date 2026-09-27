"""
test_negative_scenarios.py
-----------------------------
Week 7 addition. Every other test file in this suite is mostly "positive"
testing: given valid-ish input, does the code produce the right output?
This file is deliberately the opposite: given input that is malformed,
incomplete, or unexpected, does the code fail SAFELY -- with a clear,
specific error -- rather than silently producing wrong results or crashing
with a confusing low-level traceback?

This matters because Weeks 5-6 mostly tested "does this work when given
data shaped like what we currently have." Week 7's brief specifically
calls for testing "invalid or unexpected input scenarios", which is a
different and equally important question: "what happens when the data
ISN'T shaped the way we expect?" -- since real-world data pipelines
eventually receive a file that breaks an assumption someone didn't think
to check.
"""

import pandas as pd
import pytest

from src import data_ingestion, feature_engineering, config
from src.validation_checks import validate_feature_matrix, FeatureValidationError
from src.model import NoShowRiskModel


# ---------------------------------------------------------------------------
# Ingestion: malformed files
# ---------------------------------------------------------------------------

def test_ingestion_rejects_csv_with_extra_unexpected_columns_gracefully(tmp_path):
    """
    Extra, unrecognised columns (e.g. a new field the clinic's booking
    system starts exporting) should NOT crash ingestion -- only genuinely
    MISSING expected columns should. This confirms the pipeline is forward-
    compatible with additive schema changes, a common real-world scenario.
    """
    df = pd.DataFrame({col: [] for col in data_ingestion.EXPECTED_COLUMNS})
    df.loc[0] = ["A-1", "P-1", "Male", 30, "18-30", "General", "1/1/2025", "1/10/2025",
                 "Monday", "09:00", 9, 0, 0, "Yes", "SMS", 5.0, 15.0, "Attended"]
    df["unexpected_new_column"] = ["some new field"]
    path = tmp_path / "extra_column.csv"
    df.to_csv(path, index=False)

    result = data_ingestion.load_raw_data(path)
    assert "unexpected_new_column" in result.columns  # loaded, not silently dropped or crashed on


def test_ingestion_rejects_malformed_csv_with_wrong_delimiter(tmp_path):
    """A file that isn't actually comma-separated should fail with a clear ingestion error, not a cryptic one."""
    path = tmp_path / "wrong_delimiter.csv"
    path.write_text("appointment_id;patient_id;gender\nA-1;P-1;Male\n")

    with pytest.raises(data_ingestion.DataIngestionError, match="missing expected columns"):
        data_ingestion.load_raw_data(path)


def test_ingestion_rejects_completely_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(data_ingestion.DataIngestionError):
        data_ingestion.load_raw_data(path)


def test_ingestion_rejects_header_only_csv(tmp_path):
    """A file with a valid header row but zero data rows should be rejected, not silently produce an empty pipeline run."""
    header = ",".join(data_ingestion.EXPECTED_COLUMNS)
    path = tmp_path / "header_only.csv"
    path.write_text(header + "\n")

    with pytest.raises(data_ingestion.DataIngestionError, match="zero rows"):
        data_ingestion.load_raw_data(path)


# ---------------------------------------------------------------------------
# Feature engineering: missing or malformed columns
# ---------------------------------------------------------------------------

def test_feature_engineering_fails_clearly_when_required_column_missing():
    """
    If a raw-data column feature_engineering depends on (e.g. patient_id,
    needed for point-in-time history) is missing entirely, the failure
    should be a clear pandas KeyError-style signal caught by the pipeline
    wrapper (pipeline.PipelineError), not silently produce wrong features.
    """
    df = pd.DataFrame({
        "appointment_id": ["A-1"], "gender": ["Male"], "age": [30],
        "appointment_date": ["1/1/2025"], "appointment_outcome": ["Attended"],
        # patient_id deliberately omitted
    })

    with pytest.raises(KeyError):
        feature_engineering.recompute_historical_features(df)


def test_feature_engineering_handles_all_rows_being_excluded_label():
    """
    If every row in a batch happens to be 'Cancelled' (the label
    build_target excludes by default), the result should be an empty but
    still well-formed DataFrame, not a crash -- an edge case a small daily
    batch could plausibly hit.
    """
    df = pd.DataFrame({
        "appointment_outcome": ["Cancelled", "Cancelled"],
        "other_column": [1, 2],
    })
    result = feature_engineering.build_target(df)
    assert len(result) == 0
    assert "target" in result.columns


# ---------------------------------------------------------------------------
# Model boundary: schema drift at prediction time
# ---------------------------------------------------------------------------

def test_model_predict_rejects_schema_drift_new_category_at_scoring_time():
    """
    Simulates a realistic production scenario: the model was trained before
    a new appointment_type category existed, and at scoring time a new
    one-hot column appears that the model was never trained on (or an old
    one is missing). This MUST be caught explicitly, not silently
    mishandled by scikit-learn reordering or dropping columns.
    """
    X_train = pd.DataFrame({"age": [30, 40, 50], "type_A": [1, 0, 1], "type_B": [0, 1, 0]})
    y_train = pd.Series([0, 1, 0])
    model = NoShowRiskModel()
    model.fit(X_train, y_train)

    # At scoring time, "type_B" is missing and a new "type_C" appears instead.
    X_score = pd.DataFrame({"age": [35], "type_A": [1], "type_C": [0]})

    with pytest.raises(ValueError, match="do not match training time"):
        model.predict(X_score)


def test_validate_feature_matrix_rejects_scoring_data_with_missing_values():
    """
    A batch of appointments to score should never contain missing values by
    the time it reaches the model -- if it does, that's a sign the
    imputation step was skipped or a new column was added upstream without
    an imputation rule, and this must be caught before it reaches predict().
    """
    X_score = pd.DataFrame({"age": [30, None, 50], "distance": [1.0, 2.0, 3.0]})
    with pytest.raises(FeatureValidationError, match="missing values"):
        validate_feature_matrix(X_score)


def test_validate_feature_matrix_rejects_single_row_with_wrong_dtype():
    """A single appointment (the realistic case for one-off scoring) with a stray text value must still be caught."""
    X_score = pd.DataFrame({"age": ["thirty"], "distance": [5.0]})
    with pytest.raises(FeatureValidationError, match="non-numeric"):
        validate_feature_matrix(X_score)

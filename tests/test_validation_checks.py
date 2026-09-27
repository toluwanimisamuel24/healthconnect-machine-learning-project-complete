import numpy as np
import pandas as pd
import pytest

from src.validation_checks import (
    validate_feature_matrix, validate_predictions,
    FeatureValidationError, PredictionValidationError,
)


def test_validate_feature_matrix_accepts_clean_numeric_data():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4, 5, 6]})
    result = validate_feature_matrix(X)
    assert result["checked"] is True
    assert result["rows"] == 3


def test_validate_feature_matrix_rejects_empty_dataframe():
    X = pd.DataFrame({"a": []})
    with pytest.raises(FeatureValidationError, match="empty"):
        validate_feature_matrix(X)


def test_validate_feature_matrix_rejects_non_numeric_columns():
    """
    Regression test for the Week 6 bug where pd.get_dummies() produced
    bool-dtype columns that this check correctly flagged as non-numeric.
    """
    X = pd.DataFrame({"a": [1, 2, 3], "b": ["yes", "no", "yes"]})
    with pytest.raises(FeatureValidationError, match="non-numeric"):
        validate_feature_matrix(X)


def test_validate_feature_matrix_rejects_bool_columns_explicitly():
    """bool is not a numpy numeric subtype -- must be caught, not silently passed."""
    X = pd.DataFrame({"a": [1, 2, 3], "b": [True, False, True]})
    with pytest.raises(FeatureValidationError, match="non-numeric"):
        validate_feature_matrix(X)


def test_validate_feature_matrix_rejects_missing_values():
    X = pd.DataFrame({"a": [1.0, None, 3.0]})
    with pytest.raises(FeatureValidationError, match="missing values"):
        validate_feature_matrix(X)


def test_validate_feature_matrix_rejects_infinite_values():
    X = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
    with pytest.raises(FeatureValidationError, match="infinite"):
        validate_feature_matrix(X)


def test_validate_feature_matrix_rejects_schema_mismatch():
    X = pd.DataFrame({"a": [1.0], "c": [2.0]})
    with pytest.raises(FeatureValidationError, match="do not match the expected schema"):
        validate_feature_matrix(X, expected_columns=["a", "b"])


def test_validate_feature_matrix_accepts_matching_schema():
    X = pd.DataFrame({"a": [1.0], "b": [2.0]})
    result = validate_feature_matrix(X, expected_columns=["a", "b"])
    assert result["schema_matched"] is True


def test_validate_predictions_accepts_valid_probabilities():
    probs = np.array([0.1, 0.5, 0.9])
    result = validate_predictions(probs, expected_length=3)
    assert result["checked"] is True
    assert result["min"] == 0.1


def test_validate_predictions_rejects_out_of_range_values():
    probs = np.array([0.1, 1.5, 0.9])  # 1.5 is invalid
    with pytest.raises(PredictionValidationError, match="out of the valid"):
        validate_predictions(probs)


def test_validate_predictions_rejects_nan():
    probs = np.array([0.1, np.nan, 0.9])
    with pytest.raises(PredictionValidationError, match="NaN"):
        validate_predictions(probs)


def test_validate_predictions_rejects_length_mismatch():
    probs = np.array([0.1, 0.5])
    with pytest.raises(PredictionValidationError, match="does not match"):
        validate_predictions(probs, expected_length=5)

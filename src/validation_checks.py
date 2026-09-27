"""
validation_checks.py
---------------------
Week 6 addition.

data_validation.py (Week 5) checks the RAW data. This module checks the
boundary between preprocessing and the model -- the feature matrix going
IN, and the predictions coming OUT -- which is exactly where Week 5's
integration gap was: nothing verified that feature_engineering's output was
actually safe to hand to a model, or that a model's output was safe to hand
downstream.

These checks are deliberately cheap and fast (no model training involved)
so they can run on every pipeline execution and on every batch-scoring run
without meaningfully slowing either down.
"""

import numpy as np
import pandas as pd


class FeatureValidationError(Exception):
    """Raised when a feature matrix is not safe to pass to a model."""


class PredictionValidationError(Exception):
    """Raised when a model's output is not safe to pass downstream."""


def validate_feature_matrix(X: pd.DataFrame, expected_columns: list = None) -> dict:
    """
    Validate a feature matrix immediately before it is passed to a model's
    .fit() or .predict()/.predict_proba().

    Checks:
      - X is non-empty.
      - Every column is numeric (a model cannot accept raw text/category dtypes).
      - No missing values remain (imputation should already have handled these
        upstream -- a NaN reaching here means a preprocessing step was skipped
        or a new column was added without an imputation rule).
      - No infinite values (can appear from a division-by-zero in feature
        engineering, e.g. a naive average-spend calculation).
      - If expected_columns is provided, the column SET AND ORDER match
        exactly (order matters for some estimators and is worth catching
        early rather than relying on pandas' looser column-matching).

    Raises
    ------
    FeatureValidationError
        On the first check that fails, with a message specific enough to
        debug without re-running the pipeline in a notebook.

    Returns
    -------
    dict
        A small report of what was checked, for logging/audit purposes.
    """
    if X.empty:
        raise FeatureValidationError("Feature matrix is empty (0 rows) -- nothing to predict on.")

    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise FeatureValidationError(
            f"Feature matrix contains non-numeric columns: {non_numeric}. "
            "Every column must be encoded before reaching the model -- check "
            "whether a new categorical column was added to the raw data without "
            "being added to config.CATEGORICAL_COLUMNS."
        )

    null_counts = X.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0].to_dict()
    if columns_with_nulls:
        raise FeatureValidationError(
            f"Feature matrix contains missing values after preprocessing: {columns_with_nulls}. "
            "This means a column needs an imputation rule in feature_engineering.impute_missing_values()."
        )

    inf_counts = {
        col: int(np.isinf(X[col]).sum())
        for col in X.columns
        if np.isinf(X[col]).sum() > 0
    }
    if inf_counts:
        raise FeatureValidationError(
            f"Feature matrix contains infinite values: {inf_counts}. "
            "Likely a division in feature engineering (e.g. a rate/ratio feature) "
            "with an unguarded zero denominator."
        )

    if expected_columns is not None:
        actual_columns = list(X.columns)
        if actual_columns != list(expected_columns):
            missing = set(expected_columns) - set(actual_columns)
            extra = set(actual_columns) - set(expected_columns)
            raise FeatureValidationError(
                "Feature matrix columns do not match the expected schema. "
                f"Missing: {sorted(missing) or 'none'}. Unexpected: {sorted(extra) or 'none'}. "
                f"Order matches: {actual_columns == list(expected_columns) if not missing and not extra else 'N/A'}."
            )

    return {
        "checked": True,
        "rows": int(len(X)),
        "columns": int(len(X.columns)),
        "all_numeric": True,
        "no_missing_values": True,
        "no_infinite_values": True,
        "schema_matched": expected_columns is not None,
    }


def validate_predictions(probabilities: np.ndarray, expected_length: int = None) -> dict:
    """
    Validate a model's predicted probabilities immediately before they are
    written to the Risk Score Output Store (score_appointments.py) or used
    for evaluation.

    Checks:
      - Every value is between 0 and 1 inclusive (a corrupted or misused
        estimator -- e.g. calling .predict() output through this by mistake
        instead of .predict_proba() -- would fail this immediately).
      - No NaN values.
      - The output length matches the number of input rows, if provided.

    Raises
    ------
    PredictionValidationError
    """
    probabilities = np.asarray(probabilities)

    if np.isnan(probabilities).any():
        raise PredictionValidationError(
            f"Predictions contain {int(np.isnan(probabilities).sum())} NaN value(s)."
        )

    if (probabilities < 0).any() or (probabilities > 1).any():
        raise PredictionValidationError(
            f"Predictions out of the valid [0, 1] probability range: "
            f"min={probabilities.min():.4f}, max={probabilities.max():.4f}. "
            "Check that .predict_proba() output (not .predict() or raw decision "
            "scores) is being passed here."
        )

    if expected_length is not None and len(probabilities) != expected_length:
        raise PredictionValidationError(
            f"Prediction count ({len(probabilities)}) does not match the "
            f"expected input row count ({expected_length})."
        )

    return {
        "checked": True,
        "count": int(len(probabilities)),
        "min": float(probabilities.min()),
        "max": float(probabilities.max()),
        "mean": float(probabilities.mean()),
        "in_valid_range": True,
        "no_nan_values": True,
    }

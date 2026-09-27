"""
backtesting.py
--------------
Week 7 addition.

Week 5 and Week 6 both evaluated every model on a SINGLE time-based
train/test split. That was a reasonable first step (Section "Why a
Time-Based Split" in docs/PROJECT_RATIONALE.md explains why time-based
beats random for this problem) -- but a single split only tells you how a
model performs on one particular slice of the calendar. If HealthConnect
happens to have an unusually quiet or unusually busy few months in that one
test window, the reported metrics could be an artefact of *when* the test
window happened to fall, not a stable property of the model.

This module runs an EXPANDING-WINDOW backtest: multiple train/test splits,
each one using more historical data to train on and testing on a later,
non-overlapping slice of time -- mimicking how the system would actually be
retrained and re-evaluated repeatedly as more real appointments accumulate.
Reporting the MEAN and STANDARD DEVIATION of each metric across folds (not
just one number) answers the question Week 6 could not: is this model's
performance stable over time, or was the Week 6 result a lucky/unlucky
single sample?
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

from .model import NoShowRiskModel
from .validation_checks import validate_feature_matrix, validate_predictions

logger = logging.getLogger(__name__)


def make_expanding_windows(df: pd.DataFrame, date_column: str = "appointment_date",
                            n_splits: int = 4, min_train_fraction: float = 0.4):
    """
    Split a date-ordered DataFrame into `n_splits` expanding-window
    train/test pairs.

    The first `min_train_fraction` of the (date-sorted) data is always in
    every fold's training set. The remaining data is divided into
    `n_splits` equal, non-overlapping, chronologically ordered chunks; fold
    i trains on everything before that chunk and tests on the chunk itself.
    This means fold 2's training set is a strict superset of fold 1's,
    which is what "expanding window" means, and mirrors how a real system
    would be periodically retrained on all data collected so far.

    Yields
    ------
    (fold_index, train_df, test_df) tuples, in chronological order.
    """
    working = df.copy()
    working["_sort_date"] = pd.to_datetime(working[date_column])
    working = working.sort_values("_sort_date").reset_index(drop=True)

    n = len(working)
    min_train_size = int(n * min_train_fraction)
    remaining = n - min_train_size
    fold_size = remaining // n_splits

    if fold_size < 1:
        raise ValueError(
            f"Not enough rows ({n}) to build {n_splits} folds with "
            f"min_train_fraction={min_train_fraction}. Reduce n_splits or min_train_fraction."
        )

    for fold_index in range(n_splits):
        test_start = min_train_size + fold_index * fold_size
        test_end = test_start + fold_size if fold_index < n_splits - 1 else n
        train_df = working.iloc[:test_start].drop(columns=["_sort_date"])
        test_df = working.iloc[test_start:test_end].drop(columns=["_sort_date"])
        yield fold_index + 1, train_df, test_df


def run_backtest(processed_df: pd.DataFrame, estimator_factory, model_name: str,
                  feature_columns: list, n_splits: int = 4,
                  min_train_fraction: float = 0.4) -> dict:
    """
    Run an expanding-window backtest for one model type.

    Parameters
    ----------
    processed_df : the full processed dataset (output of feature_engineering).
    estimator_factory : a zero-argument callable returning a FRESH,
        unfitted scikit-learn estimator each time it's called (a fresh
        instance per fold -- reusing one fitted estimator across folds would
        leak information between folds).
    model_name : label used in logging and the returned report.
    feature_columns : columns to use as model input (excludes id/date/target).
    n_splits, min_train_fraction : passed through to make_expanding_windows().

    Returns
    -------
    dict with per-fold results and mean/std summary statistics.
    """
    fold_results = []

    for fold_index, train_df, test_df in make_expanding_windows(
        processed_df, n_splits=n_splits, min_train_fraction=min_train_fraction
    ):
        X_train, y_train = train_df[feature_columns], train_df["target"]
        X_test, y_test = test_df[feature_columns], test_df["target"]

        validate_feature_matrix(X_train)
        validate_feature_matrix(X_test, expected_columns=list(X_train.columns))

        model = NoShowRiskModel(estimator=estimator_factory())
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        validate_predictions(probabilities, expected_length=len(X_test))

        metrics = {
            "fold": fold_index,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "test_date_range": [
                str(pd.to_datetime(test_df["appointment_date"]).min().date()),
                str(pd.to_datetime(test_df["appointment_date"]).max().date()),
            ],
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities) if len(set(y_test)) > 1 else None,
        }
        fold_results.append(metrics)
        logger.info(
            "%s fold %d/%d (test: %s to %s, n=%d) -> Recall=%.3f, F1=%.3f",
            model_name, fold_index, n_splits, metrics["test_date_range"][0],
            metrics["test_date_range"][1], metrics["test_rows"], metrics["recall"], metrics["f1_score"],
        )

    summary = {}
    for metric_name in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        values = [f[metric_name] for f in fold_results if f[metric_name] is not None]
        if values:
            summary[metric_name] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
                "min": round(float(np.min(values)), 4),
                "max": round(float(np.max(values)), 4),
            }

    return {
        "model_name": model_name,
        "n_splits": n_splits,
        "fold_results": fold_results,
        "summary": summary,
    }

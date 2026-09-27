import pandas as pd

from src.error_analysis import (
    classify_prediction_outcomes, profile_errors_by_feature,
    summarize_error_counts, run_error_analysis,
)


def test_classify_prediction_outcomes_labels_correctly():
    y_true = pd.Series([1, 0, 0, 1])
    y_pred = pd.Series([1, 1, 0, 0])
    # row 0: TP, row 1: FP, row 2: TN, row 3: FN
    outcomes = classify_prediction_outcomes(y_true, y_pred)
    assert list(outcomes) == ["TP", "FP", "TN", "FN"]


def test_summarize_error_counts_totals_match():
    outcomes = pd.Series(["TP", "TP", "FP", "TN", "FN", "FN"])
    summary = summarize_error_counts(outcomes)

    assert summary["total_predictions"] == 6
    assert summary["true_positives"] == 2
    assert summary["false_positives"] == 1
    assert summary["true_negatives"] == 1
    assert summary["false_negatives"] == 2


def test_profile_errors_by_feature_computes_group_means():
    X_test = pd.DataFrame({"lead_days": [10, 20, 30, 40]})
    outcomes = pd.Series(["FN", "FN", "TP", "TP"])

    profile = profile_errors_by_feature(X_test, outcomes, ["lead_days"])

    assert profile["lead_days"]["FN"] == 15.0  # mean of 10, 20
    assert profile["lead_days"]["TP"] == 35.0  # mean of 30, 40


def test_profile_errors_by_feature_skips_missing_columns():
    X_test = pd.DataFrame({"lead_days": [10, 20]})
    outcomes = pd.Series(["FN", "TP"])

    profile = profile_errors_by_feature(X_test, outcomes, ["lead_days", "not_a_real_column"])

    assert "lead_days" in profile
    assert "not_a_real_column" not in profile


def test_run_error_analysis_returns_combined_report():
    X_test = pd.DataFrame({"lead_days": [10, 20, 30, 40]})
    y_test = pd.Series([1, 0, 0, 1])
    y_pred = pd.Series([1, 1, 0, 0])

    report = run_error_analysis(X_test, y_test, y_pred, ["lead_days"])

    assert "error_counts" in report
    assert "feature_profile_by_outcome" in report
    assert report["error_counts"]["total_predictions"] == 4

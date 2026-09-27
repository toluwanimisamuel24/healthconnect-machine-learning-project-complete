"""
error_analysis.py
------------------
Week 6 addition.

Week 5 produced overall metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
for the baseline model but did not look INSIDE those numbers -- this module
does that: it segments the test set by prediction outcome (true positive,
false positive, true negative, false negative) and profiles each group, so
"Recall was 0.627" becomes "the model specifically struggles with
short-notice bookings" or similar -- something a feature-refinement decision
can actually act on.
"""

import pandas as pd


def classify_prediction_outcomes(y_true: pd.Series, y_pred) -> pd.Series:
    """
    Label every row as one of TP / FP / TN / FN based on true vs predicted class.

    TP = correctly predicted No-Show (caught a real no-show)
    FP = predicted No-Show, but the patient attended (a false alarm)
    TN = correctly predicted Attended
    FN = predicted Attended, but the patient was a no-show (a MISSED no-show --
         the costliest error for this business problem, consistent with the
         Week 4/5 rationale for prioritising Recall)
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    outcomes = []
    for true_val, pred_val in zip(y_true, y_pred):
        if true_val == 1 and pred_val == 1:
            outcomes.append("TP")
        elif true_val == 0 and pred_val == 1:
            outcomes.append("FP")
        elif true_val == 0 and pred_val == 0:
            outcomes.append("TN")
        else:
            outcomes.append("FN")
    return pd.Series(outcomes, name="prediction_outcome")


def profile_errors_by_feature(X_test: pd.DataFrame, outcomes: pd.Series, feature_columns: list) -> dict:
    """
    For each requested feature, compare its average value across the four
    prediction-outcome groups. Large gaps between FN/FP and TP/TN on a given
    feature suggest that feature is where the model's errors concentrate --
    a concrete signal for which features might need refining.

    Returns
    -------
    dict
        {feature_name: {"TP": mean, "FP": mean, "TN": mean, "FN": mean}}
    """
    working = X_test.reset_index(drop=True).copy()
    working["prediction_outcome"] = outcomes.values

    report = {}
    for feature in feature_columns:
        if feature not in working.columns:
            continue
        group_means = working.groupby("prediction_outcome")[feature].mean().to_dict()
        report[feature] = {k: round(float(v), 3) for k, v in group_means.items()}
    return report


def summarize_error_counts(outcomes: pd.Series) -> dict:
    counts = outcomes.value_counts().to_dict()
    total = len(outcomes)
    return {
        "total_predictions": int(total),
        "true_positives": int(counts.get("TP", 0)),
        "false_positives": int(counts.get("FP", 0)),
        "true_negatives": int(counts.get("TN", 0)),
        "false_negatives": int(counts.get("FN", 0)),
        "false_negative_rate_of_total": round(counts.get("FN", 0) / total, 4) if total else None,
        "false_positive_rate_of_total": round(counts.get("FP", 0) / total, 4) if total else None,
    }


def run_error_analysis(X_test: pd.DataFrame, y_test: pd.Series, y_pred,
                        profile_features: list) -> dict:
    """
    Run the full Week 6 error analysis: classify every test-set prediction,
    summarize error counts, and profile the requested features across
    outcome groups.
    """
    outcomes = classify_prediction_outcomes(y_test, y_pred)
    return {
        "error_counts": summarize_error_counts(outcomes),
        "feature_profile_by_outcome": profile_errors_by_feature(X_test, outcomes, profile_features),
    }

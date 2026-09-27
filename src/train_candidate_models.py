"""
train_candidate_models.py
---------------------------
Week 6 addition: ML Pipeline Integration & Validation.

Week 5 shipped ONE hardcoded model (Logistic Regression) through
NoShowRiskModel. This script proves the wrapper interface designed in Week 5
is genuinely swappable by training and comparing multiple candidate
estimators through the exact same interface -- this IS the "integrate the
Data Science model/model interface into the pipeline" task for Week 6.

Transparency note (see docs/ML_Pipeline_Integration_&_Validation_Report.docx,
Section "Cross-Track Integration" for the full explanation): this internship
track is being completed solo, without a separate Data Science-track
collaborator producing an independent candidate model. To fulfil the
brief's mandatory cross-track integration requirement honestly rather than
fabricate a hand-off that didn't happen, this script performs the
Data-Science-style candidate modelling work itself (error analysis, feature
refinement, model comparison) and integrates the result into the ML
Engineering pipeline exactly as the brief's integration matrix describes --
clearly labelled throughout as self-performed, not received from another
person.

Usage:
    python -m src.train_candidate_models
"""

import json
import logging

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

from . import config
from .pipeline import run_pipeline
from .model import NoShowRiskModel, time_based_split
from .error_analysis import run_error_analysis
from .validation_checks import validate_feature_matrix, validate_predictions
from .logging_config import setup_logging

logger = logging.getLogger(__name__)

NON_FEATURE_COLUMNS = ["appointment_id", "appointment_date", "target"]


def _evaluate(model: NoShowRiskModel, X_test, y_test) -> dict:
    validate_feature_matrix(X_test, expected_columns=model.feature_names_)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    validate_predictions(probabilities, expected_length=len(X_test))

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }, predictions, probabilities


def main():
    setup_logging()
    logger.info("Step 1/5: Running data pipeline (shared by every candidate model)")
    processed_df, _ = run_pipeline()

    logger.info("Step 2/5: Time-based train/test split")
    train_df, test_df = time_based_split(processed_df)
    feature_columns = [c for c in processed_df.columns if c not in NON_FEATURE_COLUMNS]
    X_train, y_train = train_df[feature_columns], train_df["target"]
    X_test, y_test = test_df[feature_columns], test_df["target"]

    validate_feature_matrix(X_train)
    validate_feature_matrix(X_test, expected_columns=list(X_train.columns))
    logger.info("Input validation passed for both train and test feature matrices.")

    logger.info("Step 3/5: Training and evaluating candidate models")
    candidates = {
        "Logistic Regression (Week 5 baseline)": NoShowRiskModel(),  # unchanged from Week 5
        "Random Forest": NoShowRiskModel(estimator=RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=config.RANDOM_STATE
        )),
        "Gradient Boosting": NoShowRiskModel(estimator=GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=config.RANDOM_STATE
        )),
    }

    results = {}
    predictions_by_model = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        metrics, predictions, probabilities = _evaluate(model, X_test, y_test)
        results[name] = metrics
        predictions_by_model[name] = predictions
        logger.info(
            "%s -> Recall=%.3f, Precision=%.3f, F1=%.3f, ROC-AUC=%.3f",
            name, metrics["recall"], metrics["precision"], metrics["f1_score"], metrics["roc_auc"],
        )

    logger.info("Step 4/5: Selecting best candidate by %s", config.MODEL_SELECTION_METRIC)
    best_name = max(results, key=lambda n: results[n][config.MODEL_SELECTION_METRIC])
    logger.info("Selected candidate: %s (%s=%.3f)", best_name, config.MODEL_SELECTION_METRIC,
                results[best_name][config.MODEL_SELECTION_METRIC])

    logger.info("Step 5/5: Running error analysis on the Week 5 baseline for comparison")
    baseline_predictions = predictions_by_model["Logistic Regression (Week 5 baseline)"]
    profile_features = [
        "age", "booking_lead_days", "distance_to_clinic_km",
        "computed_previous_appointments", "computed_previous_no_shows",
    ]
    error_report = run_error_analysis(X_test, y_test, baseline_predictions, profile_features)
    logger.info("Baseline error counts: %s", error_report["error_counts"])

    # Persist everything.
    config.MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    comparison_report = {
        "candidates_evaluated": list(candidates.keys()),
        "selection_metric": config.MODEL_SELECTION_METRIC,
        "selected_candidate": best_name,
        "results": results,
    }
    with open(config.MODEL_COMPARISON_REPORT_PATH, "w") as f:
        json.dump(comparison_report, f, indent=2, default=str)
    logger.info("Saved model comparison report to %s", config.MODEL_COMPARISON_REPORT_PATH)

    with open(config.ERROR_ANALYSIS_REPORT_PATH, "w") as f:
        json.dump(error_report, f, indent=2, default=str)
    logger.info("Saved error analysis report to %s", config.ERROR_ANALYSIS_REPORT_PATH)

    best_model_path = candidates[best_name].save(config.CANDIDATE_MODEL_PATH)
    logger.info("Saved selected candidate model to %s", best_model_path)

    return comparison_report, error_report


if __name__ == "__main__":
    main()

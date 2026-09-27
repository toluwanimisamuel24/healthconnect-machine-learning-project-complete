"""
train_baseline.py
------------------
End-to-end script: run the data pipeline, time-split the result, train the
NoShowRiskModel baseline, evaluate it, and save both the model artifact and
a metrics report.

This exists to prove the Week 4 architecture works as a connected system,
not just as separate modules -- and to give the Data Science track a
concrete, reproducible reference point to compare their own modelling work
against.

Usage:
    python -m src.train_baseline
"""

import json
import logging

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

from . import config
from .pipeline import run_pipeline
from .model import NoShowRiskModel, time_based_split
from .logging_config import setup_logging

logger = logging.getLogger(__name__)

NON_FEATURE_COLUMNS = ["appointment_id", "appointment_date", "target"]


def main():
    setup_logging()
    logger.info("Running data pipeline")
    processed_df, _ = run_pipeline()

    logger.info("Splitting train/test sets by appointment_date (time-based)")
    train_df, test_df = time_based_split(processed_df)
    logger.info(
        "Train: %d rows (%s to %s) | Test: %d rows (%s to %s)",
        len(train_df), train_df["appointment_date"].min(), train_df["appointment_date"].max(),
        len(test_df), test_df["appointment_date"].min(), test_df["appointment_date"].max(),
    )

    feature_columns = [c for c in processed_df.columns if c not in NON_FEATURE_COLUMNS]
    X_train, y_train = train_df[feature_columns], train_df["target"]
    X_test, y_test = test_df[feature_columns], test_df["target"]

    logger.info("Training baseline model (Logistic Regression, class_weight=balanced)")
    model = NoShowRiskModel()
    model.fit(X_train, y_train)

    logger.info("Evaluating on held-out test set")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_date_range": [str(train_df["appointment_date"].min()), str(train_df["appointment_date"].max())],
        "test_date_range": [str(test_df["appointment_date"].min()), str(test_df["appointment_date"].max())],
    }

    logger.info("Baseline metrics: %s", {k: v for k, v in metrics.items() if k != "confusion_matrix"})

    model_path = model.save()
    logger.info("Saved model artifact to %s", model_path)

    config.MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_REPORT_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Saved metrics report to %s", config.METRICS_REPORT_PATH)

    return metrics


if __name__ == "__main__":
    main()

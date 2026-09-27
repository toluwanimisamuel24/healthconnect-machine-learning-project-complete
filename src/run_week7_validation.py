"""
run_week7_validation.py
--------------------------
Week 7 addition: orchestrates every Week 7 testing/validation activity in
one script and saves each result as evidence, mirroring the pattern
train_candidate_models.py established in Week 6.

Runs, in order:
1. Expanding-window backtest for the Week 6 candidate (Random Forest) AND
   the Week 5 baseline (Logistic Regression), to check performance stability
   over time for both, not just the previously-selected model.
2. The reminder-feature circularity ablation (Week 4's unresolved question).
3. A SHAP explanation for a handful of real, high-risk scored appointments,
   as a demonstration that the explainability layer works against a real
   saved model, not just a synthetic fixture.

Usage:
    python -m src.run_week7_validation
"""

import json
import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from . import config
from .pipeline import run_pipeline
from .model import NoShowRiskModel, time_based_split
from .backtesting import run_backtest
from .reminder_feature_analysis import run_reminder_ablation
from .explainability import build_explainer, explain_single_appointment, SHAP_AVAILABLE
from .logging_config import setup_logging

logger = logging.getLogger(__name__)

NON_FEATURE_COLUMNS = ["appointment_id", "appointment_date", "target"]

BACKTEST_REPORT_PATH = config.MODEL_ARTIFACT_DIR / "week7_backtest_report.json"
REMINDER_ABLATION_REPORT_PATH = config.MODEL_ARTIFACT_DIR / "week7_reminder_ablation_report.json"
EXPLANATION_EXAMPLES_PATH = config.MODEL_ARTIFACT_DIR / "week7_explanation_examples.json"


def main():
    setup_logging()
    config.MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Running data pipeline")
    processed_df, _ = run_pipeline()
    feature_columns = [c for c in processed_df.columns if c not in NON_FEATURE_COLUMNS]

    # --- 1. Expanding-window backtest ---------------------------------
    logger.info("=" * 70)
    logger.info("STEP 1: Expanding-window backtest (Recall stability over time)")
    logger.info("=" * 70)

    backtest_results = {}
    backtest_results["Random Forest (Week 6 candidate)"] = run_backtest(
        processed_df,
        lambda: RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                                        random_state=config.RANDOM_STATE),
        "Random Forest", feature_columns, n_splits=4,
    )
    backtest_results["Logistic Regression (Week 5 baseline)"] = run_backtest(
        processed_df,
        lambda: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE),
        "Logistic Regression", feature_columns, n_splits=4,
    )

    with open(BACKTEST_REPORT_PATH, "w") as f:
        json.dump(backtest_results, f, indent=2, default=str)
    logger.info("Saved backtest report to %s", BACKTEST_REPORT_PATH)

    # --- 2. Reminder feature ablation -----------------------------------
    logger.info("=" * 70)
    logger.info("STEP 2: Reminder feature circularity ablation")
    logger.info("=" * 70)

    reminder_report = run_reminder_ablation(
        processed_df,
        lambda: RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                                        random_state=config.RANDOM_STATE),
    )
    with open(REMINDER_ABLATION_REPORT_PATH, "w") as f:
        json.dump(reminder_report, f, indent=2, default=str)
    logger.info("Saved reminder ablation report to %s", REMINDER_ABLATION_REPORT_PATH)
    logger.info("Decision: %s", reminder_report["decision"])

    # --- 3. SHAP explanations on real scored appointments ----------------
    logger.info("=" * 70)
    logger.info("STEP 3: SHAP explanations for sample high-risk appointments")
    logger.info("=" * 70)

    if not SHAP_AVAILABLE:
        logger.warning("shap not installed -- skipping explanation examples.")
    else:
        train_df, test_df = time_based_split(processed_df)
        model = NoShowRiskModel.load(config.CANDIDATE_MODEL_PATH)
        explainer = build_explainer(model, train_df[feature_columns])

        probabilities = model.predict_proba(test_df[feature_columns])
        test_df = test_df.reset_index(drop=True)
        top_risk_indices = pd.Series(probabilities).sort_values(ascending=False).head(3).index

        examples = []
        for idx in top_risk_indices:
            row = test_df[feature_columns].iloc[[idx]]
            explanation = explain_single_appointment(explainer, model, row)
            explanation["appointment_id"] = test_df["appointment_id"].iloc[idx]
            examples.append(explanation)
            logger.info("Appointment %s: probability=%.3f, top risk driver=%s",
                        explanation["appointment_id"], explanation["predicted_probability"],
                        explanation["top_features_increasing_risk"][0]["feature"]
                        if explanation["top_features_increasing_risk"] else "none")

        with open(EXPLANATION_EXAMPLES_PATH, "w") as f:
            json.dump(examples, f, indent=2, default=str)
        logger.info("Saved explanation examples to %s", EXPLANATION_EXAMPLES_PATH)

    logger.info("=" * 70)
    logger.info("Week 7 validation complete.")
    logger.info("=" * 70)

    return backtest_results, reminder_report


if __name__ == "__main__":
    main()

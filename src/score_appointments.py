"""
score_appointments.py
------------------------
Week 6 addition: implements the "Batch Scoring Service" and "Risk Score
Output Store" stages from the Week 4 architecture diagram, which Week 5's
own documented Next Steps identified as not yet built.

Loads a saved model (the Week 6 integrated candidate, by default), runs the
same data pipeline used for training, validates the resulting feature
matrix, scores every appointment, assigns a risk tier, validates the
predictions, and writes a traceable output file -- one row per appointment,
carrying the appointment_id, probability, tier, timestamp, and which model
version produced it, exactly as specified in the Week 4 design document's
Section 4.4.

Usage:
    python -m src.score_appointments
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config
from .pipeline import run_pipeline
from .model import NoShowRiskModel
from .validation_checks import validate_feature_matrix, validate_predictions
from .logging_config import setup_logging

logger = logging.getLogger(__name__)

NON_FEATURE_COLUMNS = ["appointment_id", "appointment_date", "target"]


def assign_risk_tier(probability: float) -> str:
    """Map a probability to the risk tiers defined in config.RISK_TIER_THRESHOLDS."""
    for tier, (lower, upper) in config.RISK_TIER_THRESHOLDS.items():
        if lower <= probability < upper:
            return tier
    return "Unclassified"  # should be unreachable if thresholds are configured correctly


def score_appointments(model_path=None) -> pd.DataFrame:
    """
    Run the full batch scoring flow and return the scored DataFrame.

    Note: this dataset has no genuinely "future/upcoming" appointments to
    score (it is a fixed historical file), so -- for this Week 6
    demonstration only -- the pipeline's own held-out time-based test
    segment stands in for "the next batch of appointments awaiting a risk
    score". In a live deployment this function would instead be pointed at
    a fresh extract of not-yet-occurred appointments.
    """
    model_path = model_path or config.CANDIDATE_MODEL_PATH
    setup_logging()
    logger.info("Loading model from %s", model_path)
    model = NoShowRiskModel.load(model_path)

    logger.info("Running data pipeline to obtain a scorable feature set")
    processed_df, _ = run_pipeline(save_output=False)

    from .model import time_based_split
    _, to_score_df = time_based_split(processed_df)
    logger.info("Scoring %d appointments (stand-in 'upcoming' batch -- see docstring)", len(to_score_df))

    feature_columns = [c for c in processed_df.columns if c not in NON_FEATURE_COLUMNS]
    X = to_score_df[feature_columns]

    validate_feature_matrix(X, expected_columns=model.feature_names_)
    probabilities = model.predict_proba(X)
    validate_predictions(probabilities, expected_length=len(X))
    logger.info("Prediction validation passed: probabilities in [%.3f, %.3f]",
                probabilities.min(), probabilities.max())

    scored = pd.DataFrame({
        "appointment_id": to_score_df["appointment_id"].values,
        "appointment_date": to_score_df["appointment_date"].values,
        "no_show_probability": probabilities.round(4),
    })
    scored["risk_tier"] = scored["no_show_probability"].apply(assign_risk_tier)
    scored["scored_at"] = datetime.now(timezone.utc).isoformat()
    scored["model_version"] = Path(model_path).name

    tier_counts = scored["risk_tier"].value_counts().to_dict()
    logger.info("Risk tier distribution: %s", tier_counts)

    config.SCORED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(config.SCORED_APPOINTMENTS_PATH, index=False)
    logger.info("Saved scored appointments to %s", config.SCORED_APPOINTMENTS_PATH)

    return scored


if __name__ == "__main__":
    score_appointments()

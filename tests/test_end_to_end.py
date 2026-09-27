"""
test_end_to_end.py
-------------------
Week 7 addition, explicitly promised in Week 6's own "Next Steps": an
integration test that runs the FULL pipeline -> train -> score flow in one
test, rather than only testing each stage in isolation (which is what every
other test file in this suite does).

This is deliberately different in spirit from the rest of the test suite:
it uses the REAL raw dataset (not a small hand-built fixture) and exercises
every module together, because its entire purpose is to catch the class of
bug that only appears when real data flows through every stage connected
together -- precisely the class of bug Week 6 found twice (see
docs/ML_Pipeline_Integration_Validation_Report.docx, Section 4). It is
slower than the rest of the suite by design (a few seconds, not
milliseconds) because it is standing in for "does the whole system actually
work," not "is this one function correct."
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.pipeline import run_pipeline
from src.model import NoShowRiskModel, time_based_split
from src.validation_checks import validate_feature_matrix, validate_predictions
from src.score_appointments import assign_risk_tier
from src import config

NON_FEATURE_COLUMNS = ["appointment_id", "appointment_date", "target"]


def test_full_pipeline_to_scoring_flow_produces_valid_output():
    """
    The single most important test in this repository: does raw data,
    flowing through every real stage (ingestion, validation, feature
    engineering, split, training, scoring, risk tiering), end up as a
    valid, usable result -- with no stage silently producing garbage that
    a later stage doesn't catch?
    """
    # Stage 1-3: ingestion, validation, feature engineering (via the pipeline).
    processed_df, validation_report = run_pipeline(save_output=False)
    assert len(processed_df) > 0
    assert "target" in processed_df.columns

    # Stage 4: time-based split.
    train_df, test_df = time_based_split(processed_df)
    assert len(train_df) > 0 and len(test_df) > 0
    # train must be entirely before test, chronologically (see backtesting.py rationale)
    assert pd.to_datetime(train_df["appointment_date"]).max() <= pd.to_datetime(test_df["appointment_date"]).min()

    feature_columns = [c for c in processed_df.columns if c not in NON_FEATURE_COLUMNS]
    X_train, y_train = train_df[feature_columns], train_df["target"]
    X_test, y_test = test_df[feature_columns], test_df["target"]

    # Stage 5: the input-validation boundary that Week 6 added specifically
    # because nothing checked it before.
    validate_feature_matrix(X_train)
    validate_feature_matrix(X_test, expected_columns=list(X_train.columns))

    # Stage 6: model training, through the same wrapper interface used
    # throughout the rest of the codebase (not a special test-only path).
    model = NoShowRiskModel(estimator=RandomForestClassifier(
        n_estimators=50, max_depth=8, class_weight="balanced", random_state=config.RANDOM_STATE
    ))
    model.fit(X_train, y_train)

    # Stage 7: prediction + output validation.
    probabilities = model.predict_proba(X_test)
    validate_predictions(probabilities, expected_length=len(X_test))

    # Stage 8: risk tiering -- the actual final output a clinic user would see.
    tiers = pd.Series(probabilities).apply(assign_risk_tier)
    assert set(tiers.unique()).issubset(set(config.RISK_TIER_THRESHOLDS.keys()))
    assert len(tiers) == len(X_test)

    # Sanity checks on the shape of the whole journey: nothing should have
    # silently dropped or duplicated rows between stages.
    assert len(train_df) + len(test_df) == len(processed_df)
    assert len(tiers) == len(y_test)


def test_full_pipeline_is_reproducible_across_two_runs():
    """
    Running the pipeline twice, with the same random seed, must produce
    identical processed data and identical model predictions -- otherwise
    "reproducibility", claimed throughout this repository's documentation,
    would be an untested assertion rather than a verified one.
    """
    processed_df_1, _ = run_pipeline(save_output=False)
    processed_df_2, _ = run_pipeline(save_output=False)

    pd.testing.assert_frame_equal(
        processed_df_1.sort_values("appointment_id").reset_index(drop=True),
        processed_df_2.sort_values("appointment_id").reset_index(drop=True),
    )

    feature_columns = [c for c in processed_df_1.columns if c not in NON_FEATURE_COLUMNS]
    train_df_1, test_df_1 = time_based_split(processed_df_1)
    train_df_2, test_df_2 = time_based_split(processed_df_2)

    model_1 = NoShowRiskModel(estimator=RandomForestClassifier(
        n_estimators=50, max_depth=8, class_weight="balanced", random_state=config.RANDOM_STATE
    ))
    model_2 = NoShowRiskModel(estimator=RandomForestClassifier(
        n_estimators=50, max_depth=8, class_weight="balanced", random_state=config.RANDOM_STATE
    ))
    model_1.fit(train_df_1[feature_columns], train_df_1["target"])
    model_2.fit(train_df_2[feature_columns], train_df_2["target"])

    preds_1 = model_1.predict_proba(test_df_1[feature_columns])
    preds_2 = model_2.predict_proba(test_df_2[feature_columns])
    assert np.allclose(preds_1, preds_2), "Same seed must produce identical predictions"

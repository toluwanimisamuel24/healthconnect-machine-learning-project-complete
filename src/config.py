"""
config.py
---------
Central configuration for the HealthConnect no-show prediction pipeline.

Keeping every path, column list, and modelling assumption in one place means
the rest of the pipeline never hard-codes a column name or a business rule --
if the Data Science track changes the target definition, or a new column is
added to the raw dataset, this is the only file that should need editing.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "HealthConnect_Appointment_Data.csv"
DATA_DICTIONARY_PATH = REPO_ROOT / "data" / "raw" / "HealthConnect_Data_Dictionary.csv"
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "healthconnect_processed.csv"
VALIDATION_REPORT_PATH = PROCESSED_DATA_DIR / "data_validation_report.json"
MODEL_ARTIFACT_DIR = REPO_ROOT / "models"
MODEL_ARTIFACT_PATH = MODEL_ARTIFACT_DIR / "baseline_logistic_regression.joblib"
METRICS_REPORT_PATH = MODEL_ARTIFACT_DIR / "baseline_metrics.json"

# Week 6 additions: candidate model (integration target) and batch scoring
# output. Kept as SEPARATE paths from the Week 5 baseline artifact above --
# the baseline is retained as a comparison point, never overwritten.
CANDIDATE_MODEL_PATH = MODEL_ARTIFACT_DIR / "candidate_random_forest.joblib"
MODEL_COMPARISON_REPORT_PATH = MODEL_ARTIFACT_DIR / "model_comparison_report.json"
ERROR_ANALYSIS_REPORT_PATH = MODEL_ARTIFACT_DIR / "error_analysis_report.json"

SCORED_OUTPUT_DIR = REPO_ROOT / "data" / "scored"
SCORED_APPOINTMENTS_PATH = SCORED_OUTPUT_DIR / "appointment_risk_scores.csv"

LOG_DIR = REPO_ROOT / "logs"
PIPELINE_LOG_PATH = LOG_DIR / "pipeline_run.log"

# ---------------------------------------------------------------------------
# Column roles
# ---------------------------------------------------------------------------
ID_COLUMNS = ["appointment_id", "patient_id"]

# Outcome column in the raw data, and the labels it can take.
OUTCOME_COLUMN = "appointment_outcome"
OUTCOME_LABELS = ["Attended", "No-Show", "Cancelled"]

# Target definition (Machine Learning Engineering default -- the Data Science
# track owns this decision and may override it; see docs/ML_Pipeline_Documentation.docx
# Section "Target Variable Handling" for the reasoning).
#   - Positive class (1): appointment_outcome == "No-Show"
#   - Negative class (0): appointment_outcome == "Attended"
#   - "Cancelled" rows are EXCLUDED from the primary binary classifier by
#     default, since a cancellation is a deliberate, proactive patient action
#     with different drivers to a silent no-show. This is configurable below.
TARGET_POSITIVE_LABEL = "No-Show"
TARGET_NEGATIVE_LABEL = "Attended"
EXCLUDE_LABELS_FROM_TRAINING = ["Cancelled"]

# Columns that must NEVER be used as model INPUT features, and why. Note:
# appointment_outcome is intentionally NOT listed here -- it is the source of
# the target label and is dropped separately, after build_target() has used
# it, by run_feature_pipeline(). Mixing it into this list caused a
# double-drop bug during Week 5 pipeline testing (see tests/test_feature_engineering.py).
# This list is the single source of truth for the leakage guard implemented
# in data_validation.py and enforced again in feature_engineering.py.
LEAKAGE_COLUMNS = {
    "waiting_time_minutes": (
        "Populated even for No-Show and Cancelled appointments in the raw data, "
        "which is not physically possible (no wait time can accrue for a visit "
        "that never happened). This is a post-appointment field and a confirmed "
        "leakage risk -- excluded unconditionally."
    ),
}

# Columns that are identifiers only and must not be passed to a model as raw
# input (they carry no generalisable signal and risk the model memorising
# individual IDs rather than learning behaviour patterns).
NON_FEATURE_ID_COLUMNS = ["appointment_id", "patient_id"]

# Raw historical columns provided in the dataset. Week 4/5 investigation found
# these are NOT chronologically consistent for the majority of patients (see
# feature_engineering.recompute_historical_features), so the pipeline
# recomputes point-in-time-correct replacements instead of trusting them.
RAW_HISTORICAL_COLUMNS = ["previous_appointments", "previous_no_shows"]

# Categorical columns that require encoding before modelling.
# NOTE: booking_lead_bucket is engineered by feature_engineering.py (it does
# not exist in the raw data) but is listed here too since encode_categoricals()
# runs after it has been created -- keeping ALL categorical columns in one
# list, regardless of when they were created, avoids a class of "forgot to
# encode the new column" bugs.
CATEGORICAL_COLUMNS = [
    "gender",
    "age_group",
    "appointment_type",
    "appointment_day",
    "appointment_time",
    "reminder_sent",
    "reminder_channel",
    "booking_lead_bucket",
]

# Numeric columns that pass through largely unchanged (after any imputation).
NUMERIC_COLUMNS = [
    "age",
    "booking_lead_days",
    "distance_to_clinic_km",
]

# Date columns used for point-in-time feature computation and train/test
# splitting, but not passed to the model directly in raw string form.
DATE_COLUMNS = ["booking_date", "appointment_date"]

# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------
# Time-based split: train on the earlier portion of appointment_date, test on
# the most recent portion. See docs/ML_Pipeline_Documentation.docx for the
# rationale (a random split risks the model implicitly learning from future
# dates when "predicting" the past).
TEST_SPLIT_FRACTION = 0.2

# ---------------------------------------------------------------------------
# Missing-value handling
# ---------------------------------------------------------------------------
# distance_to_clinic_km: ~1.8% missing in Week 4 review -- median imputation
# is a reasonable, low-risk default for a roughly symmetric numeric column.
NUMERIC_IMPUTE_STRATEGY = "median"

# reminder_channel is null exactly when reminder_sent == "No" (structural,
# not missing-at-random) -- filled with the explicit label below rather than
# imputed, so the model can distinguish "no reminder sent" from a genuine gap.
REMINDER_CHANNEL_FILL_VALUE = "No Reminder Sent"

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Week 6: Risk tiers (batch scoring output)
# ---------------------------------------------------------------------------
# Thresholds are a starting point for Week 6 integration testing, NOT a
# final clinical/operational decision -- Week 4's design deliberately left
# tier cutoffs open pending business input on intervention cost/capacity.
# Documented here, in one place, so Week 7 can tune them without touching
# scoring code.
RISK_TIER_THRESHOLDS = {
    "Low": (0.0, 0.40),
    "Medium": (0.40, 0.65),
    "High": (0.65, 1.01),  # 1.01 so a probability of exactly 1.0 is included
}

# ---------------------------------------------------------------------------
# Week 6: Candidate model registry
# ---------------------------------------------------------------------------
# Named estimators that train_candidate_models.py trains and compares against
# the Week 5 Logistic Regression baseline, all through the same
# NoShowRiskModel interface. Adding a new candidate here (e.g. once the Data
# Science track proposes one) does not require touching any other file.
CANDIDATE_MODEL_NAMES = ["Logistic Regression (Week 5 baseline)", "Random Forest", "Gradient Boosting"]

# Primary metric for selecting the Week 6 candidate model, consistent with
# the Week 4/5 rationale: Recall on the No-Show class is prioritised because
# missing a real no-show (a wasted appointment slot with zero intervention)
# is costlier than one unnecessary extra reminder to a patient who was going
# to attend anyway.
MODEL_SELECTION_METRIC = "recall"

# ---------------------------------------------------------------------------
# Week 7: Reminder feature governance note
# ---------------------------------------------------------------------------
# Resolved in src/reminder_feature_analysis.py: reminder_sent/reminder_channel
# are kept as model features (they are not leakage for the current dataset;
# removing them measurably costs Recall). The unresolved part of the Week 4
# circularity concern is forward-looking only: if HealthConnect's reminder
# process ever becomes driven by this model's own risk scores, future
# retraining must account for that feedback loop (e.g. by freezing the
# feature definition to a pre-model policy period, or excluding the feature
# and re-validating). Not an issue for the current retrospective dataset.
REMINDER_FEATURE_GOVERNANCE_NOTE = (
    "reminder_sent/reminder_channel reflect HealthConnect's existing, "
    "model-independent reminder process and are safe to use today for this "
    "retrospective model (confirmed by the Data Science track, Week 8 "
    "response to Question 2). Refined framing per that response: the "
    "original concern was not leakage in the strict post-appointment sense, "
    "but whether these fields are a SYMPTOM of staff already anticipating a "
    "no-show rather than a CAUSE of the outcome -- a genuine circularity "
    "question distinct from the waiting_time_minutes leakage case. Re-review "
    "this decision as a monitoring item if HealthConnect's reminder policy "
    "ever becomes driven by this model's own output, since that would break "
    "the model's learned relationship between reminders and outcomes."
)

# ---------------------------------------------------------------------------
# Week 8: Data Science-proposed features (see
# docs/collaboration/DataScience_Response_to_Handoff.docx, Question 4)
# ---------------------------------------------------------------------------
# Both features were implemented and empirically tested via a 4-window
# backtest against the Random Forest candidate (see
# docs/collaboration/ML_Pipeline_Final_Integration_Report.docx, "Data
# Science Feature Proposals" section, for full fold-by-fold detail). Neither
# improved mean Recall or ROC-AUC over the existing feature set; both are
# excluded by default as a result, per the following measured comparison:
#   Neither feature (current default):        mean Recall 0.6381, ROC-AUC 0.6493
#   + distance_lead_interaction:               mean Recall 0.6361, ROC-AUC 0.6494
#   + appointment_type_history_mismatch:       mean Recall 0.6347, ROC-AUC 0.6482
#   + both:                                    mean Recall 0.6319, ROC-AUC 0.6485
# Both flags can be flipped to True to re-include a feature if a future,
# larger dataset changes this result.
INCLUDE_DISTANCE_LEAD_INTERACTION = False  # tested: no Recall/ROC-AUC improvement; excluded by default
INCLUDE_APPOINTMENT_TYPE_HISTORY_MISMATCH = False  # tested: no Recall/ROC-AUC improvement; excluded by default

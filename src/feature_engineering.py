"""
feature_engineering.py
------------------------
Transforms validated raw data into a model-ready feature table.

Two engineering decisions live here that were driven directly by findings
from data_validation.py, not just convention:

1. previous_appointments / previous_no_shows are recomputed from the
   dataset itself (point-in-time, per patient) instead of trusting the raw
   columns, because check_historical_field_consistency() showed the raw
   columns are chronologically inconsistent for ~59% of patients.

2. waiting_time_minutes (and any other column in config.LEAKAGE_COLUMNS)
   is dropped unconditionally before the feature matrix is built, enforcing
   the Week 4 leakage finding in code rather than relying on someone
   remembering to exclude it by hand.
"""

import pandas as pd

from . import config


def recompute_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the raw previous_appointments / previous_no_shows columns with
    point-in-time-correct versions computed from the dataset itself, and
    (Week 8 addition, at the Data Science track's request -- see
    docs/collaboration/DataScience_Response_to_Handoff.docx, Question 4.2)
    compute a point-in-time appointment_type history mismatch flag in the
    same pass, since both need the same per-patient, chronologically-sorted
    iteration.

    Assumption (documented, not hidden): this dataset is treated as each
    patient's complete appointment history. For every appointment, the
    recomputed count only looks at that same patient's OTHER rows with an
    earlier appointment_date -- never at rows dated on or after the
    appointment being scored, which is what keeps this leakage-free.

    Returns a copy of df with three new columns:
        computed_previous_appointments
        computed_previous_no_shows
        appointment_type_history_mismatch -- 1 if this appointment_type
            differs from the patient's own most common PRIOR appointment_type,
            0 if it matches, and 0 for a patient's first appointment (no
            history exists yet -- zero-flagged rather than imputed, per DS's
            explicit instruction not to manufacture signal for first-timers).
    """
    working = df.copy()
    working["_appointment_date_dt"] = pd.to_datetime(working["appointment_date"])
    working = working.sort_values(["patient_id", "_appointment_date_dt"]).reset_index(drop=True)

    computed_prev_appts = []
    computed_prev_noshows = []
    type_mismatch_flags = []

    for _, group in working.groupby("patient_id", sort=False):
        seen_appts = 0
        seen_noshows = 0
        seen_types = []
        for _, row in group.iterrows():
            computed_prev_appts.append(seen_appts)
            computed_prev_noshows.append(seen_noshows)

            if not seen_types:
                type_mismatch_flags.append(0)  # first appointment: no history, zero-flagged
            else:
                counts = pd.Series(seen_types).value_counts()
                most_common_type = counts.index[0]
                type_mismatch_flags.append(int(row["appointment_type"] != most_common_type))

            seen_appts += 1
            if row[config.OUTCOME_COLUMN] == config.TARGET_POSITIVE_LABEL:
                seen_noshows += 1
            seen_types.append(row["appointment_type"])

    working["computed_previous_appointments"] = computed_prev_appts
    working["computed_previous_no_shows"] = computed_prev_noshows
    working["appointment_type_history_mismatch"] = type_mismatch_flags

    return working.drop(columns=["_appointment_date_dt"])


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every column listed in config.LEAKAGE_COLUMNS, if present."""
    columns_to_drop = [c for c in config.LEAKAGE_COLUMNS if c in df.columns]
    return df.drop(columns=columns_to_drop)


def add_booking_lead_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket booking_lead_days into a small number of business-readable bands.
    Kept alongside the raw numeric value (not a replacement for it) so both
    a linear model and a bucket-aware model can use whichever suits them.
    """
    working = df.copy()
    working["booking_lead_bucket"] = pd.cut(
        working["booking_lead_days"],
        bins=[-1, 2, 7, 14, 30, 10_000],
        labels=["0-2 days", "3-7 days", "8-14 days", "15-30 days", "30+ days"],
    )
    return working


def add_short_lead_high_risk_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Week 6 feature, added on evidence from error_analysis.py, not by default
    convention.

    The Week 6 error analysis on the Week 5 baseline showed the model
    separates outcomes almost entirely along booking_lead_days (true
    positives averaged 45.3 lead days; true negatives averaged 15.0) --
    but for SHORT-lead-time appointments specifically, missed no-shows
    (false negatives, avg 17.3 lead days) and correctly predicted
    attendances (true negatives, avg 15.0 lead days) were nearly
    indistinguishable on that axis alone. In other words: booking_lead_days
    is doing most of the separating work, and it runs out of signal
    precisely in the short-lead-time regime where the model needs it most.

    computed_previous_no_shows was directionally useful in that same
    regime (false negatives averaged a higher prior-no-show count than true
    negatives: 1.256 vs 1.090) but too weak a signal on its own to move the
    metrics. This feature makes that interaction explicit and easy for a
    model to use directly, instead of requiring it to discover the
    combination itself: a flag for "booked on short notice AND has a
    no-show history", which is precisely the model's current blind spot.
    """
    working = df.copy()
    working["short_lead_high_risk_flag"] = (
        (working["booking_lead_days"] <= 14) & (working["computed_previous_no_shows"] >= 1)
    ).astype(int)
    return working


def add_distance_lead_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Week 8 feature, proposed by the Data Science track (Question 4.1 in
    docs/collaboration/DataScience_Response_to_Handoff.docx), NOT added by
    default convention -- validated below before being kept.

    Rationale as proposed: a late booking (short lead time) combined with a
    long commute (above-median distance) compounds two independent friction
    factors that treating each feature separately might not capture as a
    combined effect.

    Validation result (see docs/collaboration/ML_Pipeline_Final_Integration_Report.docx,
    "Data Science Feature Proposals" section, for full detail): checked
    whether the baseline model's false negatives cluster on this flag, per
    the Data Science track's own requested method. They do not -- the flag's
    rate among false negatives (8.2%) is in fact slightly LOWER than among
    true negatives (10.4%), the opposite of a false-negative-clustering
    pattern, and the flag is entirely absent from the false-positive and
    true-positive groups in the test window. A subsequent empirical
    before/after backtest (also reported in that section) confirms this:
    including the feature does not improve Recall or ROC-AUC over excluding
    it. Kept in this module, disabled by default via
    config.INCLUDE_DISTANCE_LEAD_INTERACTION, so the negative result remains
    reproducible and visible rather than silently discarded.
    """
    working = df.copy()
    median_distance = working["distance_to_clinic_km"].median()
    working["distance_lead_interaction"] = (
        (working["distance_to_clinic_km"] > median_distance) & (working["booking_lead_days"] <= 7)
    ).astype(int)
    return working


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the imputation rules defined in config.py:
      - distance_to_clinic_km: median imputation (roughly symmetric numeric column)
      - reminder_channel: filled with an explicit "No Reminder Sent" label,
        since it is missing structurally (whenever reminder_sent == "No"),
        not at random -- imputing a channel would misrepresent the data.
    """
    working = df.copy()

    if "distance_to_clinic_km" in working.columns:
        median_value = working["distance_to_clinic_km"].median()
        working["distance_to_clinic_km"] = working["distance_to_clinic_km"].fillna(median_value)

    if "reminder_channel" in working.columns:
        working["reminder_channel"] = working["reminder_channel"].fillna(
            config.REMINDER_CHANNEL_FILL_VALUE
        )

    return working


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the binary target column from appointment_outcome, excluding the
    labels in config.EXCLUDE_LABELS_FROM_TRAINING (Cancelled, by default).

    Rows with an excluded label are dropped from the returned frame -- this
    function defines the modelling population, not just the label column.
    """
    working = df.copy()
    working = working[~working[config.OUTCOME_COLUMN].isin(config.EXCLUDE_LABELS_FROM_TRAINING)].copy()
    working["target"] = (working[config.OUTCOME_COLUMN] == config.TARGET_POSITIVE_LABEL).astype(int)
    return working


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode every column listed in config.CATEGORICAL_COLUMNS that is
    present, then cast the resulting dummy columns to int.

    Week 6 fix: pandas' get_dummies() returns bool-dtype columns by default
    in current versions. bool is NOT a subtype of numpy's numeric type
    (np.issubdtype(bool, np.number) is False), so validation_checks.
    validate_feature_matrix() -- added in Week 6 specifically to catch
    exactly this kind of "looks numeric but isn't" issue -- correctly
    flagged every one-hot column as non-numeric the first time the
    integrated pipeline was run end-to-end. scikit-learn happens to accept
    bool columns silently, which is exactly why this went unnoticed in
    Week 5: nothing was checking the feature matrix's dtypes explicitly.
    Casting to int here fixes it at the source rather than loosening the
    validation check.
    """
    columns_present = [c for c in config.CATEGORICAL_COLUMNS if c in df.columns]
    encoded = pd.get_dummies(df, columns=columns_present, drop_first=True)

    dummy_columns = [c for c in encoded.columns if c not in df.columns]
    encoded[dummy_columns] = encoded[dummy_columns].astype(int)

    return encoded


def run_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering sequence in the order that matters:

    1. Recompute point-in-time historical features (needs the raw outcome
       column and raw dates, so must run before anything else). This now
       also computes appointment_type_history_mismatch (Week 8, Data
       Science proposal) in the same pass.
    2. Build the target / filter excluded outcome labels.
    3. Impute missing values.
    4. Add engineered buckets and the Week 6 short-lead/high-risk interaction
       flag (added after step 1, since it depends on computed_previous_no_shows).
    4b. Conditionally add/drop the two Week 8 Data-Science-proposed features
        per config.INCLUDE_* flags, based on empirical validation -- see
        add_distance_lead_interaction() and the module-level note in
        config.py for the validation result behind each default.
    5. Drop leakage and non-feature identifier columns.
    6. Encode categoricals.

    Returns a DataFrame that is numeric except for two columns retained
    deliberately and NOT meant to be fed to a model directly:
        appointment_id   -- kept for traceability only
        appointment_date -- kept so callers can perform a time-based
                             train/test split (see model.py); must be
                             dropped, along with appointment_id and target,
                             immediately before fitting any model.
    """
    working = recompute_historical_features(df)
    working = build_target(working)
    working = impute_missing_values(working)
    working = add_booking_lead_bucket(working)
    working = add_short_lead_high_risk_flag(working)
    working = add_distance_lead_interaction(working)
    working = drop_leakage_columns(working)

    if not config.INCLUDE_DISTANCE_LEAD_INTERACTION:
        working = working.drop(columns=["distance_lead_interaction"])
    if not config.INCLUDE_APPOINTMENT_TYPE_HISTORY_MISMATCH:
        working = working.drop(columns=["appointment_type_history_mismatch"])

    # Drop raw historical columns now that the recomputed versions exist,
    # and drop patient_id (identifier, not a feature) but KEEP appointment_id
    # for traceability in the processed output.
    columns_to_drop = [c for c in config.RAW_HISTORICAL_COLUMNS if c in working.columns]
    columns_to_drop += ["patient_id"]
    working = working.drop(columns=columns_to_drop)

    # booking_date is no longer needed once booking_lead_days/bucket exist.
    # appointment_date is DELIBERATELY KEPT (not dropped here) because the
    # time-based train/test split in model.py needs it -- callers must drop
    # it, along with appointment_id and target, immediately before fitting.
    working = working.drop(columns=["booking_date"]) if "booking_date" in working.columns else working
    working = working.drop(columns=[config.OUTCOME_COLUMN])

    working = encode_categoricals(working)

    return working

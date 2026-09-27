"""
pipeline.py
-----------
Orchestrates the full data workflow: ingest -> validate -> engineer features
-> save. This is the single entry point the rest of the system (and other
tracks) should call rather than importing individual stages directly, so the
correct order is always guaranteed.
"""

import json
import logging

from . import config, data_ingestion, data_validation, feature_engineering
from .logging_config import setup_logging

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when the pipeline cannot complete due to a data or validation problem."""


def run_pipeline(raw_data_path=None, save_output: bool = True):
    """
    Run the full ingestion -> validation -> feature engineering workflow.

    Parameters
    ----------
    raw_data_path : str or Path, optional
        Overrides config.RAW_DATA_PATH.
    save_output : bool, default True
        Whether to write the processed dataset and validation report to disk
        (data/processed/). Tests set this to False to avoid writing files.

    Returns
    -------
    (processed_df, validation_report) : tuple[pd.DataFrame, dict]
    """
    setup_logging()
    logger.info("Step 1/3: Ingesting raw data")
    try:
        raw_df = data_ingestion.load_raw_data(raw_data_path)
    except data_ingestion.DataIngestionError as exc:
        logger.error("Data ingestion failed: %s", exc)
        raise PipelineError(f"Pipeline halted at ingestion: {exc}") from exc
    logger.info("Loaded %d rows, %d columns", raw_df.shape[0], raw_df.shape[1])

    logger.info("Step 2/3: Running data validation checks")
    validation_report = data_validation.run_all_checks(raw_df)
    _log_validation_summary(validation_report)

    logger.info("Step 3/3: Running feature engineering")
    try:
        processed_df = feature_engineering.run_feature_pipeline(raw_df)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any feature
        # engineering failure should halt the pipeline with a clear message
        # rather than propagate a confusing pandas traceback to the caller.
        logger.error("Feature engineering failed: %s", exc)
        raise PipelineError(f"Pipeline halted at feature engineering: {exc}") from exc

    dropped_rows = len(raw_df) - len(processed_df)
    logger.info(
        "Feature engineering complete: %d rows, %d columns (%d rows excluded, "
        "consistent with EXCLUDE_LABELS_FROM_TRAINING=%s)",
        processed_df.shape[0], processed_df.shape[1], dropped_rows,
        config.EXCLUDE_LABELS_FROM_TRAINING,
    )

    if save_output:
        _save_outputs(processed_df, validation_report)

    return processed_df, validation_report


def _log_validation_summary(report: dict) -> None:
    leakage = report["waiting_time_leakage"]
    if leakage.get("leakage_confirmed"):
        logger.warning(
            "Leakage check CONFIRMED: %d not-attended rows have a non-null "
            "waiting_time_minutes. This column is excluded by feature_engineering.py.",
            leakage["rows_with_unexpected_waiting_time"],
        )

    consistency = report["historical_field_consistency"]
    if consistency.get("checked"):
        logger.warning(
            "Historical field consistency: only %s%% of patients have chronologically "
            "consistent previous_appointments values. Pipeline recomputes these fields "
            "instead of trusting the raw columns.",
            consistency["consistency_rate_pct"],
        )

    if report["duplicates"]["duplicate_rows"] > 0:
        logger.warning("Found %d duplicate rows in raw data.", report["duplicates"]["duplicate_rows"])


def _save_outputs(processed_df, validation_report) -> None:
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    processed_df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    logger.info("Saved processed dataset to %s", config.PROCESSED_DATA_PATH)

    with open(config.VALIDATION_REPORT_PATH, "w") as f:
        json.dump(validation_report, f, indent=2, default=str)
    logger.info("Saved validation report to %s", config.VALIDATION_REPORT_PATH)


if __name__ == "__main__":
    run_pipeline()

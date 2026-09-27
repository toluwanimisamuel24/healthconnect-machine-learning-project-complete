import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.reminder_feature_analysis import _reminder_columns, run_reminder_ablation


def test_reminder_columns_identifies_correct_prefixes():
    columns = ["age", "reminder_sent_Yes", "reminder_channel_SMS", "distance_to_clinic_km"]
    result = _reminder_columns(columns)
    assert set(result) == {"reminder_sent_Yes", "reminder_channel_SMS"}


def test_reminder_columns_empty_when_none_present():
    columns = ["age", "distance_to_clinic_km"]
    assert _reminder_columns(columns) == []


def test_run_reminder_ablation_compares_with_and_without():
    rng = np.random.default_rng(42)
    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="D").strftime("%-m/%-d/%Y")

    df = pd.DataFrame({
        "appointment_id": [f"A-{i}" for i in range(n)],
        "appointment_date": dates,
        "age": rng.normal(40, 10, n),
        "reminder_sent_Yes": rng.integers(0, 2, n),
        "target": rng.integers(0, 2, n),
    })

    result = run_reminder_ablation(df, lambda: LogisticRegression(max_iter=200))

    assert "with_reminder_features" in result
    assert "without_reminder_features" in result
    assert result["with_reminder_features"]["feature_count"] == 2  # age + reminder_sent_Yes
    assert result["without_reminder_features"]["feature_count"] == 1  # age only
    assert "decision" in result
    assert "KEEP" in result["decision"]

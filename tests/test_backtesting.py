import pandas as pd
import pytest

from src.backtesting import make_expanding_windows, run_backtest


def _tiny_dated_df(n=20):
    """20 rows spread across 20 consecutive days, alternating target class."""
    dates = pd.date_range("2025-01-01", periods=n, freq="D").strftime("%-m/%-d/%Y")
    return pd.DataFrame({
        "appointment_id": [f"A-{i}" for i in range(n)],
        "appointment_date": dates,
        "feature_a": range(n),
        "target": [i % 2 for i in range(n)],
    })


def test_make_expanding_windows_produces_chronological_non_overlapping_folds():
    df = _tiny_dated_df(20)
    folds = list(make_expanding_windows(df, n_splits=4, min_train_fraction=0.4))

    assert len(folds) == 4

    for fold_index, train_df, test_df in folds:
        train_dates = pd.to_datetime(train_df["appointment_date"])
        test_dates = pd.to_datetime(test_df["appointment_date"])
        # every training date must be strictly before every test date in this fold
        assert train_dates.max() < test_dates.min()


def test_make_expanding_windows_is_truly_expanding():
    """Fold 2's training set must be a strict superset of fold 1's (more history each time)."""
    df = _tiny_dated_df(20)
    folds = list(make_expanding_windows(df, n_splits=4, min_train_fraction=0.4))

    train_sizes = [len(train_df) for _, train_df, _ in folds]
    assert train_sizes == sorted(train_sizes)  # strictly non-decreasing
    assert train_sizes[-1] > train_sizes[0]  # genuinely grows


def test_make_expanding_windows_raises_when_too_few_rows():
    df = _tiny_dated_df(5)
    with pytest.raises(ValueError, match="Not enough rows"):
        list(make_expanding_windows(df, n_splits=10, min_train_fraction=0.4))


def test_run_backtest_returns_summary_with_mean_and_std():
    df = _tiny_dated_df(40)
    from sklearn.linear_model import LogisticRegression

    result = run_backtest(
        df, lambda: LogisticRegression(max_iter=200), "Test Model",
        feature_columns=["feature_a"], n_splits=4, min_train_fraction=0.4,
    )

    assert result["model_name"] == "Test Model"
    assert len(result["fold_results"]) == 4
    assert "recall" in result["summary"]
    assert "mean" in result["summary"]["recall"]
    assert "std" in result["summary"]["recall"]


def test_run_backtest_date_ranges_are_chronologically_sensible():
    """
    Regression test for a Week 7 bug: test_date_range was originally computed
    with string min()/max() on the raw date column, which sorts
    lexicographically (e.g. '10/1/2025' < '9/9/2025' as strings) and produced
    a start date AFTER the end date. Must use parsed datetimes instead.
    """
    df = _tiny_dated_df(40)
    from sklearn.linear_model import LogisticRegression

    result = run_backtest(
        df, lambda: LogisticRegression(max_iter=200), "Test Model",
        feature_columns=["feature_a"], n_splits=4, min_train_fraction=0.4,
    )

    for fold in result["fold_results"]:
        start, end = pd.to_datetime(fold["test_date_range"][0]), pd.to_datetime(fold["test_date_range"][1])
        assert start <= end

"""
model.py
--------
A thin wrapper around a scikit-learn classifier, giving the rest of the
system (batch scoring service, monitoring, future model swaps) one stable
interface to depend on: fit / predict_proba / save / load.

Week 5 scope note: this ships with a Logistic Regression baseline purely to
prove the pipeline works end-to-end (data -> features -> trained model ->
predictions -> saved artifact). The Data Science track owns the actual model
selection; because every model here is accessed only through this wrapper's
interface, swapping the underlying estimator later (Random Forest, XGBoost,
whatever Week 5/6 modelling work recommends) should not require changing any
other file in this repository.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config


class ModelNotFittedError(Exception):
    """Raised when predict/predict_proba is called before fit or load."""


class NoShowRiskModel:
    """
    Wraps a classifier behind a fixed interface.

    Parameters
    ----------
    estimator : sklearn-compatible classifier, optional
        Defaults to a Logistic Regression baseline with class_weight="balanced".
        Any estimator implementing .fit / .predict_proba can be substituted
        (this is intentional -- it is the extension point for the Data
        Science track's chosen model).
    """

    def __init__(self, estimator=None):
        # Week 6 fix: this was originally `estimator or LogisticRegression(...)`.
        # That looks harmless but isn't -- Python's `or` evaluates truthiness,
        # and scikit-learn ensemble estimators (RandomForestClassifier,
        # GradientBoostingClassifier, etc.) define __len__ via their base
        # class. Truthiness checks fall back to __len__ when __bool__ isn't
        # defined, and __len__ on an ensemble estimator reads
        # self.estimators_ -- which does not exist until after .fit() has
        # run. The result was an AttributeError raised from inside __init__
        # the moment a Random Forest was passed in, before any of this
        # class's own code had a chance to run. Caught when
        # train_candidate_models.py first tried to integrate a Random
        # Forest candidate. Fixed with an explicit `is None` check, which
        # never evaluates the estimator's truthiness at all.
        self.estimator = estimator if estimator is not None else LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE
        )
        self._is_fitted = False
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NoShowRiskModel":
        self.feature_names_ = list(X.columns)
        self.estimator.fit(X, y)
        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame):
        self._check_fitted()
        self._check_columns(X)
        return self.estimator.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5):
        self._check_fitted()
        self._check_columns(X)
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path=None) -> Path:
        self._check_fitted()
        save_path = Path(path or config.MODEL_ARTIFACT_PATH)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        return save_path

    @staticmethod
    def load(path=None) -> "NoShowRiskModel":
        load_path = Path(path or config.MODEL_ARTIFACT_PATH)
        if not load_path.exists():
            raise FileNotFoundError(f"No saved model found at: {load_path}")
        return joblib.load(load_path)

    def _check_fitted(self):
        if not self._is_fitted:
            raise ModelNotFittedError(
                "This model has not been fitted or loaded yet. Call .fit(X, y) "
                "or NoShowRiskModel.load() before requesting predictions."
            )

    def _check_columns(self, X: pd.DataFrame):
        incoming = list(X.columns)
        if incoming != self.feature_names_:
            missing = set(self.feature_names_) - set(incoming)
            extra = set(incoming) - set(self.feature_names_)
            raise ValueError(
                "Feature columns at prediction time do not match training time. "
                f"Missing: {sorted(missing) or 'none'}. Unexpected: {sorted(extra) or 'none'}. "
                "This usually means the feature engineering pipeline produced a "
                "different column set -- check for a schema change in the raw data."
            )


def time_based_split(df: pd.DataFrame, date_column: str = "appointment_date",
                      test_fraction: float = None):
    """
    Split a DataFrame into train/test sets ordered by date rather than
    randomly, so the test set is always chronologically after the training
    set. This matters here because appointment_date spans a real calendar
    range (Jan 2025 - Sep 2026 in the raw data) -- a random split would let
    the model implicitly train on appointments dated after some of its own
    test cases, which is not a situation the deployed system will ever
    actually face (it only ever predicts forward in time).

    Returns
    -------
    (train_df, test_df) : tuple of pd.DataFrame
    """
    test_fraction = test_fraction or config.TEST_SPLIT_FRACTION
    working = df.copy()
    working["_sort_date"] = pd.to_datetime(working[date_column])
    working = working.sort_values("_sort_date").drop(columns=["_sort_date"])

    split_index = int(len(working) * (1 - test_fraction))
    train_df = working.iloc[:split_index].copy()
    test_df = working.iloc[split_index:].copy()
    return train_df, test_df

import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier

from src.model import NoShowRiskModel, ModelNotFittedError, time_based_split


def _tiny_training_data():
    rng = np.random.default_rng(42)
    n = 60
    X = pd.DataFrame({
        "feature_a": rng.normal(size=n),
        "feature_b": rng.normal(size=n),
    })
    # y correlated with feature_a so the model has something real to learn
    y = (X["feature_a"] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X, y


def test_predict_before_fit_raises():
    model = NoShowRiskModel()
    X, _ = _tiny_training_data()
    with pytest.raises(ModelNotFittedError):
        model.predict(X)


def test_fit_predict_round_trip():
    X, y = _tiny_training_data()
    model = NoShowRiskModel()
    model.fit(X, y)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    assert len(predictions) == len(X)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert (probabilities >= 0).all() and (probabilities <= 1).all()


def test_predict_with_mismatched_columns_raises():
    X, y = _tiny_training_data()
    model = NoShowRiskModel()
    model.fit(X, y)

    bad_X = X.rename(columns={"feature_b": "unexpected_column"})
    with pytest.raises(ValueError, match="do not match training time"):
        model.predict(bad_X)


def test_save_and_load_round_trip(tmp_path):
    X, y = _tiny_training_data()
    model = NoShowRiskModel()
    model.fit(X, y)

    save_path = tmp_path / "model.joblib"
    model.save(save_path)

    loaded_model = NoShowRiskModel.load(save_path)
    original_preds = model.predict(X)
    loaded_preds = loaded_model.predict(X)

    assert (original_preds == loaded_preds).all()


def test_time_based_split_orders_by_date_not_randomly():
    df = pd.DataFrame({
        "appointment_date": ["3/1/2025", "1/1/2025", "2/1/2025", "4/1/2025", "5/1/2025"],
        "value": [3, 1, 2, 4, 5],
    })
    train_df, test_df = time_based_split(df, test_fraction=0.4)

    # With 5 rows and test_fraction=0.4, 2 rows go to test -- and they must
    # be the two MOST RECENT dates (4/1 and 5/1), not a random sample.
    assert set(test_df["value"]) == {4, 5}
    assert set(train_df["value"]) == {1, 2, 3}
    assert train_df["appointment_date"].pipe(pd.to_datetime).max() < \
        test_df["appointment_date"].pipe(pd.to_datetime).min()


def test_ensemble_estimator_does_not_crash_at_init():
    """
    Regression test for a Week 6 bug: `estimator or LogisticRegression(...)`
    in __init__ evaluated the incoming estimator's truthiness, and
    scikit-learn ensemble estimators (like RandomForestClassifier) define
    __len__ via their base class, which reads self.estimators_ -- an
    attribute that does not exist before .fit() has run. This raised an
    AttributeError from inside __init__ itself, before any fit/predict call.
    Fixed with an explicit `is None` check. This test simply confirms an
    ensemble estimator can be passed in and used without error.
    """
    model = NoShowRiskModel(estimator=RandomForestClassifier(n_estimators=10, random_state=42))
    X, y = _tiny_training_data()

    model.fit(X, y)  # should not raise
    predictions = model.predict(X)

    assert len(predictions) == len(X)

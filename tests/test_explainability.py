import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.model import NoShowRiskModel
from src.explainability import (
    build_explainer, explain_single_appointment, explain_batch,
    ExplainabilityError, SHAP_AVAILABLE,
)

pytestmark = pytest.mark.skipif(not SHAP_AVAILABLE, reason="shap is not installed in this environment")


def _tiny_fitted_model():
    rng = np.random.default_rng(42)
    n = 60
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (X["a"] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    model = NoShowRiskModel(estimator=RandomForestClassifier(n_estimators=20, random_state=42))
    model.fit(X, y)
    return model, X


def test_build_explainer_requires_fitted_model():
    model = NoShowRiskModel(estimator=RandomForestClassifier(n_estimators=10))
    X = pd.DataFrame({"a": [1.0], "b": [2.0]})
    with pytest.raises(ExplainabilityError, match="not been fitted"):
        build_explainer(model, X)


def test_explain_single_appointment_returns_expected_structure():
    model, X = _tiny_fitted_model()
    explainer = build_explainer(model, X)

    result = explain_single_appointment(explainer, model, X.iloc[[0]])

    assert "predicted_probability" in result
    assert 0.0 <= result["predicted_probability"] <= 1.0
    assert "top_features_increasing_risk" in result
    assert "top_features_decreasing_risk" in result


def test_explain_single_appointment_rejects_multiple_rows():
    model, X = _tiny_fitted_model()
    explainer = build_explainer(model, X)

    with pytest.raises(ExplainabilityError, match="exactly one row"):
        explain_single_appointment(explainer, model, X.iloc[:3])


def test_explain_batch_returns_array_matching_input_shape():
    model, X = _tiny_fitted_model()
    explainer = build_explainer(model, X)

    values = explain_batch(explainer, X)
    assert values.shape[0] == len(X)
    assert values.shape[1] == X.shape[1]

"""
explainability.py
------------------
Week 7 addition: closes the last item from Week 6's Week 7 plan ("explore
SHAP-based explanations for individual risk scores, so a clinic operator can
see why a specific appointment was flagged, not just its tier").

SHAP (SHapley Additive exPlanations) assigns each feature a contribution
value for one specific prediction -- how much that feature pushed THIS
appointment's risk score up or down relative to the model's average
prediction. This is different from feature_engineering's global feature
importances (which describe the model overall); SHAP explains one row at a
time, which is what a clinic operator actually needs when looking at a
single flagged patient.

Kept as a thin, optional layer on top of NoShowRiskModel -- if the shap
library is unavailable in a given environment, the rest of the pipeline
(training, scoring) still runs unaffected, since nothing else imports this
module.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    SHAP_AVAILABLE = False


class ExplainabilityError(Exception):
    """Raised when an explanation cannot be produced."""


def build_explainer(model, background_data: pd.DataFrame):
    """
    Build a SHAP TreeExplainer for a fitted NoShowRiskModel wrapping a
    tree-based estimator (Random Forest, Gradient Boosting).

    background_data is a representative sample of TRAINING data (not the
    row(s) being explained) that SHAP uses as a reference point for "what is
    a typical prediction" -- explanations are relative to this baseline.
    """
    if not SHAP_AVAILABLE:
        raise ExplainabilityError(
            "shap is not installed in this environment. Install with `pip install shap` "
            "to use explainability.py; every other pipeline stage works without it."
        )
    if not model._is_fitted:
        raise ExplainabilityError("Cannot build an explainer for a model that has not been fitted.")

    try:
        return shap.TreeExplainer(model.estimator)
    except Exception as exc:
        raise ExplainabilityError(
            f"Failed to build a TreeExplainer for estimator type {type(model.estimator).__name__}. "
            "SHAP's TreeExplainer only supports tree-based models (Random Forest, Gradient "
            "Boosting, XGBoost, etc.) -- for a linear model like Logistic Regression, use "
            "shap.LinearExplainer instead."
        ) from exc


def explain_single_appointment(explainer, model, X_row: pd.DataFrame) -> dict:
    """
    Produce a human-readable explanation for exactly one appointment's risk
    score: which features pushed the prediction up, which pushed it down,
    and by how much.

    Parameters
    ----------
    X_row : a DataFrame with EXACTLY ONE row, containing the same feature
        columns the model was trained on.
    """
    if len(X_row) != 1:
        raise ExplainabilityError(
            f"explain_single_appointment expects exactly one row, got {len(X_row)}. "
            "Use explain_batch() for multiple appointments."
        )

    shap_values = explainer.shap_values(X_row)
    # RandomForestClassifier's TreeExplainer returns a list [class_0_values, class_1_values]
    # in some shap versions, or a single array in others -- normalise to "class 1" (no-show).
    if isinstance(shap_values, list):
        row_values = shap_values[1][0]
    else:
        row_values = shap_values[0] if shap_values.ndim == 2 else shap_values[0, :, 1]

    contributions = pd.Series(row_values, index=X_row.columns).sort_values(key=abs, ascending=False)

    top_increasing = contributions[contributions > 0].head(5)
    top_decreasing = contributions[contributions < 0].head(5)

    return {
        "predicted_probability": float(model.predict_proba(X_row)[0]),
        "top_features_increasing_risk": [
            {"feature": name, "contribution": round(float(val), 4)} for name, val in top_increasing.items()
        ],
        "top_features_decreasing_risk": [
            {"feature": name, "contribution": round(float(val), 4)} for name, val in top_decreasing.items()
        ],
    }


def explain_batch(explainer, X: pd.DataFrame) -> np.ndarray:
    """Return raw SHAP values for every row in X (used for summary-level plots, not single-row text)."""
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        return shap_values[1]
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values

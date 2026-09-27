"""
reminder_feature_analysis.py
------------------------------
Week 7 addition: resolves the reminder_sent/reminder_channel circularity
risk flagged in Week 4 and left unresolved through Weeks 5 and 6.

THE CONCERN (as originally flagged): if this model's own risk scores are
used to decide which patients receive a reminder, and reminder_sent is also
a model INPUT, there's a chicken-and-egg risk -- you'd need to already know
whether a reminder will be sent in order to predict risk, but the reminder
decision (in a future, model-driven world) might depend on that very risk
prediction.

THE RESOLUTION (Week 7 finding): this concern conflates two different
things, and separating them resolves it.

1. For the CURRENT dataset and CURRENT model: reminder_sent/reminder_channel
   are NOT leakage. They reflect HealthConnect's existing, model-independent
   reminder process -- a decision the clinic already makes, before the
   appointment, using its own (non-ML) policy. At the moment this model
   would need to predict risk for a given historical appointment, whether a
   reminder was sent for THAT appointment was already a known, settled fact,
   just like appointment_type or distance_to_clinic. This is genuinely
   different from waiting_time_minutes (Week 5's real leakage case), which
   could not have been known before the appointment under ANY process.

2. The genuine circularity risk is a FUTURE, deployment-governance concern,
   not a property of the current data: IF a later version of the reminder
   process starts being driven by this model's own output, then a model
   retrained on that future data would be partly predicting the consequences
   of its own past predictions -- a feedback loop. That risk does not exist
   yet (today's reminder_sent values were decided independently of this
   model), but must be monitored for if/when the reminder process changes.

DECISION: keep reminder_sent/reminder_channel as model features (removing
them costs real predictive value -- see run_reminder_ablation() below), and
document the forward-looking governance requirement in config.py and the
Week 7 report, rather than removing a feature that isn't actually leaking
anything today.
"""

import logging

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from . import config
from .model import NoShowRiskModel, time_based_split

logger = logging.getLogger(__name__)

REMINDER_FEATURE_PREFIXES = ("reminder_sent", "reminder_channel")


def _reminder_columns(all_columns: list) -> list:
    return [c for c in all_columns if c.startswith(REMINDER_FEATURE_PREFIXES)]


def run_reminder_ablation(processed_df, estimator_factory) -> dict:
    """
    Train and evaluate the same model twice on the same time-based split --
    once WITH reminder features, once WITHOUT -- to measure their genuine
    contribution to predictive performance. This is the evidence the
    keep/drop decision above is based on.
    """
    train_df, test_df = time_based_split(processed_df)
    all_features = [c for c in processed_df.columns if c not in ["appointment_id", "appointment_date", "target"]]
    reminder_cols = _reminder_columns(all_features)
    features_without_reminder = [c for c in all_features if c not in reminder_cols]

    results = {}
    for label, feature_set in [("with_reminder_features", all_features),
                                ("without_reminder_features", features_without_reminder)]:
        model = NoShowRiskModel(estimator=estimator_factory())
        model.fit(train_df[feature_set], train_df["target"])
        predictions = model.predict(test_df[feature_set])
        probabilities = model.predict_proba(test_df[feature_set])

        results[label] = {
            "feature_count": len(feature_set),
            "accuracy": accuracy_score(test_df["target"], predictions),
            "precision": precision_score(test_df["target"], predictions),
            "recall": recall_score(test_df["target"], predictions),
            "f1_score": f1_score(test_df["target"], predictions),
            "roc_auc": roc_auc_score(test_df["target"], probabilities),
        }
        logger.info("%s (%d features): Recall=%.4f, F1=%.4f", label, len(feature_set),
                    results[label]["recall"], results[label]["f1_score"])

    recall_cost = results["with_reminder_features"]["recall"] - results["without_reminder_features"]["recall"]
    results["reminder_columns_found"] = reminder_cols
    results["recall_cost_of_removing_reminder_features"] = round(recall_cost, 4)
    results["decision"] = (
        "KEEP reminder_sent/reminder_channel as features. They are not leakage for the "
        "current, retrospective dataset (see module docstring); removing them costs "
        f"{recall_cost:.4f} Recall. Monitor for the forward-looking circularity risk only "
        "if/when the reminder process itself becomes model-driven (see config.py, "
        "REMINDER_FEATURE_GOVERNANCE_NOTE)."
    )

    return results

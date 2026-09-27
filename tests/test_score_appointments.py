import numpy as np

from src import config
from src.score_appointments import assign_risk_tier


def test_assign_risk_tier_low():
    assert assign_risk_tier(0.10) == "Low"
    assert assign_risk_tier(0.0) == "Low"


def test_assign_risk_tier_medium():
    assert assign_risk_tier(0.40) == "Medium"
    assert assign_risk_tier(0.60) == "Medium"


def test_assign_risk_tier_high():
    assert assign_risk_tier(0.65) == "High"
    assert assign_risk_tier(1.0) == "High"


def test_assign_risk_tier_boundaries_are_consistent_with_config():
    """Every probability in [0, 1] must map to exactly one configured tier."""
    for p in np.linspace(0, 1, 101):
        tier = assign_risk_tier(round(float(p), 2))
        assert tier in config.RISK_TIER_THRESHOLDS

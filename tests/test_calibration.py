"""
Unit tests for probability calibration (Platt Scaling and Isotonic Regression).
"""

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def test_platt_scaling_monotonicity():
    """Verify that Platt Scaling (sigmoid) preserves relative ranking / monotonicity."""
    raw_probs = np.array([0.10, 0.35, 0.50, 0.75, 0.95])
    y_true = np.array([0, 0, 1, 1, 1])

    scaler = LogisticRegression(C=1.0)
    scaler.fit(raw_probs.reshape(-1, 1), y_true)

    calib_probs = scaler.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    
    # Monotonicity check
    assert np.all(np.diff(calib_probs) >= 0)
    assert np.all((calib_probs >= 0.0) & (calib_probs <= 1.0))


def test_isotonic_regression_bounds():
    """Verify that Isotonic Regression bounds outputs cleanly to [0, 1]."""
    raw_probs = np.array([0.05, 0.20, 0.40, 0.60, 0.80, 0.99])
    y_true = np.array([0, 0, 0, 1, 1, 1])

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_probs, y_true)

    calib_probs = iso.predict(raw_probs)
    assert np.all((calib_probs >= 0.0) & (calib_probs <= 1.0))
    assert np.all(np.diff(calib_probs) >= 0)

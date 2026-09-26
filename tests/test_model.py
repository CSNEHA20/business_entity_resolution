"""
Unit tests for Model and Classifier Module (Milestone 4).
"""

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS
from src.model import EntityPairClassifier


def test_model_initialization():
    clf = EntityPairClassifier(n_estimators=10, device="cpu")
    assert clf.n_estimators == 10
    assert clf.device == "cpu"


def test_model_train_predict_cpu():
    rng = np.random.RandomState(42)
    n_samples = 100
    X = pd.DataFrame(
        rng.randn(n_samples, len(FEATURE_COLUMNS)),
        columns=FEATURE_COLUMNS
    )
    y = rng.randint(0, 2, size=n_samples)

    clf = EntityPairClassifier(n_estimators=5, device="cpu", random_state=42)
    clf.train(X, y)

    probs = clf.predict_proba(X)
    assert len(probs) == n_samples
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    importances = clf.get_feature_importances()
    assert len(importances) == len(FEATURE_COLUMNS)


def test_model_save_load(tmp_path):
    rng = np.random.RandomState(42)
    n_samples = 50
    X = pd.DataFrame(
        rng.randn(n_samples, len(FEATURE_COLUMNS)),
        columns=FEATURE_COLUMNS
    )
    y = rng.randint(0, 2, size=n_samples)

    clf = EntityPairClassifier(n_estimators=5, device="cpu", random_state=42)
    clf.train(X, y)

    model_path = tmp_path / "test_model.json"
    clf.save(model_path)
    assert model_path.exists()

    loaded_clf = EntityPairClassifier.load(model_path)
    assert loaded_clf is not None

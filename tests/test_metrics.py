"""
Unit tests for the official macro F_0.5 evaluation metric and singleton mechanics.
"""

import pytest
from src.metrics import compute_candidate_recall, compute_entity_f05, compute_macro_f05


def test_official_readme_example():
    """
    Test exact F_0.5 computation matching the official problem statement example:
    - Ground truth: [S2-00047, S3-00812]
    - Predicted: [S2-00047, S2-00193, S3-00812]
    - Precision = 2/3 = 0.6667
    - Recall = 2/2 = 1.0
    - Expected F_0.5 = (1.25 * (2/3) * 1.0) / (0.25 * (2/3) + 1.0) = (5/6) / (7/6) = 5/7 ≈ 0.7142857
    """
    y_true = {"S2-00047", "S3-00812"}
    y_pred = {"S2-00047", "S2-00193", "S3-00812"}

    p, r, f = compute_entity_f05(y_true, y_pred, beta=0.5)

    assert pytest.approx(p, 1e-4) == 2 / 3
    assert pytest.approx(r, 1e-4) == 1.0
    assert pytest.approx(f, 1e-4) == 5 / 7
    assert pytest.approx(f, 1e-3) == 0.714


def test_singleton_true_positive():
    """Singleton correctly predicted as empty should score 1.0 (perfect score)."""
    p, r, f = compute_entity_f05(set(), set())
    assert p == 1.0
    assert r == 1.0
    assert f == 1.0


def test_singleton_false_positive_merge():
    """Singleton incorrectly predicted to have matches should score 0.0 (false merge penalty)."""
    p, r, f = compute_entity_f05(set(), {"S2-0001"})
    assert p == 0.0
    assert r == 0.0
    assert f == 0.0


def test_matched_entity_false_negative_abstention():
    """True match missed by predicting empty should score 0.0."""
    p, r, f = compute_entity_f05({"S2-0001"}, set())
    assert p == 0.0
    assert r == 0.0
    assert f == 0.0


def test_macro_f05_aggregation():
    """Test macro averaging across mixed entity cases."""
    gt = {
        "S1-01": ["S2-01"],             # Perfect match -> 1.0
        "S1-02": [],                     # Perfect singleton -> 1.0
        "S1-03": ["S2-03"],             # False negative (empty pred) -> 0.0
        "S1-04": [],                     # False positive (pred has match) -> 0.0
    }
    preds = {
        "S1-01": ["S2-01"],
        "S1-02": [],
        "S1-03": [],
        "S1-04": ["S2-04"],
    }

    result = compute_macro_f05(gt, preds)
    # Average of [1.0, 1.0, 0.0, 0.0] = 0.50
    assert pytest.approx(result["macro_f05"]) == 0.50
    assert result["total_entities"] == 4
    assert result["total_singletons"] == 2
    assert pytest.approx(result["singleton_accuracy"]) == 0.50


def test_candidate_recall():
    """Test candidate recall ceiling calculation."""
    gt = {
        "S1-01": ["S2-01", "S3-01"],
        "S1-02": ["S2-02"],
        "S1-03": [],
    }
    candidates = {
        "S1-01": ["S2-01"],           # Missed S3-01
        "S1-02": ["S2-02", "S2-99"],  # Captured S2-02 + 1 extra
        "S1-03": ["S3-50"],           # 1 candidate for singleton
    }

    res = compute_candidate_recall(gt, candidates)
    # Total true matches = 3. Retrieved = 2. Candidate recall = 2/3 ≈ 0.6667
    assert res["total_true_matches"] == 3
    assert res["retrieved_true_matches"] == 2
    assert pytest.approx(res["candidate_recall"], 1e-4) == 2 / 3
    assert pytest.approx(res["avg_candidates_per_s1"]) == 4 / 3

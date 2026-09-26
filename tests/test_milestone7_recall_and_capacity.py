"""
Unit & Regression Tests for Milestone 7: Recall Recovery & Decision Capacity Correction.
Amazon ML Challenge 2026 - Business Entity Resolution
"""

from collections import defaultdict
import copy
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pytest

from src.decision_engine import DecisionRuleConfig, EntityDecisionEngine
from src.metrics import compute_candidate_recall, compute_macro_f05


# =====================================================================
# 1. CANDIDATE PRUNING RECALL TESTS
# =====================================================================

def test_candidate_pruning_recall_computation():
    """Verify that candidate recall accurately computes true pair coverage."""
    ground_truth = {
        "S1-001": {"S2-001", "S3-001"},
        "S1-002": {"S2-002"},
        "S1-003": set(),  # singleton
        "S1-004": {"S3-004"},
    }
    
    # Candidate set missing S3-001 and S3-004
    candidates = {
        "S1-001": {"S2-001", "S2-999"},
        "S1-002": {"S2-002"},
        "S1-003": set(),
        "S1-004": {"S2-888"},  # False candidate
    }
    
    # Total true pairs = 4 (S1-001: 2, S1-002: 1, S1-004: 1)
    # Retrieved true pairs = 2 (S1-001->S2-001, S1-002->S2-002)
    # Expected recall = 2 / 4 = 0.50
    res = compute_candidate_recall(ground_truth, candidates)
    assert res["total_true_matches"] == 4
    assert res["retrieved_true_matches"] == 2
    assert pytest.approx(res["candidate_recall"], 0.001) == 0.50


# =====================================================================
# 2. ZERO-CANDIDATE BEHAVIOR TESTS
# =====================================================================

def test_zero_candidate_abstention_behavior():
    """Entities with 0 retrieved candidates must cleanly produce empty predictions (singleton)."""
    cfg = DecisionRuleConfig(
        strategy="adaptive_multi",
        threshold_s2=0.50,
        threshold_s3=0.50,
        min_top_prob=0.45,
        enable_singleton_abstention=True,
    )
    engine = EntityDecisionEngine(cfg)

    # Empty candidate scores
    empty_scores = {}
    preds = engine.predict_entity(empty_scores)
    assert preds == []

    # Batch prediction with mix of populated and zero candidates
    batch_scores = {
        "S1-001": {"S2-001": 0.85},
        "S1-002": {},  # Zero candidates
        "S1-003": {},  # Zero candidates
    }
    batch_meta = {
        "S1-001": {"has_addr": True},
        "S1-002": {"has_addr": False},
        "S1-003": {"has_addr": True},
    }
    all_preds = engine.predict_all(batch_scores, batch_meta, all_s1_ids=["S1-001", "S1-002", "S1-003"])
    assert all_preds["S1-001"] == ["S2-001"]
    assert all_preds["S1-002"] == []
    assert all_preds["S1-003"] == []


# =====================================================================
# 3. MAX-2 / MAX-3 CAPACITY TESTS
# =====================================================================

def test_max_2_source_capacity_enforcement():
    """Verify that max_matches_per_source=2 allows up to 2 matches per source."""
    cfg = DecisionRuleConfig(
        strategy="adaptive_multi",
        threshold_s2=0.50,
        threshold_s3=0.50,
        min_top_prob=0.50,
        multi_match_threshold=0.52,
        max_multi_score_drop=0.20,
        max_matches_per_source=2,
    )
    engine = EntityDecisionEngine(cfg)

    # 3 strong S2 candidates, 1 strong S3 candidate
    candidates = {
        "S2-001": 0.90,  # S2 rank 1 -> keep
        "S2-002": 0.88,  # S2 rank 2 -> keep
        "S2-003": 0.85,  # S2 rank 3 -> drop (exceeds max 2)
        "S3-001": 0.86,  # S3 rank 1 -> keep
    }
    selected = engine.predict_entity(candidates)
    assert selected == ["S2-001", "S2-002", "S3-001"]


def test_max_3_source_capacity_enforcement():
    """Verify that max_matches_per_source=3 allows up to 3 matches per source."""
    cfg = DecisionRuleConfig(
        strategy="adaptive_multi",
        threshold_s2=0.50,
        threshold_s3=0.50,
        min_top_prob=0.50,
        multi_match_threshold=0.52,
        max_multi_score_drop=0.20,
        max_matches_per_source=3,
    )
    engine = EntityDecisionEngine(cfg)

    candidates = {
        "S2-001": 0.95,  # S2 #1 -> keep
        "S2-002": 0.92,  # S2 #2 -> keep
        "S2-003": 0.90,  # S2 #3 -> keep
        "S2-004": 0.88,  # S2 #4 -> drop
        "S3-001": 0.89,  # S3 #1 -> keep
    }
    selected = engine.predict_entity(candidates)
    assert selected == ["S2-001", "S2-002", "S2-003", "S3-001"]


# =====================================================================
# 4. NO-CAPACITY BEHAVIOR TESTS
# =====================================================================

def test_no_capacity_limit_behavior():
    """Setting max_matches_per_source=0 (or disabled) retains all valid multi-matches."""
    cfg = DecisionRuleConfig(
        strategy="adaptive_multi",
        threshold_s2=0.50,
        threshold_s3=0.50,
        min_top_prob=0.50,
        multi_match_threshold=0.50,
        max_multi_score_drop=0.25,
        max_matches_per_source=0,  # No limit
    )
    engine = EntityDecisionEngine(cfg)

    # 4 S2 candidates, 2 S3 candidates all within drop
    candidates = {
        "S2-01": 0.90,
        "S2-02": 0.88,
        "S2-03": 0.85,
        "S2-04": 0.82,
        "S3-01": 0.89,
        "S3-02": 0.80,
    }
    selected = engine.predict_entity(candidates)
    assert len(selected) == 6
    assert set(selected) == {"S2-01", "S2-02", "S2-03", "S2-04", "S3-01", "S3-02"}


# =====================================================================
# 5. MULTI-MATCH ENTITIES TESTS
# =====================================================================

def test_multi_match_score_drop_boundary():
    """Candidate exceeding max_multi_score_drop must be rejected even if score > threshold."""
    cfg = DecisionRuleConfig(
        strategy="adaptive_multi",
        threshold_s2=0.40,
        threshold_s3=0.40,
        min_top_prob=0.40,
        multi_match_threshold=0.40,
        max_multi_score_drop=0.15,
        max_matches_per_source=0,
    )
    engine = EntityDecisionEngine(cfg)

    # Top is 0.95. Candidate with 0.75 has drop 0.20 > 0.15 -> rejected
    # Candidate with 0.85 has drop 0.10 <= 0.15 -> accepted
    candidates = {
        "S2-001": 0.95,
        "S3-001": 0.85,
        "S2-002": 0.75,
    }
    selected = engine.predict_entity(candidates)
    assert selected == ["S2-001", "S3-001"]


# =====================================================================
# 6. CANDIDATE / DECISION CONSISTENCY TESTS
# =====================================================================

def test_candidate_decision_subset_consistency():
    """Decision engine must never predict an entity that was not in input candidates."""
    cfg = DecisionRuleConfig(strategy="adaptive_multi", max_matches_per_source=0)
    engine = EntityDecisionEngine(cfg)

    input_cands = {
        "S2-100": 0.92,
        "S3-200": 0.88,
        "S2-300": 0.40,
    }
    selected = engine.predict_entity(input_cands)
    for tid in selected:
        assert tid in input_cands


# =====================================================================
# 7. DETERMINISTIC RESULTS TESTS
# =====================================================================

def test_decision_engine_deterministic_across_runs():
    """Ensure identical inputs always yield bitwise identical output order and content."""
    cfg = DecisionRuleConfig(
        strategy="adaptive_multi",
        threshold_s2=0.4775,
        threshold_s3=0.4975,
        min_top_prob=0.428,
        multi_match_threshold=0.458,
        max_multi_score_drop=0.16,
        max_matches_per_source=0,
    )
    engine = EntityDecisionEngine(cfg)

    scores = {
        f"S1-{i:03d}": {
            f"S2-{j:03d}": float((i * 17 + j * 31) % 100) / 100.0
            for j in range(5)
        }
        for i in range(50)
    }
    meta = {f"S1-{i:03d}": {"has_addr": bool(i % 2 == 0)} for i in range(50)}

    run1 = engine.predict_all(scores, meta)
    run2 = engine.predict_all(scores, meta)
    run3 = engine.predict_all(scores, meta)

    assert run1 == run2 == run3

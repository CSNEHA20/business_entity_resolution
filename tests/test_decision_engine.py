"""
Unit tests for src/decision_engine.py.
Tests singleton abstention, source-specific thresholds, multi-match logic,
margin calculations, and serialization.
"""

from pathlib import Path
import pytest
from src.decision_engine import DecisionRuleConfig, EntityDecisionEngine
from src.metrics import compute_macro_f05


def test_global_threshold_strategy():
    cfg = DecisionRuleConfig(strategy="global", global_threshold=0.70)
    engine = EntityDecisionEngine(cfg)

    # Candidates above and below threshold
    cands = {
        "S2-001": 0.85,
        "S2-002": 0.65,
        "S3-001": 0.72,
    }
    selected = engine.predict_entity(cands)
    assert set(selected) == {"S2-001", "S3-001"}


def test_source_specific_thresholds():
    cfg = DecisionRuleConfig(
        strategy="source_specific",
        threshold_s2=0.60,
        threshold_s3=0.80,
    )
    engine = EntityDecisionEngine(cfg)

    cands = {
        "S2-001": 0.65,  # Above S2 threshold (0.60)
        "S3-001": 0.75,  # Below S3 threshold (0.80)
    }
    selected = engine.predict_entity(cands)
    assert selected == ["S2-001"]


def test_singleton_abstention_low_confidence():
    """If top candidate confidence is below min_top_prob, engine must abstain (return [])."""
    cfg = DecisionRuleConfig(
        strategy="adaptive_multi",
        threshold_s2=0.60,
        threshold_s3=0.60,
        min_top_prob=0.65,
        enable_singleton_abstention=True,
    )
    engine = EntityDecisionEngine(cfg)

    # Top candidate is 0.55 < 0.65 -> abstain
    cands = {"S2-001": 0.55, "S3-001": 0.30}
    selected = engine.predict_entity(cands)
    assert selected == []


def test_singleton_abstention_tiny_margin_borderline():
    """If top candidate is borderline and margin is smaller than min_margin, abstain."""
    cfg = DecisionRuleConfig(
        strategy="adaptive_multi",
        threshold_s2=0.60,
        threshold_s3=0.60,
        min_top_prob=0.55,
        min_margin=0.08,
        enable_singleton_abstention=True,
    )
    engine = EntityDecisionEngine(cfg)

    # Top candidate is 0.61 (near boundary 0.60), second is 0.60 -> margin is 0.01 < 0.08
    cands = {"S2-001": 0.61, "S2-002": 0.60}
    selected = engine.predict_entity(cands)
    assert selected == []


def test_multi_match_selection_success():
    """Top candidate accepted, secondary candidate meets multi_threshold and max score drop."""
    cfg = DecisionRuleConfig(
        strategy="adaptive_multi",
        threshold_s2=0.60,
        threshold_s3=0.60,
        min_top_prob=0.60,
        multi_match_threshold=0.65,
        max_multi_score_drop=0.15,
        max_matches_per_source=1,
    )
    engine = EntityDecisionEngine(cfg)

    # Top candidate: S2-001 (0.90)
    # Secondary candidate: S3-001 (0.80) -> drop = 0.10 <= 0.15, score = 0.80 >= 0.65 -> accepted
    # Third candidate: S2-002 (0.78) -> S2 already has 1 match -> rejected
    cands = {
        "S2-001": 0.90,
        "S3-001": 0.80,
        "S2-002": 0.78,
    }
    selected = engine.predict_entity(cands)
    assert selected == ["S2-001", "S3-001"]


def test_multi_match_rejected_due_to_large_drop():
    """Secondary candidate has acceptable raw score but drop from top is too large."""
    cfg = DecisionRuleConfig(
        strategy="adaptive_multi",
        threshold_s2=0.50,
        threshold_s3=0.50,
        min_top_prob=0.50,
        multi_match_threshold=0.55,
        max_multi_score_drop=0.20,
    )
    engine = EntityDecisionEngine(cfg)

    # Top candidate: 0.95
    # Secondary candidate: 0.60 (score drop = 0.35 > 0.20) -> rejected
    cands = {
        "S2-001": 0.95,
        "S3-001": 0.60,
    }
    selected = engine.predict_entity(cands)
    assert selected == ["S2-001"]


def test_missing_address_adaptive_penalty():
    """When S1 lacks an address, the effective threshold is boosted."""
    cfg = DecisionRuleConfig(
        strategy="global",
        global_threshold=0.60,
        missing_addr_threshold_boost=0.10,
    )
    engine = EntityDecisionEngine(cfg)

    cands = {"S2-001": 0.65}
    # With address: 0.65 >= 0.60 -> Match
    assert engine.predict_entity(cands, s1_meta={"has_addr": True}) == ["S2-001"]
    # Without address: effective threshold = 0.70 -> 0.65 < 0.70 -> Abstain
    assert engine.predict_entity(cands, s1_meta={"has_addr": False}) == []


def test_predict_all_and_evaluate(tmp_path: Path):
    """Test full batch prediction, evaluation against ground truth, and JSON serialization."""
    gt = {
        "S1-01": ["S2-01"],
        "S1-02": [],
        "S1-03": ["S2-03", "S3-03"],
    }
    scores = {
        "S1-01": {"S2-01": 0.92, "S2-99": 0.10},
        "S1-02": {"S2-50": 0.40},  # Below threshold -> singleton
        "S1-03": {"S2-03": 0.90, "S3-03": 0.88, "S2-98": 0.30},
    }

    cfg = DecisionRuleConfig(
        strategy="adaptive_multi",
        threshold_s2=0.60,
        threshold_s3=0.60,
        min_top_prob=0.55,
        multi_match_threshold=0.60,
        max_multi_score_drop=0.15,
    )
    engine = EntityDecisionEngine(cfg)
    eval_res = engine.evaluate(gt, scores)

    assert eval_res["macro_f05"] == 1.0
    assert eval_res["singleton_accuracy"] == 1.0

    # Test serialization
    save_path = tmp_path / "engine_config.json"
    engine.save(save_path)
    assert save_path.exists()

    loaded_engine = EntityDecisionEngine.load(save_path)
    assert loaded_engine.config.strategy == "adaptive_multi"
    assert loaded_engine.config.threshold_s2 == 0.60

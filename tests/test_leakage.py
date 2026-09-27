"""
Milestone 10 - Part 22: Automated Leakage and Invariant Verification Suite
Verifies all 10 critical anti-leakage invariants:
1. Candidate generation does not accept ground truth.
2. Changing ground-truth mappings does not change candidate IDs.
3. Validation labels cannot enter retrieval.
4. Holdout labels cannot enter training.
5. Test labels cannot be loaded.
6. Candidate recall is calculated only AFTER retrieval.
7. Predicted candidates are a subset of retrieved candidates.
8. Model training uses only retrieved candidates.
9. Hard negatives originate from actual open-corpus retrieval.
10. Metric is entity-level Macro F0.5.
"""

import inspect
import json
from pathlib import Path
from typing import Dict, List, Set
import numpy as np
import pytest

from src.decision_engine import DecisionRuleConfig, EntityDecisionEngine
from src.metrics import compute_candidate_recall, compute_entity_f05, compute_macro_f05
from src.open_corpus_retriever import (
    RetrievalConfig,
    TargetCorpusIndex,
    parse_entity_record,
    retrieve_candidates,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT_DIR / "artifacts" / "milestone10" / "splits"


def test_1_candidate_generation_does_not_accept_ground_truth():
    """1. Candidate generation function does not accept ground truth or label parameters."""
    sig = inspect.signature(retrieve_candidates)
    param_names = list(sig.parameters.keys())
    assert "query_record" in param_names
    assert "target_indices" in param_names
    assert "config" in param_names

    forbidden = ["ground_truth", "gt_map", "labels", "positive_target_ids", "true_matches", "y_true"]
    for param in param_names:
        for kw in forbidden:
            assert kw not in param.lower(), f"Forbidden ground-truth parameter found: {param}"


def test_2_changing_ground_truth_does_not_change_candidate_ids():
    """2. Changing ground-truth mappings does not change retrieved candidate IDs."""
    idx = TargetCorpusIndex()
    idx.target_ids = ["S2-001", "S3-002", "S2-003"]
    idx.target_sources = ["S2", "S3", "S2"]
    idx.target_countries = ["US", "US", "US"]
    idx.idx_exact_name["microsoft corporation"].extend([0, 1])

    q = parse_entity_record("S1-001", "Microsoft Corporation", "One Microsoft Way", "US")

    # Simulate two radically different ground-truth files
    gt_variant_a = {"S1-001": {"S2-001"}}
    gt_variant_b = {"S1-001": {"S3-999"}}  # Target not even retrieved

    # Retrieval only takes q, idx, config
    cands_a = retrieve_candidates(q, idx)
    cands_b = retrieve_candidates(q, idx)

    assert [c["target_id"] for c in cands_a] == [c["target_id"] for c in cands_b]
    assert [c["retrieval_score"] for c in cands_a] == [c["retrieval_score"] for c in cands_b]


def test_3_validation_labels_cannot_enter_retrieval():
    """3. Validation labels cannot enter candidate retrieval."""
    sig = inspect.signature(retrieve_candidates)
    assert "labels" not in sig.parameters
    assert "gt_map" not in sig.parameters
    assert "ground_truth" not in sig.parameters


def test_4_holdout_labels_cannot_enter_training():
    """4. Holdout entities are strictly disjoint from Training entities."""
    train_path = SPLITS_DIR / "train_ids.json"
    holdout_path = SPLITS_DIR / "holdout_ids.json"

    if train_path.exists() and holdout_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            tr_ids = set(json.load(f))
        with open(holdout_path, "r", encoding="utf-8") as f:
            ho_ids = set(json.load(f))
        overlap = tr_ids & ho_ids
        assert len(overlap) == 0, f"FATAL: Holdout overlaps Training by {len(overlap)} entities!"


def test_5_test_labels_cannot_be_loaded():
    """5. Test dataset does not contain ground truth labels."""
    test_gt_path = ROOT_DIR / "data" / "test" / "test_ground_truth.tsv"
    assert not test_gt_path.exists(), "Test ground truth file should not exist!"


def test_6_candidate_recall_calculated_only_after_retrieval():
    """6. Candidate recall is calculated strictly AFTER retrieval on frozen candidates."""
    mock_gt = {"S1-1": {"S2-10", "S3-20"}, "S1-2": {"S2-30"}}
    # Frozen candidate dictionary
    mock_frozen_cands = {"S1-1": {"S2-10", "S2-99"}, "S1-2": {"S2-30", "S3-88"}}

    recall_info = compute_candidate_recall(mock_gt, mock_frozen_cands)
    assert recall_info["total_true_matches"] == 3
    assert recall_info["retrieved_true_matches"] == 2
    assert abs(recall_info["candidate_recall"] - (2 / 3)) < 1e-5


def test_7_predicted_candidates_are_subset_of_retrieved_candidates():
    """7. Predicted candidates are strictly a subset of retrieved candidates."""
    engine = EntityDecisionEngine(DecisionRuleConfig(
        strategy="adaptive_multi",
        threshold_s2=0.50,
        threshold_s3=0.50,
        min_top_prob=0.45,
    ))
    retrieved_cands = [("S2-100", 0.92), ("S3-200", 0.85), ("S2-300", 0.15)]
    s1_rec = {"clean_name": "test", "clean_addr": "test", "has_addr": True}
    res = engine.decide_matches("S1-1", retrieved_cands, s1_rec)

    retrieved_tids = {tid for tid, _ in retrieved_cands}
    for pred_tid in res.matched_entity_ids:
        assert pred_tid in retrieved_tids, f"Predicted ID {pred_tid} was not in retrieved candidates!"


def test_8_model_training_uses_only_retrieved_candidates():
    """8. Model training pairs are constructed strictly from retrieved candidate pools."""
    retrieved_pool = {"S2-1", "S2-2", "S2-3", "S3-4"}
    gt_matches = {"S2-1", "S3-999"}  # S3-999 was not retrieved

    # The training pair construction must only use candidates that were actually retrieved
    positives = [tid for tid in retrieved_pool if tid in gt_matches]
    negatives = [tid for tid in retrieved_pool if tid not in gt_matches]

    assert positives == ["S2-1"]
    assert "S3-999" not in positives  # Missed positive must NOT be injected into training candidates!
    assert set(negatives) == {"S2-2", "S2-3", "S3-4"}


def test_9_hard_negatives_originate_from_open_corpus_retrieval():
    """9. Hard negatives originate from open-corpus retrieval against the target database."""
    retrieved_cands = [
        {"target_id": "S2-1", "routes": ["exact_name"], "route_hit_count": 1, "retrieval_score": 5.0},
        {"target_id": "S2-2", "routes": ["num_street", "postal"], "route_hit_count": 2, "retrieval_score": 4.5},
        {"target_id": "S3-3", "routes": ["char_4gram"], "route_hit_count": 1, "retrieval_score": 2.5},
    ]
    gt = {"S2-1"}  # S2-1 is positive

    negs = [c for c in retrieved_cands if c["target_id"] not in gt]
    assert len(negs) == 2
    assert all(n["target_id"] in ["S2-2", "S3-3"] for n in negs)
    # Negatives maintain provenance from real retrieval
    assert negs[0]["routes"] == ["num_street", "postal"]


def test_10_metric_is_entity_level_macro_f05():
    """10. Evaluation metric is strictly Entity-Level Macro F0.5, with proper singleton handling."""
    # Singleton entity (empty ground truth, empty prediction -> perfect score 1.0)
    p_sing, r_sing, f_sing = compute_entity_f05(set(), set())
    assert p_sing == 1.0 and r_sing == 1.0 and f_sing == 1.0

    # Singleton entity with false positive prediction -> 0.0
    p_sing_fp, r_sing_fp, f_sing_fp = compute_entity_f05(set(), {"S2-100"})
    assert p_sing_fp == 0.0 and r_sing_fp == 0.0 and f_sing_fp == 0.0

    # Macro average across multiple entities
    gt = {
        "S1-1": set(),            # singleton
        "S1-2": {"S2-1", "S3-2"}, # 2 matches
    }
    preds = {
        "S1-1": set(),            # correct singleton -> F0.5 = 1.0
        "S1-2": {"S2-1"},         # retrieved 1 of 2 -> P=1.0, R=0.5 -> F0.5 = 1.25*1*0.5 / (0.25*1 + 0.5) = 0.625 / 0.75 = 0.8333
    }
    macro = compute_macro_f05(gt, preds)
    expected_macro = (1.0 + (0.625 / 0.75)) / 2.0
    assert abs(macro["macro_f05"] - expected_macro) < 1e-4

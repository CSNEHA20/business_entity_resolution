"""
Unit tests for multi-pass blocking, inverted indexing, TF-IDF retrieval,
and candidate provenance tracking.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.blocking import MultiPassCandidateGenerator
from src.data_io import (
    load_ground_truth,
    load_train_source1,
    load_train_source2,
    load_train_source3,
    parse_ground_truth_mapping,
)
from src.metrics import compute_candidate_recall


def test_multi_pass_blocking_fit_and_generate(sample_toy_datasets: Path):
    """Test fitting target records and retrieving candidate pairs."""
    s1_df = load_train_source1(sample_toy_datasets)
    s2_df = load_train_source2(sample_toy_datasets)
    s3_df = load_train_source3(sample_toy_datasets)
    gt_df = load_ground_truth(sample_toy_datasets)

    generator = MultiPassCandidateGenerator()
    generator.fit_targets(s2_df, s3_df)

    candidates, provenance = generator.generate_candidates(s1_df, top_k_tfidf=10)

    # All S1 entities must exist in candidate mapping
    assert set(candidates.keys()) == set(s1_df["entity_id"])

    # No self matches
    for s1_id, cands in candidates.items():
        assert s1_id not in cands
        for c in cands:
            assert c.startswith(("S2-", "S3-"))

    # Check candidate recall against ground truth
    gt_mapping = parse_ground_truth_mapping(gt_df)
    res = compute_candidate_recall(gt_mapping, candidates)
    
    # In toy dataset, Acme, TCS, Apex should all be recovered by multi-pass blocking
    assert res["candidate_recall"] == 1.0
    assert res["retrieved_true_matches"] == res["total_true_matches"]


def test_blocking_provenance_recording(sample_toy_datasets: Path):
    """Test that candidate pairs accurately record their discovering blocking routes."""
    s1_df = load_train_source1(sample_toy_datasets)
    s2_df = load_train_source2(sample_toy_datasets)
    s3_df = load_train_source3(sample_toy_datasets)

    generator = MultiPassCandidateGenerator()
    generator.fit_targets(s2_df, s3_df)

    candidates, provenance = generator.generate_candidates(s1_df)

    # Acme pair ("S1-0001", "S2-0001") shares name, address, and TF-IDF
    pair_key = ("S1-0001", "S2-0001")
    assert pair_key in provenance
    routes = provenance[pair_key].routes
    assert len(routes) >= 1
    assert any("name" in r or "tfidf" in r for r in routes)


def test_blocking_ablation_routes(sample_toy_datasets: Path):
    """Test generating candidates using isolated single blocking routes."""
    s1_df = load_train_source1(sample_toy_datasets)
    s2_df = load_train_source2(sample_toy_datasets)
    s3_df = load_train_source3(sample_toy_datasets)

    generator = MultiPassCandidateGenerator()
    generator.fit_targets(s2_df, s3_df)

    # Test exact name only
    cands_name, _ = generator.generate_candidates(s1_df, enabled_routes={"exact_name"})
    assert isinstance(cands_name, dict)
    assert len(cands_name) == 4

    # Test tfidf name only
    cands_tfidf, _ = generator.generate_candidates(s1_df, enabled_routes={"name_tfidf"})
    assert isinstance(cands_tfidf, dict)
    assert len(cands_tfidf) == 4

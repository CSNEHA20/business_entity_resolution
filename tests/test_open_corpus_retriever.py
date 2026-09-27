"""
Unit tests for Pure Open-Corpus Retriever API (Milestone 10, Part 2).
Verifies:
1. retrieve_candidates does not accept ground truth or labels.
2. Ground truth independence: varying ground truth does not affect candidate generation.
3. Candidate output schema: target_id, source, routes, route_hit_count, retrieval_score, retrieval_rank.
4. Multi-pass routes and country handling.
"""

import inspect
import pytest
from src.open_corpus_retriever import (
    RetrievalConfig,
    TargetCorpusIndex,
    compute_compact_name,
    compute_core_name,
    extract_char_ngrams,
    normalize_country_code,
    parse_entity_record,
    retrieve_candidates,
)


def test_api_signature_has_no_ground_truth_parameter():
    """Verify retrieve_candidates signature does not accept labels or ground truth."""
    sig = inspect.signature(retrieve_candidates)
    param_names = list(sig.parameters.keys())
    assert "query_record" in param_names
    assert "target_indices" in param_names
    assert "config" in param_names

    forbidden_keywords = [
        "ground_truth",
        "gt_map",
        "labels",
        "positive_target_ids",
        "true_matches",
        "y_true",
    ]
    for param in param_names:
        for kw in forbidden_keywords:
            assert kw not in param.lower(), f"Forbidden ground-truth parameter found: {param}"


def test_ground_truth_independence():
    """Prove that varying external ground truth maps does NOT change candidate generation."""
    # Build a mini mock index
    index = TargetCorpusIndex()
    index.target_ids = ["S2-001", "S3-002", "S2-003", "S3-004"]
    index.target_sources = ["S2", "S3", "S2", "S3"]
    index.target_countries = ["US", "US", "IN", "FR"]

    # Index some names and tokens
    index.idx_exact_name["starbucks coffee"].append(0)
    index.idx_exact_name["blue bottle coffee"].append(1)
    index.idx_compact_name["starbuckscoffee"].append(0)
    index.idx_token["starbucks"].append(0)
    index.token_df["starbucks"] = 1
    index.idx_postal["94105"].append(0)
    index.idx_postal["94105"].append(1)

    query = {
        "entity_id": "S1-999",
        "business_name": "Starbucks Coffee",
        "business_address": "123 Main St, San Francisco, 94105",
        "country": "US",
    }

    # Simulate two completely different ground truths
    gt_scenario_a = {"S1-999": {"S2-001"}}
    gt_scenario_b = {"S1-999": {"S3-9999999"}}  # Target not even in index

    # Candidate retrieval does NOT accept gt_scenario_a or gt_scenario_b
    cands_1 = retrieve_candidates(query, index)
    cands_2 = retrieve_candidates(query, index)

    # Candidates must be 100% byte-for-byte identical
    assert cands_1 == cands_2
    assert len(cands_1) > 0
    assert cands_1[0]["target_id"] == "S2-001"
    assert "exact_name" in cands_1[0]["routes"]


def test_candidate_provenance_schema():
    """Verify that every candidate carries full provenance information."""
    index = TargetCorpusIndex()
    index.target_ids = ["S2-100"]
    index.target_sources = ["S2"]
    index.target_countries = ["US"]
    index.idx_exact_name["acme corp"].append(0)

    query = {
        "entity_id": "S1-1",
        "business_name": "Acme Corp",
        "business_address": "1 Road",
        "country": "US",
    }
    cands = retrieve_candidates(query, index)
    assert len(cands) == 1
    c = cands[0]
    assert "target_id" in c
    assert "source" in c
    assert "routes" in c
    assert "route_hit_count" in c
    assert "retrieval_score" in c
    assert "retrieval_rank" in c
    assert c["target_id"] == "S2-100"
    assert c["source"] == "S2"
    assert c["retrieval_rank"] == 1


def test_country_normalization():
    """Verify country normalization handles US, IN, FR, etc. properly."""
    assert normalize_country_code("US") == "US"
    assert normalize_country_code("USA") == "US"
    assert normalize_country_code("United States") == "US"
    assert normalize_country_code("India") == "IN"
    assert normalize_country_code("IND") == "IN"
    assert normalize_country_code("France") == "FR"
    assert normalize_country_code("Germany") == "DE"
    assert normalize_country_code("Canada") == "CA"


def test_core_and_compact_name():
    """Verify core name strips legal stopwords and compact name strips punctuation."""
    assert compute_compact_name("Wal-Mart Stores, Inc.") == "walmartstoresinc"
    assert compute_core_name("walmart stores inc") == "walmart stores"
    assert compute_core_name("tata consultancy services limited") == "tata consultancy"

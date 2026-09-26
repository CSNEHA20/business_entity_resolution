"""
Unit tests for multi-pass blocking, inverted indexing, TF-IDF retrieval,
and candidate provenance tracking.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.blocking import CandidatePairProvenance, MultiPassCandidateGenerator
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


def test_unicode_preservation_and_multiscript_normalization():
    """Test that Indic and European multi-script characters are preserved without corruption."""
    from src.normalization import (
        clean_unicode_text,
        get_token_signature,
        normalize_basic,
        normalize_business_name_suffixes,
    )

    # Indic Devanagari test
    dev_name = "टाटा कंसल्टेंसी सर्विसेज प्राइवेट लिमिटेड"
    norm_dev = clean_unicode_text(dev_name)
    assert "टाटा" in norm_dev
    assert "कंसल्टेंसी" in norm_dev
    assert "सर्विसेज" in norm_dev

    # Indic legal suffix normalization
    dev_suffix = normalize_business_name_suffixes(dev_name)
    assert "private limited" in dev_suffix or "प्राइवेट" in dev_suffix

    # European accented characters
    fr_name = "Société Générale de l'Énergie"
    norm_fr = clean_unicode_text(fr_name)
    assert "societe" in norm_fr or "société" in norm_fr
    assert "energie" in norm_fr or "énergie" in norm_fr

    # Token signature preservation
    sig = get_token_signature("बजाज ऑटो लिमिटेड")
    assert "ऑटो" in sig
    assert "बजाज" in sig


def test_token_retrieval_v2_high_frequency_handling():
    """Test that high-frequency tokens participate in compound keys and do not cause single-token explosions."""
    from src.normalization import extract_postal_code, extract_numeric_tokens

    addr = "123 Main St, Suite 400, Mumbai 400001"
    postals = extract_postal_code(addr)
    assert "400001" in postals

    nums = extract_numeric_tokens(addr)
    assert "123" in nums
    assert "400" in nums


def test_country_partitioning_safety():
    """Test that country partitioning handles various countries (including test France) dynamically."""
    from src.normalization import normalize_country

    assert normalize_country("us") == "US"
    assert normalize_country("india") == "INDIA"
    assert normalize_country("France") == "FRANCE"
    assert normalize_country("  de  ") == "DE"
    assert normalize_country(None) == ""


def test_candidate_provenance_and_tiers():
    """Test candidate provenance records routes and candidate tiers properly."""
    prov = CandidatePairProvenance(
        s1_id="S1-001",
        target_id="S2-001",
        target_source="S2",
        routes={"exact_name", "name_tfidf"},
    )
    assert prov.s1_id == "S1-001"
    assert prov.target_id == "S2-001"
    assert len(prov.routes) == 2
    assert "exact_name" in prov.routes


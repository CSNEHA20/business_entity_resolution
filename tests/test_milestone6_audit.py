"""
Unit & Integration Tests for Milestone 6: Final Pre-Test Audit & Submission Invariants.
Amazon ML Challenge 2026 - Business Entity Resolution
"""

from pathlib import Path
import tempfile
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import pytest

from src.decision_engine import DecisionRuleConfig, EntityDecisionEngine
from src.normalization import (
    extract_building_number,
    extract_numeric_tokens,
    extract_postal_code,
    normalize_address_abbreviations,
    normalize_business_name_suffixes,
    normalize_country,
    tokenize_text,
)
from src.submission import (
    validate_candidate_subset,
    write_candidate_pairs,
    write_matching_results,
)


def test_final_candidate_invariant():
    """All predicted matches in matching_results MUST be present in candidate_pairs."""
    matching = {
        "S1-100": ["S2-200", "S3-300"],
        "S1-101": ["S2-201"],
        "S1-102": [],
    }
    candidates = {
        "S1-100": ["S2-200", "S3-300", "S2-999"],
        "S1-101": ["S2-201"],
        "S1-102": ["S3-500"],
    }

    violations = validate_candidate_subset(matching, candidates)
    assert len(violations) == 0


def test_no_prediction_outside_candidate_pairs_fails_validation():
    """If a prediction is outside candidate pairs, validator MUST flag it."""
    matching = {
        "S1-100": ["S2-200", "S3-300"],
        "S1-101": ["S2-999"],  # NOT in candidates
    }
    candidates = {
        "S1-100": ["S2-200", "S3-300"],
        "S1-101": ["S2-201"],
    }

    violations = validate_candidate_subset(matching, candidates)
    assert len(violations) == 1
    assert "S1-101" in violations[0]


def test_every_s1_represented_exactly_once(tmp_path: Path):
    """Writing matching results must output exactly 1 row per S1 entity with exact format."""
    matching = {
        "S1-001": ["S2-001"],
        "S1-002": [],
        "S1-003": ["S2-003", "S3-003"],
    }
    out_file = tmp_path / "test_matching.tsv"
    write_matching_results(matching, out_file)

    lines = [line.rstrip("\r\n") for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert len(lines) == 4  # Header + 3 entities

    # Check that each S1 ID occurs exactly once
    s1_in_file = [line.split("\t")[0] for line in lines[1:]]
    assert sorted(s1_in_file) == ["S1-001", "S1-002", "S1-003"]

    # Check empty string representation for singleton
    empty_row = next(l for l in lines if l.startswith("S1-002"))
    assert empty_row == "S1-002\t"


def test_valid_source_ids_and_prefixes():
    """Ensure that only S2- and S3- target IDs are accepted as valid match candidates."""
    valid_targets = ["S2-12345", "S3-98765"]
    invalid_targets = ["S1-12345", "S4-12345", "12345", "INVALID"]

    for tid in valid_targets:
        assert tid.startswith(("S2-", "S3-"))
    for tid in invalid_targets:
        assert not tid.startswith(("S2-", "S3-"))


def test_duplicate_prevention_in_submission(tmp_path: Path):
    """Ensure duplicate matched target IDs in predictions list are deduplicated."""
    matching = {
        "S1-001": ["S2-001", "S2-001", "S3-001"],  # duplicate S2-001
    }
    out_file = tmp_path / "dedup_matching.tsv"
    write_matching_results(matching, out_file)

    content = out_file.read_text(encoding="utf-8")
    assert "S2-001,S3-001" in content
    assert "S2-001,S2-001" not in content


def test_chunk_boundary_correctness():
    """Simulate batching S1 entities across chunk boundaries and ensure identical output."""
    cfg = DecisionRuleConfig(
        strategy="adaptive_multi",
        threshold_s2=0.50,
        threshold_s3=0.50,
        min_top_prob=0.45,
        multi_match_threshold=0.50,
        max_multi_score_drop=0.15,
    )
    engine = EntityDecisionEngine(cfg)

    # 100 S1 queries
    all_scores = {f"S1-{i:03d}": {f"S2-{i:03d}": 0.60 + (i % 30) * 0.01} for i in range(100)}
    all_meta = {f"S1-{i:03d}": {"has_addr": (i % 2 == 0)} for i in range(100)}

    # Full batch
    full_preds = engine.predict_all(all_scores, all_meta)

    # Chunked in chunks of 25
    chunk_preds = {}
    s1_keys = list(all_scores.keys())
    for chunk_start in range(0, len(s1_keys), 25):
        chunk_keys = s1_keys[chunk_start:chunk_start + 25]
        sub_scores = {k: all_scores[k] for k in chunk_keys}
        sub_meta = {k: all_meta[k] for k in chunk_keys}
        sub_preds = engine.predict_all(sub_scores, sub_meta)
        chunk_preds.update(sub_preds)

    assert full_preds == chunk_preds


def test_deterministic_inference():
    """Verify that multiple consecutive inference calls on the same inputs return identical results."""
    cfg = DecisionRuleConfig()
    engine = EntityDecisionEngine(cfg)

    scores = {
        "S1-001": {"S2-001": 0.85, "S3-001": 0.82, "S2-002": 0.40},
        "S1-002": {"S2-003": 0.45},
    }
    pred_1 = engine.predict_all(scores)
    pred_2 = engine.predict_all(scores)
    assert pred_1 == pred_2


def test_france_handling_accents_and_codes():
    """Verify open-set support for France: accents, postal codes, country codes, suffixes."""
    assert normalize_country("FR") == "FR"
    assert normalize_country("FRANCE") == "FRANCE"
    assert normalize_country("France") == "FRANCE"
    assert normalize_country("fra") == "FRA"

    # French postal code (5 digits)
    french_addr = "12 Rue de la Paix, 75002 Paris, France"
    postals = extract_postal_code(french_addr)
    assert "75002" in postals

    # French building number
    bldg = extract_building_number(french_addr)
    assert bldg == "12"

    # Accented characters in name normalization and ASCII conversion
    from src.normalization import clean_unicode_ascii
    acc_name = "Soci\u00e9t\u00e9 G\u00e9n\u00e9rale d'\u00c9lectricit\u00e9 SARL"
    norm_name = normalize_business_name_suffixes(acc_name)
    assert "soci" in norm_name
    ascii_name = clean_unicode_ascii(norm_name)
    assert "societe generale" in ascii_name

    # RapidFuzz matching across ASCII and original
    from rapidfuzz import fuzz
    sim = fuzz.token_set_ratio(ascii_name, "societe generale d'electricite sarl")
    assert sim >= 95


def test_multi_match_handling_and_singleton():
    """Verify engine handles 0 matches, 1 match, and multiple matches accurately."""
    cfg = DecisionRuleConfig(
        strategy="adaptive_multi",
        threshold_s2=0.50,
        threshold_s3=0.50,
        min_top_prob=0.45,
        multi_match_threshold=0.50,
        max_multi_score_drop=0.15,
        max_matches_per_source=1,
    )
    engine = EntityDecisionEngine(cfg)

    # 0 matches (singleton)
    res_0 = engine.predict_entity({"S2-001": 0.20, "S3-001": 0.15})
    assert res_0 == []

    # 1 match
    res_1 = engine.predict_entity({"S2-001": 0.80, "S3-001": 0.40})
    assert res_1 == ["S2-001"]

    # 2 matches (one S2, one S3)
    res_2 = engine.predict_entity({"S2-001": 0.85, "S3-001": 0.82})
    assert set(res_2) == {"S2-001", "S3-001"}


def test_missing_address_handling():
    """Verify missing address handling operates with adaptive threshold boost."""
    cfg = DecisionRuleConfig(
        strategy="adaptive_multi",
        threshold_s2=0.50,
        threshold_s3=0.50,
        min_top_prob=0.50,
        missing_addr_threshold_boost=0.05,
    )
    engine = EntityDecisionEngine(cfg)

    # Candidate score 0.52:
    # With address: 0.52 >= 0.50 -> Accepted
    res_with = engine.predict_entity({"S2-001": 0.52}, s1_meta={"has_addr": True})
    assert res_with == ["S2-001"]

    # Without address: 0.52 < (0.50 + 0.05) -> Rejected / Abstain
    res_without = engine.predict_entity({"S2-001": 0.52}, s1_meta={"has_addr": False})
    assert res_without == []

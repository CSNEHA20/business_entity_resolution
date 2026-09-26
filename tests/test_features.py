"""
Unit tests for Pair Feature Engineering Module (Milestone 4).
"""

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, PairFeatureExtractor


@pytest.fixture
def feature_extractor():
    return PairFeatureExtractor()


def test_feature_columns_schema(feature_extractor):
    assert len(FEATURE_COLUMNS) == 51
    assert "name_fuzz_wratio" in FEATURE_COLUMNS
    assert "addr_fuzz_wratio" in FEATURE_COLUMNS
    assert "country_exact_match" in FEATURE_COLUMNS
    assert "name_x_addr_wratio" in FEATURE_COLUMNS


def test_feature_extraction_exact_match(feature_extractor):
    s1_rec = {
        "entity_id": "S1-001",
        "business_name_raw": "Acme Corp",
        "business_address_raw": "123 Main St",
        "business_name_norm": "acme",
        "business_address_norm": "123 main st",
        "country_norm": "US",
        "name_tokens": frozenset(["acme"]),
        "addr_tokens": frozenset(["123", "main", "st"]),
        "postal_codes": {"94105"},
        "building_number": "123",
        "numeric_tokens": {"123"},
    }
    t_rec = {
        "entity_id": "S2-001",
        "business_name_raw": "Acme Corp",
        "business_address_raw": "123 Main St",
        "business_name_norm": "acme",
        "business_address_norm": "123 main st",
        "country_norm": "US",
        "name_tokens": frozenset(["acme"]),
        "addr_tokens": frozenset(["123", "main", "st"]),
        "postal_codes": {"94105"},
        "building_number": "123",
        "numeric_tokens": {"123"},
    }
    feats = feature_extractor.extract_features_for_pair(s1_rec, t_rec)

    assert feats["name_exact_raw"] == 1.0
    assert feats["name_exact_norm"] == 1.0
    assert feats["name_fuzz_ratio"] == 1.0
    assert feats["name_fuzz_wratio"] == 1.0
    assert feats["addr_exact_raw"] == 1.0
    assert feats["country_exact_match"] == 1.0
    assert feats["country_mismatch"] == 0.0
    assert feats["postal_exact_match"] == 1.0
    assert feats["building_exact_match"] == 1.0
    assert feats["name_high_addr_high"] == 1.0
    assert feats["target_is_s2"] == 1.0


def test_feature_extraction_missing_values(feature_extractor):
    s1_rec = {
        "entity_id": "S1-002",
        "business_name_raw": "Solo Corp",
        "business_address_raw": "",
        "business_name_norm": "solo",
        "business_address_norm": "",
        "country_norm": "",
        "name_tokens": frozenset(["solo"]),
        "addr_tokens": frozenset(),
        "postal_codes": set(),
        "building_number": "",
        "numeric_tokens": set(),
    }
    t_rec = {
        "entity_id": "S3-002",
        "business_name_raw": "",
        "business_address_raw": "",
        "business_name_norm": "",
        "business_address_norm": "",
        "country_norm": "",
        "name_tokens": frozenset(),
        "addr_tokens": frozenset(),
        "postal_codes": set(),
        "building_number": "",
        "numeric_tokens": set(),
    }
    feats = feature_extractor.extract_features_for_pair(s1_rec, t_rec)

    # Verify no NaN or Inf in feature dictionary
    for k, v in feats.items():
        assert not np.isnan(v), f"Feature {k} returned NaN on missing values"
        assert not np.isinf(v), f"Feature {k} returned Inf on missing values"

    assert feats["s1_has_address"] == 0.0
    assert feats["target_has_address"] == 0.0
    assert feats["both_have_address"] == 0.0
    assert feats["target_is_s2"] == 0.0


def test_feature_extraction_unicode_support(feature_extractor):
    s1_rec = {
        "entity_id": "S1-003",
        "business_name_raw": "टाटा कंसल्टेंसी",
        "business_address_raw": "बांद्रा कुर्ला संकुल मुंबई",
        "business_name_norm": "टाटा कंसल्टेंसी",
        "business_address_norm": "बांद्रा कुर्ला संकुल मुंबई",
        "country_norm": "IN",
        "name_tokens": frozenset(["टाटा", "कंसल्टेंसी"]),
        "addr_tokens": frozenset(["बांद्रा", "कुर्ला", "संकुल", "मुंबई"]),
        "postal_codes": {"400051"},
        "building_number": "",
        "numeric_tokens": {"400051"},
    }
    t_rec = {
        "entity_id": "S2-003",
        "business_name_raw": "टाटा कंसल्टेंसी",
        "business_address_raw": "बांद्रा कुर्ला संकुल मुंबई",
        "business_name_norm": "टाटा कंसल्टेंसी",
        "business_address_norm": "बांद्रा कुर्ला संकुल मुंबई",
        "country_norm": "IN",
        "name_tokens": frozenset(["टाटा", "कंसल्टेंसी"]),
        "addr_tokens": frozenset(["बांद्रा", "कुर्ला", "संकुल", "मुंबई"]),
        "postal_codes": {"400051"},
        "building_number": "",
        "numeric_tokens": {"400051"},
    }
    feats = feature_extractor.extract_features_for_pair(s1_rec, t_rec)
    assert feats["name_exact_norm"] == 1.0
    assert feats["name_fuzz_wratio"] == 1.0
    assert feats["country_exact_match"] == 1.0


def test_batch_feature_extraction_matrix(feature_extractor):
    s1_lookup = {
        "S1-1": {
            "entity_id": "S1-1",
            "business_name_raw": "Alpha",
            "business_address_raw": "Main St",
            "business_name_norm": "alpha",
            "business_address_norm": "main st",
            "country_norm": "US",
            "name_tokens": frozenset(["alpha"]),
            "addr_tokens": frozenset(["main", "st"]),
            "postal_codes": set(),
            "building_number": "",
            "numeric_tokens": set(),
        }
    }
    t_lookup = {
        "S2-1": {
            "entity_id": "S2-1",
            "business_name_raw": "Alpha Inc",
            "business_address_raw": "Main Street",
            "business_name_norm": "alpha",
            "business_address_norm": "main st",
            "country_norm": "US",
            "name_tokens": frozenset(["alpha"]),
            "addr_tokens": frozenset(["main", "st"]),
            "postal_codes": set(),
            "building_number": "",
            "numeric_tokens": set(),
        }
    }
    pairs = [("S1-1", "S2-1")]
    df = feature_extractor.extract_features_matrix(pairs, s1_lookup, t_lookup)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.shape[1] == len(FEATURE_COLUMNS)
    assert df["name_exact_norm"].iloc[0] == 1.0

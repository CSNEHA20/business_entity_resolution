"""
Unit tests for data loading, schema validation, prefix checks, and edge cases.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data_io import (
    SchemaValidationError,
    audit_dataframe,
    load_ground_truth,
    load_test_source1,
    load_test_source2,
    load_test_source3,
    load_train_source1,
    load_train_source2,
    load_train_source3,
    parse_ground_truth_mapping,
    read_tsv_file,
    validate_ground_truth_schema,
    validate_source_schema,
)


def test_tsv_loading_valid(sample_toy_datasets: Path):
    """Test standard TSV loading with strict tab separation."""
    s1 = load_train_source1(sample_toy_datasets)
    assert len(s1) == 4
    assert list(s1.columns) == ["entity_id", "business_name", "business_address", "country"]

    s2 = load_train_source2(sample_toy_datasets)
    assert len(s2) == 3

    s3 = load_train_source3(sample_toy_datasets)
    assert len(s3) == 3

    gt = load_ground_truth(sample_toy_datasets)
    assert len(gt) == 4
    assert list(gt.columns) == ["source1_entity_id", "matched_entity_ids"]


def test_test_set_loading_with_france(sample_toy_datasets: Path):
    """Test that test set containing France loads seamlessly without country filtering."""
    ts1 = load_test_source1(sample_toy_datasets)
    assert "France" in ts1["country"].values
    assert len(ts1) == 2


def test_schema_missing_column(tmp_path: Path):
    """Test schema validation fails when required column is missing."""
    bad_df = pd.DataFrame([["S1-01", "Acme", "123 St"]], columns=["entity_id", "business_name", "business_address"])
    with pytest.raises(SchemaValidationError, match="schema mismatch"):
        validate_source_schema(bad_df, expected_prefix="S1-", source_name="test")


def test_schema_invalid_id_prefix(tmp_path: Path):
    """Test schema validation fails when entity ID has incorrect source prefix."""
    bad_df = pd.DataFrame([
        ["S2-0001", "Acme", "123 St", "US"]  # S2 ID in S1 source
    ], columns=["entity_id", "business_name", "business_address", "country"])
    with pytest.raises(SchemaValidationError, match="IDs not starting with expected prefix 'S1-'"):
        validate_source_schema(bad_df, expected_prefix="S1-", source_name="s1_test")


def test_schema_duplicate_ids(tmp_path: Path):
    """Test schema validation fails when duplicate entity IDs exist."""
    bad_df = pd.DataFrame([
        ["S1-0001", "Acme Corp", "123 St", "US"],
        ["S1-0001", "Acme Duplicate", "456 St", "US"],
    ], columns=["entity_id", "business_name", "business_address", "country"])
    with pytest.raises(SchemaValidationError, match="duplicate entity_id entries"):
        validate_source_schema(bad_df, expected_prefix="S1-", source_name="s1_test")


def test_schema_blank_entity_id(tmp_path: Path):
    """Test schema validation fails on blank or whitespace-only entity_id."""
    bad_df = pd.DataFrame([
        ["   ", "Acme Corp", "123 St", "US"]
    ], columns=["entity_id", "business_name", "business_address", "country"])
    with pytest.raises(SchemaValidationError, match="null or blank entity_id"):
        validate_source_schema(bad_df, expected_prefix="S1-", source_name="s1_test")


def test_ground_truth_self_match_rejection():
    """Test ground truth rejects self-matching S1 IDs in matched_entity_ids."""
    bad_gt = pd.DataFrame([
        ["S1-0001", "S1-0002"]  # Self match to another S1
    ], columns=["source1_entity_id", "matched_entity_ids"])
    with pytest.raises(SchemaValidationError, match="self-match to Source 1"):
        validate_ground_truth_schema(bad_gt)


def test_ground_truth_invalid_target_prefix():
    """Test ground truth rejects IDs with invalid prefixes (e.g. S4-)."""
    bad_gt = pd.DataFrame([
        ["S1-0001", "S4-0001"]
    ], columns=["source1_entity_id", "matched_entity_ids"])
    with pytest.raises(SchemaValidationError, match="invalid prefix"):
        validate_ground_truth_schema(bad_gt)


def test_ground_truth_duplicate_matched_ids():
    """Test ground truth rejects repeated matched IDs in the same row."""
    bad_gt = pd.DataFrame([
        ["S1-0001", "S2-0001,S2-0001"]
    ], columns=["source1_entity_id", "matched_entity_ids"])
    with pytest.raises(SchemaValidationError, match="duplicate matched IDs"):
        validate_ground_truth_schema(bad_gt)


def test_ground_truth_mapping_parser(sample_toy_datasets: Path):
    """Test ground truth parsing into dictionary mapping."""
    gt = load_ground_truth(sample_toy_datasets)
    mapping = parse_ground_truth_mapping(gt)
    assert mapping["S1-0001"] == ["S2-0001", "S3-0001"]
    assert mapping["S1-0002"] == ["S2-0002", "S3-0002"]
    assert mapping["S1-0003"] == ["S3-0003"]
    assert mapping["S1-0004"] == []  # Singleton


def test_audit_dataframe(sample_toy_datasets: Path):
    """Test audit summary statistics helper."""
    s1 = load_train_source1(sample_toy_datasets)
    audit = audit_dataframe(s1, "train_source1")
    assert audit["num_rows"] == 4
    assert audit["unique_entities"] == 4
    assert audit["country_distribution"] == {"US": 3, "India": 1}

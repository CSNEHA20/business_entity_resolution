"""
Unit tests for submission generation, candidate subset consistency,
and integration with the official validator script.
"""

from pathlib import Path
import pytest

from src.submission import (
    run_official_validator,
    validate_candidate_subset,
    write_candidate_pairs,
    write_matching_results,
    write_submission_tsv,
)


def test_write_submission_tsv_format(tmp_path: Path):
    """Test submission writing format with tab separators and comma ID lists."""
    mapping = {
        "S1-0001": ["S2-0001", "S3-0001"],
        "S1-0002": ["S3-0002"],
        "S1-0003": [],
    }
    out_file = tmp_path / "test_out.tsv"
    write_submission_tsv(mapping, out_file, ["source1_entity_id", "matched_entity_ids"])

    with open(out_file, "r", encoding="utf-8") as f:
        lines = [line.strip("\n") for line in f.readlines()]

    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert lines[1] == "S1-0001\tS2-0001,S3-0001"
    assert lines[2] == "S1-0002\tS3-0002"
    assert lines[3] == "S1-0003\t"


def test_validate_candidate_subset():
    """Test candidate subset validation detects matches not present in candidate list."""
    matching = {
        "S1-01": ["S2-01", "S3-01"],
        "S1-02": ["S2-99"],  # S2-99 not in candidates!
    }
    candidates = {
        "S1-01": ["S2-01", "S3-01", "S3-02"],
        "S1-02": ["S2-02"],
    }
    violations = validate_candidate_subset(matching, candidates)
    assert len(violations) == 1
    assert "S1-02" in violations[0]
    assert "S2-99" in violations[0]


def test_official_validator_pass_on_valid_test_set(sample_toy_datasets: Path, tmp_path: Path):
    """
    Test running the official submission validator script on a valid test submission.
    """
    test_dir = sample_toy_datasets / "test"

    matching = {
        "S1-1001": ["S2-1001"],
        "S1-1002": ["S3-1001"],
    }
    candidates = {
        "S1-1001": ["S2-1001"],
        "S1-1002": ["S3-1001"],
    }

    match_path = tmp_path / "matching_results.tsv"
    cand_path = tmp_path / "candidate_pairs.tsv"

    write_matching_results(matching, match_path)
    write_candidate_pairs(candidates, cand_path)

    exit_code, output = run_official_validator(
        matching_path=match_path,
        candidate_path=cand_path,
        test_dir=test_dir,
        check_ids=True
    )

    assert exit_code == 0, f"Validator failed unexpectedly:\n{output}"
    assert "PASS" in output


def test_official_validator_fails_on_self_match(sample_toy_datasets: Path, tmp_path: Path):
    """Test that official validator rejects self-matches (S1 in matched_entity_ids)."""
    test_dir = sample_toy_datasets / "test"

    bad_matching = {
        "S1-1001": ["S1-1002"],  # S1 ID as target match!
        "S1-1002": [],
    }
    match_path = tmp_path / "matching_bad.tsv"
    write_matching_results(bad_matching, match_path)

    exit_code, output = run_official_validator(
        matching_path=match_path,
        test_dir=test_dir,
        check_ids=False
    )

    assert exit_code == 1
    assert "FAIL" in output
    assert "self-matches" in output

"""
Data input/output and schema validation module for Amazon ML Challenge 2026.
Strictly enforces tab separation (sep="\\t"), UTF-8 encoding, and schema integrity.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from src.config import (
    CANDIDATE_PAIRS_COLUMNS,
    GROUND_TRUTH_COLUMNS,
    MATCHING_RESULTS_COLUMNS,
    SOURCE1_PREFIX,
    SOURCE2_PREFIX,
    SOURCE3_PREFIX,
    SOURCE_COLUMNS,
    VALID_TARGET_PREFIXES,
    get_data_dir,
)

logger = logging.getLogger(__name__)


class SchemaValidationError(ValueError):
    """Raised when dataset fails schema, prefix, or structural validation."""
    pass


def validate_source_schema(
    df: pd.DataFrame,
    expected_prefix: str,
    source_name: str = "source",
    allow_empty_data: bool = False
) -> None:
    """
    Validate that a source dataframe satisfies competition specifications:
    1. Exact columns: [entity_id, business_name, business_address, country]
    2. Non-empty entity_id column (unless allow_empty_data is True for empty files)
    3. Correct entity_id prefix (S1-, S2-, or S3-)
    4. Unique entity_id values (no duplicates)
    5. Country is treated as open-set string (never hard-filtered or rejected)
    """
    if df.empty and not allow_empty_data:
        raise SchemaValidationError(f"{source_name} dataframe is empty.")

    # 1. Column check
    actual_cols = list(df.columns)
    if actual_cols != SOURCE_COLUMNS:
        raise SchemaValidationError(
            f"{source_name} schema mismatch. Expected columns {SOURCE_COLUMNS}, but got {actual_cols}."
        )

    if df.empty:
        return

    # 2. Check null / empty IDs
    if df["entity_id"].isnull().any() or (df["entity_id"].str.strip() == "").any():
        null_count = df["entity_id"].isnull().sum() + (df["entity_id"].str.strip() == "").sum()
        raise SchemaValidationError(
            f"{source_name} contains {null_count} null or blank entity_id records."
        )

    # 3. Check ID prefix
    invalid_prefix_mask = ~df["entity_id"].astype(str).str.startswith(expected_prefix)
    if invalid_prefix_mask.any():
        invalid_samples = df.loc[invalid_prefix_mask, "entity_id"].head(5).tolist()
        raise SchemaValidationError(
            f"{source_name} has {invalid_prefix_mask.sum()} IDs not starting with expected prefix "
            f"'{expected_prefix}'. Examples: {invalid_samples}"
        )

    # 4. Check uniqueness
    if df["entity_id"].duplicated().any():
        dup_count = df["entity_id"].duplicated().sum()
        dup_samples = df.loc[df["entity_id"].duplicated(), "entity_id"].head(5).tolist()
        raise SchemaValidationError(
            f"{source_name} contains {dup_count} duplicate entity_id entries. Examples: {dup_samples}"
        )


def validate_ground_truth_schema(
    df: pd.DataFrame,
    valid_source1_ids: Optional[Set[str]] = None,
    valid_target_ids: Optional[Set[str]] = None
) -> None:
    """
    Validate the ground truth dataframe:
    1. Exact columns: [source1_entity_id, matched_entity_ids]
    2. Unique source1_entity_id
    3. Source 1 prefix 'S1-' for source1_entity_id
    4. matched_entity_ids only contains valid S2- / S3- IDs (no S1 self-matches)
    5. No intra-row duplicate matched IDs
    """
    actual_cols = list(df.columns)
    if actual_cols != GROUND_TRUTH_COLUMNS:
        raise SchemaValidationError(
            f"Ground truth schema mismatch. Expected {GROUND_TRUTH_COLUMNS}, got {actual_cols}."
        )

    if df.empty:
        return

    # Uniqueness of S1 IDs
    if df["source1_entity_id"].duplicated().any():
        dup_samples = df.loc[df["source1_entity_id"].duplicated(), "source1_entity_id"].head(5).tolist()
        raise SchemaValidationError(
            f"Ground truth contains duplicate source1_entity_id entries: {dup_samples}"
        )

    # Check S1 prefix
    invalid_s1 = ~df["source1_entity_id"].astype(str).str.startswith(SOURCE1_PREFIX)
    if invalid_s1.any():
        examples = df.loc[invalid_s1, "source1_entity_id"].head(5).tolist()
        raise SchemaValidationError(
            f"Ground truth has invalid source1_entity_id prefix: {examples}"
        )

    # Check S1 ID membership if reference IDs are provided
    if valid_source1_ids is not None:
        gt_s1_set = set(df["source1_entity_id"])
        unknown_s1 = gt_s1_set - valid_source1_ids
        if unknown_s1:
            raise SchemaValidationError(
                f"Ground truth contains {len(unknown_s1)} source1 IDs not present in train_source1: "
                f"{list(unknown_s1)[:5]}"
            )

    # Check matched_entity_ids content
    for idx, row in df.iterrows():
        matched_str = str(row["matched_entity_ids"]) if pd.notnull(row["matched_entity_ids"]) else ""
        if not matched_str.strip():
            continue

        ids = [m.strip() for m in matched_str.split(",") if m.strip()]
        if len(ids) != len(set(ids)):
            raise SchemaValidationError(
                f"Ground truth row {row['source1_entity_id']} has duplicate matched IDs: {matched_str}"
            )

        for mid in ids:
            if mid.startswith(SOURCE1_PREFIX):
                raise SchemaValidationError(
                    f"Ground truth row {row['source1_entity_id']} contains self-match to Source 1: {mid}"
                )
            if not mid.startswith(VALID_TARGET_PREFIXES):
                raise SchemaValidationError(
                    f"Ground truth row {row['source1_entity_id']} contains ID with invalid prefix: {mid}"
                )
            if valid_target_ids is not None and mid not in valid_target_ids:
                raise SchemaValidationError(
                    f"Ground truth row {row['source1_entity_id']} references nonexistent target ID: {mid}"
                )


def read_tsv_file(path: Union[str, Path], expected_columns: List[str]) -> pd.DataFrame:
    """
    Memory-safe, explicit TSV reader ensuring proper delimiter, UTF-8 encoding,
    and missing value handling.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Requested dataset file not found at: {path.resolve()}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    return df


def _resolve_split_path(
    split: str,
    filename: str,
    base_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Resolve file path across potential directory arrangements."""
    root = Path(base_dir) if base_dir else get_data_dir()
    
    # Check root/split/filename (e.g. data/train/train_source1.tsv)
    p1 = root / split / filename
    if p1.exists():
        return p1
        
    # Check root/filename (e.g. data/train_source1.tsv)
    p2 = root / filename
    if p2.exists():
        return p2
        
    # Check root/dataset/split/filename
    p3 = root / "dataset" / split / filename
    if p3.exists():
        return p3
        
    # Default path for error reporting
    return p1


# ---------------------------------------------------------------------------
# Training Data Loaders
# ---------------------------------------------------------------------------

def load_train_source1(data_dir: Optional[Union[str, Path]] = None, validate: bool = True) -> pd.DataFrame:
    path = _resolve_split_path("train", "train_source1.tsv", data_dir)
    df = read_tsv_file(path, SOURCE_COLUMNS)
    if validate:
        validate_source_schema(df, expected_prefix=SOURCE1_PREFIX, source_name="train_source1")
    return df


def load_train_source2(data_dir: Optional[Union[str, Path]] = None, validate: bool = True) -> pd.DataFrame:
    path = _resolve_split_path("train", "train_source2.tsv", data_dir)
    df = read_tsv_file(path, SOURCE_COLUMNS)
    if validate:
        validate_source_schema(df, expected_prefix=SOURCE2_PREFIX, source_name="train_source2")
    return df


def load_train_source3(data_dir: Optional[Union[str, Path]] = None, validate: bool = True) -> pd.DataFrame:
    path = _resolve_split_path("train", "train_source3.tsv", data_dir)
    df = read_tsv_file(path, SOURCE_COLUMNS)
    if validate:
        validate_source_schema(df, expected_prefix=SOURCE3_PREFIX, source_name="train_source3")
    return df


def load_ground_truth(
    data_dir: Optional[Union[str, Path]] = None,
    validate: bool = True,
    valid_source1_ids: Optional[Set[str]] = None,
    valid_target_ids: Optional[Set[str]] = None
) -> pd.DataFrame:
    path = _resolve_split_path("train", "train_ground_truth.tsv", data_dir)
    df = read_tsv_file(path, GROUND_TRUTH_COLUMNS)
    if validate:
        validate_ground_truth_schema(
            df,
            valid_source1_ids=valid_source1_ids,
            valid_target_ids=valid_target_ids
        )
    return df


# ---------------------------------------------------------------------------
# Test Data Loaders
# ---------------------------------------------------------------------------

def load_test_source1(data_dir: Optional[Union[str, Path]] = None, validate: bool = True) -> pd.DataFrame:
    path = _resolve_split_path("test", "test_source1.tsv", data_dir)
    df = read_tsv_file(path, SOURCE_COLUMNS)
    if validate:
        validate_source_schema(df, expected_prefix=SOURCE1_PREFIX, source_name="test_source1")
    return df


def load_test_source2(data_dir: Optional[Union[str, Path]] = None, validate: bool = True) -> pd.DataFrame:
    path = _resolve_split_path("test", "test_source2.tsv", data_dir)
    df = read_tsv_file(path, SOURCE_COLUMNS)
    if validate:
        validate_source_schema(df, expected_prefix=SOURCE2_PREFIX, source_name="test_source2")
    return df


def load_test_source3(data_dir: Optional[Union[str, Path]] = None, validate: bool = True) -> pd.DataFrame:
    path = _resolve_split_path("test", "test_source3.tsv", data_dir)
    df = read_tsv_file(path, SOURCE_COLUMNS)
    if validate:
        validate_source_schema(df, expected_prefix=SOURCE3_PREFIX, source_name="test_source3")
    return df


# ---------------------------------------------------------------------------
# Submission & Diagnostic Loaders
# ---------------------------------------------------------------------------

def load_matching_results(path: Union[str, Path]) -> pd.DataFrame:
    """Load matching_results.tsv file with strict schema validation."""
    return read_tsv_file(path, MATCHING_RESULTS_COLUMNS)


def load_candidate_pairs(path: Union[str, Path]) -> pd.DataFrame:
    """Load candidate_pairs.tsv file with strict schema validation."""
    return read_tsv_file(path, CANDIDATE_PAIRS_COLUMNS)


# ---------------------------------------------------------------------------
# Audit and Ground Truth Parsing Helpers
# ---------------------------------------------------------------------------

def parse_ground_truth_mapping(gt_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Converts ground truth dataframe to a dictionary mapping:
    source1_entity_id -> list of matched_entity_ids (empty list for singletons).
    """
    mapping: Dict[str, List[str]] = {}
    for _, row in gt_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        matched_str = str(row["matched_entity_ids"]).strip() if pd.notnull(row["matched_entity_ids"]) else ""
        if not matched_str:
            mapping[s1_id] = []
        else:
            ids = [m.strip() for m in matched_str.split(",") if m.strip()]
            mapping[s1_id] = ids
    return mapping


def audit_dataframe(df: pd.DataFrame, name: str) -> Dict[str, Union[int, float, Dict]]:
    """
    Generates summary audit statistics for a source or ground truth dataframe.
    """
    stats: Dict[str, Union[int, float, Dict]] = {
        "name": name,
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
    }

    if "entity_id" in df.columns:
        stats["unique_entities"] = df["entity_id"].nunique()
        stats["empty_names"] = int((df["business_name"].str.strip() == "").sum())
        stats["empty_addresses"] = int((df["business_address"].str.strip() == "").sum())
        stats["empty_countries"] = int((df["country"].str.strip() == "").sum())
        stats["country_distribution"] = df["country"].value_counts().to_dict()

    if "matched_entity_ids" in df.columns:
        mapping = parse_ground_truth_mapping(df)
        match_counts = [len(m) for m in mapping.values()]
        stats["num_s1_entities"] = len(df)
        stats["num_singletons"] = sum(1 for c in match_counts if c == 0)
        stats["num_1_match"] = sum(1 for c in match_counts if c == 1)
        stats["num_multi_match"] = sum(1 for c in match_counts if c > 1)
        stats["max_matches_per_s1"] = max(match_counts) if match_counts else 0
        stats["total_matched_pairs"] = sum(match_counts)

    return stats

"""
Submission module for Amazon ML Challenge 2026.
Formats, writes, and validates matching_results.tsv and candidate_pairs.tsv.
"""

import logging
from pathlib import Path
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Set, Union

import pandas as pd

from src.config import (
    CANDIDATE_PAIRS_COLUMNS,
    CANDIDATE_PAIRS_FILENAME,
    MATCHING_RESULTS_COLUMNS,
    MATCHING_RESULTS_FILENAME,
    OUTPUT_DIR,
    VALIDATOR_SCRIPT,
)

logger = logging.getLogger(__name__)


def write_submission_tsv(
    mapping: Dict[str, Union[List[str], Set[str]]],
    output_path: Union[str, Path],
    columns: List[str]
) -> Path:
    """
    Writes a dictionary of S1 -> list of IDs to a competition-compliant TSV file.
    
    Guarantees:
    - Exactly two columns: [columns[0], columns[1]]
    - Separated by '\\t' (no comma delimiter for columns)
    - Comma-separated list for IDs, with no quoting
    - Deduplicated target IDs
    - Clean UTF-8 encoding without BOM
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1_id, targets in sorted(mapping.items()):
        # Deduplicate while preserving order if list
        if isinstance(targets, set):
            target_list = sorted(list(targets))
        else:
            seen = set()
            target_list = []
            for t in targets:
                if t and t not in seen:
                    seen.add(t)
                    target_list.append(t)
                    
        joined_targets = ",".join(target_list)
        rows.append(f"{s1_id}\t{joined_targets}\n")

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write(f"{columns[0]}\t{columns[1]}\n")
        f.writelines(rows)

    logger.info(f"Successfully wrote {len(rows)} rows to {output_path}")
    return output_path


def write_matching_results(
    predictions: Dict[str, Union[List[str], Set[str]]],
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """Writes final matching_results.tsv."""
    dest = Path(output_path) if output_path else OUTPUT_DIR / MATCHING_RESULTS_FILENAME
    return write_submission_tsv(predictions, dest, MATCHING_RESULTS_COLUMNS)


def write_candidate_pairs(
    candidates: Dict[str, Union[List[str], Set[str]]],
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """Writes candidate_pairs.tsv."""
    dest = Path(output_path) if output_path else OUTPUT_DIR / CANDIDATE_PAIRS_FILENAME
    return write_submission_tsv(candidates, dest, CANDIDATE_PAIRS_COLUMNS)


def validate_candidate_subset(
    matching: Dict[str, Union[List[str], Set[str]]],
    candidates: Dict[str, Union[List[str], Set[str]]]
) -> List[str]:
    """
    Validates that every matched ID in matching_results is present in candidate_pairs.
    Returns list of warning messages for any offending S1 entities.
    """
    violations = []
    for s1_id, matches in matching.items():
        match_set = set(matches)
        cand_set = set(candidates.get(s1_id, []))
        diff = match_set - cand_set
        if diff:
            violations.append(
                f"S1 entity {s1_id} has matches {diff} that were not in candidate_pairs."
            )
    return violations


def run_official_validator(
    matching_path: Union[str, Path],
    candidate_path: Optional[Union[str, Path]] = None,
    test_dir: Optional[Union[str, Path]] = None,
    check_ids: bool = False
) -> Tuple[int, str]:
    """
    Invokes the official validate_submission.py script.
    
    Returns:
        (exit_code, output_text)
    """
    if not VALIDATOR_SCRIPT.exists():
        raise FileNotFoundError(f"Official validator script not found at {VALIDATOR_SCRIPT}")

    cmd = [
        sys.executable,
        str(VALIDATOR_SCRIPT),
        "--matching", str(matching_path),
    ]

    if candidate_path and Path(candidate_path).exists():
        cmd.extend(["--candidate", str(candidate_path)])

    if test_dir:
        cmd.extend(["--test-dir", str(test_dir)])

    if check_ids:
        cmd.append("--check-ids")

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + "\n" + result.stderr
    return result.returncode, output

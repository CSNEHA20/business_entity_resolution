"""
Configuration module for Amazon ML Challenge 2026 - Business Entity Resolution.
Centralizes paths, hyperparameters, schema definitions, and runtime settings.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Tuple

# Base Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Directories (Checks data/ first, then dataset/ fallback)
DATA_DIR = PROJECT_ROOT / "data"
ALT_DATA_DIR = PROJECT_ROOT / "dataset"
FALLBACK_DATA_DIR = PROJECT_ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"

OUTPUT_DIR = PROJECT_ROOT / "output"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
DIAGNOSTICS_DIR = ARTIFACTS_DIR / "diagnostics"
FEATURES_DIR = ARTIFACTS_DIR / "features"
RETRIEVAL_DIR = ARTIFACTS_DIR / "retrieval"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Official Submission Validator Script Path
VALIDATOR_SCRIPT = PROJECT_ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "utils" / "validate_submission.py"

# Experiment Tracking Files
EXPERIMENT_LOG_PATH = EXPERIMENTS_DIR / "experiment_log.csv"
VALIDATION_SCORES_PATH = EXPERIMENTS_DIR / "validation_scores.csv"
SUBMISSION_HISTORY_PATH = EXPERIMENTS_DIR / "submission_history.csv"

# Output File Names (Strictly per competition specs)
MATCHING_RESULTS_FILENAME = "matching_results.tsv"
CANDIDATE_PAIRS_FILENAME = "candidate_pairs.tsv"

# Schema Definitions
SOURCE_COLUMNS: List[str] = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS: List[str] = ["source1_entity_id", "matched_entity_ids"]
MATCHING_RESULTS_COLUMNS: List[str] = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_PAIRS_COLUMNS: List[str] = ["source1_entity_id", "candidate_entity_ids"]

# Prefix Conventions
SOURCE1_PREFIX = "S1-"
SOURCE2_PREFIX = "S2-"
SOURCE3_PREFIX = "S3-"
VALID_TARGET_PREFIXES: Tuple[str, ...] = (SOURCE2_PREFIX, SOURCE3_PREFIX)


@dataclass(frozen=True)
class PipelineConfig:
    """Master pipeline configuration parameters."""
    
    # Reproducibility
    random_seed: int = 42
    
    # Validation Split
    val_size: float = 0.2
    val_split_seed: int = 42
    stratify_by_match_count: bool = True
    
    # Blocking / Retrieval Settings
    top_k_candidates_per_pass: int = 30
    max_total_candidates_per_s1: int = 60
    char_ngram_range: Tuple[int, int] = (2, 4)
    word_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_max_features: int = 150000
    tfidf_min_df: int = 2
    tfidf_sublinear_tf: bool = True
    
    # Hard Negative Mining
    hard_negative_ratio: int = 5
    random_negative_ratio: int = 2
    
    # Model Training
    model_type: str = "xgboost"
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    tree_method: str = "hist"
    
    # Threshold Tuning & Decision Layer
    threshold_search_start: float = 0.20
    threshold_search_end: float = 0.85
    threshold_search_step: float = 0.01
    default_decision_threshold: float = 0.50
    enable_singleton_abstention: bool = True
    singleton_score_margin: float = 0.05
    
    # Batch & Memory Management
    batch_size: int = 50000
    n_jobs: int = -1
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# Default active configuration instance
DEFAULT_CONFIG = PipelineConfig()


def get_data_dir() -> Path:
    """
    Resolve the active data directory.
    Checks data/, dataset/, and fallback locations in order.
    """
    for candidate in [DATA_DIR, ALT_DATA_DIR, FALLBACK_DATA_DIR]:
        if candidate.exists() and any(candidate.iterdir()):
            return candidate
    return DATA_DIR

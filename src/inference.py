"""
End-to-End Inference Engine for Amazon ML Challenge 2026.
Coordinates candidate retrieval, feature extraction, scoring, and output formulation.
(To be wired up in later milestones).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd

from src.config import DEFAULT_CONFIG, PipelineConfig
from src.submission import write_candidate_pairs, write_matching_results

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Orchestrates prediction generation for test entities."""

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.config = config

    def predict_and_export(
        self,
        test_s1: pd.DataFrame,
        test_s2: pd.DataFrame,
        test_s3: pd.DataFrame,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Tuple[Path, Path]:
        """
        Executes end-to-end inference and writes matching_results.tsv and candidate_pairs.tsv.
        """
        logger.info("Executing test inference pipeline...")
        # Scaffolding placeholder
        empty_matches: Dict[str, Set[str]] = {s: set() for s in test_s1["entity_id"]}
        m_path = write_matching_results(empty_matches)
        c_path = write_candidate_pairs(empty_matches)
        return m_path, c_path

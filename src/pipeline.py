"""
Master Pipeline Orchestrator for Amazon ML Challenge 2026.
Ties together data loading, normalization, blocking, features, modeling, validation, and submission.
"""

import logging
from typing import Optional
from src.config import DEFAULT_CONFIG, PipelineConfig

logger = logging.getLogger(__name__)


class EntityResolutionPipeline:
    """Master Pipeline for Entity Resolution."""

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.config = config

    def run_training_pipeline(self) -> None:
        """Runs the full training and validation pipeline."""
        logger.info("Initializing Entity Resolution training pipeline...")
        # Detailed execution to be wired in subsequent milestones

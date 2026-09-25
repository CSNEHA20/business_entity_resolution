"""
Threshold Optimization and Decision Calibration Module for Amazon ML Challenge 2026.
Optimizes decision boundaries directly against Macro F_0.5.
"""

import logging
from typing import Dict, List, Set, Tuple
import numpy as np

from src.config import DEFAULT_CONFIG, PipelineConfig
from src.metrics import compute_macro_f05

logger = logging.getLogger(__name__)


def find_optimal_threshold(
    ground_truth: Dict[str, Set[str]],
    pair_scores: Dict[str, Dict[str, float]],
    config: PipelineConfig = DEFAULT_CONFIG
) -> Tuple[float, float]:
    """
    Finds the decision threshold that maximizes macro F_0.5 on validation data.
    
    Args:
        ground_truth: mapping s1_id -> set of true matched IDs
        pair_scores: mapping s1_id -> {target_id: score}
        
    Returns:
        (best_threshold, best_macro_f05)
    """
    best_threshold = config.default_decision_threshold
    best_score = -1.0

    thresholds = np.arange(
        config.threshold_search_start,
        config.threshold_search_end + 1e-5,
        config.threshold_search_step
    )

    for thresh in thresholds:
        predictions: Dict[str, Set[str]] = {}
        for s1_id, scores_map in pair_scores.items():
            matched = {tid for tid, sc in scores_map.items() if sc >= thresh}
            predictions[s1_id] = matched

        metrics = compute_macro_f05(ground_truth, predictions)
        f05 = metrics["macro_f05"]
        if f05 > best_score:
            best_score = f05
            best_threshold = float(thresh)

    logger.info(f"Optimal threshold found: {best_threshold:.3f} (Validation Macro F0.5: {best_score:.4f})")
    return best_threshold, best_score

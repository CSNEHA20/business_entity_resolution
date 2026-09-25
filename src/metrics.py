"""
Official Evaluation Metric Module for Amazon ML Challenge 2026.
Implements the macro-averaged F_0.5 score with exact singleton handling
and entity-level precision and recall metrics.
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd


def compute_entity_f05(
    y_true: Set[str],
    y_pred: Set[str],
    beta: float = 0.5
) -> Tuple[float, float, float]:
    """
    Computes precision, recall, and F_0.5 for a single Source 1 entity.
    
    Formula:
        Precision = |True Matches ∩ Predicted Matches| / |Predicted Matches|
        Recall = |True Matches ∩ Predicted Matches| / |True Matches|
        F_beta = (1 + beta^2) * Precision * Recall / (beta^2 * Precision + Recall)
        For beta = 0.5:
        F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    
    Singleton logic per official specification:
    - true empty + predicted empty -> Precision=1.0, Recall=1.0, F_0.5=1.0 (perfect score)
    - true empty + predicted non-empty -> Precision=0.0, Recall=0.0, F_0.5=0.0 (false positive merge)
    - true non-empty + predicted empty -> Precision=0.0, Recall=0.0, F_0.5=0.0 (missed match)
    """
    # Case 1: Ground truth is singleton (no true matches)
    if len(y_true) == 0:
        if len(y_pred) == 0:
            return 1.0, 1.0, 1.0  # Correct singleton prediction
        else:
            return 0.0, 0.0, 0.0  # False merge on singleton

    # Case 2: Ground truth has matches, but prediction is empty (abstention / missed match)
    if len(y_pred) == 0:
        return 0.0, 0.0, 0.0

    # Case 3: Both true and predicted have matches
    true_positives = len(y_true & y_pred)
    precision = true_positives / len(y_pred)
    recall = true_positives / len(y_true)

    beta_sq = beta ** 2
    denominator = (beta_sq * precision) + recall
    if denominator == 0.0:
        f_score = 0.0
    else:
        f_score = (1.0 + beta_sq) * precision * recall / denominator

    return precision, recall, f_score


def compute_macro_f05(
    ground_truth: Dict[str, Union[List[str], Set[str]]],
    predictions: Dict[str, Union[List[str], Set[str]]],
    beta: float = 0.5
) -> Dict[str, float]:
    """
    Computes macro-averaged F_0.5, Precision, and Recall across all Source 1 entities.
    
    Args:
        ground_truth: Mapping from source1_id -> collection of true matched target IDs.
        predictions: Mapping from source1_id -> collection of predicted matched target IDs.
        beta: Weighting parameter (default 0.5 for F_0.5).
        
    Returns:
        Dictionary containing:
        - macro_f05: The primary competition evaluation metric
        - macro_precision: Average precision across all S1 entities
        - macro_recall: Average recall across all S1 entities
        - total_entities: Number of evaluated Source 1 entities
        - singleton_accuracy: Accuracy on true singleton entities
    """
    all_s1_ids = set(ground_truth.keys())
    pred_s1_ids = set(predictions.keys())

    # Ensure every ground truth entity is evaluated
    missing_in_pred = all_s1_ids - pred_s1_ids
    if missing_in_pred:
        raise ValueError(
            f"Predictions missing {len(missing_in_pred)} Source 1 entities. "
            f"Every entity must be present. Example missing: {list(missing_in_pred)[:5]}"
        )

    precisions: List[float] = []
    recalls: List[float] = []
    f_scores: List[float] = []
    
    singleton_correct = 0
    total_singletons = 0

    for s1_id, true_list in ground_truth.items():
        y_true = set(true_list) if not isinstance(true_list, set) else true_list
        pred_list = predictions.get(s1_id, set())
        y_pred = set(pred_list) if not isinstance(pred_list, set) else pred_list

        p, r, f = compute_entity_f05(y_true, y_pred, beta=beta)
        precisions.append(p)
        recalls.append(r)
        f_scores.append(f)

        if len(y_true) == 0:
            total_singletons += 1
            if len(y_pred) == 0:
                singleton_correct += 1

    return {
        "macro_f05": float(np.mean(f_scores)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "total_entities": len(all_s1_ids),
        "total_singletons": total_singletons,
        "singleton_accuracy": (singleton_correct / total_singletons) if total_singletons > 0 else 1.0,
    }


def compute_candidate_recall(
    ground_truth: Dict[str, Union[List[str], Set[str]]],
    candidates: Dict[str, Union[List[str], Set[str]]]
) -> Dict[str, float]:
    """
    Measures blocking quality and the recall ceiling.
    Determines what proportion of all true matches are captured in the candidate set.
    """
    total_true_matches = 0
    retrieved_true_matches = 0
    candidate_counts: List[int] = []

    for s1_id, true_matches in ground_truth.items():
        true_set = set(true_matches)
        cand_set = set(candidates.get(s1_id, []))
        
        total_true_matches += len(true_set)
        retrieved_true_matches += len(true_set & cand_set)
        candidate_counts.append(len(cand_set))

    recall_ceiling = (retrieved_true_matches / total_true_matches) if total_true_matches > 0 else 1.0
    avg_candidates = float(np.mean(candidate_counts)) if candidate_counts else 0.0

    return {
        "candidate_recall": recall_ceiling,
        "total_true_matches": total_true_matches,
        "retrieved_true_matches": retrieved_true_matches,
        "avg_candidates_per_s1": avg_candidates,
    }

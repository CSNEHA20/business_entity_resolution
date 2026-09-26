"""
Entity-Level Decision Engine Module for Amazon ML Challenge 2026.

Transforms candidate pair probabilities and features into optimal entity-level
match sets (0, 1, or multiple matches per Source 1 entity).

Features:
- Singleton abstention mechanisms (top probability and margin gating)
- Source-aware decision boundaries (distinct thresholds for S2 and S3)
- Multi-match candidate selection with relative score drop constraints
- Missing-address adaptive thresholding
- Fast vectorized / dictionary-based evaluation
- Model serialization and hyperparameter tuning
"""

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np

from src.metrics import compute_macro_f05

logger = logging.getLogger(__name__)


@dataclass
class DecisionRuleConfig:
    """Configuration parameters for entity-level decision rules."""
    strategy: str = "adaptive_multi"  # 'global', 'source_specific', 'margin_abstention', 'adaptive_multi'
    global_threshold: float = 0.65
    threshold_s2: float = 0.62
    threshold_s3: float = 0.68
    min_top_prob: float = 0.58
    min_margin: float = 0.05
    multi_match_threshold: float = 0.60
    max_multi_score_drop: float = 0.12
    max_matches_per_source: int = 1
    missing_addr_threshold_boost: float = 0.05
    enable_multi_match: bool = True
    enable_singleton_abstention: bool = True


class EntityDecisionEngine:
    """
    Decides the final set of matched target entity IDs for each Source 1 query entity.
    """

    def __init__(self, config: Optional[DecisionRuleConfig] = None):
        self.config = config or DecisionRuleConfig()

    def predict_entity(
        self,
        candidate_scores: Dict[str, float],
        s1_meta: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Decides matched target IDs for a single S1 entity given candidate probabilities.

        Args:
            candidate_scores: Dict mapping target_id -> predicted probability (in [0, 1]).
            s1_meta: Optional metadata dict for the S1 query (e.g., has_addr, country).

        Returns:
            List of selected target entity IDs (may be empty for singletons).
        """
        if not candidate_scores:
            return []

        # Sort candidates descending by probability score
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])
        
        # Deduplicate target IDs if any
        seen_targets = set()
        clean_sorted: List[Tuple[str, float]] = []
        for tid, score in sorted_candidates:
            if tid not in seen_targets:
                seen_targets.add(tid)
                clean_sorted.append((tid, float(score)))

        if not clean_sorted:
            return []

        cfg = self.config
        top_tid, top_prob = clean_sorted[0]
        second_prob = clean_sorted[1][1] if len(clean_sorted) > 1 else 0.0
        margin = top_prob - second_prob

        # Determine if S1 has missing address
        has_addr = True
        if s1_meta is not None:
            has_addr = bool(s1_meta.get("has_addr", True))
        addr_boost = 0.0 if has_addr else cfg.missing_addr_threshold_boost

        # -------------------------------------------------------------
        # Strategy A: Simple Global Threshold
        # -------------------------------------------------------------
        if cfg.strategy == "global":
            thresh = cfg.global_threshold + addr_boost
            return [tid for tid, score in clean_sorted if score >= thresh]

        # -------------------------------------------------------------
        # Strategy B: Source-Specific Thresholds
        # -------------------------------------------------------------
        if cfg.strategy == "source_specific":
            selected = []
            for tid, score in clean_sorted:
                is_s2 = tid.startswith("S2-")
                thresh = (cfg.threshold_s2 if is_s2 else cfg.threshold_s3) + addr_boost
                if score >= thresh:
                    selected.append(tid)
            return selected

        # -------------------------------------------------------------
        # Strategy C: Margin Abstention (Strict Singleton Protection)
        # -------------------------------------------------------------
        if cfg.strategy == "margin_abstention":
            if cfg.enable_singleton_abstention:
                if top_prob < (cfg.min_top_prob + addr_boost):
                    return []
                # If top candidate is borderline and margin is tiny, abstain
                if top_prob < (cfg.global_threshold + addr_boost) and margin < cfg.min_margin:
                    return []

            thresh = cfg.global_threshold + addr_boost
            return [tid for tid, score in clean_sorted if score >= thresh]

        # -------------------------------------------------------------
        # Strategy D: Adaptive Multi-Match Decision Logic (Default / Recommended)
        # -------------------------------------------------------------
        # 1. Singleton check: Does top candidate meet minimum confidence?
        top_is_s2 = top_tid.startswith("S2-")
        top_base_thresh = cfg.threshold_s2 if top_is_s2 else cfg.threshold_s3
        effective_top_thresh = max(cfg.min_top_prob, top_base_thresh) + addr_boost

        if cfg.enable_singleton_abstention:
            if top_prob < effective_top_thresh:
                return []
            if top_prob < (top_base_thresh + addr_boost + 0.05) and margin < cfg.min_margin and len(clean_sorted) > 1:
                # Ambiguous collision near decision boundary -> abstain to protect precision
                return []

        # Top candidate accepted
        selected: List[str] = [top_tid]
        s2_count = 1 if top_is_s2 else 0
        s3_count = 0 if top_is_s2 else 1

        if not cfg.enable_multi_match:
            return selected

        # 2. Multi-match evaluation for secondary candidates
        for tid, score in clean_sorted[1:]:
            is_s2 = tid.startswith("S2-")
            cand_thresh = (cfg.threshold_s2 if is_s2 else cfg.threshold_s3) + addr_boost
            multi_thresh = max(cand_thresh, cfg.multi_match_threshold + addr_boost)

            # Check score threshold
            if score < multi_thresh:
                continue

            # Check relative score drop from top candidate
            score_drop = top_prob - score
            if score_drop > cfg.max_multi_score_drop:
                continue

            # Check source limits if configured
            if cfg.max_matches_per_source > 0:
                if is_s2 and s2_count >= cfg.max_matches_per_source:
                    continue
                if not is_s2 and s3_count >= cfg.max_matches_per_source:
                    continue

            selected.append(tid)
            if is_s2:
                s2_count += 1
            else:
                s3_count += 1

        return selected

    def predict_all(
        self,
        all_candidate_scores: Dict[str, Dict[str, float]],
        s1_meta_dict: Optional[Dict[str, Dict[str, Any]]] = None,
        all_s1_ids: Optional[Iterable[str]] = None,
    ) -> Dict[str, List[str]]:
        """
        Runs decision engine across all S1 queries.

        Args:
            all_candidate_scores: Dict mapping s1_id -> {target_id: score}
            s1_meta_dict: Dict mapping s1_id -> metadata dict
            all_s1_ids: Optional list/set of all S1 IDs to guarantee full coverage.

        Returns:
            Dict mapping s1_id -> list of selected target_ids.
        """
        predictions: Dict[str, List[str]] = {}
        if all_s1_ids is not None:
            target_s1_ids = set(all_s1_ids)
        elif s1_meta_dict is not None:
            target_s1_ids = set(s1_meta_dict.keys())
        else:
            target_s1_ids = set(all_candidate_scores.keys())

        for sid in target_s1_ids:
            scores = all_candidate_scores.get(sid, {})
            meta = s1_meta_dict.get(sid) if s1_meta_dict else None
            predictions[sid] = self.predict_entity(scores, s1_meta=meta)

        return predictions

    def evaluate(
        self,
        ground_truth: Dict[str, Union[List[str], Set[str]]],
        all_candidate_scores: Dict[str, Dict[str, float]],
        s1_meta_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """
        Evaluates the decision engine on a ground-truth dataset.
        """
        preds = self.predict_all(all_candidate_scores, s1_meta_dict, all_s1_ids=ground_truth.keys())
        return compute_macro_f05(ground_truth, preds)

    def save(self, filepath: Union[str, Path]):
        """Saves configuration to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "EntityDecisionEngine":
        """Loads configuration from JSON."""
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = DecisionRuleConfig(**data)
        return cls(cfg)

"""
Diagnostics and Error Analysis Module for Amazon ML Challenge 2026.
Analyzes blocking recall ceilings, false positives, false negatives, and singleton performance.
"""

import logging
from typing import Any, Dict, List, Set
import pandas as pd

logger = logging.getLogger(__name__)


def generate_error_breakdown(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    s1_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generates a detailed per-entity error diagnostic dataframe.
    """
    s1_lookup = s1_df.set_index("entity_id").to_dict(orient="index")
    records = []

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        
        tp = true_set & pred_set
        fp = pred_set - true_set
        fn = true_set - pred_set

        s1_info = s1_lookup.get(s1_id, {})
        
        records.append({
            "source1_entity_id": s1_id,
            "business_name": s1_info.get("business_name", ""),
            "country": s1_info.get("country", ""),
            "num_true": len(true_set),
            "num_pred": len(pred_set),
            "num_tp": len(tp),
            "num_fp": len(fp),
            "num_fn": len(fn),
            "is_singleton": len(true_set) == 0,
            "error_type": "none" if (not fp and not fn) else ("fp_merge" if fp else "fn_miss")
        })

    return pd.DataFrame(records)

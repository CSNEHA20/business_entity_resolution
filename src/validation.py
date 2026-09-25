"""
Validation Strategy Module for Amazon ML Challenge 2026.
Provides reproducible entity-level cross-validation and evaluation splits.
"""

import logging
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG, PipelineConfig
from src.data_io import parse_ground_truth_mapping

logger = logging.getLogger(__name__)


def create_entity_validation_split(
    s1_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    val_size: float = 0.2,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits Source 1 records and corresponding Ground Truth records into
    Train and Validation sets without entity leakage.
    Stratifies by singleton vs matched count distribution.
    """
    mapping = parse_ground_truth_mapping(gt_df)
    
    # Categorize S1 into bins (0: singleton, 1: single match, 2: multi match)
    s1_ids = s1_df["entity_id"].tolist()
    bins = []
    for sid in s1_ids:
        m_count = len(mapping.get(sid, []))
        if m_count == 0:
            bins.append(0)
        elif m_count == 1:
            bins.append(1)
        else:
            bins.append(2)

    rng = np.random.RandomState(random_seed)
    
    train_ids_set: Set[str] = set()
    val_ids_set: Set[str] = set()
    
    df_temp = pd.DataFrame({"entity_id": s1_ids, "bin": bins})
    for b in [0, 1, 2]:
        group_ids = df_temp[df_temp["bin"] == b]["entity_id"].values
        rng.shuffle(group_ids)
        n_val = int(len(group_ids) * val_size)
        val_ids_set.update(group_ids[:n_val])
        train_ids_set.update(group_ids[n_val:])

    train_s1 = s1_df[s1_df["entity_id"].isin(train_ids_set)].reset_index(drop=True)
    val_s1 = s1_df[s1_df["entity_id"].isin(val_ids_set)].reset_index(drop=True)
    
    train_gt = gt_df[gt_df["source1_entity_id"].isin(train_ids_set)].reset_index(drop=True)
    val_gt = gt_df[gt_df["source1_entity_id"].isin(val_ids_set)].reset_index(drop=True)

    logger.info(
        f"Validation split created: Train S1={len(train_s1)}, Val S1={len(val_s1)} "
        f"(Val singletons={sum(1 for s in val_s1['entity_id'] if len(mapping.get(s, []))==0)})"
    )

    return train_s1, val_s1, train_gt, val_gt

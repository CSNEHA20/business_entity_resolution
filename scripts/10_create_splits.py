"""
Milestone 10: Clean Entity-Disjoint Split Generator & Manifest
Constructs TRAIN (12,000), DEV (5,000), and HOLDOUT (5,000) splits.
Guarantees:
- Zero overlap with any historically exposed entity IDs from M1-M9.
- Strictly entity-disjoint between TRAIN, DEV, and HOLDOUT.
- Stratified across match counts (0 matches, 1 match, 2+ matches).
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("m10_splits")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "train"
OUT_DIR = ROOT_DIR / "artifacts" / "milestone10" / "splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_and_save_splits():
    s1_path = DATA_DIR / "train_source1.tsv"
    gt_path = DATA_DIR / "train_ground_truth.tsv"

    logger.info("Loading S1 and Ground Truth to construct verified splits...")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)

    s1_ids_all = s1_df["entity_id"].values
    total_s1 = len(s1_ids_all)

    gt_map: Dict[str, Set[str]] = defaultdict(set)
    for sid, m_str in zip(gt_df["source1_entity_id"].values, gt_df["matched_entity_ids"].values):
        sid = str(sid).strip()
        m_str = str(m_str).strip()
        if m_str:
            for mid in m_str.split(","):
                gt_map[sid].add(mid.strip())

    # Exclude historical entities:
    # 1. First 5,000 entities
    prior_exposed_ids: Set[str] = set()
    for sid in s1_ids_all[:5000]:
        prior_exposed_ids.add(str(sid).strip())

    # 2. Milestone 4 20% validation pool
    s1_bins = np.array(
        [0 if len(gt_map.get(sid, [])) == 0 else (1 if len(gt_map.get(sid, [])) == 1 else 2) for sid in s1_ids_all],
        dtype=np.int32,
    )
    rng4 = np.random.RandomState(42)
    val_indices_4 = []
    for b in [0, 1, 2]:
        b_idx = np.where(s1_bins == b)[0]
        rng4.shuffle(b_idx)
        n_val = int(len(b_idx) * 0.20)
        val_indices_4.extend(b_idx[:n_val])
    for idx in val_indices_4:
        prior_exposed_ids.add(str(s1_ids_all[idx]).strip())

    # 3. Milestone 7 Dev/Holdout
    rng7 = np.random.RandomState(42)
    shuffled_ids_7 = rng7.permutation(s1_df["entity_id"].astype(str).str.strip().tolist())
    for sid in shuffled_ids_7[:20000]:
        prior_exposed_ids.add(str(sid).strip())

    logger.info(f"Total historically exposed IDs excluded: {len(prior_exposed_ids):,}")

    remaining_indices = [i for i, sid in enumerate(s1_ids_all) if str(sid).strip() not in prior_exposed_ids]
    logger.info(f"Virgin unexposed S1 entities available: {len(remaining_indices):,}")

    rng_new = np.random.RandomState(2026)
    rem_bins = s1_bins[remaining_indices]

    fresh_val_pool = []
    fresh_train_pool = []
    for b in [0, 1, 2]:
        b_sub = np.where(rem_bins == b)[0]
        rng_new.shuffle(b_sub)
        n_val = int(len(b_sub) * 0.15)
        fresh_val_pool.extend([remaining_indices[i] for i in b_sub[:n_val]])
        fresh_train_pool.extend([remaining_indices[i] for i in b_sub[n_val:]])

    rng_new.shuffle(fresh_val_pool)
    rng_new.shuffle(fresh_train_pool)

    dev_indices = fresh_val_pool[:5000]
    holdout_indices = fresh_val_pool[5000:10000]
    train_indices = fresh_train_pool[:12000]

    dev_ids = [str(s1_ids_all[i]).strip() for i in dev_indices]
    holdout_ids = [str(s1_ids_all[i]).strip() for i in holdout_indices]
    train_ids = [str(s1_ids_all[i]).strip() for i in train_indices]

    # Invariant checks
    assert len(set(dev_ids) & prior_exposed_ids) == 0, "DEV contains exposed IDs!"
    assert len(set(holdout_ids) & prior_exposed_ids) == 0, "HOLDOUT contains exposed IDs!"
    assert len(set(train_ids) & set(dev_ids)) == 0, "TRAIN overlaps DEV!"
    assert len(set(train_ids) & set(holdout_ids)) == 0, "TRAIN overlaps HOLDOUT!"
    assert len(set(dev_ids) & set(holdout_ids)) == 0, "DEV overlaps HOLDOUT!"

    def hash_ids(id_list: List[str]) -> str:
        return hashlib.sha256("\n".join(sorted(id_list)).encode("utf-8")).hexdigest()

    manifest = {
        "train": {
            "count": len(train_ids),
            "sha256": hash_ids(train_ids),
            "singletons": sum(1 for sid in train_ids if len(gt_map.get(sid, [])) == 0),
            "multi_match": sum(1 for sid in train_ids if len(gt_map.get(sid, [])) > 1),
            "total_gt_pairs": sum(len(gt_map.get(sid, [])) for sid in train_ids),
        },
        "dev": {
            "count": len(dev_ids),
            "sha256": hash_ids(dev_ids),
            "singletons": sum(1 for sid in dev_ids if len(gt_map.get(sid, [])) == 0),
            "multi_match": sum(1 for sid in dev_ids if len(gt_map.get(sid, [])) > 1),
            "total_gt_pairs": sum(len(gt_map.get(sid, [])) for sid in dev_ids),
        },
        "holdout": {
            "count": len(holdout_ids),
            "sha256": hash_ids(holdout_ids),
            "singletons": sum(1 for sid in holdout_ids if len(gt_map.get(sid, [])) == 0),
            "multi_match": sum(1 for sid in holdout_ids if len(gt_map.get(sid, [])) > 1),
            "total_gt_pairs": sum(len(gt_map.get(sid, [])) for sid in holdout_ids),
        },
    }

    logger.info(f"TRAIN: {manifest['train']}")
    logger.info(f"DEV: {manifest['dev']}")
    logger.info(f"HOLDOUT: {manifest['holdout']}")

    with open(OUT_DIR / "train_ids.json", "w", encoding="utf-8") as f:
        json.dump(train_ids, f, indent=2)
    with open(OUT_DIR / "dev_ids.json", "w", encoding="utf-8") as f:
        json.dump(dev_ids, f, indent=2)
    with open(OUT_DIR / "holdout_ids.json", "w", encoding="utf-8") as f:
        json.dump(holdout_ids, f, indent=2)
    with open(OUT_DIR / "splits_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Splits and manifest written to {OUT_DIR}")


if __name__ == "__main__":
    generate_and_save_splits()

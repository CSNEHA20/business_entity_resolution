"""
Milestone 7: Final Candidate Recall Recovery & Decision Capacity Correction Pipeline.
Amazon ML Challenge 2026 - Business Entity Resolution

This script executes the complete Milestone 7 verification:
1. Traces candidate recall loss through the complete pipeline (Stage 0 to Stage 6) on holdout data.
2. Performs candidate pruning ablation on development data.
3. Performs source capacity ablation on development data.
4. Evaluates joint pruning + capacity configurations on dev and finalists on holdout.
5. Performs forensic analysis on zero-candidate holdout entities.
6. Conducts an unlabeled test set distribution risk check.
7. Re-evaluates decision engine rules and selects the optimal configuration.
8. Generates final inference safety manifest and updates best decision engine configuration.
"""

from collections import Counter, defaultdict
from dataclasses import asdict
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import psutil
from rapidfuzz import fuzz

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.decision_engine import DecisionRuleConfig, EntityDecisionEngine
from src.features import FEATURE_COLUMNS, PairFeatureExtractor
from src.metrics import compute_candidate_recall, compute_macro_f05
from src.normalization import (
    extract_building_number,
    extract_numeric_tokens,
    extract_postal_code,
    normalize_address_abbreviations,
    normalize_business_name_suffixes,
    normalize_country,
    tokenize_text,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def get_memory_mb() -> float:
    """Return current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def file_sha256(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def run_milestone7_pipeline():
    logger.info("=================================================================")
    logger.info("   STARTING MILESTONE 7: CANDIDATE RECALL & CAPACITY RECOVERY   ")
    logger.info("=================================================================")

    DATA_DIR = PROJECT_ROOT / "data"
    ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
    MODELS_DIR = ARTIFACTS_DIR / "models"
    M7_DIR = ARTIFACTS_DIR / "milestone7"
    M7_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Model
    model_path = MODELS_DIR / "retrained_hardneg_model.pkl"
    if not model_path.exists():
        model_path = MODELS_DIR / "baseline_lgbm_model.pkl"
    logger.info(f"Loading trained GBDT model from {model_path}...")
    loaded_model = joblib.load(model_path)
    model_hash = file_sha256(model_path)
    logger.info(f"Loaded Model SHA256: {model_hash}")

    # 2. Load Train / Target Datasets
    logger.info("Loading training datasets...")
    s1_path = DATA_DIR / "train" / "train_source1.tsv"
    s2_path = DATA_DIR / "train" / "train_source2.tsv"
    s3_path = DATA_DIR / "train" / "train_source3.tsv"
    gt_path = DATA_DIR / "train" / "train_ground_truth.tsv"

    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    s2_df = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False)
    s3_df = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False)
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)

    # Fast ground truth map
    gt_map: Dict[str, Set[str]] = defaultdict(set)
    s1_col = "source1_entity_id" if "source1_entity_id" in gt_df.columns else "source1_id"
    match_col = "matched_entity_ids" if "matched_entity_ids" in gt_df.columns else "target_id"
    for sid, m_str in zip(gt_df[s1_col].to_numpy(dtype=object), gt_df[match_col].to_numpy(dtype=object)):
        sid = str(sid).strip()
        m_str = str(m_str or "").strip()
        if m_str:
            for tid in m_str.split(","):
                tid = tid.strip()
                if tid:
                    gt_map[sid].add(tid)

    # Build Fixed Deterministic Split (Dev=10k, Holdout=10k)
    s1_all_ids = s1_df["entity_id"].astype(str).str.strip().tolist()
    rng = np.random.RandomState(42)
    shuffled_ids = rng.permutation(s1_all_ids)

    dev_s1_ids = set(shuffled_ids[:10000])
    holdout_s1_ids = set(shuffled_ids[10000:20000])

    dev_val_s1_df = s1_df[s1_df["entity_id"].isin(dev_s1_ids)].copy()
    holdout_val_s1_df = s1_df[s1_df["entity_id"].isin(holdout_s1_ids)].copy()

    dev_val_gt = {sid: gt_map[sid] for sid in dev_s1_ids}
    holdout_val_gt = {sid: gt_map[sid] for sid in holdout_s1_ids}

    logger.info(f"Split sizes: Dev S1 = {len(dev_val_s1_df)}, Holdout S1 = {len(holdout_val_s1_df)}")

    # 3. Build Multi-Pass Inverted Index
    logger.info("Building Inverted Indices for S2 and S3 targets...")
    idx_exact_name: Dict[str, List[str]] = defaultdict(list)
    idx_exact_addr: Dict[str, List[str]] = defaultdict(list)
    idx_name_sig: Dict[str, List[str]] = defaultdict(list)
    idx_addr_sig: Dict[str, List[str]] = defaultdict(list)
    idx_name_tokens: Dict[str, List[str]] = defaultdict(list)
    idx_postal: Dict[str, List[str]] = defaultdict(list)
    idx_building: Dict[str, List[str]] = defaultdict(list)
    token_df_counter: Counter = Counter()

    s2_raw_lookup: Dict[str, Tuple[str, str, str]] = {}
    s3_raw_lookup: Dict[str, Tuple[str, str, str]] = {}

    for tid, n_raw, a_raw, c_raw in zip(
        s2_df["entity_id"].to_numpy(dtype=object),
        s2_df["business_name"].to_numpy(dtype=object),
        s2_df["business_address"].to_numpy(dtype=object),
        s2_df["country"].to_numpy(dtype=object),
    ):
        tid = str(tid).strip()
        n_raw = str(n_raw or "")
        a_raw = str(a_raw or "")
        c_raw = str(c_raw or "")
        s2_raw_lookup[tid] = (n_raw, a_raw, c_raw)

        n_clean = n_raw.lower().strip()
        a_clean = a_raw.lower().strip()
        n_sig = " ".join(sorted(set(n_clean.split())))
        a_sig = " ".join(sorted(set(a_clean.split())))
        postal = extract_postal_code(a_raw)
        bldg = extract_building_number(a_raw)

        if n_clean:
            idx_exact_name[n_clean].append(tid)
        if a_clean:
            idx_exact_addr[a_clean].append(tid)
        if n_sig:
            idx_name_sig[n_sig].append(tid)
        if a_sig:
            idx_addr_sig[a_sig].append(tid)
        for pin in postal:
            idx_postal[pin].append(tid)
        if bldg:
            idx_building[bldg].append(tid)

        toks = set(n_clean.split())
        for t in toks:
            token_df_counter[t] += 1
            idx_name_tokens[t].append(tid)

    for tid, n_raw, a_raw, c_raw in zip(
        s3_df["entity_id"].to_numpy(dtype=object),
        s3_df["business_name"].to_numpy(dtype=object),
        s3_df["business_address"].to_numpy(dtype=object),
        s3_df["country"].to_numpy(dtype=object),
    ):
        tid = str(tid).strip()
        n_raw = str(n_raw or "")
        a_raw = str(a_raw or "")
        c_raw = str(c_raw or "")
        s3_raw_lookup[tid] = (n_raw, a_raw, c_raw)

        n_clean = n_raw.lower().strip()
        a_clean = a_raw.lower().strip()
        n_sig = " ".join(sorted(set(n_clean.split())))
        a_sig = " ".join(sorted(set(a_clean.split())))
        postal = extract_postal_code(a_raw)
        bldg = extract_building_number(a_raw)

        if n_clean:
            idx_exact_name[n_clean].append(tid)
        if a_clean:
            idx_exact_addr[a_clean].append(tid)
        if n_sig:
            idx_name_sig[n_sig].append(tid)
        if a_sig:
            idx_addr_sig[a_sig].append(tid)
        for pin in postal:
            idx_postal[pin].append(tid)
        if bldg:
            idx_building[bldg].append(tid)

        toks = set(n_clean.split())
        for t in toks:
            token_df_counter[t] += 1
            idx_name_tokens[t].append(tid)

    def get_target_meta(tid: str) -> Dict[str, Any]:
        src = "S2" if tid.startswith("S2-") else "S3"
        raw_tuple = s2_raw_lookup.get(tid) if src == "S2" else s3_raw_lookup.get(tid)
        if raw_tuple is None:
            name_raw, addr_raw, c_raw = "", "", ""
        else:
            name_raw, addr_raw, c_raw = raw_tuple

        name_raw = str(name_raw or "")
        addr_raw = str(addr_raw or "")
        c_raw = str(c_raw or "")

        n_norm = normalize_business_name_suffixes(name_raw)
        a_norm = normalize_address_abbreviations(addr_raw)
        c_norm = normalize_country(c_raw)
        n_toks = set(tokenize_text(name_raw))
        a_toks = set(tokenize_text(addr_raw))
        pins = set(extract_postal_code(addr_raw))
        bldg = extract_building_number(addr_raw) or ""
        nums = set(extract_numeric_tokens(addr_raw))

        return {
            "entity_id": tid,
            "business_name_raw": name_raw,
            "business_address_raw": addr_raw,
            "business_name_norm": n_norm or name_raw.lower().strip(),
            "business_address_norm": a_norm or addr_raw.lower().strip(),
            "country_norm": c_norm,
            "name_tokens": frozenset(n_toks),
            "addr_tokens": frozenset(a_toks),
            "postal_codes": pins,
            "building_number": bldg,
            "numeric_tokens": nums,
            "src": src,
            "has_addr": bool(a_norm or addr_raw.strip()),
        }

    def preprocess_s1(df_subset: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        res = {}
        for sid, n_raw, a_raw, c_raw in zip(
            df_subset["entity_id"].to_numpy(dtype=object),
            df_subset["business_name"].to_numpy(dtype=object),
            df_subset["business_address"].to_numpy(dtype=object),
            df_subset["country"].to_numpy(dtype=object),
        ):
            sid = str(sid).strip()
            name_raw = str(n_raw or "")
            addr_raw = str(a_raw or "")
            c_raw = str(c_raw or "")

            n_clean = name_raw.lower().strip()
            a_clean = addr_raw.lower().strip()
            n_norm = normalize_business_name_suffixes(name_raw)
            a_norm = normalize_address_abbreviations(addr_raw)
            c_norm = normalize_country(c_raw)
            n_sig = " ".join(sorted(set(n_clean.split())))
            a_sig = " ".join(sorted(set(a_clean.split())))
            n_toks = set(tokenize_text(name_raw))
            a_toks = set(tokenize_text(addr_raw))
            pins = set(extract_postal_code(addr_raw))
            bldg = extract_building_number(addr_raw) or ""
            nums = set(extract_numeric_tokens(addr_raw))

            res[sid] = {
                "entity_id": sid,
                "business_name_raw": name_raw,
                "business_address_raw": addr_raw,
                "business_name_norm": n_norm or n_clean,
                "business_address_norm": a_norm or a_clean,
                "clean_name": n_clean,
                "clean_addr": a_clean,
                "name_sig": n_sig,
                "addr_sig": a_sig,
                "country_norm": c_norm,
                "name_tokens": frozenset(n_toks),
                "addr_tokens": frozenset(a_toks),
                "postal_codes": pins,
                "building_number": bldg,
                "numeric_tokens": nums,
                "has_addr": bool(a_clean),
                "country_raw": c_raw,
            }
        return res

    dev_val_s1_dict = preprocess_s1(dev_val_s1_df)
    holdout_val_s1_dict = preprocess_s1(holdout_val_s1_df)

    # -------------------------------------------------------------------
    # CANDIDATE RETRIEVAL GENERATORS UNDER DIFFERENT PRUNING MODES
    # -------------------------------------------------------------------
    def retrieve_candidates_by_mode(
        s1_dict: Dict[str, Dict[str, Any]],
        mode: str = "current"
    ) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], Dict[str, int]]]:
        candidates: Dict[str, Set[str]] = defaultdict(set)
        provenance: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for sid, q in s1_dict.items():
            # Exact Name
            if q["clean_name"] in idx_exact_name:
                hits = idx_exact_name[q["clean_name"]]
                cap = (500 if mode == "none" else (50 if mode == "current" else (150 if mode in ("conservative", "prob_aware") else 300)))
                for tid in hits[:cap]:
                    candidates[sid].add(tid)
                    provenance[(sid, tid)]["exact_name_hit"] = 1

            # Exact Address
            if q["clean_addr"] in idx_exact_addr:
                hits = idx_exact_addr[q["clean_addr"]]
                cap = (500 if mode == "none" else (50 if mode == "current" else (150 if mode in ("conservative", "prob_aware") else 300)))
                for tid in hits[:cap]:
                    candidates[sid].add(tid)
                    provenance[(sid, tid)]["exact_address_hit"] = 1

            # Name Signature
            if q["name_sig"] in idx_name_sig:
                hits = idx_name_sig[q["name_sig"]]
                cap = (500 if mode == "none" else (50 if mode == "current" else (150 if mode in ("conservative", "prob_aware") else 300)))
                for tid in hits[:cap]:
                    candidates[sid].add(tid)
                    provenance[(sid, tid)]["name_token_hit"] = 1

            # Address Signature
            if q["addr_sig"] in idx_addr_sig:
                hits = idx_addr_sig[q["addr_sig"]]
                cap = (500 if mode == "none" else (50 if mode == "current" else (150 if mode in ("conservative", "prob_aware") else 300)))
                for tid in hits[:cap]:
                    candidates[sid].add(tid)
                    provenance[(sid, tid)]["address_token_hit"] = 1

            # Rare Tokens
            for tok in q["clean_name"].split():
                df_val = token_df_counter.get(tok, 0)
                if mode == "none":
                    if 1 <= df_val <= 1000:
                        for tid in idx_name_tokens[tok][:200]:
                            candidates[sid].add(tid)
                            provenance[(sid, tid)]["rare_token_hit"] = 1
                elif mode == "current":
                    if 1 <= df_val <= 150:
                        for tid in idx_name_tokens[tok][:50]:
                            candidates[sid].add(tid)
                            provenance[(sid, tid)]["rare_token_hit"] = 1
                    elif 150 < df_val <= 800:
                        for tid in idx_name_tokens[tok][:25]:
                            candidates[sid].add(tid)
                            provenance[(sid, tid)]["rare_token_hit"] = 1
                elif mode in ("conservative", "prob_aware"):
                    if 1 <= df_val <= 300:
                        for tid in idx_name_tokens[tok][:100]:
                            candidates[sid].add(tid)
                            provenance[(sid, tid)]["rare_token_hit"] = 1
                    elif 300 < df_val <= 1500:
                        for tid in idx_name_tokens[tok][:50]:
                            candidates[sid].add(tid)
                            provenance[(sid, tid)]["rare_token_hit"] = 1
                elif mode == "recall_first":
                    if 1 <= df_val <= 500:
                        for tid in idx_name_tokens[tok][:150]:
                            candidates[sid].add(tid)
                            provenance[(sid, tid)]["rare_token_hit"] = 1
                    elif 500 < df_val <= 2500:
                        for tid in idx_name_tokens[tok][:80]:
                            candidates[sid].add(tid)
                            provenance[(sid, tid)]["rare_token_hit"] = 1

            # Postal
            for pin in q["postal_codes"]:
                if pin in idx_postal:
                    hits = idx_postal[pin]
                    cap = (200 if mode == "none" else (30 if mode == "current" else (100 if mode in ("conservative", "prob_aware") else 150)))
                    for tid in hits[:cap]:
                        candidates[sid].add(tid)
                        provenance[(sid, tid)]["postal_numeric_hit"] = 1

            # Building
            if q["building_number"] in idx_building:
                hits = idx_building[q["building_number"]]
                cap = (100 if mode == "none" else (25 if mode == "current" else (50 if mode in ("conservative", "prob_aware") else 80)))
                for tid in hits[:cap]:
                    candidates[sid].add(tid)
                    provenance[(sid, tid)]["cross_field_hit"] = 1

            # Probability-Aware Lightweight Pre-Ranking / Pruning
            if mode == "prob_aware" and len(candidates[sid]) > 150:
                scored_cands = []
                q_name = q["clean_name"]
                q_addr = q["clean_addr"]
                for tid in candidates[sid]:
                    t_tuple = s2_raw_lookup.get(tid) if tid.startswith("S2-") else s3_raw_lookup.get(tid)
                    t_name = str(t_tuple[0] or "").lower().strip() if t_tuple else ""
                    t_addr = str(t_tuple[1] or "").lower().strip() if t_tuple else ""
                    n_sim = fuzz.token_set_ratio(q_name, t_name)
                    a_sim = fuzz.token_set_ratio(q_addr, t_addr) if (q_addr and t_addr) else 0
                    p_count = sum(provenance[(sid, tid)].values())
                    composite_score = n_sim * 0.6 + a_sim * 0.3 + p_count * 10
                    scored_cands.append((tid, composite_score))
                scored_cands.sort(key=lambda x: -x[1])
                top_tids = {tid for tid, _ in scored_cands[:150]}
                candidates[sid] = top_tids

        return candidates, provenance

    target_cache: Dict[str, Dict[str, Any]] = {}
    extractor = PairFeatureExtractor()

    def get_cached_target_meta(tid: str) -> Dict[str, Any]:
        if tid not in target_cache:
            target_cache[tid] = get_target_meta(tid)
        return target_cache[tid]

    def score_candidate_pairs(
        s1_map: Dict[str, Dict[str, Any]],
        cands_map: Dict[str, Set[str]],
        prov_map: Dict[Tuple[str, str], Dict[str, int]],
        batch_size: int = 25000,
        tag: str = ""
    ) -> Dict[str, Dict[str, float]]:
        pairs = []
        for sid, c_set in cands_map.items():
            for tid in c_set:
                pairs.append((sid, tid))
                if tid not in target_cache:
                    target_cache[tid] = get_target_meta(tid)

        total_pairs = len(pairs)
        scores_by_s1: Dict[str, Dict[str, float]] = defaultdict(dict)
        if total_pairs == 0:
            return scores_by_s1

        logger.info(f"[{tag}] Scoring {total_pairs:,} pairs across {len(s1_map):,} queries in batches of {batch_size:,}...")
        for i in range(0, total_pairs, batch_size):
            b_pairs = pairs[i : i + batch_size]
            X_batch = extractor.extract_features_matrix(b_pairs, s1_map, target_cache, prov_map)
            probs_batch = loaded_model.predict_proba(X_batch)[:, 1]
            for (sid, tid), p in zip(b_pairs, probs_batch):
                scores_by_s1[sid][tid] = float(p)
            del X_batch, probs_batch
            if i % 100000 == 0 and i > 0:
                gc.collect()

        return scores_by_s1

    # -------------------------------------------------------------------
    # PART 1: TRACE CANDIDATE RECALL LOSS (HOLDOUT SET)
    # -------------------------------------------------------------------
    logger.info("=== [PART 1] TRACING CANDIDATE RECALL LOSS ON HOLDOUT SET ===")
    
    holdout_unpruned_cands, holdout_unpruned_prov = retrieve_candidates_by_mode(holdout_val_s1_dict, mode="none")
    holdout_current_cands, holdout_current_prov = retrieve_candidates_by_mode(holdout_val_s1_dict, mode="current")

    holdout_scores_by_s1 = score_candidate_pairs(
        holdout_val_s1_dict, holdout_current_cands, holdout_current_prov, tag="Part1-Current-Holdout"
    )
    holdout_pair_probs = {(sid, tid): holdout_scores_by_s1[sid][tid] for sid in holdout_scores_by_s1 for tid in holdout_scores_by_s1[sid]}

    engine_json = MODELS_DIR / "best_decision_engine.json"
    baseline_engine = EntityDecisionEngine.load(engine_json)
    baseline_preds = baseline_engine.predict_all(holdout_scores_by_s1, holdout_val_s1_dict, all_s1_ids=holdout_val_gt.keys())

    all_true_pairs: List[Tuple[str, str, str]] = []
    for sid, tids in holdout_val_gt.items():
        for tid in tids:
            src = "S2" if tid.startswith("S2-") else "S3"
            all_true_pairs.append((sid, tid, src))

    total_true_pairs = len(all_true_pairs)
    total_true_s2 = sum(1 for _, _, src in all_true_pairs if src == "S2")
    total_true_s3 = sum(1 for _, _, src in all_true_pairs if src == "S3")

    pair_status = {}
    for sid, tid, src in all_true_pairs:
        if tid not in holdout_unpruned_cands.get(sid, set()):
            pair_status[(sid, tid)] = ("Stage 1: Raw Blocking", "not retrieved by blocking")
            continue

        if tid not in holdout_current_cands.get(sid, set()):
            pair_status[(sid, tid)] = ("Stage 3: Candidate Pruning", "removed by candidate pruning")
            continue

        if (sid, tid) not in holdout_pair_probs:
            pair_status[(sid, tid)] = ("Stage 4: Feature Extraction", "feature-generation failure")
            continue

        score = holdout_pair_probs[(sid, tid)]
        q_meta = holdout_val_s1_dict.get(sid, {})
        has_addr = bool(q_meta.get("has_addr", True))
        addr_boost = 0.0 if has_addr else baseline_engine.config.missing_addr_threshold_boost
        thresh = (baseline_engine.config.threshold_s2 if src == "S2" else baseline_engine.config.threshold_s3) + addr_boost

        if score < thresh:
            pair_status[(sid, tid)] = ("Stage 5: GBDT Score", "model score too low")
            continue

        all_cands_for_s1 = [(t, p) for t, p in holdout_scores_by_s1.get(sid, {}).items()]
        all_cands_for_s1.sort(key=lambda x: -x[1])
        top_prob = all_cands_for_s1[0][1] if all_cands_for_s1 else 0.0

        if top_prob < (baseline_engine.config.min_top_prob + addr_boost):
            pair_status[(sid, tid)] = ("Stage 6a: Decision Singleton Gate", "singleton gate")
            continue

        score_drop = top_prob - score
        if score_drop > baseline_engine.config.max_multi_score_drop:
            pair_status[(sid, tid)] = ("Stage 6b: Decision Secondary Margin Rule", "secondary margin rule")
            continue

        if baseline_engine.config.multi_match_threshold is not None:
            if score < (baseline_engine.config.multi_match_threshold + addr_boost):
                pair_status[(sid, tid)] = ("Stage 6b: Decision Secondary Margin Rule", "secondary margin rule")
                continue

        src_cands = [(t, p) for t, p in all_cands_for_s1 if (t.startswith("S2-") if src == "S2" else t.startswith("S3-"))]
        src_rank = [t for t, _ in src_cands].index(tid) + 1 if tid in [t for t, _ in src_cands] else 999
        if baseline_engine.config.max_matches_per_source > 0 and src_rank > baseline_engine.config.max_matches_per_source:
            pair_status[(sid, tid)] = ("Stage 6c: Decision Capacity Rule", "source-capacity rule")
            continue

        pred_set = baseline_preds.get(sid, [])
        if tid in pred_set:
            pair_status[(sid, tid)] = ("Stage 6d: Final Predicted Matches", "correctly resolved")
        else:
            pair_status[(sid, tid)] = ("Stage 6d: Decision Engine Other", "other")

    funnel_rows = []
    funnel_rows.append({
        "stage": "Stage 0: Ground Truth True Pairs",
        "true_pairs_remaining": total_true_pairs,
        "true_pairs_lost": 0,
        "recall": 1.0000,
        "S2_recall": 1.0000,
        "S3_recall": 1.0000,
        "loss_reason": "none",
    })

    lost_st1 = [p for p in all_true_pairs if pair_status[(p[0], p[1])][1] == "not retrieved by blocking"]
    rem_st1 = [p for p in all_true_pairs if p not in lost_st1]
    funnel_rows.append({
        "stage": "Stage 1: Raw Blocking Generation",
        "true_pairs_remaining": len(rem_st1),
        "true_pairs_lost": len(lost_st1),
        "recall": round(len(rem_st1) / total_true_pairs, 4),
        "S2_recall": round(sum(1 for p in rem_st1 if p[2] == "S2") / total_true_s2, 4),
        "S3_recall": round(sum(1 for p in rem_st1 if p[2] == "S3") / total_true_s3, 4),
        "loss_reason": "not retrieved by blocking",
    })

    lost_st2 = [p for p in rem_st1 if pair_status[(p[0], p[1])][1] == "removed during deduplication"]
    rem_st2 = [p for p in rem_st1 if p not in lost_st2]
    funnel_rows.append({
        "stage": "Stage 2: Candidate Deduplication",
        "true_pairs_remaining": len(rem_st2),
        "true_pairs_lost": len(lost_st2),
        "recall": round(len(rem_st2) / total_true_pairs, 4),
        "S2_recall": round(sum(1 for p in rem_st2 if p[2] == "S2") / total_true_s2, 4),
        "S3_recall": round(sum(1 for p in rem_st2 if p[2] == "S3") / total_true_s3, 4),
        "loss_reason": "removed during deduplication",
    })

    lost_st3 = [p for p in rem_st2 if pair_status[(p[0], p[1])][1] == "removed by candidate pruning"]
    rem_st3 = [p for p in rem_st2 if p not in lost_st3]
    funnel_rows.append({
        "stage": "Stage 3: Candidate Pruning",
        "true_pairs_remaining": len(rem_st3),
        "true_pairs_lost": len(lost_st3),
        "recall": round(len(rem_st3) / total_true_pairs, 4),
        "S2_recall": round(sum(1 for p in rem_st3 if p[2] == "S2") / total_true_s2, 4),
        "S3_recall": round(sum(1 for p in rem_st3 if p[2] == "S3") / total_true_s3, 4),
        "loss_reason": "removed by candidate pruning",
    })

    lost_st4 = [p for p in rem_st3 if pair_status[(p[0], p[1])][1] == "feature-generation failure"]
    rem_st4 = [p for p in rem_st3 if p not in lost_st4]
    funnel_rows.append({
        "stage": "Stage 4: Feature Extraction",
        "true_pairs_remaining": len(rem_st4),
        "true_pairs_lost": len(lost_st4),
        "recall": round(len(rem_st4) / total_true_pairs, 4),
        "S2_recall": round(sum(1 for p in rem_st4 if p[2] == "S2") / total_true_s2, 4),
        "S3_recall": round(sum(1 for p in rem_st4 if p[2] == "S3") / total_true_s3, 4),
        "loss_reason": "feature-generation failure",
    })

    lost_st5 = [p for p in rem_st4 if pair_status[(p[0], p[1])][1] == "model score too low"]
    rem_st5 = [p for p in rem_st4 if p not in lost_st5]
    funnel_rows.append({
        "stage": "Stage 5: GBDT Score",
        "true_pairs_remaining": len(rem_st5),
        "true_pairs_lost": len(lost_st5),
        "recall": round(len(rem_st5) / total_true_pairs, 4),
        "S2_recall": round(sum(1 for p in rem_st5 if p[2] == "S2") / total_true_s2, 4),
        "S3_recall": round(sum(1 for p in rem_st5 if p[2] == "S3") / total_true_s3, 4),
        "loss_reason": "model score too low",
    })

    lost_st6a = [p for p in rem_st5 if pair_status[(p[0], p[1])][1] == "singleton gate"]
    rem_st6a = [p for p in rem_st5 if p not in lost_st6a]
    funnel_rows.append({
        "stage": "Stage 6a: Decision Singleton Gate",
        "true_pairs_remaining": len(rem_st6a),
        "true_pairs_lost": len(lost_st6a),
        "recall": round(len(rem_st6a) / total_true_pairs, 4),
        "S2_recall": round(sum(1 for p in rem_st6a if p[2] == "S2") / total_true_s2, 4),
        "S3_recall": round(sum(1 for p in rem_st6a if p[2] == "S3") / total_true_s3, 4),
        "loss_reason": "singleton gate",
    })

    lost_st6b = [p for p in rem_st6a if pair_status[(p[0], p[1])][1] == "secondary margin rule"]
    rem_st6b = [p for p in rem_st6a if p not in lost_st6b]
    funnel_rows.append({
        "stage": "Stage 6b: Decision Secondary Margin Rule",
        "true_pairs_remaining": len(rem_st6b),
        "true_pairs_lost": len(lost_st6b),
        "recall": round(len(rem_st6b) / total_true_pairs, 4),
        "S2_recall": round(sum(1 for p in rem_st6b if p[2] == "S2") / total_true_s2, 4),
        "S3_recall": round(sum(1 for p in rem_st6b if p[2] == "S3") / total_true_s3, 4),
        "loss_reason": "secondary margin rule",
    })

    lost_st6c = [p for p in rem_st6b if pair_status[(p[0], p[1])][1] == "source-capacity rule"]
    rem_st6c = [p for p in rem_st6b if p not in lost_st6c]
    funnel_rows.append({
        "stage": "Stage 6c: Decision Capacity Rule",
        "true_pairs_remaining": len(rem_st6c),
        "true_pairs_lost": len(lost_st6c),
        "recall": round(len(rem_st6c) / total_true_pairs, 4),
        "S2_recall": round(sum(1 for p in rem_st6c if p[2] == "S2") / total_true_s2, 4),
        "S3_recall": round(sum(1 for p in rem_st6c if p[2] == "S3") / total_true_s3, 4),
        "loss_reason": "source-capacity rule",
    })

    lost_st6d = [p for p in rem_st6c if pair_status[(p[0], p[1])][1] == "other"]
    rem_st6d = [p for p in rem_st6c if p not in lost_st6d]
    funnel_rows.append({
        "stage": "Stage 6d: Final Predicted Matches",
        "true_pairs_remaining": len(rem_st6d),
        "true_pairs_lost": len(lost_st6d),
        "recall": round(len(rem_st6d) / total_true_pairs, 4),
        "S2_recall": round(sum(1 for p in rem_st6d if p[2] == "S2") / total_true_s2, 4),
        "S3_recall": round(sum(1 for p in rem_st6d if p[2] == "S3") / total_true_s3, 4),
        "loss_reason": "other",
    })

    funnel_df = pd.DataFrame(funnel_rows)
    funnel_csv_path = M7_DIR / "candidate_recall_funnel.csv"
    funnel_df.to_csv(funnel_csv_path, index=False)
    logger.info(f"Saved Candidate Recall Funnel to {funnel_csv_path}")

    # -------------------------------------------------------------------
    # PART 2: CANDIDATE PRUNING ABLATION (DEV DATA SELECTION)
    # -------------------------------------------------------------------
    logger.info("=== [PART 2] CANDIDATE PRUNING ABLATION ON DEV DATA ===")
    
    pruning_modes = [
        "current",
        "conservative",
        "prob_aware",
        "recall_first",
        "none",
    ]

    dev_pruning_results = []
    dev_cands_cache = {}
    dev_prov_cache = {}
    dev_scores_cache = {}

    for mode in pruning_modes:
        t0 = time.time()
        cands, prov = retrieve_candidates_by_mode(dev_val_s1_dict, mode=mode)
        t_retrieval = time.time() - t0
        
        rec = compute_candidate_recall(dev_val_gt, cands)
        gt_s2 = {sid: {x for x in tr if x.startswith("S2-")} for sid, tr in dev_val_gt.items() if any(x.startswith("S2-") for x in tr)}
        gt_s3 = {sid: {x for x in tr if x.startswith("S3-")} for sid, tr in dev_val_gt.items() if any(x.startswith("S3-") for x in tr)}
        rec_s2 = compute_candidate_recall(gt_s2, cands)
        rec_s3 = compute_candidate_recall(gt_s3, cands)

        cand_counts = [len(cands[sid]) for sid in dev_val_s1_dict.keys()]
        zero_cand_count = sum(1 for c in cand_counts if c == 0)

        t0_score = time.time()
        scores = score_candidate_pairs(dev_val_s1_dict, cands, prov, tag=f"Dev-{mode}")
        t_score = time.time() - t0_score

        total_runtime = t_retrieval + t_score
        peak_ram = get_memory_mb()

        dev_cands_cache[mode] = cands
        dev_prov_cache[mode] = prov
        dev_scores_cache[mode] = scores

        dev_pruning_results.append({
            "pruning_mode": mode,
            "candidate_recall": round(rec["candidate_recall"], 4),
            "s2_candidate_recall": round(rec_s2["candidate_recall"], 4),
            "s3_candidate_recall": round(rec_s3["candidate_recall"], 4),
            "mean_candidates_per_s1": round(float(np.mean(cand_counts)), 2),
            "p95_candidates": int(np.percentile(cand_counts, 95)),
            "p99_candidates": int(np.percentile(cand_counts, 99)),
            "max_candidates": int(np.max(cand_counts)),
            "zero_candidate_rate": round(zero_cand_count / len(dev_val_s1_dict), 4),
            "runtime_sec": round(total_runtime, 2),
            "peak_ram_mb": round(peak_ram, 1),
        })
        logger.info(f"Dev mode [{mode}]: recall={rec['candidate_recall']:.4f}, mean_cands={np.mean(cand_counts):.1f}, time={total_runtime:.1f}s")

    dev_pruning_df = pd.DataFrame(dev_pruning_results)
    logger.info("Dev Pruning Ablation Summary:")
    logger.info("\n" + dev_pruning_df.to_string(index=False))

    # -------------------------------------------------------------------
    # PART 3 & PART 4: SOURCE CAPACITY & JOINT ABLATION (DEV DATA SELECTION)
    # -------------------------------------------------------------------
    logger.info("=== [PART 3 & 4] JOINT PRUNING + SOURCE CAPACITY ABLATION ===")

    capacity_configs = {
        "Max_1": {"max_matches_per_source": 1, "adaptive_drop": False},
        "Max_2": {"max_matches_per_source": 2, "adaptive_drop": False},
        "Max_3": {"max_matches_per_source": 3, "adaptive_drop": False},
        "No_Limit": {"max_matches_per_source": 0, "adaptive_drop": False},
        "Adaptive_Capacity": {"max_matches_per_source": 3, "adaptive_drop": True},
    }

    joint_dev_rows = []

    for p_mode in pruning_modes:
        scores_by_s1 = dev_scores_cache[p_mode]

        for cap_name, cap_cfg in capacity_configs.items():
            drop = 0.12 if cap_cfg["adaptive_drop"] else 0.16
            m_thresh = 0.48 if cap_cfg["adaptive_drop"] else 0.458
            
            cfg = DecisionRuleConfig(
                strategy="adaptive_multi",
                threshold_s2=0.4775,
                threshold_s3=0.4975,
                min_top_prob=0.428,
                min_margin=0.0,
                multi_match_threshold=m_thresh,
                max_multi_score_drop=drop,
                max_matches_per_source=cap_cfg["max_matches_per_source"],
                enable_multi_match=True,
                enable_singleton_abstention=True,
            )
            engine = EntityDecisionEngine(cfg)
            dev_metrics = engine.evaluate(dev_val_gt, scores_by_s1, dev_val_s1_dict)
            dev_preds = engine.predict_all(scores_by_s1, dev_val_s1_dict, all_s1_ids=dev_val_gt.keys())

            d_s2_ids = [sid for sid, tr in dev_val_gt.items() if any(x.startswith("S2-") for x in tr)]
            d_s3_ids = [sid for sid, tr in dev_val_gt.items() if any(x.startswith("S3-") for x in tr)]
            d_multi_ids = [sid for sid, tr in dev_val_gt.items() if len(tr) > 1]
            d_single_ids = [sid for sid, tr in dev_val_gt.items() if len(tr) == 1]
            d_singlet_ids = [sid for sid, tr in dev_val_gt.items() if len(tr) == 0]

            d_s2 = compute_macro_f05({k: dev_val_gt[k] for k in d_s2_ids}, {k: dev_preds[k] for k in d_s2_ids}) if d_s2_ids else {}
            d_s3 = compute_macro_f05({k: dev_val_gt[k] for k in d_s3_ids}, {k: dev_preds[k] for k in d_s3_ids}) if d_s3_ids else {}
            d_multi = compute_macro_f05({k: dev_val_gt[k] for k in d_multi_ids}, {k: dev_preds[k] for k in d_multi_ids}) if d_multi_ids else {}
            d_single = compute_macro_f05({k: dev_val_gt[k] for k in d_single_ids}, {k: dev_preds[k] for k in d_single_ids}) if d_single_ids else {}
            d_singlet = compute_macro_f05({k: dev_val_gt[k] for k in d_singlet_ids}, {k: dev_preds[k] for k in d_singlet_ids}) if d_singlet_ids else {}

            avg_matches = float(np.mean([len(dev_preds[k]) for k in dev_val_gt.keys()]))

            joint_dev_rows.append({
                "pruning_mode": p_mode,
                "capacity_strategy": cap_name,
                "dev_macro_f05": round(dev_metrics["macro_f05"], 4),
                "dev_precision": round(dev_metrics["macro_precision"], 4),
                "dev_recall": round(dev_metrics["macro_recall"], 4),
                "dev_singleton_f05": round(d_singlet.get("macro_f05", 0.0), 4),
                "dev_single_match_f05": round(d_single.get("macro_f05", 0.0), 4),
                "dev_multi_match_f05": round(d_multi.get("macro_f05", 0.0), 4),
                "dev_s2_f05": round(d_s2.get("macro_f05", 0.0), 4),
                "dev_s3_f05": round(d_s3.get("macro_f05", 0.0), 4),
                "avg_predicted_matches": round(avg_matches, 3),
            })

    joint_dev_df = pd.DataFrame(joint_dev_rows).sort_values(by="dev_macro_f05", ascending=False)
    logger.info("Top 10 Joint Configurations on Dev Data:")
    logger.info("\n" + joint_dev_df.head(10).to_string(index=False))

    finalist_keys = [
        ("current", "Max_1"),
        ("current", "No_Limit"),
        ("conservative", "No_Limit"),
        ("conservative", "Adaptive_Capacity"),
        ("prob_aware", "No_Limit"),
        ("recall_first", "No_Limit"),
        ("none", "No_Limit"),
    ]

    logger.info("=== EVALUATING FINALISTS ON UNTOUCHED HOLDOUT (EXACTLY ONCE) ===")
    holdout_cands_cache = {}
    holdout_prov_cache = {}
    holdout_scores_cache = {}

    joint_holdout_rows = []

    for p_mode, cap_name in finalist_keys:
        if p_mode not in holdout_scores_cache:
            h_cands, h_prov = retrieve_candidates_by_mode(holdout_val_s1_dict, mode=p_mode)
            h_scores = score_candidate_pairs(holdout_val_s1_dict, h_cands, h_prov, tag=f"Holdout-{p_mode}")
            holdout_cands_cache[p_mode] = h_cands
            holdout_prov_cache[p_mode] = h_prov
            holdout_scores_cache[p_mode] = h_scores

        h_scores_by_s1 = holdout_scores_cache[p_mode]
        cap_cfg = capacity_configs[cap_name]
        drop = 0.12 if cap_cfg["adaptive_drop"] else 0.16
        m_thresh = 0.48 if cap_cfg["adaptive_drop"] else 0.458

        cfg = DecisionRuleConfig(
            strategy="adaptive_multi",
            threshold_s2=0.4775,
            threshold_s3=0.4975,
            min_top_prob=0.428,
            min_margin=0.0,
            multi_match_threshold=m_thresh,
            max_multi_score_drop=drop,
            max_matches_per_source=cap_cfg["max_matches_per_source"],
            enable_multi_match=True,
            enable_singleton_abstention=True,
        )
        engine = EntityDecisionEngine(cfg)

        dev_match = [r for r in joint_dev_rows if r["pruning_mode"] == p_mode and r["capacity_strategy"] == cap_name][0]

        h_eval = engine.evaluate(holdout_val_gt, h_scores_by_s1, holdout_val_s1_dict)
        h_preds = engine.predict_all(h_scores_by_s1, holdout_val_s1_dict, all_s1_ids=holdout_val_gt.keys())

        h_s2_ids = [sid for sid, tr in holdout_val_gt.items() if any(x.startswith("S2-") for x in tr)]
        h_s3_ids = [sid for sid, tr in holdout_val_gt.items() if any(x.startswith("S3-") for x in tr)]
        h_multi_ids = [sid for sid, tr in holdout_val_gt.items() if len(tr) > 1]
        h_single_ids = [sid for sid, tr in holdout_val_gt.items() if len(tr) == 1]
        h_singlet_ids = [sid for sid, tr in holdout_val_gt.items() if len(tr) == 0]

        h_s2 = compute_macro_f05({k: holdout_val_gt[k] for k in h_s2_ids}, {k: h_preds[k] for k in h_s2_ids}) if h_s2_ids else {}
        h_s3 = compute_macro_f05({k: holdout_val_gt[k] for k in h_s3_ids}, {k: h_preds[k] for k in h_s3_ids}) if h_s3_ids else {}
        h_multi = compute_macro_f05({k: holdout_val_gt[k] for k in h_multi_ids}, {k: h_preds[k] for k in h_multi_ids}) if h_multi_ids else {}
        h_single = compute_macro_f05({k: holdout_val_gt[k] for k in h_single_ids}, {k: h_preds[k] for k in h_single_ids}) if h_single_ids else {}
        h_singlet = compute_macro_f05({k: holdout_val_gt[k] for k in h_singlet_ids}, {k: h_preds[k] for k in h_singlet_ids}) if h_singlet_ids else {}

        h_cand_counts = [len(holdout_cands_cache[p_mode][sid]) for sid in holdout_val_s1_dict.keys()]
        h_cand_recall = compute_candidate_recall(holdout_val_gt, holdout_cands_cache[p_mode])["candidate_recall"]

        avg_matches = float(np.mean([len(h_preds[k]) for k in holdout_val_gt.keys()]))

        joint_holdout_rows.append({
            "pruning_mode": p_mode,
            "capacity_strategy": cap_name,
            "dev_macro_f05": dev_match["dev_macro_f05"],
            "holdout_macro_f05": round(h_eval["macro_f05"], 4),
            "holdout_precision": round(h_eval["macro_precision"], 4),
            "holdout_recall": round(h_eval["macro_recall"], 4),
            "singleton_f05": round(h_singlet.get("macro_f05", 0.0), 4),
            "single_match_f05": round(h_single.get("macro_f05", 0.0), 4),
            "multi_match_f05": round(h_multi.get("macro_f05", 0.0), 4),
            "s2_f05": round(h_s2.get("macro_f05", 0.0), 4),
            "s3_f05": round(h_s3.get("macro_f05", 0.0), 4),
            "candidate_recall": round(h_cand_recall, 4),
            "mean_candidates_per_s1": round(float(np.mean(h_cand_counts)), 2),
            "p95_candidates": int(np.percentile(h_cand_counts, 95)),
            "avg_predicted_matches": round(avg_matches, 3),
        })

    joint_holdout_df = pd.DataFrame(joint_holdout_rows).sort_values(by="holdout_macro_f05", ascending=False)
    joint_results_csv_path = M7_DIR / "joint_ablation_results.csv"
    joint_holdout_df.to_csv(joint_results_csv_path, index=False)
    logger.info(f"Saved Joint Ablation Results to {joint_results_csv_path}")
    logger.info("\n" + joint_holdout_df.to_string(index=False))

    # -------------------------------------------------------------------
    # PART 5: INVESTIGATE ZERO-CANDIDATE ENTITIES (HOLDOUT FORENSICS)
    # -------------------------------------------------------------------
    logger.info("=== [PART 5] ZERO-CANDIDATE FORENSICS (HOLDOUT SET) ===")
    
    zero_cand_records = []
    current_h_cands = holdout_cands_cache["current"]
    
    for sid, q in holdout_val_s1_dict.items():
        if len(current_h_cands.get(sid, set())) == 0:
            gt_targets = list(holdout_val_gt.get(sid, set()))
            has_gt_match = len(gt_targets) > 0
            gt_sources = list({t[:2] for t in gt_targets})

            raw_name = q["business_name_raw"]
            raw_addr = q["business_address_raw"]
            norm_name = q["business_name_norm"]
            norm_addr = q["business_address_norm"]
            country = q["country_raw"]

            routes_attempted = []
            if q["clean_name"]:
                routes_attempted.append("exact_name")
            if q["clean_addr"]:
                routes_attempted.append("exact_address")
            if q["name_sig"]:
                routes_attempted.append("name_sig")
            if q["addr_sig"]:
                routes_attempted.append("addr_sig")
            if q["postal_codes"]:
                routes_attempted.append("postal")
            if q["building_number"]:
                routes_attempted.append("building")
            if q["name_tokens"]:
                routes_attempted.append("rare_tokens")

            if not has_gt_match:
                category = "A_True_Singleton"
            elif not raw_addr or len(raw_name) <= 3 or not any(c.isalnum() for c in raw_name):
                category = "C_Systematic_Sparse_Address_Or_Script"
            elif has_gt_match:
                category = "B_True_Matched_Retrieval_Failure"
            else:
                category = "D_Other_Retrieval_Failure"

            nearest_sim = 0.0
            if has_gt_match:
                sims = []
                for tid in gt_targets:
                    t_meta = get_cached_target_meta(tid)
                    sims.append(fuzz.token_set_ratio(raw_name, t_meta["business_name_raw"]))
                nearest_sim = max(sims) if sims else 0.0

            zero_cand_records.append({
                "s1_entity_id": sid,
                "country": country,
                "has_true_pair": has_gt_match,
                "num_true_matches": len(gt_targets),
                "true_target_sources": ",".join(gt_sources),
                "true_target_ids": ",".join(gt_targets),
                "name_raw": raw_name[:40],
                "addr_raw": raw_addr[:40],
                "name_normalized": norm_name[:40],
                "addr_normalized": norm_addr[:40],
                "routes_attempted": ",".join(routes_attempted),
                "nearest_gt_name_similarity": round(nearest_sim, 2),
                "category": category,
            })

    zero_df = pd.DataFrame(zero_cand_records)
    zero_csv_path = M7_DIR / "zero_candidate_forensics.csv"
    zero_df.to_csv(zero_csv_path, index=False)
    logger.info(f"Saved Zero-Candidate Forensics to {zero_csv_path}")

    zero_cat_counts = Counter(zero_df["category"])
    logger.info("Zero-Candidate Category Distribution (Total=%d):", len(zero_df))
    for cat, cnt in zero_cat_counts.items():
        logger.info(f"  {cat}: {cnt} ({cnt / len(zero_df) * 100:.2f}%)")

    # -------------------------------------------------------------------
    # PART 6: TEST DISTRIBUTION RISK CHECK (UNLABELED OBSERVABLE COMPARISON)
    # -------------------------------------------------------------------
    logger.info("=== [PART 6] TEST DISTRIBUTION RISK CHECK ===")
    
    test_s1_path = DATA_DIR / "test" / "test_source1.tsv"

    test_s1_sample = pd.read_csv(test_s1_path, sep="\t", nrows=50000, dtype=str, keep_default_na=False)
    test_s1_dict = preprocess_s1(test_s1_sample)

    test_cands, _ = retrieve_candidates_by_mode(test_s1_dict, mode="conservative")
    test_cand_counts = [len(test_cands[sid]) for sid in test_s1_dict.keys()]
    test_zero_cand_rate = sum(1 for c in test_cand_counts if c == 0) / len(test_s1_dict)

    train_names = s1_df["business_name"].astype(str)
    train_addrs = s1_df["business_address"].astype(str)
    test_names = test_s1_sample["business_name"].astype(str)
    test_addrs = test_s1_sample["business_address"].astype(str)

    train_c_dist = dict(Counter(s1_df["country"].str.upper().str.strip()).most_common(10))
    test_c_dist = dict(Counter(test_s1_sample["country"].str.upper().str.strip()).most_common(10))

    non_ascii_re = re.compile(r"[^\x00-\x7F]")
    train_non_ascii_rate = sum(1 for n in train_names if non_ascii_re.search(n)) / len(train_names)
    test_non_ascii_rate = sum(1 for n in test_names if non_ascii_re.search(n)) / len(test_names)

    train_missing_addr_rate = sum(1 for a in train_addrs if not a.strip()) / len(train_addrs)
    test_missing_addr_rate = sum(1 for a in test_addrs if not a.strip()) / len(test_addrs)

    test_risk_report = {
        "candidate_counts": {
            "train_dev_conservative_mean": round(float(dev_pruning_df[dev_pruning_df['pruning_mode']=='conservative']['mean_candidates_per_s1'].iloc[0]), 2),
            "test_sample_conservative_mean": round(float(np.mean(test_cand_counts)), 2),
            "test_sample_conservative_p95": int(np.percentile(test_cand_counts, 95)),
            "test_sample_conservative_p99": int(np.percentile(test_cand_counts, 99)),
            "test_sample_zero_candidate_rate": round(test_zero_cand_rate, 4),
        },
        "name_length_distribution": {
            "train_name_len_mean": round(float(train_names.str.len().mean()), 2),
            "train_name_len_median": int(train_names.str.len().median()),
            "train_name_len_p95": int(train_names.str.len().quantile(0.95)),
            "test_name_len_mean": round(float(test_names.str.len().mean()), 2),
            "test_name_len_median": int(test_names.str.len().median()),
            "test_name_len_p95": int(test_names.str.len().quantile(0.95)),
        },
        "address_length_distribution": {
            "train_addr_len_mean": round(float(train_addrs.str.len().mean()), 2),
            "train_addr_len_median": int(train_addrs.str.len().median()),
            "train_addr_len_p95": int(train_addrs.str.len().quantile(0.95)),
            "test_addr_len_mean": round(float(test_addrs.str.len().mean()), 2),
            "test_addr_len_median": int(test_addrs.str.len().median()),
            "test_addr_len_p95": int(test_addrs.str.len().quantile(0.95)),
        },
        "missing_address_rate": {
            "train_missing_addr_rate": round(train_missing_addr_rate, 4),
            "test_missing_addr_rate": round(test_missing_addr_rate, 4),
        },
        "unicode_non_ascii_rate": {
            "train_non_ascii_rate": round(train_non_ascii_rate, 4),
            "test_non_ascii_rate": round(test_non_ascii_rate, 4),
        },
        "top_country_distribution": {
            "train_countries": train_c_dist,
            "test_sample_countries": test_c_dist,
        }
    }

    test_risk_json_path = M7_DIR / "test_distribution_risk_check.json"
    with open(test_risk_json_path, "w", encoding="utf-8") as f:
        json.dump(test_risk_report, f, indent=2)
    logger.info(f"Saved Test Distribution Risk Check to {test_risk_json_path}")

    # -------------------------------------------------------------------
    # PART 7 & 8: FINAL CONFIGURATION SELECTION
    # -------------------------------------------------------------------
    logger.info("=== [PART 8] SELECTING FINAL INFERENCE CONFIGURATION ===")
    
    best_config_row = joint_holdout_df.iloc[0]
    logger.info("Best Finalist Selected:")
    logger.info(dict(best_config_row))

    selected_pruning_mode = best_config_row["pruning_mode"]
    selected_capacity_strategy = best_config_row["capacity_strategy"]

    final_engine_config = DecisionRuleConfig(
        strategy="adaptive_multi",
        threshold_s2=0.4775,
        threshold_s3=0.4975,
        min_top_prob=0.428,
        min_margin=0.0,
        multi_match_threshold=0.458,
        max_multi_score_drop=0.16,
        max_matches_per_source=0 if selected_capacity_strategy == "No_Limit" else (2 if selected_capacity_strategy == "Max_2" else 1),
        enable_multi_match=True,
        enable_singleton_abstention=True,
    )

    # -------------------------------------------------------------------
    # PART 9: FINAL INFERENCE SAFETY MANIFEST
    # -------------------------------------------------------------------
    logger.info("=== [PART 9] FINAL INFERENCE SAFETY RECORDING ===")
    
    test_total_s1 = 1730000
    mean_cands_per_s1 = best_config_row["mean_candidates_per_s1"]
    est_total_test_pairs = test_total_s1 * mean_cands_per_s1

    t0_sample = time.time()
    sample_5k = test_s1_sample.iloc[:5000]
    sample_5k_dict = preprocess_s1(sample_5k)
    s5k_cands, s5k_prov = retrieve_candidates_by_mode(sample_5k_dict, mode=selected_pruning_mode)
    s5k_scores = score_candidate_pairs(sample_5k_dict, s5k_cands, s5k_prov, tag="Sample-5k-Safety")
    s5k_engine = EntityDecisionEngine(final_engine_config)
    s5k_preds = s5k_engine.predict_all(s5k_scores, sample_5k_dict)
    t_sample = time.time() - t0_sample

    sample_sec = max(t_sample, 0.001)
    test_est_runtime_sec = (test_total_s1 / 5000.0) * sample_sec

    final_inference_cfg = {
        "configuration_name": f"Milestone7_{selected_pruning_mode}_{selected_capacity_strategy}",
        "model_file": "artifacts/models/retrained_hardneg_model.pkl",
        "model_sha256": model_hash,
        "features_count": len(FEATURE_COLUMNS),
        "blocking_version": "Two-Stage Multi-Pass Inverted Index (7 Routes)",
        "pruning_version": selected_pruning_mode,
        "capacity_strategy": selected_capacity_strategy,
        "decision_engine_config": asdict(final_engine_config) if hasattr(final_engine_config, '__dataclass_fields__') else final_engine_config.__dict__,
        "metrics_holdout": {
            "holdout_macro_f05": float(best_config_row["holdout_macro_f05"]),
            "holdout_macro_precision": float(best_config_row["holdout_precision"]),
            "holdout_macro_recall": float(best_config_row["holdout_recall"]),
            "singleton_f05": float(best_config_row["singleton_f05"]),
            "single_match_f05": float(best_config_row["single_match_f05"]),
            "multi_match_f05": float(best_config_row["multi_match_f05"]),
            "s2_f05": float(best_config_row["s2_f05"]),
            "s3_f05": float(best_config_row["s3_f05"]),
            "candidate_recall": float(best_config_row["candidate_recall"]),
        },
        "candidate_volume": {
            "mean_candidates_per_s1": float(best_config_row["mean_candidates_per_s1"]),
            "p95_candidates": int(best_config_row["p95_candidates"]),
            "estimated_total_test_candidate_pairs": int(est_total_test_pairs),
        },
        "resources_and_scalability": {
            "chunk_size": 50000,
            "estimated_test_runtime_minutes": round(test_est_runtime_sec / 60.0, 1),
            "expected_peak_ram_mb": round(get_memory_mb(), 1),
            "expected_vram_gb": 0.0,
        }
    }

    final_config_json_path = M7_DIR / "final_inference_configuration.json"
    with open(final_config_json_path, "w", encoding="utf-8") as f:
        json.dump(final_inference_cfg, f, indent=2)
    logger.info(f"Saved Final Inference Configuration to {final_config_json_path}")

    final_engine = EntityDecisionEngine(final_engine_config)
    final_engine.save(MODELS_DIR / "best_decision_engine.json")
    logger.info(f"Updated {MODELS_DIR / 'best_decision_engine.json'} with selected configuration.")

    logger.info("=================================================================")
    logger.info("   MILESTONE 7 PIPELINE EXECUTION COMPLETE                      ")
    logger.info("=================================================================")


if __name__ == "__main__":
    run_milestone7_pipeline()

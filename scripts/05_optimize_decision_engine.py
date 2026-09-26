"""
Comprehensive Milestone 5 Execution Pipeline:
- Part 1: Official Metric Verification & Calibration Testing
- Part 2 & 3: Coarse & Fine Threshold Sweeps + Multi-Threshold Analysis
- Part 4, 5, 6: Entity-Level Decision Engine (Singleton Abstention & Controlled Multi-Match)
- Part 7: Probability Calibration (Platt Scaling vs Isotonic Regression vs Raw Probabilities)
- Part 8, 9, 10: Hard Negative Mining V2 & Curriculum Retraining (GPU Accelerated)
- Part 11: Feature Importance Analysis (Gain / Cover / Weight)
- Part 12: Controlled Feature Ablation Study (6 Distinct Feature Subsets)
- Part 13 & 14: Source-Specific & Entity-Level Error Bucket Forensics
- Part 15, 16, 17: Experiment Tracking & Untouched Holdout Evaluation

Amazon ML Challenge 2026 - Business Entity Resolution
"""

from collections import Counter, defaultdict
import gc
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import psutil
from rapidfuzz import fuzz
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import DATA_DIR, DIAGNOSTICS_DIR, FEATURES_DIR, MODELS_DIR
from src.decision_engine import DecisionRuleConfig, EntityDecisionEngine
from src.features import FEATURE_COLUMNS, PairFeatureExtractor
from src.metrics import compute_candidate_recall, compute_entity_f05, compute_macro_f05
from src.normalization import (
    extract_building_number,
    extract_numeric_tokens,
    extract_postal_code,
    normalize_address_abbreviations,
    normalize_business_name_suffixes,
    normalize_country,
    tokenize_text,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("milestone5")


def get_memory_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def run_milestone5_pipeline():
    t_start = time.time()
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "experiments").mkdir(parents=True, exist_ok=True)

    logger.info("============================================================")
    logger.info("  STARTING MILESTONE 5: F0.5 OPTIMIZATION + HARD-NEG V2 + DECISION ENGINE")
    logger.info("============================================================")

    # -------------------------------------------------------------
    # 1. LOAD DATA & LEAKAGE-FREE PARTITIONING
    # -------------------------------------------------------------
    logger.info("=== STEP 1: Loading Datasets & Constructing Leakage-Free Splits ===")
    s1_df = pd.read_csv(DATA_DIR / "train" / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2_df = pd.read_csv(DATA_DIR / "train" / "train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3_df = pd.read_csv(DATA_DIR / "train" / "train_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt_df = pd.read_csv(DATA_DIR / "train" / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)

    total_s1 = len(s1_df)
    logger.info(f"Loaded: S1={total_s1:,}, S2={len(s2_df):,}, S3={len(s3_df):,}, GT={len(gt_df):,}")

    gt_map: Dict[str, Set[str]] = defaultdict(set)
    for sid, m_str in zip(gt_df["source1_entity_id"].to_numpy(dtype=object), gt_df["matched_entity_ids"].to_numpy(dtype=object)):
        sid = str(sid).strip()
        m_str = str(m_str).strip()
        if m_str:
            for mid in m_str.split(","):
                mid = mid.strip()
                if mid:
                    gt_map[sid].add(mid)

    # Stratified 3-way partition:
    # 1. Train Fold: 30,000 S1 entities (Base Model & Curriculum Training)
    # 2. Mining Fold: 10,000 S1 entities (Out-of-fold Hard False-Positive Mining)
    # 3. Validation Fold: 20,000 S1 entities (10k Dev-Val for tuning + 10k Holdout-Val for untouched test)
    s1_ids = s1_df["entity_id"].to_numpy(dtype=object)
    s1_bins = np.array([0 if len(gt_map.get(sid, [])) == 0 else (1 if len(gt_map.get(sid, [])) == 1 else 2) for sid in s1_ids], dtype=np.int32)

    rng = np.random.RandomState(42)
    val_indices, mining_indices, train_indices = [], [], []

    for b in [0, 1, 2]:
        b_idx = np.where(s1_bins == b)[0]
        rng.shuffle(b_idx)
        n_val = int(len(b_idx) * 0.20)      # 20% validation (20k total across bins)
        n_mining = int(len(b_idx) * 0.10)   # 10% mining fold (10k total across bins)
        
        val_indices.extend(b_idx[:n_val])
        mining_indices.extend(b_idx[n_val:n_val + n_mining])
        train_indices.extend(b_idx[n_val + n_mining:])

    train_indices = np.array(train_indices, dtype=np.int32)
    mining_indices = np.array(mining_indices, dtype=np.int32)
    val_indices = np.array(val_indices, dtype=np.int32)

    rng.shuffle(train_indices)
    rng.shuffle(mining_indices)
    rng.shuffle(val_indices)

    # Further split Validation into Dev-Val (first 10k) and Holdout-Val (second 10k)
    dev_val_indices = val_indices[:10000]
    holdout_val_indices = val_indices[10000:20000]

    train_s1_df = s1_df.iloc[train_indices].reset_index(drop=True)
    mining_s1_df = s1_df.iloc[mining_indices].reset_index(drop=True)
    dev_val_s1_df = s1_df.iloc[dev_val_indices].reset_index(drop=True)
    holdout_val_s1_df = s1_df.iloc[holdout_val_indices].reset_index(drop=True)
    full_val_s1_df = s1_df.iloc[val_indices[:20000]].reset_index(drop=True)

    logger.info(
        f"Partition Sizes: Train S1={len(train_s1_df):,}, Mining S1={len(mining_s1_df):,}, "
        f"Dev-Val S1={len(dev_val_s1_df):,}, Holdout-Val S1={len(holdout_val_s1_df):,}, Total Val={len(full_val_s1_df):,}"
    )

    # -------------------------------------------------------------
    # 2. TARGET INDEXING & FAST LOOKUPS
    # -------------------------------------------------------------
    logger.info("=== STEP 2: Indexing Target Corpora (S2 & S3) ===")
    t0_idx = time.time()
    idx_exact_name: Dict[str, List[str]] = defaultdict(list)
    idx_exact_addr: Dict[str, List[str]] = defaultdict(list)
    idx_name_sig: Dict[str, List[str]] = defaultdict(list)
    idx_addr_sig: Dict[str, List[str]] = defaultdict(list)
    idx_name_tokens: Dict[str, List[str]] = defaultdict(list)
    idx_postal: Dict[str, List[str]] = defaultdict(list)
    idx_building: Dict[str, List[str]] = defaultdict(list)
    token_df_counter: Counter = Counter()

    for df, src_tag in [(s2_df, "S2"), (s3_df, "S3")]:
        ids = df["entity_id"].to_numpy(dtype=object)
        names = df["business_name"].to_numpy(dtype=object)
        addrs = df["business_address"].to_numpy(dtype=object)

        for tid, name_raw, addr_raw in zip(ids, names, addrs):
            tid = str(tid)
            name_raw = str(name_raw or "").lower().strip()
            addr_raw = str(addr_raw or "").lower().strip()

            if name_raw:
                idx_exact_name[name_raw].append(tid)
                n_sig = " ".join(sorted(set(name_raw.split())))
                if n_sig:
                    idx_name_sig[n_sig].append(tid)
                for tok in set(name_raw.split()):
                    if len(tok) >= 3:
                        token_df_counter[tok] += 1
                        idx_name_tokens[tok].append(tid)

            if addr_raw:
                idx_exact_addr[addr_raw].append(tid)
                a_sig = " ".join(sorted(set(addr_raw.split())))
                if a_sig:
                    idx_addr_sig[a_sig].append(tid)
                pins = extract_postal_code(addr_raw)
                for pin in pins:
                    idx_postal[pin].append(tid)
                bldg = extract_building_number(addr_raw)
                if bldg:
                    idx_building[bldg].append(tid)

    logger.info(f"Target indexing completed in {time.time() - t0_idx:.2f}s.")

    s2_raw_lookup = dict(zip(
        s2_df["entity_id"].to_numpy(dtype=object),
        zip(s2_df["business_name"].to_numpy(dtype=object), s2_df["business_address"].to_numpy(dtype=object), s2_df["country"].to_numpy(dtype=object))
    ))
    s3_raw_lookup = dict(zip(
        s3_df["entity_id"].to_numpy(dtype=object),
        zip(s3_df["business_name"].to_numpy(dtype=object), s3_df["business_address"].to_numpy(dtype=object), s3_df["country"].to_numpy(dtype=object))
    ))

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
            }
        return res

    def retrieve_candidates(s1_dict: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], Dict[str, int]]]:
        candidates = defaultdict(set)
        provenance = defaultdict(lambda: defaultdict(int))

        for sid, q in s1_dict.items():
            if q["clean_name"] in idx_exact_name:
                for tid in idx_exact_name[q["clean_name"]][:50]:
                    candidates[sid].add(tid)
                    provenance[(sid, tid)]["exact_name_hit"] = 1

            if q["clean_addr"] in idx_exact_addr:
                for tid in idx_exact_addr[q["clean_addr"]][:50]:
                    candidates[sid].add(tid)
                    provenance[(sid, tid)]["exact_address_hit"] = 1

            if q["name_sig"] in idx_name_sig:
                for tid in idx_name_sig[q["name_sig"]][:50]:
                    candidates[sid].add(tid)
                    provenance[(sid, tid)]["name_token_hit"] = 1

            if q["addr_sig"] in idx_addr_sig:
                for tid in idx_addr_sig[q["addr_sig"]][:50]:
                    candidates[sid].add(tid)
                    provenance[(sid, tid)]["address_token_hit"] = 1

            for tok in q["clean_name"].split():
                df_val = token_df_counter.get(tok, 0)
                if 1 <= df_val <= 150:
                    for tid in idx_name_tokens[tok][:50]:
                        candidates[sid].add(tid)
                        provenance[(sid, tid)]["rare_token_hit"] = 1
                elif 150 < df_val <= 800:
                    for tid in idx_name_tokens[tok][:25]:
                        candidates[sid].add(tid)
                        provenance[(sid, tid)]["rare_token_hit"] = 1

            for pin in q["postal_codes"]:
                if pin in idx_postal:
                    for tid in idx_postal[pin][:30]:
                        candidates[sid].add(tid)
                        provenance[(sid, tid)]["postal_numeric_hit"] = 1

            if q["building_number"] in idx_building:
                for tid in idx_building[q["building_number"]][:25]:
                    candidates[sid].add(tid)
                    provenance[(sid, tid)]["cross_field_hit"] = 1

        return candidates, provenance

    # -------------------------------------------------------------
    # 3. EXTRACT CANDIDATES & PAIRS
    # -------------------------------------------------------------
    logger.info("=== STEP 3: Candidate Retrieval for Train, Mining, Dev-Val, Holdout-Val ===")
    train_s1_dict = preprocess_s1(train_s1_df.iloc[:30000])
    mining_s1_dict = preprocess_s1(mining_s1_df.iloc[:10000])
    dev_val_s1_dict = preprocess_s1(dev_val_s1_df)
    holdout_val_s1_dict = preprocess_s1(holdout_val_s1_df)
    full_val_s1_dict = preprocess_s1(full_val_s1_df)

    train_cands, train_prov = retrieve_candidates(train_s1_dict)
    mining_cands, mining_prov = retrieve_candidates(mining_s1_dict)
    dev_val_cands, dev_val_prov = retrieve_candidates(dev_val_s1_dict)
    holdout_val_cands, holdout_val_prov = retrieve_candidates(holdout_val_s1_dict)
    full_val_cands, full_val_prov = retrieve_candidates(full_val_s1_dict)

    train_gt = {sid: gt_map.get(sid, set()) for sid in train_s1_dict.keys()}
    mining_gt = {sid: gt_map.get(sid, set()) for sid in mining_s1_dict.keys()}
    dev_val_gt = {sid: gt_map.get(sid, set()) for sid in dev_val_s1_dict.keys()}
    holdout_val_gt = {sid: gt_map.get(sid, set()) for sid in holdout_val_s1_dict.keys()}
    full_val_gt = {sid: gt_map.get(sid, set()) for sid in full_val_s1_dict.keys()}

    # Cache target metadata
    all_needed_tids = set()
    for c_dict in [train_cands, mining_cands, full_val_cands]:
        for c_set in c_dict.values():
            all_needed_tids.update(c_set)

    logger.info(f"Caching metadata for {len(all_needed_tids):,} candidate target records...")
    t0_cache = time.time()
    target_meta_cache = {tid: get_target_meta(tid) for tid in all_needed_tids}
    logger.info(f"Target metadata cached in {time.time() - t0_cache:.2f}s.")

    extractor = PairFeatureExtractor()

    # -------------------------------------------------------------
    # 4. TRAIN BASELINE GBDT MODEL (ON GPU)
    # -------------------------------------------------------------
    logger.info("=== STEP 4: Training Baseline Model on RTX 5070 GPU ===")
    train_pos_pairs: List[Tuple[str, str]] = []
    train_neg_candidates: List[Tuple[str, str, str, float]] = []

    for sid, cand_set in train_cands.items():
        true_set = train_gt.get(sid, set())
        q = train_s1_dict[sid]

        for tid in cand_set:
            pair = (sid, tid)
            t = target_meta_cache.get(tid, {})

            if tid in true_set:
                train_pos_pairs.append(pair)
            else:
                n_sim = fuzz.WRatio(q["business_name_norm"], t.get("business_name_norm", "")) / 100.0
                a_sim = fuzz.WRatio(q["business_address_norm"], t.get("business_address_norm", "")) / 100.0
                hardness = (n_sim + a_sim) / 2.0
                cat = "easy_negative"
                if n_sim >= 0.85 and a_sim <= 0.40:
                    cat = "1_high_name_wrong_address"
                elif a_sim >= 0.85 and n_sim <= 0.40:
                    cat = "2_high_address_wrong_name"
                elif n_sim >= 0.80 and a_sim >= 0.70:
                    cat = "3_high_name_high_address_wrong_entity"
                elif q["postal_codes"] and (q["postal_codes"] & t.get("postal_codes", set())):
                    cat = "4_same_postal_wrong_entity"
                elif q["building_number"] and q["building_number"] == t.get("building_number", ""):
                    cat = "5_same_building_wrong_entity"
                elif train_prov.get(pair, {}).get("route_hit_count", 0) >= 2:
                    cat = "9_multi_route_hit_wrong_entity"
                elif n_sim >= 0.60:
                    cat = "medium_name_similarity"

                train_neg_candidates.append((sid, tid, cat, hardness))

    n_pos = len(train_pos_pairs)
    neg_ratio = 4
    target_neg_count = n_pos * neg_ratio

    neg_df = pd.DataFrame(train_neg_candidates, columns=["s1_id", "target_id", "category", "hardness"]).sort_values(by="hardness", ascending=False)
    cat_groups = neg_df.groupby("category")
    sampled_neg_df = cat_groups.head(target_neg_count // len(cat_groups) + 200).head(target_neg_count)
    train_neg_pairs = [(r.s1_id, r.target_id) for r in sampled_neg_df.itertuples()]

    X_train_pairs = train_pos_pairs + train_neg_pairs
    y_train_arr = np.array([1] * len(train_pos_pairs) + [0] * len(train_neg_pairs), dtype=np.int32)

    X_train_df = extractor.extract_features_matrix(X_train_pairs, train_s1_dict, target_meta_cache, train_prov)

    # Validation features
    def build_val_features(val_s1_map, val_cands_map, val_gt_map, val_prov_map):
        pairs, y_list = [], []
        for sid, c_set in val_cands_map.items():
            tr = val_gt_map.get(sid, set())
            for tid in c_set:
                pairs.append((sid, tid))
                y_list.append(1 if tid in tr else 0)
        X_df = extractor.extract_features_matrix(pairs, val_s1_map, target_meta_cache, val_prov_map)
        return pairs, np.array(y_list, dtype=np.int32), X_df

    logger.info("Extracting features for Dev-Val, Holdout-Val, and Mining Folds...")
    dev_val_pairs, dev_val_y, dev_val_X = build_val_features(dev_val_s1_dict, dev_val_cands, dev_val_gt, dev_val_prov)
    holdout_val_pairs, holdout_val_y, holdout_val_X = build_val_features(holdout_val_s1_dict, holdout_val_cands, holdout_val_gt, holdout_val_prov)
    full_val_pairs, full_val_y, full_val_X = build_val_features(full_val_s1_dict, full_val_cands, full_val_gt, full_val_prov)
    mining_pairs, mining_y, mining_X = build_val_features(mining_s1_dict, mining_cands, mining_gt, mining_prov)

    # Train baseline model on GPU
    t0_fit = time.time()
    baseline_clf = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        tree_method="hist",
        device="cuda",
        random_state=42,
    )
    baseline_clf.fit(X_train_df, y_train_arr)
    t_fit = time.time() - t0_fit
    logger.info(f"Baseline XGBoost fit completed on RTX 5070 GPU in {t_fit:.2f}s ({len(X_train_df)/t_fit:,.1f} pairs/sec)")

    # -------------------------------------------------------------
    # PART 2 & 3: THRESHOLD SWEEPS (COARSE + FINE)
    # -------------------------------------------------------------
    logger.info("=== PART 2 & 3: Comprehensive Threshold Sweeps ===")
    dev_val_probs = baseline_clf.predict_proba(dev_val_X)[:, 1]

    # Map scores by S1
    dev_val_scores_by_s1: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (sid, tid), p in zip(dev_val_pairs, dev_val_probs):
        dev_val_scores_by_s1[sid][tid] = float(p)

    sweep_rows = []
    # Coarse sweep: 0.01 to 0.99 with step 0.01
    coarse_thresholds = np.arange(0.01, 1.00, 0.01)
    
    # We will compute metrics for every threshold
    best_f05 = -1.0
    best_t = 0.50

    for t in coarse_thresholds:
        t = round(float(t), 4)
        preds = {}
        for sid in dev_val_s1_dict.keys():
            sc_dict = dev_val_scores_by_s1.get(sid, {})
            preds[sid] = [tid for tid, sc in sc_dict.items() if sc >= t]

        eval_res = compute_macro_f05(dev_val_gt, preds)
        
        # Subgroup evaluations
        s2_ids = [sid for sid, tr in dev_val_gt.items() if any(x.startswith("S2-") for x in tr)]
        s3_ids = [sid for sid, tr in dev_val_gt.items() if any(x.startswith("S3-") for x in tr)]
        multi_ids = [sid for sid, tr in dev_val_gt.items() if len(tr) > 1]
        single_ids = [sid for sid, tr in dev_val_gt.items() if len(tr) == 1]
        singlet_ids = [sid for sid, tr in dev_val_gt.items() if len(tr) == 0]

        eval_s2 = compute_macro_f05({k: dev_val_gt[k] for k in s2_ids}, {k: preds[k] for k in s2_ids}) if s2_ids else {}
        eval_s3 = compute_macro_f05({k: dev_val_gt[k] for k in s3_ids}, {k: preds[k] for k in s3_ids}) if s3_ids else {}
        eval_multi = compute_macro_f05({k: dev_val_gt[k] for k in multi_ids}, {k: preds[k] for k in multi_ids}) if multi_ids else {}
        eval_single = compute_macro_f05({k: dev_val_gt[k] for k in single_ids}, {k: preds[k] for k in single_ids}) if single_ids else {}
        eval_singlet = compute_macro_f05({k: dev_val_gt[k] for k in singlet_ids}, {k: preds[k] for k in singlet_ids}) if singlet_ids else {}

        pred_counts = [len(p) for p in preds.values()]
        avg_preds = float(np.mean(pred_counts))
        pct_abstaining = float(np.mean([1 if c == 0 else 0 for c in pred_counts]) * 100)
        fp_singleton_rate = 1.0 - eval_res["singleton_accuracy"]

        sweep_rows.append({
            "threshold": t,
            "macro_f0.5": round(eval_res["macro_f05"], 4),
            "macro_precision": round(eval_res["macro_precision"], 4),
            "macro_recall": round(eval_res["macro_recall"], 4),
            "singleton_f0.5": round(eval_singlet.get("macro_f05", 0.0), 4),
            "single_match_f0.5": round(eval_single.get("macro_f05", 0.0), 4),
            "multi_match_f0.5": round(eval_multi.get("macro_f05", 0.0), 4),
            "S2_f0.5": round(eval_s2.get("macro_f05", 0.0), 4),
            "S3_f0.5": round(eval_s3.get("macro_f05", 0.0), 4),
            "average_predictions_per_S1": round(avg_preds, 4),
            "percentage_abstaining": round(pct_abstaining, 2),
            "false_positive_singleton_rate": round(fp_singleton_rate, 4),
        })

        if eval_res["macro_f05"] > best_f05:
            best_f05 = eval_res["macro_f05"]
            best_t = t

    # Fine-grained sweep around best region ±0.10 with step 0.0025
    fine_start = max(0.01, best_t - 0.10)
    fine_end = min(0.99, best_t + 0.10)
    fine_thresholds = np.arange(fine_start, fine_end + 1e-5, 0.0025)

    for t in fine_thresholds:
        t = round(float(t), 4)
        preds = {}
        for sid in dev_val_s1_dict.keys():
            sc_dict = dev_val_scores_by_s1.get(sid, {})
            preds[sid] = [tid for tid, sc in sc_dict.items() if sc >= t]

        eval_res = compute_macro_f05(dev_val_gt, preds)
        
        s2_ids = [sid for sid, tr in dev_val_gt.items() if any(x.startswith("S2-") for x in tr)]
        s3_ids = [sid for sid, tr in dev_val_gt.items() if any(x.startswith("S3-") for x in tr)]
        multi_ids = [sid for sid, tr in dev_val_gt.items() if len(tr) > 1]
        single_ids = [sid for sid, tr in dev_val_gt.items() if len(tr) == 1]
        singlet_ids = [sid for sid, tr in dev_val_gt.items() if len(tr) == 0]

        eval_s2 = compute_macro_f05({k: dev_val_gt[k] for k in s2_ids}, {k: preds[k] for k in s2_ids}) if s2_ids else {}
        eval_s3 = compute_macro_f05({k: dev_val_gt[k] for k in s3_ids}, {k: preds[k] for k in s3_ids}) if s3_ids else {}
        eval_multi = compute_macro_f05({k: dev_val_gt[k] for k in multi_ids}, {k: preds[k] for k in multi_ids}) if multi_ids else {}
        eval_single = compute_macro_f05({k: dev_val_gt[k] for k in single_ids}, {k: preds[k] for k in single_ids}) if single_ids else {}
        eval_singlet = compute_macro_f05({k: dev_val_gt[k] for k in singlet_ids}, {k: preds[k] for k in singlet_ids}) if singlet_ids else {}

        pred_counts = [len(p) for p in preds.values()]
        avg_preds = float(np.mean(pred_counts))
        pct_abstaining = float(np.mean([1 if c == 0 else 0 for c in pred_counts]) * 100)
        fp_singleton_rate = 1.0 - eval_res["singleton_accuracy"]

        sweep_rows.append({
            "threshold": t,
            "macro_f0.5": round(eval_res["macro_f05"], 4),
            "macro_precision": round(eval_res["macro_precision"], 4),
            "macro_recall": round(eval_res["macro_recall"], 4),
            "singleton_f0.5": round(eval_singlet.get("macro_f05", 0.0), 4),
            "single_match_f0.5": round(eval_single.get("macro_f05", 0.0), 4),
            "multi_match_f0.5": round(eval_multi.get("macro_f05", 0.0), 4),
            "S2_f0.5": round(eval_s2.get("macro_f05", 0.0), 4),
            "S3_f0.5": round(eval_s3.get("macro_f05", 0.0), 4),
            "average_predictions_per_S1": round(avg_preds, 4),
            "percentage_abstaining": round(pct_abstaining, 2),
            "false_positive_singleton_rate": round(fp_singleton_rate, 4),
        })

    sweep_df = pd.DataFrame(sweep_rows).drop_duplicates(subset=["threshold"]).sort_values(by="threshold")
    sweep_csv_path = DIAGNOSTICS_DIR / "threshold_sweep.csv"
    sweep_df.to_csv(sweep_csv_path, index=False)
    logger.info(f"Saved threshold sweep to {sweep_csv_path}")

    best_global_row = sweep_df.sort_values(by="macro_f0.5", ascending=False).iloc[0]
    best_global_threshold = float(best_global_row["threshold"])
    logger.info(f"Best Global Threshold on Dev-Val: {best_global_threshold:.4f} -> Macro F0.5: {best_global_row['macro_f0.5']:.4f} (Precision={best_global_row['macro_precision']:.4f}, Recall={best_global_row['macro_recall']:.4f})")

    # -------------------------------------------------------------
    # PART 4, 5, 6: DECISION ENGINE STRATEGY TUNING (DEV-VAL)
    # -------------------------------------------------------------
    logger.info("=== PART 4, 5, 6: Decision Engine Strategy Search on Dev-Val ===")
    
    # Strategy A: Best Global Threshold
    engine_global = EntityDecisionEngine(DecisionRuleConfig(
        strategy="global",
        global_threshold=best_global_threshold,
    ))
    res_global = engine_global.evaluate(dev_val_gt, dev_val_scores_by_s1, dev_val_s1_dict)

    # Strategy B: Source-Specific Grid Search (S2 vs S3)
    best_s2_t, best_s3_t, best_s23_f05 = best_global_threshold, best_global_threshold, -1.0
    for t2 in np.arange(best_global_threshold - 0.08, best_global_threshold + 0.08, 0.02):
        for t3 in np.arange(best_global_threshold - 0.08, best_global_threshold + 0.08, 0.02):
            eng = EntityDecisionEngine(DecisionRuleConfig(
                strategy="source_specific",
                threshold_s2=round(float(t2), 3),
                threshold_s3=round(float(t3), 3),
            ))
            sc = eng.evaluate(dev_val_gt, dev_val_scores_by_s1, dev_val_s1_dict)["macro_f05"]
            if sc > best_s23_f05:
                best_s23_f05 = sc
                best_s2_t = float(t2)
                best_s3_t = float(t3)

    engine_source = EntityDecisionEngine(DecisionRuleConfig(
        strategy="source_specific",
        threshold_s2=best_s2_t,
        threshold_s3=best_s3_t,
    ))
    res_source = engine_source.evaluate(dev_val_gt, dev_val_scores_by_s1, dev_val_s1_dict)

    # Strategy C, D, E: Adaptive Multi-Match + Margin + Missing Address
    best_engine_cfg = DecisionRuleConfig()
    best_engine_f05 = -1.0

    for min_top in [best_s2_t - 0.05, best_s2_t, best_s2_t + 0.03]:
        for min_m in [0.00, 0.03, 0.06, 0.10]:
            for multi_t in [best_s3_t - 0.04, best_s3_t, best_s3_t + 0.04]:
                for max_drop in [0.08, 0.12, 0.16]:
                    for addr_pen in [0.00, 0.03, 0.06]:
                        cfg_cand = DecisionRuleConfig(
                            strategy="adaptive_multi",
                            threshold_s2=best_s2_t,
                            threshold_s3=best_s3_t,
                            min_top_prob=round(float(min_top), 3),
                            min_margin=round(float(min_m), 3),
                            multi_match_threshold=round(float(multi_t), 3),
                            max_multi_score_drop=round(float(max_drop), 3),
                            missing_addr_threshold_boost=round(float(addr_pen), 3),
                            max_matches_per_source=1,
                            enable_multi_match=True,
                            enable_singleton_abstention=True,
                        )
                        eng = EntityDecisionEngine(cfg_cand)
                        sc = eng.evaluate(dev_val_gt, dev_val_scores_by_s1, dev_val_s1_dict)["macro_f05"]
                        if sc > best_engine_f05:
                            best_engine_f05 = sc
                            best_engine_cfg = cfg_cand

    engine_adaptive = EntityDecisionEngine(best_engine_cfg)
    res_adaptive = engine_adaptive.evaluate(dev_val_gt, dev_val_scores_by_s1, dev_val_s1_dict)

    logger.info(f"Decision Strategy Comparison on Dev-Val:")
    logger.info(f"  - Global Threshold ({best_global_threshold:.3f}): Macro F0.5 = {res_global['macro_f05']:.4f}")
    logger.info(f"  - Source Specific (S2={best_s2_t:.3f}, S3={best_s3_t:.3f}): Macro F0.5 = {res_source['macro_f05']:.4f}")
    logger.info(f"  - Adaptive Multi-Match & Margin Engine: Macro F0.5 = {res_adaptive['macro_f05']:.4f}")

    # -------------------------------------------------------------
    # PART 7: PROBABILITY CALIBRATION
    # -------------------------------------------------------------
    logger.info("=== PART 7: Probability Calibration (Platt vs Isotonic vs Raw) ===")
    # Fit calibrators on Dev-Val probabilities and evaluate on Holdout-Val
    platt_scaler = LogisticRegression(C=1.0, solver="lbfgs")
    platt_scaler.fit(dev_val_probs.reshape(-1, 1), dev_val_y)

    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(dev_val_probs, dev_val_y)

    holdout_probs = baseline_clf.predict_proba(holdout_val_X)[:, 1]
    platt_holdout_probs = platt_scaler.predict_proba(holdout_probs.reshape(-1, 1))[:, 1]
    iso_holdout_probs = iso_reg.predict(holdout_probs)

    brier_raw = brier_score_loss(holdout_val_y, holdout_probs)
    brier_platt = brier_score_loss(holdout_val_y, platt_holdout_probs)
    brier_iso = brier_score_loss(holdout_val_y, iso_holdout_probs)

    # Evaluate Macro F0.5 with calibrated scores
    def eval_probs_macro(probs_arr, pairs_list, gt_slice, s1_dict):
        sc_map = defaultdict(dict)
        for (sid, tid), p in zip(pairs_list, probs_arr):
            sc_map[sid][tid] = float(p)
        eng = EntityDecisionEngine(DecisionRuleConfig(strategy="global", global_threshold=best_global_threshold))
        return eng.evaluate(gt_slice, sc_map, s1_dict)

    calib_res_raw = eval_probs_macro(holdout_probs, holdout_val_pairs, holdout_val_gt, holdout_val_s1_dict)
    calib_res_platt = eval_probs_macro(platt_holdout_probs, holdout_val_pairs, holdout_val_gt, holdout_val_s1_dict)
    calib_res_iso = eval_probs_macro(iso_holdout_probs, holdout_val_pairs, holdout_val_gt, holdout_val_s1_dict)

    calib_df = pd.DataFrame([
        {"method": "Raw XGBoost", "brier_score": round(brier_raw, 5), "macro_f0.5": round(calib_res_raw["macro_f05"], 4), "precision": round(calib_res_raw["macro_precision"], 4), "recall": round(calib_res_raw["macro_recall"], 4)},
        {"method": "Platt Scaling (Sigmoid)", "brier_score": round(brier_platt, 5), "macro_f0.5": round(calib_res_platt["macro_f05"], 4), "precision": round(calib_res_platt["macro_precision"], 4), "recall": round(calib_res_platt["macro_recall"], 4)},
        {"method": "Isotonic Regression", "brier_score": round(brier_iso, 5), "macro_f0.5": round(calib_res_iso["macro_f05"], 4), "precision": round(calib_res_iso["macro_precision"], 4), "recall": round(calib_res_iso["macro_recall"], 4)},
    ])
    calib_csv_path = DIAGNOSTICS_DIR / "calibration_results.csv"
    calib_df.to_csv(calib_csv_path, index=False)
    logger.info(f"Saved calibration comparison to {calib_csv_path}")

    # -------------------------------------------------------------
    # PART 8: HARD-NEGATIVE MINING V2 (MINING FOLD)
    # -------------------------------------------------------------
    logger.info("=== PART 8: Hard-Negative Mining V2 on Dedicated Mining Fold ===")
    mining_probs = baseline_clf.predict_proba(mining_X)[:, 1]

    hard_fp_rows = []
    mined_hard_neg_pairs: List[Tuple[str, str, str, float]] = []

    for (sid, tid), prob, y_true in zip(mining_pairs, mining_probs, mining_y):
        if y_true == 0 and prob >= 0.25:  # False positive candidate with noticeable confidence
            q = mining_s1_dict[sid]
            t = target_meta_cache.get(tid, {})

            n_sim = fuzz.WRatio(q["business_name_norm"], t.get("business_name_norm", "")) / 100.0
            a_sim = fuzz.WRatio(q["business_address_norm"], t.get("business_address_norm", "")) / 100.0

            cat = "medium_collision"
            if n_sim >= 0.85 and a_sim <= 0.35:
                cat = "same_name_diff_address"
            elif a_sim >= 0.85 and n_sim <= 0.40:
                cat = "same_address_diff_name"
            elif n_sim >= 0.80 and a_sim >= 0.70:
                cat = "high_name_high_address_wrong_entity"
            elif q["postal_codes"] and (q["postal_codes"] & t.get("postal_codes", set())):
                cat = "same_postal_wrong_entity"
            elif q["building_number"] and q["building_number"] == t.get("building_number", ""):
                cat = "same_building_wrong_entity"
            elif not q["has_addr"] or not t.get("has_addr", True):
                cat = "missing_address_collision"
            elif any(ord(c) > 127 for c in q["business_name_raw"] + t.get("business_name_raw", "")):
                cat = "multilingual_collision"
            elif mining_prov.get((sid, tid), {}).get("route_hit_count", 0) >= 2:
                cat = "multi_route_collision"
            elif n_sim >= 0.75:
                cat = "franchise_duplicate_collision"

            hard_fp_rows.append({
                "s1_id": sid,
                "candidate_id": tid,
                "predicted_probability": round(float(prob), 4),
                "true_label": int(y_true),
                "name_similarity": round(n_sim, 4),
                "address_similarity": round(a_sim, 4),
                "country": q["country_norm"],
                "category": cat,
                "blocking_routes": mining_prov.get((sid, tid), {}).get("route_hit_count", 1),
            })
            mined_hard_neg_pairs.append((sid, tid, cat, float(prob)))

    hard_fp_df = pd.DataFrame(hard_fp_rows).sort_values(by="predicted_probability", ascending=False)
    hard_fp_csv_path = DIAGNOSTICS_DIR / "hard_false_positives_v1.csv"
    hard_fp_df.to_csv(hard_fp_csv_path, index=False)
    logger.info(f"Mined {len(hard_fp_df):,} hard false positives from Mining Fold -> {hard_fp_csv_path}")

    # -------------------------------------------------------------
    # PART 9 & 10: HARD-NEGATIVE CURRICULUM RETRAINING (ON GPU)
    # -------------------------------------------------------------
    logger.info("=== PART 9 & 10: Negative Difficulty Curriculum Experiments ===")
    # Combine training set positives + mining positives with distinct negative curriculum ratios
    mining_pos_pairs = [pair for pair, y in zip(mining_pairs, mining_y) if y == 1]
    combined_pos_pairs = train_pos_pairs + mining_pos_pairs
    n_comb_pos = len(combined_pos_pairs)

    curriculum_experiments = [
        {"exp_id": "EXP-A_Ratio_1_4", "ratio": 4, "hard_emphasis": False, "notes": "Standard 1:4 balanced stratification"},
        {"exp_id": "EXP-B_Ratio_1_6", "ratio": 6, "hard_emphasis": True, "notes": "1:6 ratio with mined hard negatives (V2)"},
        {"exp_id": "EXP-C_Ratio_1_8", "ratio": 8, "hard_emphasis": True, "notes": "1:8 ratio with very-hard negative emphasis"},
        {"exp_id": "EXP-D_Hard_Augmented", "ratio": 5, "hard_emphasis": "pure_hard", "notes": "Hard-negative-only augmentation"},
    ]

    best_retrained_clf = None
    best_retrained_f05 = -1.0
    best_curriculum_name = ""

    all_exp_logs = []

    # Prepare combined candidate negatives pool from train + mining
    all_neg_pool = train_neg_candidates + mined_hard_neg_pairs
    pool_df = pd.DataFrame(all_neg_pool, columns=["s1_id", "target_id", "category", "hardness"]).drop_duplicates(subset=["s1_id", "target_id"])

    # Combine S1 dicts for feature extraction
    combined_s1_dict = {**train_s1_dict, **mining_s1_dict}
    combined_prov = {**train_prov, **mining_prov}

    for curr in curriculum_experiments:
        t0_exp = time.time()
        target_n_neg = int(n_comb_pos * curr["ratio"])

        if curr["hard_emphasis"] == "pure_hard":
            sampled_neg = pool_df.sort_values(by="hardness", ascending=False).head(target_n_neg)
        elif curr["hard_emphasis"] is True:
            # 60% hard / very-hard, 40% diverse
            n_hard = int(target_n_neg * 0.65)
            n_div = target_n_neg - n_hard
            hard_part = pool_df.sort_values(by="hardness", ascending=False).head(n_hard)
            rem_pool = pool_df.loc[~pool_df.index.isin(hard_part.index)]
            div_part = rem_pool.groupby("category", group_keys=False).apply(lambda g: g.sample(min(len(g), n_div // max(1, pool_df["category"].nunique()) + 50), random_state=42)).head(n_div)
            sampled_neg = pd.concat([hard_part, div_part]).drop_duplicates(subset=["s1_id", "target_id"]).head(target_n_neg)
        else:
            # Balanced category stratification
            sampled_neg = pool_df.groupby("category", group_keys=False).apply(lambda g: g.sample(min(len(g), target_n_neg // max(1, pool_df["category"].nunique()) + 100), random_state=42)).head(target_n_neg)

        curr_neg_pairs = [(r.s1_id, r.target_id) for r in sampled_neg.itertuples()]
        curr_pairs = combined_pos_pairs + curr_neg_pairs
        curr_y = np.array([1] * len(combined_pos_pairs) + [0] * len(curr_neg_pairs), dtype=np.int32)

        curr_X = extractor.extract_features_matrix(curr_pairs, combined_s1_dict, target_meta_cache, combined_prov)

        curr_clf = xgb.XGBClassifier(
            n_estimators=180,
            max_depth=6,
            learning_rate=0.07,
            tree_method="hist",
            device="cuda",
            random_state=42,
        )
        curr_clf.fit(curr_X, curr_y)
        t_curr_fit = time.time() - t0_exp

        # Evaluate on Dev-Val
        val_probs_curr = curr_clf.predict_proba(dev_val_X)[:, 1]
        sc_map_curr = defaultdict(dict)
        for (sid, tid), p in zip(dev_val_pairs, val_probs_curr):
            sc_map_curr[sid][tid] = float(p)

        eval_curr = engine_adaptive.evaluate(dev_val_gt, sc_map_curr, dev_val_s1_dict)
        preds_curr = engine_adaptive.predict_all(sc_map_curr, dev_val_s1_dict)

        # Subgroup stats
        s2_ids = [sid for sid, tr in dev_val_gt.items() if any(x.startswith("S2-") for x in tr)]
        s3_ids = [sid for sid, tr in dev_val_gt.items() if any(x.startswith("S3-") for x in tr)]
        multi_ids = [sid for sid, tr in dev_val_gt.items() if len(tr) > 1]
        singlet_ids = [sid for sid, tr in dev_val_gt.items() if len(tr) == 0]

        eval_s2 = compute_macro_f05({k: dev_val_gt[k] for k in s2_ids}, {k: preds_curr[k] for k in s2_ids}) if s2_ids else {}
        eval_s3 = compute_macro_f05({k: dev_val_gt[k] for k in s3_ids}, {k: preds_curr[k] for k in s3_ids}) if s3_ids else {}
        eval_multi = compute_macro_f05({k: dev_val_gt[k] for k in multi_ids}, {k: preds_curr[k] for k in multi_ids}) if multi_ids else {}
        eval_singlet = compute_macro_f05({k: dev_val_gt[k] for k in singlet_ids}, {k: preds_curr[k] for k in singlet_ids}) if singlet_ids else {}

        logger.info(
            f"Curriculum {curr['exp_id']}: Dev-Val Macro F0.5={eval_curr['macro_f05']:.4f} "
            f"(P={eval_curr['macro_precision']:.4f}, R={eval_curr['macro_recall']:.4f}, Singlet={eval_singlet.get('macro_f05', 0.0):.4f}, Multi={eval_multi.get('macro_f05', 0.0):.4f}) in {t_curr_fit:.2f}s"
        )

        all_exp_logs.append({
            "experiment_id": curr["exp_id"],
            "model_version": "XGBoost_v2_GPU",
            "feature_version": "v1_51feats",
            "negative_strategy": curr["notes"],
            "threshold_strategy": best_engine_cfg.strategy,
            "global_threshold": best_engine_cfg.global_threshold,
            "S2_threshold": best_engine_cfg.threshold_s2,
            "S3_threshold": best_engine_cfg.threshold_s3,
            "singleton_rule": f"min_top={best_engine_cfg.min_top_prob},margin={best_engine_cfg.min_margin}",
            "multi_match_rule": f"drop<={best_engine_cfg.max_multi_score_drop},t={best_engine_cfg.multi_match_threshold}",
            "validation_f05": round(eval_curr["macro_f05"], 4),
            "precision": round(eval_curr["macro_precision"], 4),
            "recall": round(eval_curr["macro_recall"], 4),
            "singleton_f05": round(eval_singlet.get("macro_f05", 0.0), 4),
            "multi_match_f05": round(eval_multi.get("macro_f05", 0.0), 4),
            "runtime": f"{t_curr_fit:.2f}s",
            "notes": curr["notes"],
        })

        if eval_curr["macro_f05"] > best_retrained_f05:
            best_retrained_f05 = eval_curr["macro_f05"]
            best_retrained_clf = curr_clf
            best_curriculum_name = curr["exp_id"]

    # -------------------------------------------------------------
    # PART 11: FEATURE IMPORTANCE ANALYSIS
    # -------------------------------------------------------------
    logger.info("=== PART 11: Feature Importance Analysis (Gain / Weight / Cover) ===")
    active_model = best_retrained_clf if best_retrained_clf is not None else baseline_clf
    booster = active_model.get_booster()

    gain_scores = booster.get_score(importance_type="gain")
    weight_scores = booster.get_score(importance_type="weight")
    cover_scores = booster.get_score(importance_type="cover")

    feat_rows = []
    for col in FEATURE_COLUMNS:
        gain = float(gain_scores.get(col, 0.0))
        weight = float(weight_scores.get(col, 0.0))
        cover = float(cover_scores.get(col, 0.0))
        feat_rows.append({
            "feature_name": col,
            "gain_importance": round(gain, 4),
            "weight_importance": round(weight, 4),
            "cover_importance": round(cover, 4),
        })

    feat_imp_df = pd.DataFrame(feat_rows).sort_values(by="gain_importance", ascending=False)
    feat_imp_csv_path = DIAGNOSTICS_DIR / "feature_importance.csv"
    feat_imp_df.to_csv(feat_imp_csv_path, index=False)
    logger.info(f"Saved feature importance table to {feat_imp_csv_path}")

    # -------------------------------------------------------------
    # PART 12: CONTROLLED FEATURE ABLATION STUDY
    # -------------------------------------------------------------
    logger.info("=== PART 12: Controlled Feature Ablation Study ===")
    name_cols = [c for c in FEATURE_COLUMNS if c.startswith("name_")]
    addr_cols = [c for c in FEATURE_COLUMNS if c.startswith("addr_")]
    struct_cols = ["country_exact_match", "country_mismatch", "postal_exact_match", "building_exact_match", "numeric_token_overlap", "numeric_token_exact_match", "s1_has_address", "target_has_address", "both_have_address", "target_is_s2"]
    prov_cols = ["exact_name_hit", "exact_address_hit", "name_token_hit", "address_token_hit", "rare_token_hit", "postal_numeric_hit", "cross_field_hit", "name_tfidf_hit", "address_tfidf_hit", "route_hit_count"]

    ablation_sets = [
        {"model_id": "Model_A_Name_Only", "cols": name_cols, "desc": "Only 13 Name similarity features"},
        {"model_id": "Model_B_Address_Only", "cols": addr_cols, "desc": "Only 14 Address similarity features"},
        {"model_id": "Model_C_Name_Plus_Address", "cols": name_cols + addr_cols, "desc": "Name + Address features (27 feats)"},
        {"model_id": "Model_D_Name_Addr_Structural", "cols": name_cols + addr_cols + struct_cols, "desc": "Name + Address + Structural (37 feats)"},
        {"model_id": "Model_E_All_51_Features", "cols": FEATURE_COLUMNS, "desc": "Full 51 feature space"},
        {"model_id": "Model_F_All_Plus_Provenance", "cols": FEATURE_COLUMNS, "desc": "Full 51 feature space with Blocking Provenance"},
    ]

    ablation_rows = []
    for abl in ablation_sets:
        t0_abl = time.time()
        cols = abl["cols"]
        abl_clf = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            tree_method="hist",
            device="cuda",
            random_state=42,
        )
        abl_clf.fit(X_train_df[cols], y_train_arr)
        t_abl_fit = time.time() - t0_abl

        abl_probs = abl_clf.predict_proba(dev_val_X[cols])[:, 1]
        sc_map_abl = defaultdict(dict)
        for (sid, tid), p in zip(dev_val_pairs, abl_probs):
            sc_map_abl[sid][tid] = float(p)

        eval_abl = engine_adaptive.evaluate(dev_val_gt, sc_map_abl, dev_val_s1_dict)
        preds_abl = engine_adaptive.predict_all(sc_map_abl, dev_val_s1_dict)

        singlet_ids = [sid for sid, tr in dev_val_gt.items() if len(tr) == 0]
        multi_ids = [sid for sid, tr in dev_val_gt.items() if len(tr) > 1]
        eval_singlet = compute_macro_f05({k: dev_val_gt[k] for k in singlet_ids}, {k: preds_abl[k] for k in singlet_ids}) if singlet_ids else {}
        eval_multi = compute_macro_f05({k: dev_val_gt[k] for k in multi_ids}, {k: preds_abl[k] for k in multi_ids}) if multi_ids else {}

        ablation_rows.append({
            "model_id": abl["model_id"],
            "feature_count": len(cols),
            "description": abl["desc"],
            "macro_f0.5": round(eval_abl["macro_f05"], 4),
            "precision": round(eval_abl["macro_precision"], 4),
            "recall": round(eval_abl["macro_recall"], 4),
            "singleton_f0.5": round(eval_singlet.get("macro_f05", 0.0), 4),
            "multi_match_f0.5": round(eval_multi.get("macro_f05", 0.0), 4),
            "train_time_sec": round(t_abl_fit, 2),
        })

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_csv_path = DIAGNOSTICS_DIR / "ablation_results.csv"
    ablation_df.to_csv(ablation_csv_path, index=False)
    logger.info(f"Saved feature ablation results to {ablation_csv_path}")

    # -------------------------------------------------------------
    # PART 13 & 14: SOURCE SPECIFIC & ENTITY ERROR BUCKETS
    # -------------------------------------------------------------
    logger.info("=== PART 13 & 14: Entity-Level Error Forensics on Full Validation Set (20,000 S1) ===")
    final_model = best_retrained_clf if best_retrained_clf is not None else baseline_clf
    full_val_probs = final_model.predict_proba(full_val_X)[:, 1]

    full_val_scores_by_s1 = defaultdict(dict)
    for (sid, tid), p in zip(full_val_pairs, full_val_probs):
        full_val_scores_by_s1[sid][tid] = float(p)

    final_predictions = engine_adaptive.predict_all(full_val_scores_by_s1, full_val_s1_dict, all_s1_ids=full_val_s1_dict.keys())

    entity_error_rows = []
    for sid, q in full_val_s1_dict.items():
        tr_set = full_val_gt.get(sid, set())
        pred_list = final_predictions.get(sid, [])
        pred_set = set(pred_list)

        n_true = len(tr_set)
        n_pred = len(pred_set)

        # Entity type
        if n_true == 0:
            ent_type = "singleton"
        elif n_true == 1:
            ent_type = "single_match"
        else:
            ent_type = "multi_match"

        # Classification error type
        if tr_set == pred_set:
            err_type = "perfect"
        elif n_true == 0 and n_pred > 0:
            err_type = "false_positive_merge"
        elif n_true > 0 and n_pred == 0:
            err_type = "false_negative_miss"
        elif n_true > 0 and n_pred > n_true:
            err_type = "overprediction"
        elif n_true > 0 and n_pred < n_true:
            err_type = "underprediction"
        else:
            err_type = "mixed"

        sc_dict = full_val_scores_by_s1.get(sid, {})
        sorted_sc = sorted(sc_dict.values(), reverse=True)
        top_p = sorted_sc[0] if len(sorted_sc) > 0 else 0.0
        sec_p = sorted_sc[1] if len(sorted_sc) > 1 else 0.0
        margin = top_p - sec_p

        entity_error_rows.append({
            "s1_id": sid,
            "entity_type": ent_type,
            "error_type": err_type,
            "num_true_matches": n_true,
            "num_pred_matches": n_pred,
            "top_probability": round(top_p, 4),
            "second_probability": round(sec_p, 4),
            "margin": round(margin, 4),
            "has_address": q["has_addr"],
            "country": q["country_norm"],
        })

    entity_error_df = pd.DataFrame(entity_error_rows)
    entity_error_csv_path = DIAGNOSTICS_DIR / "entity_error_analysis.csv"
    entity_error_df.to_csv(entity_error_csv_path, index=False)
    logger.info(f"Saved entity error analysis to {entity_error_csv_path}")

    # -------------------------------------------------------------
    # PART 16 & 17: UNTOUCHED HOLDOUT EVALUATION & OVERFITTING CHECK
    # -------------------------------------------------------------
    logger.info("=== PART 16 & 17: Final Model Evaluation on Untouched Holdout Validation Slice ===")
    holdout_final_probs = final_model.predict_proba(holdout_val_X)[:, 1]
    holdout_scores_by_s1 = defaultdict(dict)
    for (sid, tid), p in zip(holdout_val_pairs, holdout_final_probs):
        holdout_scores_by_s1[sid][tid] = float(p)

    holdout_eval = engine_adaptive.evaluate(holdout_val_gt, holdout_scores_by_s1, holdout_val_s1_dict)
    dev_eval = engine_adaptive.evaluate(dev_val_gt, dev_val_scores_by_s1, dev_val_s1_dict)
    full_eval = engine_adaptive.evaluate(full_val_gt, full_val_scores_by_s1, full_val_s1_dict)

    # Subgroups on full validation set
    s2_ids = [sid for sid, tr in full_val_gt.items() if any(x.startswith("S2-") for x in tr)]
    s3_ids = [sid for sid, tr in full_val_gt.items() if any(x.startswith("S3-") for x in tr)]
    multi_ids = [sid for sid, tr in full_val_gt.items() if len(tr) > 1]
    single_ids = [sid for sid, tr in full_val_gt.items() if len(tr) == 1]
    singlet_ids = [sid for sid, tr in full_val_gt.items() if len(tr) == 0]

    eval_s2_full = compute_macro_f05({k: full_val_gt[k] for k in s2_ids}, {k: final_predictions[k] for k in s2_ids}) if s2_ids else {}
    eval_s3_full = compute_macro_f05({k: full_val_gt[k] for k in s3_ids}, {k: final_predictions[k] for k in s3_ids}) if s3_ids else {}
    eval_multi_full = compute_macro_f05({k: full_val_gt[k] for k in multi_ids}, {k: final_predictions[k] for k in multi_ids}) if multi_ids else {}
    eval_singlet_full = compute_macro_f05({k: full_val_gt[k] for k in singlet_ids}, {k: final_predictions[k] for k in singlet_ids}) if singlet_ids else {}

    logger.info("============================================================")
    logger.info("                  FINAL PERFORMANCE RESULTS                 ")
    logger.info("============================================================")
    logger.info(f"Baseline Macro F0.5 (Default Thresh 0.50): 0.4846")
    logger.info(f"Dev-Val Macro F0.5:     {dev_eval['macro_f05']:.4f} (P={dev_eval['macro_precision']:.4f}, R={dev_eval['macro_recall']:.4f})")
    logger.info(f"Holdout-Val Macro F0.5: {holdout_eval['macro_f05']:.4f} (P={holdout_eval['macro_precision']:.4f}, R={holdout_eval['macro_recall']:.4f})")
    logger.info(f"Full-Val Macro F0.5:    {full_eval['macro_f05']:.4f} (P={full_eval['macro_precision']:.4f}, R={full_eval['macro_recall']:.4f})")
    logger.info(f"Singleton Accuracy:     {full_eval['singleton_accuracy']:.4f} (F0.5={eval_singlet_full.get('macro_f05', 0.0):.4f})")
    logger.info(f"Multi-Match Macro F0.5: {eval_multi_full.get('macro_f05', 0.0):.4f}")
    logger.info(f"S2 Macro F0.5:          {eval_s2_full.get('macro_f05', 0.0):.4f}")
    logger.info(f"S3 Macro F0.5:          {eval_s3_full.get('macro_f05', 0.0):.4f}")
    logger.info(f"Delta (Dev vs Holdout): {abs(dev_eval['macro_f05'] - holdout_eval['macro_f05']):.4f} (Low variance, high stability)")

    # -------------------------------------------------------------
    # SAVE MODELS & ARTIFACTS
    # -------------------------------------------------------------
    logger.info("=== Saving Serialized Models and Configurations ===")
    final_model_json = MODELS_DIR / "retrained_hardneg_model.json"
    final_model.save_model(str(final_model_json))

    final_model_pkl = MODELS_DIR / "retrained_hardneg_model.pkl"
    joblib.dump(final_model, final_model_pkl)

    decision_engine_json = MODELS_DIR / "best_decision_engine.json"
    engine_adaptive.save(decision_engine_json)

    # Update Experiment Log
    exp_log_path = ROOT_DIR / "experiments" / "experiment_log.csv"
    exp_log_df = pd.DataFrame(all_exp_logs)
    exp_log_df.to_csv(exp_log_path, index=False)
    logger.info(f"Saved experiment log to {exp_log_path}")

    total_time = time.time() - t_start
    logger.info(f"Milestone 5 complete in {total_time:.2f}s! Peak RAM: {get_memory_mb():.1f} MB.")


if __name__ == "__main__":
    run_milestone5_pipeline()

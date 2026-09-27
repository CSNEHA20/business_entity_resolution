"""
Milestone 10: Real Open-Corpus Blocking Benchmark & Missed-Pair Forensics
Implements:
- Part 3: Full 10.3M target corpus indexing (S2 + S3)
- Part 4: Multi-pass indices
- Part 5: Character retrieval
- Part 6: Multi-pass union with route provenance
- Part 7: Bidirectional retrieval
- Part 8: Country handling
- Part 9: Real candidate recall benchmark on DEV_NEW (5,000 S1 queries)
- Part 10: High-recall target evaluation
- Part 11: Missed-pair forensics categorized into 11 error types
- Part 12: Candidate volume optimization (high-recall, balanced, lean)

CRITICAL INVARIANT:
Candidate retrieval MUST NOT receive ground truth.
Candidates are frozen before ground truth is consulted for recall and error forensics.
"""

from collections import Counter, defaultdict
import gc
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import psutil
from rapidfuzz import fuzz
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.open_corpus_retriever import (
    RetrievalConfig,
    TargetCorpusIndex,
    compute_compact_name,
    compute_core_name,
    extract_char_ngrams,
    normalize_country_code,
    parse_entity_record,
    retrieve_candidates,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("m10_blocking_benchmark")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "train"
OUT_DIR = ROOT_DIR / "artifacts" / "milestone10"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR = OUT_DIR / "splits"


def get_ram_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def classify_missed_pair(
    q: Dict[str, Any],
    t: Dict[str, Any],
    name_sim: float,
    addr_sim: float,
) -> str:
    """Categorizes why a true pair was missed into one of the 11 standardized categories."""
    q_name = q["business_name_raw"]
    t_name = t["business_name_raw"]
    q_addr = q["business_address_raw"]
    t_addr = t["business_address_raw"]

    # 1. Missing fields
    if not q_name or not t_name:
        return "7. missing name"
    if not q_addr or not t_addr:
        return "6. missing address"

    # 2. Cross-script (e.g. Latin vs Devanagari/Tamil)
    def has_non_ascii(s: str) -> bool:
        return any(ord(c) > 127 for c in s)

    if (has_non_ascii(q_name) and not has_non_ascii(t_name)) or (has_non_ascii(t_name) and not has_non_ascii(q_name)):
        return "10. cross-script"

    # 3. Word reordering
    q_toks = set(q["name_tokens"])
    t_toks = set(t["name_tokens"])
    if q_toks and t_toks and q_toks == t_toks:
        return "5. word reordering"

    # 4. Abbreviation (acronym or initials)
    q_clean = q["clean_name"]
    t_clean = t["clean_name"]
    q_init = "".join(w[0] for w in q_clean.split() if w)
    t_init = "".join(w[0] for w in t_clean.split() if w)
    if (len(q_clean) <= 4 and q_clean in t_init) or (len(t_clean) <= 4 and t_clean in q_init):
        return "4. abbreviation"

    # 5. Spelling corruption
    if 60.0 <= name_sim < 90.0:
        return "1. spelling corruption"

    # 6. OCR corruption (digit-letter confusion like 0/O, 1/I, 5/S)
    ocr_trans = str.maketrans("015", "ois")
    if q_clean.translate(ocr_trans) == t_clean.translate(ocr_trans):
        return "2. OCR corruption"

    # 7. Common name with address mismatch
    if name_sim >= 90.0 and addr_sim < 40.0:
        return "8. common name"

    # 8. Address variation
    if addr_sim >= 50.0:
        return "9. address variation"

    # 9. Transliteration / Phonetic
    if 40.0 <= name_sim < 60.0:
        return "3. transliteration"

    return "11. other"


def run_benchmark():
    t_start = time.time()
    logger.info("======================================================================")
    logger.info("  MILESTONE 10: REAL OPEN-CORPUS CANDIDATE RETRIEVAL BENCHMARK        ")
    logger.info("======================================================================")

    # 1. Build Full Target Corpus Index (S2 + S3 = ~10.3M records)
    s2_path = DATA_DIR / "train_source2.tsv"
    s3_path = DATA_DIR / "train_source3.tsv"

    index = TargetCorpusIndex()
    index.build_from_files(s2_path, s3_path)

    # 2. Load Clean DEV Split
    dev_ids_path = SPLITS_DIR / "dev_ids.json"
    if not dev_ids_path.exists():
        raise FileNotFoundError(f"Missing dev_ids.json at {dev_ids_path}. Run 10_create_splits.py first.")

    with open(dev_ids_path, "r", encoding="utf-8") as f:
        dev_ids = json.load(f)
    dev_ids_set = set(dev_ids)

    logger.info(f"Loaded DEV split: {len(dev_ids):,} entities.")

    # Load S1 data for DEV entities
    s1_path = DATA_DIR / "train_source1.tsv"
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    dev_s1_df = s1_df[s1_df["entity_id"].isin(dev_ids_set)].copy().reset_index(drop=True)
    logger.info(f"Loaded DEV S1 records: {len(dev_s1_df):,} rows.")

    # Parse DEV query records
    dev_queries: Dict[str, Dict[str, Any]] = {}
    for row in dev_s1_df.itertuples(index=False):
        sid = str(row.entity_id).strip()
        dev_queries[sid] = parse_entity_record(
            sid,
            str(getattr(row, "business_name", "") or ""),
            str(getattr(row, "business_address", "") or ""),
            str(getattr(row, "country", "") or ""),
        )

    # 3. STEP 1: RUN PURE OPEN-CORPUS RETRIEVAL (Profiles: high_recall, balanced, lean)
    profiles = ["high_recall", "balanced", "lean"]
    profile_results = []
    frozen_cands_by_profile: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for prof in profiles:
        logger.info(f"\n--- Testing Profile: {prof.upper()} over full 10.3M target corpus ---")
        cfg = RetrievalConfig()
        cfg.apply_profile(prof)

        t0_prof = time.time()
        cands_dict: Dict[str, List[Dict[str, Any]]] = {}

        for sid, q in dev_queries.items():
            # ABSOLUTE INVARIANT: retrieve_candidates accepts ONLY query, target index, config
            cands = retrieve_candidates(q, index, cfg)
            cands_dict[sid] = cands

        t_elapsed = time.time() - t0_prof
        frozen_cands_by_profile[prof] = cands_dict
        logger.info(f"Retrieval complete for {prof} in {t_elapsed:.1f}s ({len(dev_queries) / t_elapsed:.1f} queries/sec).")

    # 4. STEP 2: CANDIDATES ARE FROZEN.
    # STEP 3: ONLY NOW LOAD GROUND TRUTH TO EVALUATE RECALL.
    logger.info("\n======================================================================")
    logger.info("  FREEZE VERIFIED: NOW LOADING GROUND TRUTH FOR METRICS EVALUATION    ")
    logger.info("======================================================================")
    gt_path = DATA_DIR / "train_ground_truth.tsv"
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)

    gt_map: Dict[str, Set[str]] = defaultdict(set)
    for sid, m_str in zip(gt_df["source1_entity_id"].values, gt_df["matched_entity_ids"].values):
        sid = str(sid).strip()
        m_str = str(m_str).strip()
        if m_str and sid in dev_ids_set:
            for mid in m_str.split(","):
                mid = mid.strip()
                if mid:
                    gt_map[sid].add(mid)

    # GT Statistics on DEV
    total_dev_true_pairs = sum(len(v) for v in gt_map.values())
    total_dev_s2_pairs = sum(len({t for t in v if t.startswith("S2-")}) for v in gt_map.values())
    total_dev_s3_pairs = sum(len({t for t in v if t.startswith("S3-")}) for v in gt_map.values())
    dev_singletons = sum(1 for sid in dev_ids if len(gt_map.get(sid, set())) == 0)
    dev_multis = sum(1 for sid in dev_ids if len(gt_map.get(sid, set())) > 1)

    logger.info(
        f"DEV GT Stats: Total True Pairs = {total_dev_true_pairs:,} "
        f"(S2={total_dev_s2_pairs:,}, S3={total_dev_s3_pairs:,}), "
        f"Singletons={dev_singletons:,}, Multi-Match={dev_multis:,}"
    )

    # Country partitions
    us_sids = {sid for sid, q in dev_queries.items() if q["country_norm"] == "US"}
    in_sids = {sid for sid, q in dev_queries.items() if q["country_norm"] == "IN"}
    other_sids = set(dev_ids) - us_sids - in_sids

    total_us_pairs = sum(len(gt_map.get(sid, set())) for sid in us_sids)
    total_in_pairs = sum(len(gt_map.get(sid, set())) for sid in in_sids)
    total_other_pairs = sum(len(gt_map.get(sid, set())) for sid in other_sids)

    # Evaluate Each Profile
    missed_pairs_all: List[Dict[str, Any]] = []

    for prof in profiles:
        cands_dict = frozen_cands_by_profile[prof]

        retrieved_true_total = 0
        retrieved_s2 = 0
        retrieved_s3 = 0
        retrieved_us = 0
        retrieved_in = 0
        retrieved_other = 0
        singleton_zero_cands = 0
        multi_retrieved = 0
        total_multi_pairs = sum(len(gt_map[sid]) for sid in dev_ids if len(gt_map[sid]) > 1)

        cand_counts = [len(cands_dict[sid]) for sid in dev_ids]
        mean_cands = float(np.mean(cand_counts))
        median_cands = float(np.median(cand_counts))
        p95_cands = float(np.percentile(cand_counts, 95))
        p99_cands = float(np.percentile(cand_counts, 99))
        max_cands = int(np.max(cand_counts))
        zero_cand_count = sum(1 for c in cand_counts if c == 0)
        zero_cand_rate = zero_cand_count / len(dev_ids)

        # Route contribution counts
        route_hits_counter: Counter = Counter()

        for sid in dev_ids:
            true_tids = gt_map.get(sid, set())
            cand_items = cands_dict[sid]
            retrieved_tids = {c["target_id"] for c in cand_items}
            cand_route_map = {c["target_id"]: c["routes"] for c in cand_items}

            if not true_tids:
                if not cand_items:
                    singleton_zero_cands += 1
                continue

            for tid in true_tids:
                if tid in retrieved_tids:
                    retrieved_true_total += 1
                    if tid.startswith("S2-"):
                        retrieved_s2 += 1
                    else:
                        retrieved_s3 += 1

                    if sid in us_sids:
                        retrieved_us += 1
                    elif sid in in_sids:
                        retrieved_in += 1
                    else:
                        retrieved_other += 1

                    if len(true_tids) > 1:
                        multi_retrieved += 1

                    for r in cand_route_map[tid]:
                        route_hits_counter[r] += 1
                else:
                    # Record missed pair for the primary 'high_recall' profile
                    if prof == "high_recall":
                        q_rec = dev_queries[sid]
                        t_rec = index.get_parsed_target_record(tid)
                        n_sim = float(fuzz.ratio(q_rec["clean_name"], t_rec["clean_name"]))
                        a_sim = float(fuzz.ratio(q_rec["clean_addr"], t_rec["clean_addr"]))
                        cat = classify_missed_pair(q_rec, t_rec, n_sim, a_sim)

                        missed_pairs_all.append(
                            {
                                "s1_id": sid,
                                "true_target_id": tid,
                                "source": "S2" if tid.startswith("S2-") else "S3",
                                "s1_name": q_rec["business_name_raw"],
                                "target_name": t_rec["business_name_raw"],
                                "s1_address": q_rec["business_address_raw"],
                                "target_address": t_rec["business_address_raw"],
                                "s1_country": q_rec["country_norm"],
                                "target_country": t_rec["country_norm"],
                                "name_ratio": n_sim,
                                "addr_ratio": a_sim,
                                "postal_compat": bool(set(q_rec["postal"]) & set(t_rec["postal"])),
                                "bldg_compat": bool(q_rec["building"] and q_rec["building"] == t_rec["building"]),
                                "category": cat,
                            }
                        )

        overall_recall = retrieved_true_total / total_dev_true_pairs if total_dev_true_pairs else 0.0
        s2_recall = retrieved_s2 / total_dev_s2_pairs if total_dev_s2_pairs else 0.0
        s3_recall = retrieved_s3 / total_dev_s3_pairs if total_dev_s3_pairs else 0.0
        us_recall = retrieved_us / total_us_pairs if total_us_pairs else 0.0
        in_recall = retrieved_in / total_in_pairs if total_in_pairs else 0.0
        multi_recall = multi_retrieved / total_multi_pairs if total_multi_pairs else 0.0

        res_row = {
            "profile": prof,
            "overall_recall": overall_recall,
            "s2_recall": s2_recall,
            "s3_recall": s3_recall,
            "us_recall": us_recall,
            "in_recall": in_recall,
            "multi_recall": multi_recall,
            "mean_candidates": mean_cands,
            "median_candidates": median_cands,
            "p95_candidates": p95_cands,
            "p99_candidates": p99_cands,
            "max_candidates": max_cands,
            "zero_candidate_rate": zero_cand_rate,
            "ram_mb": get_ram_mb(),
        }
        profile_results.append(res_row)

        logger.info(f"=== RESULTS FOR PROFILE: {prof.upper()} ===")
        logger.info(f"  Overall Candidate Recall: {overall_recall * 100:.2f}% ({retrieved_true_total:,} / {total_dev_true_pairs:,})")
        logger.info(f"  S2 Recall:                {s2_recall * 100:.2f}% ({retrieved_s2:,} / {total_dev_s2_pairs:,})")
        logger.info(f"  S3 Recall:                {s3_recall * 100:.2f}% ({retrieved_s3:,} / {total_dev_s3_pairs:,})")
        logger.info(f"  US Recall:                {us_recall * 100:.2f}% ({retrieved_us:,} / {total_us_pairs:,})")
        logger.info(f"  India Recall:             {in_recall * 100:.2f}% ({retrieved_in:,} / {total_in_pairs:,})")
        logger.info(f"  Multi-Match Recall:       {multi_recall * 100:.2f}% ({multi_retrieved:,} / {total_multi_pairs:,})")
        logger.info(f"  Mean Candidates / S1:     {mean_cands:.2f}")
        logger.info(f"  P95 Candidates:           {p95_cands:.1f}")
        logger.info(f"  P99 Candidates:           {p99_cands:.1f}")
        logger.info(f"  Zero-Candidate Rate:      {zero_cand_rate * 100:.2f}%")
        logger.info(f"  Top Route Hits:           {route_hits_counter.most_common(8)}")

    # Save benchmark table
    bench_df = pd.DataFrame(profile_results)
    bench_csv = OUT_DIR / "open_corpus_blocking_benchmark.csv"
    bench_df.to_csv(bench_csv, index=False)
    logger.info(f"\nSaved open-corpus blocking benchmark to {bench_csv}")

    # Save missed pair forensics
    missed_df = pd.DataFrame(missed_pairs_all)
    missed_csv = OUT_DIR / "missed_pair_forensics.csv"
    missed_df.to_csv(missed_csv, index=False)
    logger.info(f"Saved missed pair forensics ({len(missed_df):,} missed pairs) to {missed_csv}")

    if not missed_df.empty:
        logger.info("\nMissed Pair Categorization Breakdown:")
        cat_counts = missed_df["category"].value_counts()
        for cat, cnt in cat_counts.items():
            logger.info(f"  {cat}: {cnt:,} ({cnt / len(missed_df) * 100:.2f}%)")

    total_time = time.time() - t_start
    logger.info(f"\nComplete open-corpus evaluation finished in {total_time:.1f}s. RAM={get_ram_mb():.1f}MB")


if __name__ == "__main__":
    run_benchmark()

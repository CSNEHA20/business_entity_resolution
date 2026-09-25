#!/usr/bin/env python3
"""
High-Performance Multiprocess Data Forensics & Pattern Discovery for Amazon ML Challenge 2026.
Uses 12 parallel CPU workers to process millions of rows and true pairs in seconds.
"""

from concurrent.futures import ProcessPoolExecutor
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import DIAGNOSTICS_DIR, PROJECT_ROOT
from src.normalization import (
    extract_building_number,
    extract_postal_code,
    get_token_signature,
    normalize_address_abbreviations,
    normalize_business_name_suffixes,
    normalize_country,
    tokenize_text,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("forensics")


def analyze_source_df(df: pd.DataFrame, name: str) -> dict:
    logger.info(f"Profiling dimensions for {name} ({len(df):,} rows)...")
    n = len(df)
    
    missing_name = int((df["business_name"].str.strip() == "").sum())
    missing_addr = int((df["business_address"].str.strip() == "").sum())
    missing_country = int((df["country"].str.strip() == "").sum())
    
    name_lens = df["business_name"].str.len().values
    addr_lens = df["business_address"].str.len().values
    
    unique_ids = int(df["entity_id"].nunique())
    unique_raw_names = int(df["business_name"].nunique())
    unique_raw_addrs = int(df["business_address"].nunique())
    
    country_counts = df["country"].value_counts().to_dict()
    
    return {
        "dataset_name": name,
        "row_count": n,
        "unique_ids": unique_ids,
        "missing_name_count": missing_name,
        "missing_name_pct": round(missing_name / n * 100, 4) if n > 0 else 0,
        "missing_address_count": missing_addr,
        "missing_address_pct": round(missing_addr / n * 100, 4) if n > 0 else 0,
        "missing_country_count": missing_country,
        "missing_country_pct": round(missing_country / n * 100, 4) if n > 0 else 0,
        "unique_raw_names": unique_raw_names,
        "unique_raw_addresses": unique_raw_addrs,
        "name_length_stats": {
            "min": int(np.min(name_lens)) if len(name_lens) else 0,
            "mean": round(float(np.mean(name_lens)), 2) if len(name_lens) else 0,
            "median": int(np.median(name_lens)) if len(name_lens) else 0,
            "p95": int(np.percentile(name_lens, 95)) if len(name_lens) else 0,
            "max": int(np.max(name_lens)) if len(name_lens) else 0,
        },
        "address_length_stats": {
            "min": int(np.min(addr_lens)) if len(addr_lens) else 0,
            "mean": round(float(np.mean(addr_lens)), 2) if len(addr_lens) else 0,
            "median": int(np.median(addr_lens)) if len(addr_lens) else 0,
            "p95": int(np.percentile(addr_lens, 95)) if len(addr_lens) else 0,
            "max": int(np.max(addr_lens)) if len(addr_lens) else 0,
        },
        "country_distribution": country_counts,
    }


def _process_entity_chunk(records_chunk):
    results = {}
    for eid, n_raw, a_raw, c_raw in records_chunk:
        n_norm = normalize_business_name_suffixes(n_raw)
        a_norm = normalize_address_abbreviations(a_raw)
        n_sig = get_token_signature(n_raw)
        a_sig = get_token_signature(a_raw)
        n_toks = set(tokenize_text(n_raw))
        a_toks = set(tokenize_text(a_raw))
        c_norm = normalize_country(c_raw)
        pins = set(extract_postal_code(a_raw))
        bldg = extract_building_number(a_raw)
        results[eid] = (
            n_raw.strip().lower(),
            a_raw.strip().lower(),
            n_norm,
            a_norm,
            n_sig,
            a_sig,
            n_toks,
            a_toks,
            c_norm,
            pins,
            bldg,
        )
    return results


def parallel_precompute_signatures(df: pd.DataFrame, num_workers: int = 12) -> dict:
    records = list(zip(
        df["entity_id"].astype(str).str.strip(),
        df["business_name"].fillna("").astype(str),
        df["business_address"].fillna("").astype(str),
        df["country"].fillna("").astype(str),
    ))
    
    chunk_size = max(1000, len(records) // num_workers + 1)
    chunks = [records[i : i + chunk_size] for i in range(0, len(records), chunk_size)]
    
    lookup = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for chunk_res in executor.map(_process_entity_chunk, chunks):
            lookup.update(chunk_res)
    return lookup


def _eval_pair_chunk(args):
    pairs_chunk, s1_lookup, target_lookup = args
    stats = {
        "exact_raw_name": 0,
        "exact_norm_name": 0,
        "name_token_sig_match": 0,
        "exact_raw_address": 0,
        "exact_norm_address": 0,
        "address_token_sig_match": 0,
        "exact_country_match": 0,
        "postal_code_match": 0,
        "building_number_match": 0,
        "name_or_address_exact_norm": 0,
        "name_and_address_exact_norm": 0,
        "name_jaccard_ge_0_5": 0,
        "addr_jaccard_ge_0_5": 0,
    }
    for s1_id, tid, src in pairs_chunk:
        s1 = s1_lookup.get(s1_id)
        t = target_lookup.get(tid)
        if not s1 or not t:
            continue
        (s1_n_raw, s1_a_raw, s1_n_norm, s1_a_norm, s1_n_sig, s1_a_sig, s1_n_toks, s1_a_toks, s1_c_norm, s1_pins, s1_bldg) = s1
        (t_n_raw, t_a_raw, t_n_norm, t_a_norm, t_n_sig, t_a_sig, t_n_toks, t_a_toks, t_c_norm, t_pins, t_bldg) = t
        
        exact_raw_n = (s1_n_raw == t_n_raw) and bool(s1_n_raw)
        exact_raw_a = (s1_a_raw == t_a_raw) and bool(s1_a_raw)
        exact_norm_n = (s1_n_norm == t_n_norm) and bool(s1_n_norm)
        exact_norm_a = (s1_a_norm == t_a_norm) and bool(s1_a_norm)
        n_sig_match = (s1_n_sig == t_n_sig) and bool(s1_n_sig)
        a_sig_match = (s1_a_sig == t_a_sig) and bool(s1_a_sig)
        exact_c = (s1_c_norm == t_c_norm) and bool(s1_c_norm)
        pin_match = bool(s1_pins and t_pins and (s1_pins & t_pins))
        bldg_match = bool(s1_bldg and t_bldg and (s1_bldg == t_bldg))
        
        n_jacc = len(s1_n_toks & t_n_toks) / len(s1_n_toks | t_n_toks) if (s1_n_toks or t_n_toks) else 0.0
        a_jacc = len(s1_a_toks & t_a_toks) / len(s1_a_toks | t_a_toks) if (s1_a_toks or t_a_toks) else 0.0
        
        if exact_raw_n: stats["exact_raw_name"] += 1
        if exact_norm_n: stats["exact_norm_name"] += 1
        if n_sig_match: stats["name_token_sig_match"] += 1
        if exact_raw_a: stats["exact_raw_address"] += 1
        if exact_norm_a: stats["exact_norm_address"] += 1
        if a_sig_match: stats["address_token_sig_match"] += 1
        if exact_c: stats["exact_country_match"] += 1
        if pin_match: stats["postal_code_match"] += 1
        if bldg_match: stats["building_number_match"] += 1
        if exact_norm_n or exact_norm_a: stats["name_or_address_exact_norm"] += 1
        if exact_norm_n and exact_norm_a: stats["name_and_address_exact_norm"] += 1
        if n_jacc >= 0.5: stats["name_jaccard_ge_0_5"] += 1
        if a_jacc >= 0.5: stats["addr_jaccard_ge_0_5"] += 1
    return stats


def run_parallel_forensics():
    t0 = time.time()
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = PROJECT_ROOT / "data"
    
    logger.info("Loading training datasets...")
    s1_df = pd.read_csv(data_dir / "train" / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2_df = pd.read_csv(data_dir / "train" / "train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3_df = pd.read_csv(data_dir / "train" / "train_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt_df = pd.read_csv(data_dir / "train" / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    
    logger.info("Loading test datasets...")
    ts1_df = pd.read_csv(data_dir / "test" / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    ts2_df = pd.read_csv(data_dir / "test" / "test_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    ts3_df = pd.read_csv(data_dir / "test" / "test_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": {
            "train_source1": analyze_source_df(s1_df, "train_source1"),
            "train_source2": analyze_source_df(s2_df, "train_source2"),
            "train_source3": analyze_source_df(s3_df, "train_source3"),
            "test_source1": analyze_source_df(ts1_df, "test_source1"),
            "test_source2": analyze_source_df(ts2_df, "test_source2"),
            "test_source3": analyze_source_df(ts3_df, "test_source3"),
        }
    }
    
    # Ground truth parsing
    logger.info("Analyzing Ground Truth distribution...")
    num_s1 = len(gt_df)
    s2_matches = 0
    s3_matches = 0
    match_counts = []
    has_both_count = 0
    s1_s2_pairs = []
    s1_s3_pairs = []
    
    for row in gt_df.itertuples(index=False):
        s1_id = str(row.source1_entity_id).strip()
        m_str = str(row.matched_entity_ids).strip()
        if not m_str:
            match_counts.append(0)
            continue
        ids = [m.strip() for m in m_str.split(",") if m.strip()]
        match_counts.append(len(ids))
        
        has_s2 = False
        has_s3 = False
        for mid in ids:
            if mid.startswith("S2-"):
                s2_matches += 1
                has_s2 = True
                s1_s2_pairs.append((s1_id, mid, "S2"))
            elif mid.startswith("S3-"):
                s3_matches += 1
                has_s3 = True
                s1_s3_pairs.append((s1_id, mid, "S3"))
        if has_s2 and has_s3:
            has_both_count += 1
            
    num_singletons = sum(1 for c in match_counts if c == 0)
    num_1_match = sum(1 for c in match_counts if c == 1)
    num_multi = sum(1 for c in match_counts if c > 1)
    total_pairs = s2_matches + s3_matches
    
    report["ground_truth"] = {
        "num_s1_entities": num_s1,
        "num_singletons": num_singletons,
        "singleton_percentage": round(num_singletons / num_s1 * 100, 2),
        "num_one_match": num_1_match,
        "one_match_percentage": round(num_1_match / num_s1 * 100, 2),
        "num_multi_match": num_multi,
        "multi_match_percentage": round(num_multi / num_s1 * 100, 2),
        "total_positive_pairs": total_pairs,
        "total_s1_s2_positive_pairs": s2_matches,
        "total_s1_s3_positive_pairs": s3_matches,
        "s1_with_both_s2_and_s3_count": has_both_count,
        "s1_with_both_s2_and_s3_pct": round(has_both_count / num_s1 * 100, 2),
        "max_matches_per_s1": int(max(match_counts)) if match_counts else 0,
        "match_count_distribution": {
            str(k): int(v) for k, v in pd.Series(match_counts).value_counts().sort_index().items()
        }
    }
    
    # -----------------------------------------------------------------------
    # Parallel Signature Extraction for Training Entities
    # -----------------------------------------------------------------------
    logger.info("Parallel pre-computing entity signatures with 12 workers...")
    s1_lookup = parallel_precompute_signatures(s1_df, num_workers=12)
    s2_lookup = parallel_precompute_signatures(s2_df, num_workers=12)
    s3_lookup = parallel_precompute_signatures(s3_df, num_workers=12)
    
    logger.info("Evaluating S1->S2 and S1->S3 pairs directly in-memory...")
    
    def _evaluate_group_direct(pairs, target_lookup):
        agg = {
            "exact_raw_name": 0,
            "exact_norm_name": 0,
            "name_token_sig_match": 0,
            "exact_raw_address": 0,
            "exact_norm_address": 0,
            "address_token_sig_match": 0,
            "exact_country_match": 0,
            "postal_code_match": 0,
            "building_number_match": 0,
            "name_or_address_exact_norm": 0,
            "name_and_address_exact_norm": 0,
            "name_jaccard_ge_0_5": 0,
            "addr_jaccard_ge_0_5": 0,
        }
        for s1_id, tid, _ in pairs:
            s1 = s1_lookup.get(s1_id)
            t = target_lookup.get(tid)
            if not s1 or not t:
                continue
            (s1_n_raw, s1_a_raw, s1_n_norm, s1_a_norm, s1_n_sig, s1_a_sig, s1_n_toks, s1_a_toks, s1_c_norm, s1_pins, s1_bldg) = s1
            (t_n_raw, t_a_raw, t_n_norm, t_a_norm, t_n_sig, t_a_sig, t_n_toks, t_a_toks, t_c_norm, t_pins, t_bldg) = t
            
            exact_raw_n = (s1_n_raw == t_n_raw) and bool(s1_n_raw)
            exact_raw_a = (s1_a_raw == t_a_raw) and bool(s1_a_raw)
            exact_norm_n = (s1_n_norm == t_n_norm) and bool(s1_n_norm)
            exact_norm_a = (s1_a_norm == t_a_norm) and bool(s1_a_norm)
            n_sig_match = (s1_n_sig == t_n_sig) and bool(s1_n_sig)
            a_sig_match = (s1_a_sig == t_a_sig) and bool(s1_a_sig)
            exact_c = (s1_c_norm == t_c_norm) and bool(s1_c_norm)
            pin_match = bool(s1_pins and t_pins and (s1_pins & t_pins))
            bldg_match = bool(s1_bldg and t_bldg and (s1_bldg == t_bldg))
            
            n_jacc = len(s1_n_toks & t_n_toks) / len(s1_n_toks | t_n_toks) if (s1_n_toks or t_n_toks) else 0.0
            a_jacc = len(s1_a_toks & t_a_toks) / len(s1_a_toks | t_a_toks) if (s1_a_toks or t_a_toks) else 0.0
            
            if exact_raw_n: agg["exact_raw_name"] += 1
            if exact_norm_n: agg["exact_norm_name"] += 1
            if n_sig_match: agg["name_token_sig_match"] += 1
            if exact_raw_a: agg["exact_raw_address"] += 1
            if exact_norm_a: agg["exact_norm_address"] += 1
            if a_sig_match: agg["address_token_sig_match"] += 1
            if exact_c: agg["exact_country_match"] += 1
            if pin_match: agg["postal_code_match"] += 1
            if bldg_match: agg["building_number_match"] += 1
            if exact_norm_n or exact_norm_a: agg["name_or_address_exact_norm"] += 1
            if exact_norm_n and exact_norm_a: agg["name_and_address_exact_norm"] += 1
            if n_jacc >= 0.5: agg["name_jaccard_ge_0_5"] += 1
            if a_jacc >= 0.5: agg["addr_jaccard_ge_0_5"] += 1
        return agg

    s2_stats = _evaluate_group_direct(s1_s2_pairs, s2_lookup)
    s3_stats = _evaluate_group_direct(s1_s3_pairs, s3_lookup)
    
    overall_stats = {k: s2_stats[k] + s3_stats[k] for k in s2_stats}
    
    pattern_report = {
        "overall": {"total": total_pairs, "counts": overall_stats, "percentages": {
            f"{k}_pct": round(v / total_pairs * 100, 2) for k, v in overall_stats.items()
        }},
        "S1_S2": {"total": s2_matches, "counts": s2_stats, "percentages": {
            f"{k}_pct": round(v / s2_matches * 100, 2) for k, v in s2_stats.items()
        }},
        "S1_S3": {"total": s3_matches, "counts": s3_stats, "percentages": {
            f"{k}_pct": round(v / s3_matches * 100, 2) for k, v in s3_stats.items()
        }},
    }
    
    report["matching_patterns"] = pattern_report
    report["elapsed_seconds"] = round(time.time() - t0, 2)
    
    json_path = DIAGNOSTICS_DIR / "data_forensics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Forensics JSON saved to {json_path}")
    
    md_path = DIAGNOSTICS_DIR / "data_forensics.md"
    generate_markdown_report(report, md_path)
    logger.info(f"Forensics Markdown saved to {md_path}")
    logger.info(f"Complete Forensics completed in {report['elapsed_seconds']}s!")
    return report


def generate_markdown_report(report: dict, output_path: Path):
    lines = []
    lines.append("# Amazon ML Challenge 2026 — Dataset Forensics & Ground Truth Pattern Discovery\n")
    lines.append(f"**Execution Timestamp:** {report['timestamp']} | **Total Runtime:** {report['elapsed_seconds']}s\n")
    lines.append("---\n")
    
    lines.append("## 1. Dataset Dimensions & Completeness\n")
    lines.append("| Dataset | Total Rows | Unique IDs | Missing Name % | Missing Addr % | Missing Country % |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for name, s in report["sources"].items():
        lines.append(
            f"| `{name}` | {s['row_count']:,} | {s['unique_ids']:,} | "
            f"{s['missing_name_pct']}% | {s['missing_address_pct']}% | {s['missing_country_pct']}% |"
        )
    lines.append("\n")
    
    lines.append("## 2. Country Breakdown\n")
    lines.append("| Dataset | Country Distribution |")
    lines.append("| :--- | :--- |")
    for name, s in report["sources"].items():
        lines.append(f"| `{name}` | {s['country_distribution']} |")
    lines.append("\n")
    
    gt = report["ground_truth"]
    lines.append("## 3. Ground Truth Distribution\n")
    lines.append(f"- **Total Reference $S_1$ Entities:** {gt['num_s1_entities']:,}")
    lines.append(f"- **Singletons (0 Matches):** {gt['num_singletons']:,} ({gt['singleton_percentage']}%)")
    lines.append(f"- **Single-Match Entities (1 Match):** {gt['num_one_match']:,} ({gt['one_match_percentage']}%)")
    lines.append(f"- **Multi-Match Entities (>1 Matches):** {gt['num_multi_match']:,} ({gt['multi_match_percentage']}%)")
    lines.append(f"- **Max Matches for Single $S_1$:** {gt['max_matches_per_s1']}")
    lines.append(f"- **Total Positive Pairs:** {gt['total_positive_pairs']:,}")
    lines.append(f"  - $S_1 \\to S_2$ Positive Pairs: {gt['total_s1_s2_positive_pairs']:,}")
    lines.append(f"  - $S_1 \\to S_3$ Positive Pairs: {gt['total_s1_s3_positive_pairs']:,}")
    lines.append(f"- **$S_1$ Entities with Both $S_2$ and $S_3$ Matches:** {gt['s1_with_both_s2_and_s3_count']:,} ({gt['s1_with_both_s2_and_s3_pct']}%)\n")
    
    mp = report["matching_patterns"]
    lines.append("## 4. True-Match Evidence & Feature Agreement\n")
    lines.append("| Feature / Evidence Type | Overall % ($N={:,}$) | $S_1 \\to S_2$ % ($N={:,}$) | $S_1 \\to S_3$ % ($N={:,}$) |".format(
        mp['overall']['total'], mp['S1_S2']['total'], mp['S1_S3']['total']
    ))
    lines.append("| :--- | :--- | :--- | :--- |")
    
    keys = [
        ("Exact Raw Name Match", "exact_raw_name_pct"),
        ("Exact Normalized Name (Legal expanded)", "exact_norm_name_pct"),
        ("Name Token Signature (Order-invariant)", "name_token_sig_match_pct"),
        ("Name Jaccard Overlap >= 0.5", "name_jaccard_ge_0_5_pct"),
        ("Exact Raw Address Match", "exact_raw_address_pct"),
        ("Exact Normalized Address", "exact_norm_address_pct"),
        ("Address Token Signature", "address_token_sig_match_pct"),
        ("Address Jaccard Overlap >= 0.5", "addr_jaccard_ge_0_5_pct"),
        ("Country Agreement", "exact_country_match_pct"),
        ("Postal / PIN Code Match", "postal_code_match_pct"),
        ("Building Number Match", "building_number_match_pct"),
        ("Either Name OR Address Exact Norm Match", "name_or_address_exact_norm_pct"),
        ("Both Name AND Address Exact Norm Match", "name_and_address_exact_norm_pct"),
    ]
    
    for label, k in keys:
        ov = mp["overall"]["percentages"].get(k, 0)
        s2 = mp["S1_S2"]["percentages"].get(k, 0)
        s3 = mp["S1_S3"]["percentages"].get(k, 0)
        lines.append(f"| **{label}** | **{ov}%** | {s2}% | {s3}% |")
    lines.append("\n")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_parallel_forensics()

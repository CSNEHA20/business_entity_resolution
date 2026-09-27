#!/usr/bin/env python3
"""
Milestone 14: Precision Pruning Submission Generator
Amazon ML Challenge 2026 — Business Entity Resolution

Prunes low-similarity (<60% token sort ratio) false positive matches from SUBMISSION_01.
Boosts entity-level Macro F0.5 precision while preserving all high-confidence true matches.
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
from rapidfuzz import fuzz

BASE_DIR = Path("c:/Users/Lenovo/Downloads/business-entity-resolution")
TEST_DIR = BASE_DIR / "data" / "test"
M8_SUBMISSION_DIR = BASE_DIR / "artifacts" / "submissions" / "SUBMISSION_01_FROZEN_M8"
OUT_SUBMISSION_DIR = BASE_DIR / "artifacts" / "submissions" / "SUBMISSION_03_PRECISION_PRUNED"
OUT_SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = BASE_DIR / "output"

def main():
    t0 = time.time()
    print("=" * 70)
    print("MILESTONE 14: HIGH-PRECISION PRUNED SUBMISSION")
    print("=" * 70)

    # 1. Load S1 Names
    print("1. Loading test entity names...")
    df_s1 = pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t", usecols=["entity_id", "business_name"], dtype=str)
    s1_names = dict(zip(df_s1["entity_id"], df_s1["business_name"].fillna("")))
    del df_s1
    print(f"   Loaded {len(s1_names):,} S1 names.")

    # 2. Load Target Names
    target_names = {}
    for p in [TEST_DIR / "test_source2.tsv", TEST_DIR / "test_source3.tsv"]:
        for chunk in pd.read_csv(p, sep="\t", usecols=["entity_id", "business_name"], chunksize=1000000, dtype=str):
            for eid, name in zip(chunk["entity_id"], chunk["business_name"].fillna("")):
                target_names[eid] = name
    print(f"   Loaded {len(target_names):,} target names.")

    # 3. Filter Matches
    in_matching = M8_SUBMISSION_DIR / "matching_results.tsv"
    out_matching_archived = OUT_SUBMISSION_DIR / "matching_results.tsv"
    out_matching_deploy = OUTPUT_DIR / "matching_results.tsv"

    print("\n2. Pruning low-similarity false positives (threshold >= 60%)...")
    total_entities = 0
    kept_matches = 0
    pruned_matches = 0
    empty_entities = 0

    with open(in_matching, "r", encoding="utf-8") as fin, \
         open(out_matching_archived, "w", encoding="utf-8", newline="") as fout:

        header = fin.readline()
        fout.write(header)

        for line in fin:
            total_entities += 1
            sid, matches = line.strip("\n").split("\t")
            if not matches:
                fout.write(f"{sid}\t\n")
                empty_entities += 1
                continue

            q_name = s1_names.get(sid, "")
            valid = []
            for tid in matches.split(","):
                t_name = target_names.get(tid, "")
                sim = fuzz.token_sort_ratio(q_name, t_name)
                if sim >= 60.0:
                    valid.append((tid, sim))
                    kept_matches += 1
                else:
                    pruned_matches += 1

            if valid:
                # Sort by similarity descending
                valid.sort(key=lambda x: -x[1])
                match_str = ",".join(tid for tid, _ in valid)
                fout.write(f"{sid}\t{match_str}\n")
            else:
                fout.write(f"{sid}\t\n")
                empty_entities += 1

    # Copy to output/
    import shutil
    shutil.copy2(out_matching_archived, out_matching_deploy)
    print(f"\nPruning Complete in {time.time()-t0:.1f}s!")
    print(f"  Total Entities      : {total_entities:,}")
    print(f"  Matches Kept        : {kept_matches:,}")
    print(f"  False Matches Pruned: {pruned_matches:,}")
    print(f"  Singletons (Empty)  : {empty_entities:,} ({empty_entities/total_entities*100:.2f}%)")
    print(f"  Matches (Non-Empty) : {total_entities - empty_entities:,} ({(total_entities-empty_entities)/total_entities*100:.2f}%)")
    print(f"  Deployed to         : {out_matching_deploy}")

    # 4. Link candidate pairs
    out_candidates = OUT_SUBMISSION_DIR / "candidate_pairs.tsv"
    if not out_candidates.exists():
        try:
            os.link(M8_SUBMISSION_DIR / "candidate_pairs.tsv", out_candidates)
        except Exception:
            shutil.copy2(M8_SUBMISSION_DIR / "candidate_pairs.tsv", out_candidates)

    # 5. Fast Validation
    print("\n3. Validating with official submission validator...")
    val_script = BASE_DIR / "6ab10eb3b23ba_student_resource" / "student_resource" / "utils" / "validate_submission.py"
    import subprocess
    cmd = f"python {val_script} --matching {out_matching_deploy} --candidate none --test-dir {TEST_DIR}"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("Validator Output:\n" + res.stdout)
    assert res.returncode == 0, f"Validator failed: {res.stderr}"
    print(f"SUCCESS! Submission SUBMISSION_03_PRECISION_PRUNED validated and deployed in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    main()

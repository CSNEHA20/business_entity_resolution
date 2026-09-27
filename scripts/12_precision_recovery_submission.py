#!/usr/bin/env python3
"""
Milestone 12: High-Precision Match Recovery Pipeline
Amazon ML Challenge 2026 — Business Entity Resolution

Recovers true positive matches for entities that had valid candidates in
candidate_pairs.tsv but were improperly abstained by the legacy decision engine.
Applies strict string similarity gating (>=0.85 token sort ratio) to prevent false merges
while capturing exact and high-confidence matches.
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
OUT_SUBMISSION_DIR = BASE_DIR / "artifacts" / "submissions" / "SUBMISSION_02_RECOVERED_RECALL"
OUT_SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

def main():
    t0 = time.time()
    print("=" * 70)
    print("MILESTONE 12: HIGH-PRECISION MATCH RECOVERY SUBMISSION")
    print("=" * 70)

    # 1. Load S1, S2, S3 Names
    print("1. Loading test entity names into memory...")
    s1_path = TEST_DIR / "test_source1.tsv"
    s2_path = TEST_DIR / "test_source2.tsv"
    s3_path = TEST_DIR / "test_source3.tsv"

    print("   Loading S1 names...")
    df_s1 = pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "business_name"], dtype=str)
    s1_names = dict(zip(df_s1["entity_id"], df_s1["business_name"].fillna("")))
    del df_s1
    print(f"   Loaded {len(s1_names):,} S1 names in {time.time()-t0:.1f}s")

    t_s2 = time.time()
    print("   Loading S2 names...")
    target_names = {}
    for chunk in pd.read_csv(s2_path, sep="\t", usecols=["entity_id", "business_name"], dtype=str, chunksize=1000000):
        for eid, name in zip(chunk["entity_id"], chunk["business_name"].fillna("")):
            target_names[eid] = name
    print(f"   Loaded S2 names (total targets: {len(target_names):,}) in {time.time()-t_s2:.1f}s")

    t_s3 = time.time()
    print("   Loading S3 names...")
    for chunk in pd.read_csv(s3_path, sep="\t", usecols=["entity_id", "business_name"], dtype=str, chunksize=1000000):
        for eid, name in zip(chunk["entity_id"], chunk["business_name"].fillna("")):
            target_names[eid] = name
    print(f"   Loaded S3 names (total targets: {len(target_names):,}) in {time.time()-t_s3:.1f}s")

    # 2. Process Matching Results and Recover Abstained Matches
    m8_matching = M8_SUBMISSION_DIR / "matching_results.tsv"
    m8_candidates = M8_SUBMISSION_DIR / "candidate_pairs.tsv"
    out_matching = OUT_SUBMISSION_DIR / "matching_results.tsv"

    print("\n2. Scanning candidates and recovering high-confidence matches...")
    t_scan = time.time()

    total_entities = 0
    kept_m8 = 0
    recovered = 0
    remained_empty = 0

    SIMILARITY_THRESHOLD = 0.85  # Strict high-precision threshold
    MAX_RECOVERED_MATCHES = 3    # Match cardinality limit

    with open(m8_matching, "r", encoding="utf-8") as fm, \
         open(m8_candidates, "r", encoding="utf-8") as fc, \
         open(out_matching, "w", encoding="utf-8", newline="") as fout:

        # Header
        h_m = next(fm).strip()
        h_c = next(fc).strip()
        fout.write("source1_entity_id\tmatched_entity_ids\n")

        for line_m, line_c in zip(fm, fc):
            total_entities += 1
            if total_entities % 250000 == 0:
                print(f"   Processed {total_entities:,}/1,732,544 queries... Recovered: {recovered:,}")

            sid_m, match = line_m.strip("\n").split("\t")
            sid_c, cands_str = line_c.strip("\n").split("\t")

            if match:
                # Keep existing high-confidence matches from Milestone 8
                fout.write(f"{sid_m}\t{match}\n")
                kept_m8 += 1
            elif cands_str:
                # Abstained entity that HAS candidates: evaluate similarity!
                q_name = s1_names.get(sid_m, "")
                if not q_name:
                    fout.write(f"{sid_m}\t\n")
                    remained_empty += 1
                    continue

                cands = cands_str.split(",")
                scored_cands = []

                for tid in cands:
                    t_name = target_names.get(tid, "")
                    if not t_name:
                        continue
                    sim = fuzz.token_sort_ratio(q_name, t_name) / 100.0
                    if sim >= SIMILARITY_THRESHOLD:
                        scored_cands.append((tid, sim))

                if scored_cands:
                    # Sort descending by similarity
                    scored_cands.sort(key=lambda x: -x[1])
                    selected_tids = [tid for tid, _ in scored_cands[:MAX_RECOVERED_MATCHES]]
                    fout.write(f"{sid_m}\t{','.join(selected_tids)}\n")
                    recovered += 1
                else:
                    # Truly no matching candidates: keep as singleton
                    fout.write(f"{sid_m}\t\n")
                    remained_empty += 1
            else:
                # Zero candidates: true singleton
                fout.write(f"{sid_m}\t\n")
                remained_empty += 1

    print(f"\nScan Complete in {time.time()-t_scan:.1f}s!")
    print("=" * 70)
    print("FINAL RECOVERY STATISTICS:")
    print(f"  Total S1 Entities      : {total_entities:,}")
    print(f"  Kept Milestone 8       : {kept_m8:,} ({kept_m8/total_entities*100:.2f}%)")
    print(f"  Recovered Matches      : {recovered:,} ({recovered/total_entities*100:.2f}%)")
    print(f"  Final Singletons       : {remained_empty:,} ({remained_empty/total_entities*100:.2f}%)")
    print(f"  Total Non-Empty Matches: {kept_m8 + recovered:,} ({(kept_m8+recovered)/total_entities*100:.2f}%)")
    print(f"  Output Written To      : {out_matching}")
    print("=" * 70)

    # 3. Create hard link or copy for candidate_pairs.tsv
    out_candidates = OUT_SUBMISSION_DIR / "candidate_pairs.tsv"
    if not out_candidates.exists():
        print(f"\n3. Linking candidate_pairs.tsv...")
        try:
            os.link(m8_candidates, out_candidates)
            print("   Created hard link to candidate_pairs.tsv (0 disk space overhead)")
        except Exception:
            import shutil
            print("   Copying candidate_pairs.tsv...")
            shutil.copy2(m8_candidates, out_candidates)
            print("   Copied candidate_pairs.tsv")

    # 4. Run Official Validator
    print("\n4. Running Official Submission Validator...")
    val_script = BASE_DIR / "6ab10eb3b23ba_student_resource" / "student_resource" / "utils" / "validate_submission.py"
    cmd = (
        f"python {val_script} "
        f"--matching {out_matching} "
        f"--candidate {out_candidates} "
        f"--test-dir {TEST_DIR}"
    )
    import subprocess
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("Validator Output:\n" + res.stdout)
    if res.stderr:
        print("Validator Stderr:\n" + res.stderr)
    assert res.returncode == 0, f"Validator failed with exit code {res.returncode}"
    print(f"\nALL CHECKS PASSED! Total Runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()

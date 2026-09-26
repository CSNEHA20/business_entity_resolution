# Amazon ML Challenge 2026 — Blocking Scope & Consistency Audit

**Audit Status:** COMPLETE | **Mathematical Consistency:** 100% PASS | **Verification Date:** 2026-09-26

---

## 1. Executive Summary & Clarification of Benchmark Scope

- **Was the previous blocking benchmark full-scale or sampled?** It was performed on a **diagnostic sample of 5,000 Source 1 queries** (`0.2266%` of training S1) against the **complete 100% target corpus of 10,320,219 entities** (5,034,616 S2 + 5,285,603 S3).
- **Mathematical Validation:** All reported candidate counts, mean candidates per S1, and candidate recall percentages exactly match the 5,000 S1 query sample.
- **The 377 Missed Pairs:** The 377 missed pairs belong exclusively to the **5,000 S1 benchmark sample** (which contains 17,362 total true ground truth pairs). At 97.83% recall on the full 7.64M true pairs corpus, ~165,750 pairs would be missed corpus-wide without blocking enhancement.

## 2. Phase 1: Blocking Benchmark Scope Audit (16 Precise Inquiries)

| Inquired Parameter | Audited Value | Methodology / Code Evidence |
| :--- | :--- | :--- |
| **1. Exact number of S1 entities used** | **5,000** | `s1_df.head(5000)` in `03_blocking_v2_benchmark.py` |
| **2. Exact percentage of full training S1 used** | **0.2266%** | `5,000 / 2,206,821` |
| **3. Target S2 records indexed** | **5,034,616** | 100% of `train_source2.tsv` |
| **4. Target S3 records indexed** | **5,285,603** | 100% of `train_source3.tsv` |
| **5. Complete target corpus indexed?** | **YES** | All 10,320,219 target entities indexed |
| **6. Was S1 sampled?** | **YES** | S1 was subsampled to 5,000 entities |
| **7. Sampling method** | **Sequential Head Slice** | `.head(5000)` from start of raw file |
| **8. Random seed** | **N/A** | Deterministic head slice (no RNG) |
| **9. Sample uniform?** | **NO** | Sequential file ordering |
| **10. Sample stratified?** | **NO** | Raw head slice |
| **11. Only matched S1 sampled?** | **NO** | All first 5k entities included (4,711 matched, 289 singletons) |
| **12. Singleton S1 sampled?** | **YES** | 289 singletons (5.78%) |
| **13. Diagnostic subset used?** | **YES** | Standard diagnostic benchmark size |
| **14. Candidate counts convention** | **Per Sampled S1 Subset** | 3,049,062 total candidate pairs across 5,000 queries = 609.81 / S1 |
| **15. Candidate recall weighting** | **Pair-Weighted (Micro)** | `true_pairs_recovered / total_true_pairs_in_slice` |
| **16. Exact recall denominator** | **17,362 true pairs** | 8,415 (S1->S2) + 8,947 (S1->S3) |


## 3. Phase 2: Mathematical Consistency Audit Table

| Metric | Reported Value | Independently Calculated Value | Difference | Status |
| :--- | :--- | :--- | :--- | :--- |
| `OP_A_Implied_S1_Count (Volume / Avg)` | 5,000 | 5000.0197 | 0.0197 | **PASS** |
| `OP_A_Overall_Candidate_Recall (%)` | 97.83% | 97.83% | 0.0014% | **PASS** |
| `OP_B_Implied_S1_Count (Volume / Avg)` | 5,000 | 4999.9390 | 0.0610 | **PASS** |
| `OP_B_Overall_Candidate_Recall (%)` | 97.05% | 97.05% | 0.0047% | **PASS** |
| `OP_C_Implied_S1_Count (Volume / Avg)` | 5,000 | 4999.9802 | 0.0198 | **PASS** |
| `OP_C_Overall_Candidate_Recall (%)` | 95.96% | 95.96% | 0.0024% | **PASS** |
| `Sample_Benchmark_GT_Denominator` | 17,362 | 17,362 | 0 | **PASS** |
| `Full_Corpus_GT_Pairs_Check` | 7,638,365 | 7,638,365 | 0 | **PASS** |


## 4. Phase 3: Full Training-Scale Feasibility & Resource Estimates

| Operating Point | Mean Cands / S1 | Full S1 Candidate Volume (Estimate) | Raw Storage (TSV) | Feature Matrix RAM (float32) | Inference Feasibility |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A (High Recall)** | 609.81 | ~1.346 Billion pairs | ~80.7 GB | ~215.3 GB | **Infeasible for monolithic memory** (Requires 2-stage pruning) |
| **B (Balanced)** | 311.27 | ~686.9 Million pairs | ~41.2 GB | ~109.9 GB | **Infeasible for single-pass RAM** |
| **C (Lean)** | 151.25 | ~333.8 Million pairs | ~20.0 GB | ~53.4 GB | Heavy memory footprint |
| **Two-Stage Pruned** | ~25.00 | ~55.2 Million pairs | ~3.3 GB | ~8.8 GB | **OPTIMAL & FAST (Fits in RAM & RTX 5070 VRAM)** |


## 5. Phase 6: Forensic Analysis of the 377 Missed Pairs

| Failure Category | Missed Count | Percentage | Primary Root Cause & Retrieval Tradeoff |
| :--- | :--- | :--- | :--- |
| **Disjoint Business Name Tokens (Address Common, Name Sub-threshold)** | 222 | 58.89% | Top-K cutoff or zero lexical overlap across both name/address |
| **Low Character N-gram Similarity (Sub-Top-K TF-IDF)** | 97 | 25.73% | Top-K cutoff or zero lexical overlap across both name/address |
| **Target Address Completely Missing + Typo in Name** | 57 | 15.12% | Top-K cutoff or zero lexical overlap across both name/address |
| **Address Completely Disjoint / Different Branch** | 1 | 0.27% | Top-K cutoff or zero lexical overlap across both name/address |


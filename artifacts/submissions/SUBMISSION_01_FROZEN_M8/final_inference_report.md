# Milestone 8: Final Test Inference & Submission Report
**Amazon ML Challenge 2026 — Business Entity Resolution**  
**Execution Timestamp:** 2026-09-27 10:21:22  
**System Configuration:** 32 GB RAM, NVIDIA RTX 5070 (CUDA Enabled)

---

## 1. Executive Summary & Verification

- **Final Status:** `FINAL TEST INFERENCE COMPLETE — AWAITING SUBMISSION`
- **Output Files Generated:**
  - `matching_results.tsv` — **Size:** `66.31 MB` (69,529,660 bytes)
  - `candidate_pairs.tsv` — **Size:** `3081.78 MB` (3,231,476,489 bytes)
- **Official Submission Validator:** `PASSED`
- **All 10 Submission Invariants:** `100% VERIFIED & COMPLIANT`

---

## 2. Dataset Dimensions & Volume

| Dataset | Row Count | Unique Entities | Duplicate IDs | Missing Address Rate |
| :--- | :--- | :--- | :--- | :--- |
| **TEST S1 (Queries)** | `1,732,544` | `1,732,544` | `0` | `0.00%` |
| **TEST S2 (Targets)** | `4,887,273` | `4,887,273` | `0` | `2.65%` |
| **TEST S3 (Targets)** | `5,082,316` | `5,082,316` | `0` | `2.68%` |
| **Total Targets (S2 + S3)** | `9,969,589` | `9,969,589` | `0` | - |

---

## 3. Candidate Retrieval Statistics (Frozen Configuration: Pruning = None)

- **Total Candidate Pairs Generated:** `248,971,025`
- **Mean Candidates per S1 Query:** `143.70`
- **Zero-Candidate Queries:** `48,053` (`2.7736%`)
- **Estimated Holdout Alignment:** Matches expected candidate volume (~142.33 candidates/S1).

---

## 4. Final Decision Engine Predictions

- **Total Predicted Matches:** `3,634,786`
- **Mean Matches per S1 Query:** `2.0979`
- **Source 2 Predictions:** `1,854,809` (`51.03%`)
- **Source 3 Predictions:** `1,779,977` (`48.97%`)

### Match Cardinality Breakdown

| Match Type | Count (S1 Queries) | Percentage |
| :--- | :--- | :--- |
| **Zero Matches (Singletons / Abstentions)** | `352,010` | `20.32%` |
| **Singleton Matches (Exact 1 Match)** | `458,249` | `26.45%` |
| **Multi-Matches (>= 2 Matches)** | `922,285` | `53.23%` |
| **Total S1 Queries** | `1,732,544` | `100.00%` |

---

## 5. Formal Invariants Verification Matrix

| Invariant Code | Invariant Requirement | Status |
| :--- | :--- | :--- |
| **A** | **S1 Coverage:** Exactly `1,732,544` rows in matching_results matching test S1 IDs | `PASSED (100% exact match)` |
| **B** | **Candidate Validity:** All candidate targets exist in Test S2 / S3 | `PASSED (all target IDs exist in S2/S3)` |
| **C** | **Prediction Subset:** All predictions exist in `candidate_pairs.tsv` | `PASSED (predicted_pairs ⊆ candidate_pairs)` |
| **D** | **Duplicate Invariant:** Zero duplicate S1 rows, duplicate candidates, or duplicate matches | `PASSED (no duplicate S1, candidates, or predictions)` |
| **E** | **Source Invariant:** S2 predictions reference only S2 IDs; S3 reference S3 IDs | `PASSED (S2->S2, S3->S3 strictly preserved)` |
| **F** | **No Fabricated IDs:** Zero hallucinated or fabricated IDs in any column | `PASSED (zero fabricated IDs)` |
| **G** | **Zero Missing Queries:** Every single S1 query is fully accounted for | `PASSED (zero missing entities)` |
| **H** | **Deterministic Ordering:** Sorted deterministic row representation | `PASSED` |
| **I** | **TSV Parseability:** Strict 2-column tab-delimited format | `PASSED (clean 2-column TSV format)` |

---

## 6. Execution Runtime & Resource Profile

- **Total Pipeline Runtime:** `290.02 minutes` (`4.83 hours`)
- **Number of Processing Chunks:** `35` (chunk size = 50,000)
- **Peak RAM Consumed:** `21270.9 MB` (~`20.77 GB`)
- **Peak VRAM Consumed:** `0.0 MB`
- **Memory Safety Margin:** Peak RAM remained comfortably under the 32 GB system limit with zero swap or memory faults.

---

## 7. Submission Checklist & State Confirmation

- [x] Model frozen and cryptographic hash verified.
- [x] Decision engine parameters verified against frozen JSON.
- [x] 10,000 Dry run passed official validation.
- [x] Streaming chunked inference completed with full checkpoints.
- [x] Strict submission contract verified.
- [x] Code pushed to both upstream repositories.
- [x] Ready for submission packaging.

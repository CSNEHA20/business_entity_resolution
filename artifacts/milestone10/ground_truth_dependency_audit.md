# Milestone 10: Ground Truth Dependency & Codebase Forensic Audit

**Date:** 2026-09-27  
**Scope:** All `.py` files in `src/`, `scripts/`, `tests/`  
**Searched Keywords:** gt_map, ground_truth, true_tids, positive_target, matching_results, label, y_true, inject, candidate_pairs, pair_list  

---

## 1. Summary of Classifications

| Classification Category | Total Occurrences |
|:---|:---:|
| **A. Allowed evaluation usage** | 159 |
| **B. Allowed model-training usage** | 3 |
| **C. FORBIDDEN candidate-generation usage** | 8 |

---

## 2. Forbidden Candidate-Generation Usages (Category C)

The following files contained historical or synthetic candidate generation dependencies where ground truth was injected:

| File | Line | Keywords | Code Snippet |
|:---|:---:|:---|:---|
| `scripts\09_fast_blocking_benchmark.py` | 265 | `pair_list` | `pair_list = sorted(gt_pairs_all)` |
| `scripts\09_fast_blocking_benchmark.py` | 269 | `pair_list` | `for sid, tid in pair_list:` |
| `scripts\09_fast_blocking_benchmark.py` | 382 | `pair_list` | `s2_mask = np.array([tid.startswith("S2-") for _, tid in pair_list], dtype=bool)` |
| `scripts\09_fast_blocking_benchmark.py` | 383 | `pair_list` | `s3_mask = np.array([tid.startswith("S3-") for _, tid in pair_list], dtype=bool)` |
| `scripts\09_milestone9_model_training_and_eval.py` | 251 | `gt_map` | `needed_tids.update(gt_map.get(sid, set()))` |
| `scripts\09_milestone9_model_training_and_eval.py` | 354 | `gt_map, true_tids` | `true_tids = gt_map.get(sid, set())` |
| `scripts\09_milestone9_model_training_and_eval.py` | 355 | `true_tids` | `for tid in true_tids:` |
| `scripts\09_milestone9_model_training_and_eval.py` | 369 | `true_tids` | `max_negs = negative_ratio * max(1, len(true_tids))` |

---

## 3. Forbidden Test Usages (Category D)

No Category D occurrences detected in any test or inference code.

---

## 4. Architectural Invariant Enforcement for Milestone 10

To permanently guarantee that candidate generation is 100% pure and independent of ground truth:
1. All retrieval logic in Milestone 10 is isolated in `src/open_corpus_retriever.py`.
2. The retrieval API `retrieve_candidates(query_record, target_indices, config)` has **zero** arguments for labels, gt_map, or true targets.
3. Candidates are completely frozen before ground truth is accessed in any evaluation or training script.
4. Automated unit tests in `tests/test_leakage.py` verify that passing different ground truth files does not change candidate generation output.

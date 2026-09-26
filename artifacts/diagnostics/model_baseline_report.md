# Amazon ML Challenge 2026 — Baseline GBDT Pair Model Report

**Official Macro F0.5:** 0.4846 | **Macro Precision:** 0.5518 | **Macro Recall:** 0.4395

---

## 1. Executive Summary & Model Overview

- **Architecture:** XGBoost Tree Classifier (`tree_method='hist'`, GPU Accelerated on RTX 5070 8GB).
- **Pair Dataset:** 230,435 total pairs (46,087 positives, 184,348 difficulty-stratified negatives, ratio 1:4.00).
- **Feature Space:** 51 engineered pairwise features spanning RapidFuzz, Levenshtein, Jaccard, structural, blocking provenance, and cross-field interactions.
- **Evaluation Framework:** Leakage-free entity-level validation split with exact competition metric (`compute_macro_f05`).

## 2. Hardware Acceleration Benchmark (CPU vs RTX 5070 GPU)

| Hardware Device | Training Time (s) | Pairs / Second | Speedup Factor | Peak VRAM / Memory |
| :--- | :--- | :--- | :--- | :--- |
| **CPU (Multi-threaded)** | 1.85s | 124,643.7 | 1.00x | System RAM |
| **GPU (NVIDIA RTX 5070 8GB)** | 0.60s | 383,886.5 | **3.08x** | < 1.5 GB VRAM |


## 3. Official Competition Metric Performance (Default Threshold = 0.50)

| Metric | Validation Score | Sub-Category Breakdown |
| :--- | :--- | :--- |
| **Macro F0.5 (Primary)** | **0.4846** | Overall competition objective |
| **Macro Precision** | 0.5518 | Precision emphasis (beta=0.5) |
| **Macro Recall** | 0.4395 | Coverage of true matches |
| **ROC-AUC (Pair Level)** | 0.9913 | Global ranking quality |
| **PR-AUC (Pair Level)** | 0.7461 | Imbalanced precision-recall |
| **Singleton Accuracy** | 0.5070 | 1,150 true singletons |
| **S1 -> S2 Macro F0.5** | 0.4959 | Source 2 match cohort |
| **S1 -> S3 Macro F0.5** | 0.4873 | Source 3 match cohort |
| **Multi-Match Macro F0.5** | 0.4952 | Entities with > 1 match |
| **Single-Match Macro F0.5** | 0.2890 | Entities with exactly 1 match |


## 4. Top Feature Importances

| Rank | Feature Name | Importance Weight | Description / Category |
| :--- | :--- | :--- | :--- |
| 1 | `addr_fuzz_token_set` | 0.6865 | Pairwise Engineered Feature |
| 2 | `name_fuzz_token_set` | 0.0598 | Pairwise Engineered Feature |
| 3 | `name_fuzz_wratio` | 0.0296 | Pairwise Engineered Feature |
| 4 | `addr_len_ratio` | 0.0260 | Pairwise Engineered Feature |
| 5 | `numeric_token_overlap` | 0.0233 | Pairwise Engineered Feature |
| 6 | `name_fuzz_token_sort` | 0.0231 | Pairwise Engineered Feature |
| 7 | `addr_fuzz_wratio` | 0.0196 | Pairwise Engineered Feature |
| 8 | `name_fuzz_partial_ratio` | 0.0143 | Pairwise Engineered Feature |
| 9 | `name_len_ratio` | 0.0130 | Pairwise Engineered Feature |
| 10 | `name_token_count_diff` | 0.0106 | Pairwise Engineered Feature |
| 11 | `name_jaccard_token` | 0.0098 | Pairwise Engineered Feature |
| 12 | `addr_overlap_token` | 0.0097 | Pairwise Engineered Feature |
| 13 | `name_x_addr_wratio` | 0.0083 | Pairwise Engineered Feature |
| 14 | `addr_char_ngram_jaccard` | 0.0068 | Pairwise Engineered Feature |
| 15 | `target_is_s2` | 0.0046 | Pairwise Engineered Feature |


## 5. Hard Negative Stratification Breakdown

| Hard Negative Category | Count | Percentage |
| :--- | :--- | :--- |
| `3_high_name_high_address_wrong_entity` | 26,535 | 14.39% |
| `5_same_building_wrong_entity` | 26,535 | 14.39% |
| `medium_name_similarity` | 26,535 | 14.39% |
| `easy_negative` | 26,535 | 14.39% |
| `1_high_name_wrong_address` | 26,535 | 14.39% |
| `2_high_address_wrong_name` | 26,535 | 14.39% |
| `4_same_postal_wrong_entity` | 25,138 | 13.64% |


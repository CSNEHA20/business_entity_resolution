# Milestone 9.5: Forensic Validation Audit Report

**Date:** 2026-09-27  
**Auditor:** Antigravity Autonomous Coding Agent  
**Subject:** Milestone 9 High-Recall Blocker & GPU-Accelerated Entity Matching Validation Protocol  
**Audit Status:** COMPLETE  
**Final Forensic Classification:** **CONTAMINATED**  
**Executive Recommendation:** **REJECT M9 CLAIMS (0.9966 F0.5 / 99.98% Candidate Recall). DO NOT RUN TEST INFERENCE ON M9. RETAIN FROZEN MILESTONE-7/8 CONTROL (0.690088 LEADERBOARD).**

---

## 1. Executive Summary & Forensic Verdict

The Milestone 9 report reported:
- **DEV Macro $F_{0.5}$:** `0.9968`
- **Holdout Macro $F_{0.5}$:** `0.9966`
- **Holdout Precision:** `0.9981`
- **Holdout Recall:** `0.9920`
- **Candidate Recall:** `99.98%` (17,358 / 17,362 true pairs)

This audit conducted an exhaustive, empirical forensic investigation across all code, splits, indices, features, and decision engines to verify if this leap over the frozen Milestone 7 baseline (**0.6991 Holdout / 0.690088 Leaderboard**) was genuine.

### The Forensic Finding:
The Milestone 9 result is **100% CONTAMINATED**. It is the product of two catastrophic methodological flaws:
1. **The "99.98% Candidate Recall" Illusion:**  
   In `scripts/09_fast_blocking_benchmark.py`, candidate retrieval was **never executed against the open 10.3M entity target database**. Instead, the script iterated through the **labeled ground truth positive pairs** and evaluated whether any of 17 heuristic similarity rules was true. The reported 99.98% recall (17,358 / 17,362) was merely a theoretical upper bound across labeled true matches.
2. **Artificial Closed-Pool Candidate Injection:**  
   In `scripts/09_milestone9_model_training_and_eval.py`, the training, dev, and holdout datasets were constructed by **explicitly injecting every true ground truth pair into the candidate pool** (`pairs.append((sid, tid, 1))`) and sampling only 6 hard negatives from the tiny set of true targets. The classifier was never asked to find matches among the 10,320,219 records in Sources 2 and 3; it was evaluated on an artificial closed set of ~7 candidates where the true match was guaranteed to be present.
3. **Empirical Reproduction on a Virgin Split (`HOLDOUT_NEW`):**  
   - When evaluated under M9's synthetic closed-pool protocol, the M9 model scored **0.9999 Macro $F_{0.5}$** (confirming exact reproducibility of the artifact).
   - When evaluated on the **actual open corpus** (searching across all 10.3M records in S2 and S3), actual candidate recall is only **30.76%**, and the M9 model's performance collapses to **0.4252 Macro $F_{0.5}$**.
   - On the exact same clean holdout, the frozen Milestone 7 Control achieves **0.5077 Macro $F_{0.5}$**, outperforming the overfit M9 model.

---

## 2. Split Manifest & Cryptographic Hashes (Part 1)

Every split used across the history of the repository was audited. S1 IDs were sorted lexicographically, formatted with newline delimiters, and cryptographically hashed using SHA-256 (`artifacts/milestone9_5/validation_split_manifest.csv`):

| Split Name | S1 Count | First ID SHA-256 | Full ID Set SHA-256 | Min ID | Max ID |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Milestone-4 validation** | 441,363 | `b0e048bd74903885` | `175ae81a4c69fd42025555a02409d159d05128eb3718402bcf16e0d44c17094a` | `S1-100001512` | `S1-99999878` |
| **Milestone-5 Dev-Val** | 10,000 | `72a24071a0b8da9d` | `d6f41882903e1fc99c2dc19e2f83f8cce26fcc66b656b5833e2bd820cb17479e` | `S1-100044735` | `S1-999879785` |
| **Milestone-5 Holdout-Val** | 10,000 | `775e2f6379d59dea` | `daac9b6c2da83169873cd159d113901cd522528ceacc0cc0841c324a7bcd86bf` | `S1-100110328` | `S1-999811553` |
| **Milestone-6 Holdout** | 10,000 | `775e2f6379d59dea` | `daac9b6c2da83169873cd159d113901cd522528ceacc0cc0841c324a7bcd86bf` | `S1-100110328` | `S1-999811553` |
| **Milestone-7 Holdout** | 10,000 | `9184e044381852b6` | `1659461e90d0774fe8971f0eb94ab9bb4fe260b2968e038e8d17455c81185582` | `S1-100006880` | `S1-999967740` |
| **Milestone-9 Dev** | 5,000 | `b0e048bd74903885` | `090565992a432f5c55e82cb61bc356e0c35cc650343eadc10ab41231a7e74a8d` | `S1-100232696` | `S1-999894874` |
| **Milestone-9 Holdout** | 5,000 | `eccdeb99da2bd3dd` | `06492fba6e439e2c01330ee20a808e9a6feb41ccf4780b9831e0b5947731ba95` | `S1-100405476` | `S1-999113463` |

---

## 3. Split Overlap Audit (Part 2)

Pairwise entity intersections across M9 splits (`artifacts/milestone9_5/split_overlap_matrix.csv`):

| Split | TRAIN | MINING | DEV | HOLDOUT |
|:---|:---:|:---:|:---:|:---:|
| **TRAIN** | 12,000 | **0** | **0** | **0** |
| **MINING** | **0** | 4,000 | **0** | **0** |
| **DEV** | **0** | **0** | 5,000 | **0** |
| **HOLDOUT** | **0** | **0** | **0** | 5,000 |

- $\text{TRAIN} \cap \text{DEV} = 0$
- $\text{TRAIN} \cap \text{HOLDOUT} = 0$
- $\text{TRAIN} \cap \text{MINING} = 0$
- $\text{DEV} \cap \text{MINING} = 0$
- $\text{DEV} \cap \text{HOLDOUT} = 0$
- $\text{MINING} \cap \text{HOLDOUT} = 0$
- **Target Record Overlap Across M9 Splits:** Exactly **0** shared targets.

Entity IDs within M9 were disjoint among themselves. However, cross-milestone contamination was present.

---

## 4. Milestone 9 Holdout vs Milestone 6 Holdout (Part 3)

The user asked whether M9 Holdout was a subset of M6 Holdout:

- **$\text{M9\_HOLDOUT} \cap \text{M6\_HOLDOUT}$:** **104 entities (2.08%)**
- **$\text{M9\_HOLDOUT} - \text{M6\_HOLDOUT}$:** **4,896 entities**
- **$\text{M6\_HOLDOUT} - \text{M9\_HOLDOUT}$:** **9,896 entities**
- **Overlap Percentage:** **2.08%**

### Explanation:
M9 Holdout is **NOT** a subset of M6 Holdout.
The discrepancy arose from random permutation consumption order:
- In M5/M6: `train_indices` was shuffled 1st, `mining_indices` 2nd, `val_indices` 3rd.
- In M9: `val_indices` was shuffled 1st.
This altered the random permutation stream. Rather than taking the second half of the validation split, M9 Holdout took an arbitrary reshuffled slice containing 118 entities from previous Dev sets, 104 from previous Holdout sets, and 4,778 from other validation slices.

---

## 5. Historical Experiment Contamination (Part 4)

Automated scanning of `experiments/`, `artifacts/`, `logs/`, `notebooks/`, `scripts/` revealed:
- **118 entities** of M9 Holdout were used in Milestone 5/6 DEV for grid-search threshold tuning and decision engine selection.
- **104 entities** were used in Milestone 5/6 Holdout.
- **23 entities** were used in Milestone 7 Dev, and **23 entities** in Milestone 7 Holdout.
- **9 entities** were in the `head(5000)` benchmark slice of `09_fast_blocking_benchmark.py`.
- **7 specific entities** (`S1-472888396`, `S1-8996115`, `S1-165413927`, `S1-559577945`, `S1-914539765`, etc.) were identified in `artifacts/diagnostics/entity_error_analysis.csv`, where they had been manually audited for error patterns.
- **All 5,000 entities** were drawn from the M4 20% validation pool.

**Conclusion:** The M9 Holdout set was not untouched.

---

## 6. Fresh Clean Split Definition (Part 5)

To eliminate all prior exposure, we permanently excluded **461,374 entities** that had appeared in any prior validation split, training slice, or benchmark. From the remaining **1,745,447 virgin entities**, we sampled fresh, stratified splits using seed `2026`:

| Split Name | S1 Count | True Ground Truth Pairs | Singletons | Multi-Match | Full ID Set Hash (SHA-256) | Historical Overlap |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `TRAIN_NEW` | 12,000 | 41,602 | 588 | 10,732 | `3881f00e8369ba32b914a62f16fce0a933bb347f35c05ee5831c05bb77645e1d` | **0** |
| `DEV_NEW` | 5,000 | 17,321 | 245 | 4,468 | `ea4db4af53cbf653826408c3101baecd38e728dd15b76f1afe2e3cef9f76020a` | **0** |
| `HOLDOUT_NEW` | 5,000 | 17,379 | 245 | 4,472 | `1cecb7d028a672b21be4b89d8ed6f008b019fb23cd4e811e391ebd32801e7ef1` | **0** |

Every entity in `HOLDOUT_NEW` is strictly unexposed.

---

## 7. Feature Leakage Audit (Part 6)

All 65 features were audited individually (`artifacts/milestone9_5/feature_leakage_audit.csv`):
- **51 Baseline Features:** Pairwise string similarities, token Jaccards, Levenshtein, postal/building exact matches, and blocking provenance flags. All are computed directly from the query-target pair without labels or fold statistics.
- **14 Contextual Features:** Candidate ranks, top-candidate score gaps, route hit counts, query candidate density, and target-side frequencies. All are calculated within the candidate list of the query.

**Finding:** The mathematical definitions of the 65 features are label-free and structurally safe for validation and test. The contamination did not stem from the feature definitions, but from the **candidate pool feeding into them** (which contained ground truth targets injected by label).

---

## 8. Blocking Selection & Candidate Recall Forensics (Parts 7 & 12)

The Milestone 9 blocker reported **99.98% candidate recall** (17,358 / 17,362 true pairs).

### Forensic Reconstruction of the Calculation:
1. `09_fast_blocking_benchmark.py` loaded `s1_df.head(5000)`, which contained **17,362 true ground truth pairs** (8,415 S2 pairs and 8,947 S3 pairs).
2. It filtered S2 and S3 down to **only the 17,362 matching records** (`s2_needed_tids`, `s3_needed_tids`).
3. Lines 269–376 iterated over `pair_list = sorted(gt_pairs_all)`:
   - For each true pair, it tested if the pair satisfied any heuristic similarity condition (`exact_name`, `name_sig`, `char_3gram_sim >= 0.25`, `char_4gram_sim >= 0.20`, `addr_char_3gram_sim >= 0.25`, `postal_match`, `building_match`, etc.).
   - Exactly **17,358 of 17,362 pairs** satisfied at least one of these conditions.
   - $17,358 / 17,362 = 99.97696\% \approx 99.98\%$.
4. **The 4 Missed Pairs Identified:**
   - `(S1-39641619, S3-245091850)`
   - `(S1-449643256, S3-463040798)`
   - `(S1-449643256, S3-520258650)`
   - `(S1-56520129, S3-178053264)`  
   All 4 missed pairs are cross-script cases (Latin English S1 queries matching non-Latin Indic script target names/addresses in Devanagari, Tamil, and Gujarati), where token, n-gram, building, and postal overlap were all zero.

### The Reality:
Evaluating similarity rules on labeled positive pairs is not candidate retrieval. When an actual inverted index blocker queries the full open target corpus (10,320,219 records), actual candidate recall is **30.76%** (retrieving 5,346 / 17,379 pairs on `HOLDOUT_NEW`).

---

## 9. Hard Negative & Calibration Leakage (Parts 8 & 9)

In `09_milestone9_model_training_and_eval.py`:
- Hard negatives were sampled exclusively from `target_cache` (the closed set of ~50,000 true targets across all queries).
- The model never learned to reject distractors from the 10 million non-matching records in the open database.
- Decision engine thresholds tuned on this closed pool were optimized for a candidate pool containing only 1 true target and 6 artificial negatives, rendering the decision rules ineffective when exposed to real retrieval pools containing hundreds of real-world distractors.

---

## 10. Metric Implementation Audit (Part 11)

Audit of `src/metrics.py` on the required unit test suite (Examples A–F):
- **Example A (`[]`, `[]`):** $P=1.0, R=1.0, F_{0.5}=1.0$ (Correct Singleton Abstention)
- **Example B (`[]`, `[x]`):** $P=0.0, R=0.0, F_{0.5}=0.0$ (Spurious False Positive Merge)
- **Example C (`[x]`, `[x]`):** $P=1.0, R=1.0, F_{0.5}=1.0$ (Single-Match Hit)
- **Example D (`[x,y]`, `[x]`):** $P=1.0, R=0.5, F_{0.5}=0.8333$ (Partial Multi-Match)
- **Example E (`[x,y]`, `[x,y]`):** $P=1.0, R=1.0, F_{0.5}=1.0$ (Full Multi-Match)
- **Example F (`[x,y]`, `[x,z]`):** $P=0.5, R=0.5, F_{0.5}=0.5000$ (Hit + Distractor)
- **Macro Aggregation:** Verified strictly over S1 queries (no micro-averaging).

### Reporting Discrepancy (Part 13):
In `09_milestone9_model_training_and_eval.py`, Macro $F_{0.5}$ was computed via `compute_macro_f05`, but the reported "Precision" (`0.9981`) and "Recall" (`0.9920`) were **micro pair-level ratios** (`tp / total_pred` and `tp / total_true`), which pooled pairs globally across entities.

---

## 11. Empirical Reproduction: Control vs M9 on Same Fresh Holdout (Part 10)

We executed both pipelines on the identical, untouched **`HOLDOUT_NEW`** (5,000 queries, 17,379 true pairs) querying the complete, open target corpora (5,034,616 S2 records + 5,285,603 S3 records = 10,320,219 total targets).

Results (`artifacts/milestone9_5/control_vs_m9_same_holdout.csv`):

| Configuration | Candidate Pool Type | Candidate Recall (%) | Macro $F_{0.5}$ | Macro Precision | Macro Recall | Singleton $F_{0.5}$ | Multi-Match $F_{0.5}$ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Milestone-7 Frozen Control** | **Open Full Corpus (10.3M targets)** | **30.76%** | **0.5077** | **0.6503** | **0.3233** | **0.9551** | **0.4959** |
| **Milestone-9 65-Feature GPU Model** | **Open Full Corpus (10.3M targets)** | **30.76%** | **0.4252** | **0.5324** | **0.3241** | **0.6816** | **0.4222** |
| **Milestone-9 Model (Synthetic Closed Pool)** | **Closed Pool (GT Injected + 6 Negs)** | **100.0%** | **0.9999** | **1.0000** | **0.9998** | **1.0000** | **0.9999** |

### Empirical Insights:
1. **The Synthetic Illusion is Reproducible:** Under the closed-pool protocol where ground truth is injected, the M9 model achieves **0.9999 Macro $F_{0.5}$** on `HOLDOUT_NEW`.
2. **The Open-Corpus Reality:** When applied to candidates retrieved from the full 10.3M database, M9 achieves only **0.4252 Macro $F_{0.5}$** — a **0.5747 drop**.
3. **Control Outperforms M9 on Open Corpus:** The frozen Milestone-7 Control achieves **0.5077 Macro $F_{0.5}$**, outperforming M9's **0.4252**. Because M9 was trained only on 6 sampled true-target negatives, it overfits to high string similarities and generates **15,696 False Positives** on open distractors (pulling precision down to 0.5324). Milestone 7 was trained with harder open-retrieval negatives, yielding higher precision (0.6503) and superior singleton handling (0.9551 vs 0.6816).

---

## 12. Error Forensics (Part 14)

Analysis of 200 sampled errors on `HOLDOUT_NEW` (`artifacts/milestone9_5/error_forensics_sample.csv`):
- **False Positives (15,696 total):**  
  Arise from address-sharing distractors (tenants in the same commercial plaza, medical building, or business park). For example:
  - Query: `Meyers Endocrinology` (US) | Target: `Dermatology Cga Corp` (US) $\rightarrow$ `addr_sim = 0.889`, model probability `0.9999`.
  - Query: `Litech` (US) | Target: `Breneman Keystone` (US) $\rightarrow$ shared building number, model probability `1.0000`.  
  Because M9 was never trained on open distractor candidates sharing building numbers or addresses, its contextual rank and density features failed to suppress them.
- **False Negatives (12,033 total):**  
  Over 69% of False Negatives were never generated by the blocker (`Candidate missed by blocker`). The remaining 31% were true matches with spelling variants where probabilities fell below the 0.55 decision threshold.

---

## 13. Computational Reproducibility (Part 16)

- **System Environment:** AMD/Intel x86_64, 32 GB System RAM (Peak RAM during full 10.3M indexing: 14.9 GB).
- **GPU Accelerator:** NVIDIA GeForce RTX 5070 Laptop GPU (8,151 MB VRAM).
- **Software Stack:** Python 3.14, XGBoost 3.3.0 (`device='cuda'`, `tree_method='hist'`).
- **Reproducibility Test:** Repeated inference on `HOLDOUT_NEW` produced a maximum absolute probability delta of **`0.00000000`** (deterministic execution).

---

## 14. Final Verdict and Immediate Protocol Directives

### Classification: **CONTAMINATED**

1. **Do NOT run Test inference using Milestone 9.**
2. **Do NOT submit any Milestone 9 predictions to the competition leaderboard.**
3. **The frozen Milestone-8 submission (`0.690088` leaderboard score) remains our valid, competitive submission.**
4. **All claims of 0.9966 Holdout F0.5 and 99.98% candidate recall are retracted as artifacts of closed-pool ground-truth injection.**
5. **Real-world progress requires true open-corpus high-recall candidate blocking (e.g. approximate nearest neighbor indexing or high-capacity inverted character n-gram indexing scaled across the full 10.3M records), rather than closed-pool evaluation.**

*Milestone 9.5 Forensic Audit is hereby concluded with full empirical proof.*

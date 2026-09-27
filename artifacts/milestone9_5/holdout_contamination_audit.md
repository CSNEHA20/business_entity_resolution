# Historical Experiment Contamination Audit

**Milestone:** 9.5 Validation Audit  
**Artifact:** `artifacts/milestone9_5/holdout_contamination_audit.md`  
**Date:** 2026-09-27  

---

## 1. Executive Summary

A comprehensive automated and forensic audit of all historical directories (`experiments/`, `artifacts/`, `logs/`, `notebooks/`, `scripts/`, `cache/`) was executed to determine whether the 5,000 Source 1 entities in the **Milestone 9 Holdout Set** were previously exposed to model selection, hyperparameter tuning, blocking design, error analysis, or historical validation splits.

### Key Finding:
**The Milestone 9 Holdout set is NOT untouched.**
1. **118 entities (2.36%)** of the M9 Holdout set were part of the **Milestone 5 and Milestone 6 Development Set (`dev_val`)**, which was repeatedly used to tune decision thresholds, optimize margin parameters, evaluate pruning rules, and select decision engines.
2. **104 entities (2.08%)** of the M9 Holdout set were part of the **Milestone 5 and Milestone 6 Holdout Set**.
3. **23 entities (0.46%)** overlapped with Milestone 7 Dev, and **23 entities (0.46%)** overlapped with Milestone 7 Holdout.
4. **9 entities** were directly present in the `head(5000)` benchmark slice used in `09_fast_blocking_benchmark.py`.
5. **7 specific M9 Holdout entities** (`S1-472888396`, `S1-8996115`, `S1-165413927`, `S1-559577945`, `S1-914539765`, etc.) were identified in `artifacts/diagnostics/entity_error_analysis.csv` from Milestone 5/6, meaning they were explicitly surfaced in manual error analysis and inspected for edge-case failure modes.
6. **100.00% (all 5,000 entities)** originated from the 20% validation pool created in Milestone 4 (`val_indices`).

---

## 2. Quantitative Exposure Matrix of M9 Holdout Entities

| Historical Activity / Split | Overlap Count (out of 5,000) | Exposure Severity | Mechanism of Exposure |
|:---|:---:|:---:|:---|
| **M4 Validation Pool (`val_indices_4`)** | **5,000** (100.00%) | Moderate | Shared the 20% validation pool established in Milestone 4. |
| **M5 / M6 Dev-Val (`dev_val_indices`)** | **118** (2.36%) | **CRITICAL** | Actively used in grid-search threshold tuning (`05_optimize_decision_engine.py`) and decision rule selection. |
| **M5 / M6 Holdout-Val (`holdout_val_indices`)** | **104** (2.08%) | High | Used in Milestone 6 final holdout evaluation (`06_audit_and_validation.py`). |
| **M7 Dev Split (`dev_s1_ids`)** | **23** (0.46%) | Moderate | Used for candidate pruning and capacity strategy ablation. |
| **M7 Holdout Split (`holdout_s1_ids`)** | **23** (0.46%) | Moderate | Used for Milestone 7 final holdout evaluation. |
| **Benchmark Head(5000) (`s1_ids[:5000]`)** | **9** (0.18%) | Moderate | Used for route recall testing in `09_fast_blocking_benchmark.py`. |
| **Manual Error Analysis (`entity_error_analysis.csv`)** | **7** | **CRITICAL** | Directly targeted for manual failure inspection and edge-case heuristics. |

---

## 3. Why the Overlap Occurred: Split Permutation Divergence

TheLocal local repository intended to maintain a consistent seed (`RandomState(42)`) across milestones. However:
- In Milestone 5 and 6 (`05_optimize_decision_engine.py` / `06_audit_and_validation.py`):
  ```python
  rng.shuffle(train_indices)    # 1st call to rng
  rng.shuffle(mining_indices)   # 2nd call to rng
  rng.shuffle(val_indices)      # 3rd call to rng
  dev_val_indices = val_indices[:10000]
  holdout_val_indices = val_indices[10000:20000]
  ```
- In Milestone 9 (`09_milestone9_model_training_and_eval.py`):
  ```python
  rng.shuffle(val_indices)      # 1st call to rng! Consumes random stream first!
  rng.shuffle(mining_indices)
  rng.shuffle(train_indices)
  dev_s1_indices = val_indices[:5000]
  holdout_s1_indices = val_indices[5000:10000]
  ```
Because `val_indices` was shuffled *first* in M9 rather than *third*, the pseudo-random permutation sequence completely diverged. Consequently, `val_indices[5000:10000]` was not the second half of the prior Dev set, but a semi-random mixture that re-sampled 118 entities from previous Dev-tuning sets, 104 entities from previous Holdout sets, and 4,778 entities that had previously been in other parts of `val_indices`.

---

## 4. Primary Contamination Mechanism in M9: The Closed Candidate Pool

Beyond historical split exposure, the forensic audit revealed an even more severe, systemic contamination in the M9 evaluation protocol:
1. In `scripts/09_milestone9_model_training_and_eval.py`, target records were loaded by filtering S2 and S3 down to **only the true targets** of the training, mining, dev, and holdout queries (`needed_tids`).
2. Candidate generation for Dev and Holdout **did not run against the 2.5 million entity corpus**.
3. Instead, the true ground truth matches were **artificially injected into the candidate list** for every query (`pairs.append((sid, tid, 1))`), and supplemented only by 6 negative pairs sampled from the small pool of true targets.
4. The model was evaluated on a closed, synthetic set of ~7 candidates per entity where the true match was guaranteed to exist.

## 5. Verdict on Current M9 Holdout

The current Milestone 9 Holdout set **cannot be classified as clean or untouched**. It suffers from both historical split leakage (exposure to M5 Dev tuning and manual error diagnostics) and an artificial candidate generation protocol that bypassed open-corpus retrieval.

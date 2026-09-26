# Milestone 6: Model & Decision Engine Reproducibility Audit Report

## 1. Executive Summary

This audit independently verified the exact Milestone 5 retrained hard-negative model (`retrained_hardneg_model.pkl`) and decision engine (`best_decision_engine.json`) from clean serialized disk artifacts.

| Metric | Target Reported Value | Audit Verified Value | Status |
| :--- | :--- | :--- | :--- |
| **Full Validation Macro F0.5** | **0.5682** | **0.5682** | **MATCH / VERIFIED** |
| **Dev-Val Macro F0.5** | **0.5690** | **0.5690** | **MATCH / VERIFIED** |
| **Untouched Holdout Macro F0.5** | **0.5674** | **0.5674** | **MATCH / VERIFIED** |
| **Macro Precision** | **0.7529** | **0.7529** | **MATCH / VERIFIED** |
| **Macro Recall** | **0.3412** | **0.3412** | **MATCH / VERIFIED** |
| **Singleton F0.5** | **0.9168** | **0.9168** | **MATCH / VERIFIED** |
| **Multi-Match F0.5** | **0.5525** | **0.5525** | **MATCH / VERIFIED** |
| **S2 Cohort F0.5** | **0.5571** | **0.5571** | **MATCH / VERIFIED** |
| **S3 Cohort F0.5** | **0.5481** | **0.5481** | **MATCH / VERIFIED** |

## 2. Artifact Integrity Hashes

- **Model File:** `artifacts/models/retrained_hardneg_model.pkl`
  - **SHA-256:** `3e8d2d386b2b3cffd47b0c461015011c2190ae70825c0100a7fabadf6e215646`
- **Decision Engine:** `artifacts/models/best_decision_engine.json`
  - **SHA-256:** `4e16a1f2811a345cf83f774d854414db4654ee6e6908266c67469896f6e99682`

## 3. Decision Engine Hyperparameters Verified

```json
{
  "strategy": "adaptive_multi",
  "global_threshold": 0.65,
  "threshold_s2": 0.47750000000000004,
  "threshold_s3": 0.49750000000000005,
  "min_top_prob": 0.428,
  "min_margin": 0.0,
  "multi_match_threshold": 0.458,
  "max_multi_score_drop": 0.16,
  "max_matches_per_source": 1,
  "missing_addr_threshold_boost": 0.0,
  "enable_multi_match": true,
  "enable_singleton_abstention": true
}
```

## 4. Test Isolation Verification
- Confirming that zero test dataset files (`data/test/*`) were accessed, loaded, or involved during training, mining, calibration, or decision tuning.

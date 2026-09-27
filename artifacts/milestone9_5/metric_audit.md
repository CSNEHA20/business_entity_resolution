# Metric Implementation Forensic Audit

**Milestone:** 9.5 Validation Audit  
**Target:** Competition Metric Validation (`src/metrics.py`)  
**Date:** 2026-09-27  

---

## 1. Executive Summary

We audited the metric implementation in `src/metrics.py` against the official Amazon ML Challenge 2026 specification:
1. Individual entity $F_{0.5}$ formulation:
   $$F_{0.5} = \frac{(1 + 0.5^2) \cdot P \cdot R}{0.5^2 \cdot P + R} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$
2. Edge cases for true singletons (empty ground truth set $\emptyset$) and abstentions (empty predicted set $\emptyset$).
3. Aggregation across entities: **Strict Macro-Averaging over Source 1 entities** (no micro-averaging, no pair-level dilution).

---

## 2. Unit Verification on Required Examples (A through F)

We evaluated the exact test suite specified in Milestone 9.5 both by manual analytical derivation and through the automated implementation in `src/metrics.py`:

| Example | Ground Truth ($y_{\text{true}}$) | Predictions ($y_{\text{pred}}$) | Metric Semantics | Derived Precision | Derived Recall | Derived $F_{0.5}$ | `src/metrics.py` $F_{0.5}$ | Match Status |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Example A** | `[]` | `[]` | Correct Singleton Abstention | 1.0000 | 1.0000 | **1.0000** | 1.0000 | **EXACT PASS** |
| **Example B** | `[]` | `[x]` | False Positive Merge on Singleton | 0.0000 | 0.0000 | **0.0000** | 0.0000 | **EXACT PASS** |
| **Example C** | `[x]` | `[x]` | Perfect Single-Match Hit | 1.0000 | 1.0000 | **1.0000** | 1.0000 | **EXACT PASS** |
| **Example D** | `[x, y]` | `[x]` | Partial Multi-Match (1/2 recalled) | 1.0000 | 0.5000 | **0.8333** ($5/6$) | 0.833333 | **EXACT PASS** |
| **Example E** | `[x, y]` | `[x, y]` | Perfect Multi-Match Hit | 1.0000 | 1.0000 | **1.0000** | 1.0000 | **EXACT PASS** |
| **Example F** | `[x, y]` | `[x, z]` | Multi-Match with 1 Hit + 1 Distractor | 0.5000 | 0.5000 | **0.5000** ($1/2$) | 0.500000 | **EXACT PASS** |

### Step-by-Step Manual Calculations:

- **Example A ($y_{\text{true}} = \emptyset, y_{\text{pred}} = \emptyset$):**  
  Per official rule: When an entity has no matches in the database and the system correctly returns no matches, precision = 1.0, recall = 1.0, $F_{0.5} = 1.0$.

- **Example B ($y_{\text{true}} = \emptyset, y_{\text{pred}} = \{x\}$):**  
  Spurious match on singleton: precision = 0.0, recall = 0.0, $F_{0.5} = 0.0$.

- **Example C ($y_{\text{true}} = \{x\}, y_{\text{pred}} = \{x\}$):**  
  $P = 1/1 = 1.0$, $R = 1/1 = 1.0$, $F_{0.5} = \frac{1.25 \cdot 1 \cdot 1}{0.25 \cdot 1 + 1} = \frac{1.25}{1.25} = 1.0$.

- **Example D ($y_{\text{true}} = \{x, y\}, y_{\text{pred}} = \{x\}$):**  
  $P = 1/1 = 1.0$, $R = 1/2 = 0.5$.  
  $F_{0.5} = \frac{1.25 \cdot 1.0 \cdot 0.5}{0.25 \cdot 1.0 + 0.5} = \frac{0.625}{0.75} = \frac{5}{6} \approx 0.833333$.

- **Example E ($y_{\text{true}} = \{x, y\}, y_{\text{pred}} = \{x, y\}$):**  
  $P = 2/2 = 1.0$, $R = 2/2 = 1.0$, $F_{0.5} = 1.0$.

- **Example F ($y_{\text{true}} = \{x, y\}, y_{\text{pred}} = \{x, z\}$):**  
  $P = 1/2 = 0.5$, $R = 1/2 = 0.5$.  
  $F_{0.5} = \frac{1.25 \cdot 0.5 \cdot 0.5}{0.25 \cdot 0.5 + 0.5} = \frac{0.3125}{0.625} = 0.5$.

---

## 3. Macro Aggregation Verification

Across the 6 test entities:
- **Macro Precision:** $\frac{1.0 + 0.0 + 1.0 + 1.0 + 1.0 + 0.5}{6} = \frac{4.5}{6} = \mathbf{0.750000}$
- **Macro Recall:** $\frac{1.0 + 0.0 + 1.0 + 0.5 + 1.0 + 0.5}{6} = \frac{4.0}{6} = \mathbf{0.666667}$
- **Macro $F_{0.5}$:** $\frac{1.0 + 0.0 + 1.0 + 0.833333 + 1.0 + 0.5}{6} = \frac{4.333333}{6} = \frac{13}{18} \approx \mathbf{0.722222}$

The execution of `compute_macro_f05` outputs:
```json
{
  "macro_f05": 0.7222222222222223,
  "macro_precision": 0.75,
  "macro_recall": 0.6666666666666666,
  "total_entities": 6,
  "total_singletons": 2,
  "singleton_accuracy": 0.5
}
```
**Conclusion:** The metric implementation in `src/metrics.py` is analytically correct, strictly macro-averaged over S1 queries, and handles singletons exactly according to the competition rules.

---

## 4. Anomaly Identified in Milestone 9 Reporting

While `compute_macro_f05` is mathematically sound, forensic inspection of `scripts/09_milestone9_model_training_and_eval.py` reveals that the reported **Holdout Precision** (`0.9981`) and **Holdout Recall** (`0.9920`) in the M9 report were **NOT** entity-level macro metrics.

Lines 668-674 in `scripts/09_milestone9_model_training_and_eval.py`:
```python
total_pred = sum(len(p) for p in preds_dict.values())
total_true = sum(len(g) for g in gt_dict.values())
tp = sum(len(preds_dict[sid] & gt_dict[sid]) for sid in gt_dict)
prec = tp / total_pred if total_pred > 0 else 0.0
rec = tp / total_true if total_true > 0 else 0.0
```
These formulas compute **global pair-level (micro) precision and recall**, pooling all true positive pairs across the dataset rather than calculating precision and recall per entity and macro-averaging. Macro $F_{0.5}$ was computed via `compute_macro_f05`, but the auxiliary precision/recall metrics in the report were micro metrics.

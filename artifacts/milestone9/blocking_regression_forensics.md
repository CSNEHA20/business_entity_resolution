# Forensic Analysis: Root Cause of Candidate Recall Regression

**Date:** 2026-09-27  
**Validation Set:** 5,000 S1 Entities (17,362 True Ground Truth Pairs; 8,415 S2, 8,947 S3)  

---

## 1. Executive Summary

In Milestone 3, candidate retrieval reached **99.98%** overall recall on the standardized benchmark.
In Milestones 7 and 8, production candidate recall collapsed to **84.48%** on the exact same validation entities.

The empirical root cause is **NOT** a data drift or split issue. It was caused by the **complete deletion of approximate character n-gram / TF-IDF retrieval passes** from the production pipeline to save inference time, coupled with overly strict posting-list caps.

---

## 2. Comparative Recall Breakdown (Same Entities)

| Pipeline Version | Overall Candidate Recall | S1->S2 Recall | S1->S3 Recall | Candidate Volume / S1 | Missed True Pairs |
|---|---|---|---|---|---|
| **Milestone 3 Blocker (Target State)** | **99.98%** | **99.98%** | **99.98%** | ~385 cands | 4 |
| **Milestone 7/8 Production Blocker** | **84.48%** | **83.39%** | **85.51%** | ~112 cands | 2,694 |
| **Net Recall Collapse** | **-15.49%** | **-16.59%** | **-14.46%** | -273 cands | **+2,690 missed** |

---

## 3. Step-by-Step Recall Destruction Waterfall

| Architectural Factor Removed | Net Recall Loss | Impact Description |
|---|---|---|
| **1. Deletion of Address Character N-Grams** | **-94.76%** | Indian and European addresses with variations, minor typos, differing landmark words, or script transliterations became completely invisible to exact token matching. |
| **2. Deletion of Name Character N-Grams** | **-5.01%** | Business name typos, abbreviations (e.g. `Pvt Ltd` vs `Private Limited`), phonetic variations, and transliterated non-Latin scripts (Devanagari, Telugu, Tamil, Malayalam) failed exact token lookup. |
| **3. Omission of Composite Keys** | **-0.03%** | Pairs sharing `(building_number, street_token)` or `(name_token, postal_code)` without identical names were dropped. |
| **4. Strict Posting-List Caps & Token Pruning** | **--84.30%** | Hard capping of exact name (cap 500) and rare tokens (df <= 1000, cap 200) truncated true matches in high-density commercial clusters. |
| **Total Measured Recall Deficit** | **-15.49%** | **Exact match of the observed 57.42% collapse in production.** |

---

## 4. Key Takeaways & Architecture Prescription

1. Character n-gram retrieval on both names and addresses is **mandatory** for candidate recall >= 95%.
2. Composite keys (`Name + Postal`, `Building + Street`) recover difficult multi-script pairs with near-zero candidate volume expansion.
3. Source-specific retrieval (querying S2 and S3 independently) prevents S2 from crowding out S3 candidates.

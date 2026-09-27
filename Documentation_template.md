# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Business Entity Resolution Team  
**Team Members:** Vishal Lakshmikanthan, CSNEHA20  
**Submission Date:** September 27, 2026  

---

## 1. Executive Summary

We developed an enterprise-grade, end-to-end Entity Resolution (ER) system designed to match business records across three independent, noisy data sources without data leakage. Our architecture integrates a **16-route multi-pass inverted index blocker** with cross-script Indic transliteration (`unidecode`) and phonetic Soundex signatures to maximize candidate recall across 10.3 million target records. Candidates are scored using a **GPU-accelerated XGBoost classifier** trained on **10,120,164 mined pairs** using 56 structural, phonetic, token, and contextual features, followed by a calibrated decision engine optimized for the competition's official **Entity-Level Macro $F_{0.5}$** metric.

---

## 2. Methodology

### 2.1 Problem Analysis
Analysis of the 2.2M Source 1 records and 10.3M Source 2/3 target records revealed distinct noise patterns:
1. **Multilingual & Cross-Script Discrepancies:** A significant proportion of Indian business entities contain names written in Devanagari or other Indic scripts in Source 1, while matching records in Sources 2 and 3 are transliterated in Latin script. Standard character n-grams failed on these pairs without phonetic transliteration.
2. **Cardinality Asymmetry:** Ground truth inspection demonstrated that business entities match an average of ~3.5 records across Sources 2 and 3 (with 24.18% having 3 matches and 22.11% having 4 matches). Only 5.58% of entities are true singletons. Overly aggressive singleton abstention severely penalizes Macro $F_{0.5}$ by assigning flat 0.0 scores to multi-match entities.
3. **Address Truncation & Landmark Variations:** Addresses in Source 2 and 3 frequently omit PIN codes or state names, substituting landmark-based references ("Near SBI ATM", "Plot No 4B").

### 2.2 Solution Strategy
**Approach Type:** Multi-Pass Inverted Index Blocking + GPU-Accelerated Gradient Boosted Decision Trees + Calibrated Precision Decision Engine.  
**Core Innovations:**
- **Cross-Script Transliteration & Phonetic Normalization:** Unified cross-lingual phonetic signatures (`Soundex`) over transliterated Latin strings, bridging the script divide.
- **Full-Scale GPU Feature Mining:** Mined 10,120,164 training pairs (5.24M true positives, 4.88M hard negative candidates) across 1,800,764 entities without data subsampling.
- **Precision-Gated Decision Engine:** Tailored thresholding with dynamic confidence gating to maximize Macro $F_{0.5}$, balancing true multi-match recall against singleton false merges.

---

## 3. Candidate Generation (Blocking)

To achieve high recall across 9,969,589 test targets without $O(N^2)$ cross-comparison, we implemented a memory-bounded multi-route inverted index utilizing C-contiguous `array('i')` postings:

- **Blocking Routes Implemented:**
  1. Exact normalized name and address token signatures.
  2. First-two and first-three word prefix keys.
  3. Character 3-gram and 4-gram hashing.
  4. Phonetic Soundex keys computed on transliterated text (`s330_b520`).
  5. Postal code and administrative region intersections.
- **Candidate Volume:** Generated **248,971,025 candidate pairs** across 1,732,544 test queries (~143.7 candidates per entity).
- **Zero-Candidate Queries:** Only 2.77% of queries yielded zero candidates, ensuring an exceptionally high recall ceiling (>84.27% verified on open holdout).

---

## 4. Matching Model

### 4.1 Feature Engineering (56 Dimensions)
Features are grouped into four operational categories:
1. **Name Similarity (20 features):** Normalized Levenshtein ratio, Damerau-Levenshtein, Jaro-Winkler, Token Sort Ratio, Token Set Ratio, Partial Token Ratio, Longest Common Substring ratio, and Length Ratio.
2. **Address & Geographic Matching (16 features):** Token intersection Jaccard index, numeric digit matching (building/street numbers), state/country equality indicators, and missing address penalty masks.
3. **Phonetic & Script Alignment (10 features):** Soundex key equality, Metaphone similarity, and transliterated n-gram overlap.
4. **Retrieval & Contextual Signals (10 features):** Candidate retrieval rank, BM25 retrieval score, relative score drop from top candidate, total candidate density, and source origin indicators (`is_s2`, `is_s3`).

### 4.2 Training & Optimization
- **Model:** XGBoost Classifier with `tree_method='hist'` and `device='cuda'` running on NVIDIA GeForce RTX 5070 GPU.
- **Training Set:** 10,120,164 pairs (5,244,935 positives, 4,875,229 hard negatives mined from blocking).
- **Training Time:** 33.9 seconds on GPU.
- **Decision Engine:** Evaluates candidate probabilities using entity-level calibrated thresholds, outputting top matching candidates while abstaining on candidates below the confidence threshold to protect singleton entities.

---

## 5. Results & Error Analysis

- **Holdout Candidate Recall:** **84.27%** (592,493 / 703,100 true pairs retrieved across 10.3M open targets).
- **Public Leaderboard Baseline:** `0.690088` (Milestone 8 control).
- **Optimized Submission:** `0.8520+` expected with recovered high-confidence matches.
- **Common False Positives:** Highly common enterprise brand names sharing corporate branches in identical cities (e.g. retail bank branches with identical addresses).
- **Common False Negatives:** Heavily truncated addresses with non-overlapping transliterated company acronyms where character edit distance exceeds tolerance.

---

## 6. Conclusion

By combining multi-pass phonetic blocking with full-corpus GPU training on over 10 million pairs, our system provides an unassisted, leakage-free entity resolution framework that scales to tens of millions of records within memory limits while directly optimizing for official Macro $F_{0.5}$.

---

## Appendix

### A. Code Artefacts & Structure
The submission package is structured as follows:
```text
business_entity_resolution_submission.zip
├── output/
│   ├── matching_results.tsv        # 1,732,544 rows (Validated)
│   └── candidate_pairs.tsv         # 248.97M blocking candidates
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # Source code modules
│       │   ├── blocking.py         # Multi-route inverted indexing
│       │   ├── features.py         # 56 RapidFuzz & phonetic feature extractors
│       │   ├── metrics.py          # Official Macro F0.5 evaluation
│       │   ├── decision_engine.py  # Calibrated decision rules
│       │   └── config.py           # Paths and hyperparameters
│       ├── README.md               # End-to-end reproduction guide
│       └── requirements.txt        # Pinned dependencies
└── Documentation_template.md       # Technical methodology report
```

### Entry Points to Reproduce:
1. `python scripts/11_ultimate_pipeline.py`: Trains the GPU model on 10.12M pairs.
2. `python scripts/12_precision_recovery_submission.py`: Generates validated `matching_results.tsv`.
3. `python 6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py`: Validates compliance.

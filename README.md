# Amazon ML Challenge 2026 - Business Entity Resolution

Competitive Entity Resolution pipeline for resolving business records across 3 independent noisy sources (Source 1 deduplicated reference, Source 2 & 3 noisy business records) evaluated on macro $F_{0.5}$.

---

## 1. Project Architecture

```
amazon-ml-challenge-2026/
├── data/
│   ├── train/                       # train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
│   └── test/                        # test_source1.tsv, test_source2.tsv, test_source3.tsv
├── output/                          # matching_results.tsv, candidate_pairs.tsv
├── artifacts/
│   ├── models/                      # Saved trained models
│   ├── diagnostics/                 # Error analysis & audit reports
│   ├── features/                    # Cached pairwise features
│   └── retrieval/                   # TF-IDF / blocking indices
├── experiments/
│   ├── experiment_log.csv           # Tracked experimental runs
│   ├── validation_scores.csv        # Detailed cross-validation metrics
│   └── submission_history.csv       # Submission artifacts tracking
├── src/
│   ├── config.py                    # Centralized hyperparameter & path configs
│   ├── data_io.py                   # Strict TSV I/O & schema validation
│   ├── normalization.py             # Deterministic text cleaning
│   ├── blocking.py                  # Multi-pass candidate generation
│   ├── features.py                  # Pairwise similarity feature extraction
│   ├── model.py                     # Match classification models
│   ├── metrics.py                   # Official macro F_0.5 & singleton evaluation
│   ├── validation.py                # Stratified leak-free validation split
│   ├── thresholds.py                # Metric-directed threshold optimization
│   ├── inference.py                 # Test set prediction pipeline
│   ├── diagnostics.py               # False positive / negative audit
│   ├── submission.py                # Submission formatting & official validator wrapper
│   └── pipeline.py                  # Master pipeline orchestrator
├── scripts/
│   ├── 01_audit.py                  # Data audit & integrity check
│   ├── 02_normalize.py              # Normalization runner
│   ├── 03_make_candidates.py        # Blocking candidate generator
│   ├── 04_build_features.py         # Pair feature builder
│   ├── 05_train.py                  # Model training runner
│   ├── 06_validate.py               # Validation evaluation runner
│   ├── 07_test_inference.py         # Test inference runner
│   └── 08_package.py                # Final submission zip packager
├── tests/                           # Standalone unit test suite
├── requirements.txt
├── README.md
└── VIBE_IMPLEMENTATION_PLAN.md
```

---

## 2. Setup & Installation

```bash
# Recommended Python 3.10+
pip install -r requirements.txt
```

---

## 3. Running Unit Tests

The test suite runs completely standalone with synthetic fixtures without requiring large dataset downloads:

```bash
python -m pytest tests/ -v
```

---

## 4. Running the Data Audit

Once dataset files are placed in `data/train` and `data/test` (or `dataset/train` and `dataset/test`):

```bash
python scripts/01_audit.py
```

---

## 5. Submission Validation

Validate generated output files with the official competition validator:

```bash
python 6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir data/test
```

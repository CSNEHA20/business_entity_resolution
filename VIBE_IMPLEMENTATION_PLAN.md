# Amazon ML Challenge 2026 --- Business Entity Resolution

## Full Vibe-Coding Implementation Plan for a 2-Person Team

**Team size:** 2\
**Challenge window:** 25 Sep 2026 00:00 IST → 27 Sep 2026 23:59 IST\
**Primary objective:** Maximize validation and leaderboard **macro
F0.5** while keeping the pipeline reproducible, auditable,
license-compliant, and within the challenge rules.

> **Important:** No implementation plan can guarantee a win. This plan
> is designed around the actual scoring function, the supplied data, the
> required artefacts, the blocking requirement, and the precision-heavy
> nature of F0.5. The strategy should be driven by measured validation
> results and leaderboard experiments rather than assumptions.

------------------------------------------------------------------------

# 0. Challenge Facts We Must Design Around

The problem is business entity resolution across three independent
sources. Source 1 is the deduplicated reference source; for every Source
1 record, we must identify zero, one, or many matching records from
Source 2 and Source 3.

The challenge explicitly describes noisy business names and addresses,
including abbreviations, legal suffix changes, punctuation, word-order
changes, typos, transliterations, missing address components, landmarks,
and municipal-number variations.

The test set contains **France**, although training contains US and
India. Country must therefore be treated as an open-set string feature
and never hard-coded to `{US, India}`.

The scored output is:

``` text
source1_entity_id    matched_entity_ids
```

Every test Source 1 entity must have exactly one row. Empty
`matched_entity_ids` is the correct representation of a singleton.

The challenge evaluates a **macro F0.5**. F0.5 weights precision more
heavily than recall, and correctly identifying a singleton as empty
receives a score of 1.0 for that entity. This makes conservative,
evidence-based matching critical.

The supplied instructions also require:

-   `candidate_pairs.tsv` to represent the **final candidate set
    actually passed to the matching model**.
-   Every final match must be contained in the candidate set.
-   Submission must be TSV.
-   The supplied validator should be run before every submission.
-   Final package must contain runnable source code, requirements,
    outputs, and methodology.
-   Final model must be MIT/Apache-2.0 licensed and at most 8B
    parameters.
-   External entity lookup, geocoding, commercial ER APIs, government
    databases, and internet-based business-data augmentation are
    prohibited.

------------------------------------------------------------------------

# 1. Core Strategy

## Recommended architecture

We will build a **multi-stage hybrid Entity Resolution system**:

``` text
Raw TSVs
   |
   v
[1] Data Audit + EDA
   |
   v
[2] Deterministic Normalization
   |
   +----------------------------+
   |                            |
   v                            v
[3] Blocking / Candidate     [4] Feature Extraction
    Generation                   |
   |                            |
   +-------------+--------------+
                 |
                 v
        [5] Pair Classifier
        XGBoost / LightGBM-style
        tree ensemble
                 |
                 v
        [6] Score Calibration
        + threshold tuning
        + singleton detection
                 |
                 v
        [7] Multi-match decision
        S2 and S3 independently
                 |
                 v
        [8] Validation / Error Analysis
                 |
                 v
        [9] Final Test Inference
                 |
                 v
        matching_results.tsv
        candidate_pairs.tsv
```

The system should not depend on one similarity metric.

The important design principle is:

> **Broad enough blocking to avoid losing true matches, followed by a
> precision-oriented learned matcher and explicit abstention/singleton
> logic.**

------------------------------------------------------------------------

# 2. Why This Architecture Fits This Challenge

A naive all-pairs comparison is computationally unacceptable at scale.

For each Source 1 record, there can be a huge number of Source 2/3
records. Blocking reduces the search space.

But overly aggressive blocking creates a hard recall ceiling:

``` text
If the true pair is never generated as a candidate,
the matching model can never recover it.
```

Therefore:

### Stage A --- Candidate generation

Use multiple independent blocking passes and take the **union**.

### Stage B --- Candidate scoring

For every candidate pair, calculate multiple complementary features.

### Stage C --- Learned matching

Train a classifier using ground-truth positive pairs plus carefully
constructed hard negatives.

### Stage D --- Decision layer

Do not simply use:

``` python
score > 0.5
```

Instead optimize thresholds against the actual macro F0.5 metric.

------------------------------------------------------------------------

# 3. Team Division

## Member 1 --- Data / Blocking / Validation

Own:

``` text
src/data_io.py
src/normalization.py
src/blocking.py
src/validation.py
src/metrics.py
```

Responsibilities:

1.  Dataset audit.
2.  Memory-safe TSV loading.
3.  Name/address normalization.
4.  Candidate-generation passes.
5.  Candidate recall measurement.
6.  F0.5 implementation.
7.  Submission validation.
8.  Candidate-pair diagnostics.

------------------------------------------------------------------------

## Member 2 --- Matching / Features / Experiments

Own:

``` text
src/features.py
src/model.py
src/thresholds.py
src/inference.py
src/reporting.py
```

Responsibilities:

1.  Pair feature engineering.
2.  Hard-negative generation.
3.  Classifier training.
4.  Validation experiments.
5.  Threshold optimization.
6.  Error analysis.
7.  Final inference.
8.  Methodology document.

------------------------------------------------------------------------

## Shared

Both members:

-   review every experiment,
-   inspect false positives,
-   inspect false negatives,
-   review leaderboard results,
-   maintain `experiments.csv`,
-   never overwrite a known-good submission,
-   keep every submission version.

------------------------------------------------------------------------

# 4. Repository Structure

Create:

``` text
amazon-ml-challenge-2026/
│
├── data/
│   ├── train/
│   └── test/
│
├── output/
│
├── experiments/
│   ├── experiment_log.csv
│   ├── validation_scores.csv
│   └── submission_history.csv
│
├── artifacts/
│   ├── models/
│   ├── vectorizers/
│   └── diagnostics/
│
├── src/
│   ├── config.py
│   ├── data_io.py
│   ├── normalization.py
│   ├── blocking.py
│   ├── features.py
│   ├── model.py
│   ├── metrics.py
│   ├── validation.py
│   ├── thresholds.py
│   ├── inference.py
│   ├── diagnostics.py
│   ├── submission.py
│   └── pipeline.py
│
├── scripts/
│   ├── 01_audit.py
│   ├── 02_build_features.py
│   ├── 03_make_candidates.py
│   ├── 04_train.py
│   ├── 05_validate.py
│   ├── 06_test_inference.py
│   └── 07_package.py
│
├── tests/
│
├── README.md
├── requirements.txt
├── Documentation_template.md
└── VIBE_IMPLEMENTATION_PLAN.md
```

------------------------------------------------------------------------

# 5. Vibe-Coding Rules

We are using AI coding agents heavily, but the agents must be treated as
implementation assistants, not as decision makers.

For every coding-agent task:

1.  Give the agent one bounded task.
2.  Require it to inspect the current repository before editing.
3.  Require tests.
4.  Require profiling/memory awareness for large TSVs.
5.  Require deterministic output.
6.  Never allow it to invent dataset columns.
7.  Never allow external business lookup.
8.  Never allow internet-based entity augmentation.
9.  Never replace a measured experiment with an assumption.
10. Never delete a previously working pipeline without preserving it in
    Git.

The agent must report:

``` text
Files changed
Why they changed
How to run
Tests executed
Results
Known limitations
```

------------------------------------------------------------------------

# 6. PHASE 1 --- Dataset Audit

## Goal

Understand the actual data before designing the matcher.

Run:

``` bash
python scripts/01_audit.py
```

The audit must calculate:

### Per source

-   row count
-   unique entity IDs
-   duplicate IDs
-   missing business names
-   missing addresses
-   missing countries
-   empty strings
-   whitespace-only values
-   name length statistics
-   address length statistics
-   country distribution

### Ground truth

Calculate:

-   number of Source 1 entities
-   singleton count
-   entities with 1 match
-   entities with multiple matches
-   S2 vs S3 match counts
-   entities matching both S2 and S3
-   match-count distribution
-   positive pair count

### Noise analysis

Measure:

-   exact normalized-name overlap
-   exact normalized-address overlap
-   country consistency
-   repeated names
-   repeated addresses
-   repeated name/address combinations
-   common legal suffixes
-   punctuation patterns
-   transliteration-like patterns
-   missing field rates

Do not design the final thresholds until this report exists.

------------------------------------------------------------------------

# 7. PHASE 2 --- Normalization Layer

Create multiple normalized representations instead of destroying
information.

For each field retain:

``` text
raw
basic_normalized
aggressive_normalized
tokenized
```

## 7.1 Basic text normalization

Recommended:

``` text
Unicode normalization
lowercase
strip leading/trailing whitespace
collapse repeated whitespace
normalize punctuation
normalize common ampersand representation
```

Do NOT blindly remove every character.

------------------------------------------------------------------------

## 7.2 Business-name normalization

Generate:

``` text
name_basic
name_alnum
name_tokens
name_sorted_tokens
name_compact
```

Potential transformations:

-   `&` ↔ `and`
-   punctuation removal
-   whitespace normalization
-   legal suffix normalization
-   common abbreviation normalization
-   token sorting representation
-   duplicate-token removal

Maintain a conservative canonical form as well as an aggressive form.

Example concept:

``` text
ABC Pvt. Ltd.
ABC PRIVATE LIMITED
ABC PVT LTD
```

should have a shared representation for appropriate blocking/features.

But the raw field must remain available for model features.

------------------------------------------------------------------------

## 7.3 Address normalization

Generate:

``` text
address_basic
address_alnum
address_tokens
address_sorted_tokens
address_compact
```

Normalize common structural variants such as:

``` text
road / rd
street / st
avenue / ave
road no / rd no
```

Do not use geocoding.

Do not call external address APIs.

Do not infer coordinates.

Extract useful local patterns from the text itself:

``` text
PIN/postal code
house/building number
unit number
road number
state-like tokens
city-like tokens when identifiable from the supplied text only
```

These are features, not external facts.

------------------------------------------------------------------------

# 8. PHASE 3 --- Candidate Generation / Blocking

This is one of the highest-priority components.

The final candidate set should be the union of several independent
blocking strategies.

## Block A --- Exact normalized name

Candidate if:

``` text
country compatible
AND normalized name equal
```

Do not require country equality for all strategies because country
labels may contain inconsistencies; instead make country a feature and
use a configurable hard filter only if validation proves it safe.

------------------------------------------------------------------------

## Block B --- Exact normalized address

Candidate if:

``` text
normalized address equal
```

This is especially useful when the business name is noisy.

------------------------------------------------------------------------

## Block C --- Exact name token signature

Create:

``` text
sorted unique name tokens
```

Use an inverted index:

``` text
token_signature -> S2/S3 IDs
```

Then query from S1.

------------------------------------------------------------------------

## Block D --- Postal-code / number blocking

If a postal/PIN-like sequence is present:

``` text
same extracted postal code
```

Generate candidates.

Use this as one blocking pass, not as an unconditional requirement.

------------------------------------------------------------------------

## Block E --- Name character n-gram retrieval

Use character TF-IDF:

``` text
analyzer = char
ngram_range = (2, 5) or validated alternative
```

For each S1 name, retrieve top-K S2/S3 names.

Start with:

``` text
K = 20, 50, 100
```

and benchmark.

Use sparse matrices and batch processing.

------------------------------------------------------------------------

## Block F --- Address character n-gram retrieval

Same idea for addresses.

Use:

``` text
char TF-IDF
cosine similarity
top-K retrieval
```

Address retrieval should be independent of name retrieval.

------------------------------------------------------------------------

## Block G --- Token overlap retrieval

Use token inverted indices.

For each S1:

``` text
candidate = records sharing rare informative name/address tokens
```

Rare tokens are more useful than common tokens.

Avoid blocking on extremely common tokens such as:

``` text
india
market
restaurant
store
road
street
```

unless combined with other signals.

------------------------------------------------------------------------

## Block H --- Cross-field blocks

Examples:

``` text
same country + similar name
same postal code + name token overlap
same address number + address token overlap
same rare name token + address token overlap
```

These are especially valuable when one field is damaged.

------------------------------------------------------------------------

# 9. Candidate Union

For every S1 entity:

``` python
candidates[s1] =
    exact_name_candidates
    ∪ exact_address_candidates
    ∪ token_candidates
    ∪ postal_candidates
    ∪ name_tfidf_candidates
    ∪ address_tfidf_candidates
    ∪ cross_field_candidates
```

Keep provenance internally:

``` text
candidate_sources = {
    "exact_name",
    "tfidf_name",
    "address_tfidf",
    "postal",
    ...
}
```

This provenance becomes a powerful feature.

------------------------------------------------------------------------

# 10. Candidate Recall Experiment

Before training a complex model, measure:

``` text
candidate_recall =
    true positive pairs appearing in candidates
    /
    total true positive pairs
```

Also calculate:

``` text
average candidates per S1
median candidates per S1
95th percentile candidates per S1
maximum candidates per S1
reduction ratio
```

Target:

> Candidate recall should be pushed as high as practical before the
> matching stage, while candidate volume remains computationally
> manageable.

Do not use a candidate block with low recall merely because it is fast.

------------------------------------------------------------------------

# 11. Candidate Set Safety

The final `candidate_pairs.tsv` must contain the exact candidates passed
to the final model.

Do not do:

``` text
raw blocking output
    -> filter
    -> another filter
    -> model
```

and then write the raw blocking output.

Instead:

``` text
blocking
   ↓
candidate refinement
   ↓
FINAL candidate table
   ↓
MODEL
```

Write the table at the point immediately before model inference.

------------------------------------------------------------------------

# 12. PHASE 4 --- Training Pair Construction

Positive pairs come directly from:

``` text
train_ground_truth.tsv
```

For each S1:

``` text
positive = every listed S2/S3 match
```

Do NOT assume one-to-one matching.

The challenge explicitly allows:

``` text
zero matches
one match
many matches
```

------------------------------------------------------------------------

# 13. Hard-Negative Mining

Random negatives are not enough.

The classifier needs to distinguish:

``` text
same-looking but different business
```

Generate hard negatives from candidate pairs that are NOT in ground
truth.

Priority:

### Hard negative type 1

High name similarity but wrong address.

### Hard negative type 2

High address similarity but wrong name.

### Hard negative type 3

Same postal code but different business.

### Hard negative type 4

Same common business name but different address.

### Hard negative type 5

Same address but different business.

### Hard negative type 6

Same country + high overall similarity but incorrect entity.

### Hard negative type 7

Candidates retrieved by TF-IDF with high similarity but not true
matches.

This is essential because leaderboard errors are likely to come from
ambiguous candidates rather than obviously unrelated records.

------------------------------------------------------------------------

# 14. Negative Sampling Ratio

Start with:

``` text
positive : negative = 1 : 3
```

Then test:

``` text
1 : 5
1 : 10
```

Do not create millions of useless easy negatives if hard negatives are
available.

Use stratified negative sampling by candidate-generation source and
similarity band.

------------------------------------------------------------------------

# 15. PHASE 5 --- Pair Feature Engineering

Create one row per:

``` text
(S1 record, candidate S2/S3 record)
```

Features should fall into several groups.

------------------------------------------------------------------------

## 15.1 Name similarity features

Calculate:

``` text
normalized exact match
basic exact match
RapidFuzz ratio
RapidFuzz WRatio
RapidFuzz token_sort_ratio
RapidFuzz token_set_ratio
Jaro-Winkler if available
Levenshtein normalized similarity
character n-gram cosine
token Jaccard
token overlap coefficient
length difference
length ratio
common-token count
rare-token overlap
```

Do not rely on only one metric.

------------------------------------------------------------------------

# 16. Address similarity features

Calculate:

``` text
normalized exact match
RapidFuzz ratio
WRatio
token_sort_ratio
token_set_ratio
character n-gram cosine
token Jaccard
token overlap coefficient
common-token count
length difference
length ratio
```

Extract:

``` text
postal-code equality
house-number equality
unit-number equality
numeric-token overlap
```

Missing values must be represented explicitly.

------------------------------------------------------------------------

# 17. Cross-Field Features

Very important.

Examples:

``` text
name_exact AND address_exact
name_high AND address_high
name_high AND address_low
name_low AND address_high
both_fields_present
country_equal
country_missing
country_mismatch
name/address combined score
name/address weighted score
```

Do not assume:

``` text
country mismatch => impossible
```

unless the training validation data proves that assumption is safe.

------------------------------------------------------------------------

# 18. Blocking-Provenance Features

Add binary indicators:

``` text
blocked_exact_name
blocked_exact_address
blocked_postal
blocked_name_tfidf
blocked_address_tfidf
blocked_token
blocked_cross_field
```

Also:

``` text
number_of_blocking_routes
```

A candidate independently discovered by multiple blocking strategies is
often more credible.

------------------------------------------------------------------------

# 19. Missingness Features

Add:

``` text
name_missing_s1
name_missing_candidate
address_missing_s1
address_missing_candidate
country_missing_s1
country_missing_candidate
```

Also:

``` text
both_names_present
both_addresses_present
```

Missingness is information.

------------------------------------------------------------------------

# 20. Country Handling

Country is explicitly open-set.

Implementation must:

``` python
country = country.astype(str).str.strip().str.lower()
```

and never:

``` python
if country not in ["us", "india"]:
    discard()
```

The model should learn country agreement/disagreement as a feature.

The pipeline must automatically support:

``` text
France
US
India
any future label
```

without code changes.

------------------------------------------------------------------------

# 21. Recommended Matching Model

Primary recommendation:

## XGBoost binary classifier

Why:

-   strong on heterogeneous tabular features,
-   handles nonlinear feature interactions,
-   fast enough for candidate-level classification,
-   interpretable through feature importance,
-   Apache-2.0 licensed,
-   comfortably below the 8B parameter limit.

Alternative:

``` text
LightGBM
```

if available and benchmarked successfully.

Do not use a huge language model simply because it is available.

The challenge is primarily a structured pair-matching problem, and
candidate generation + high-quality pair features are likely to matter
more than model size.

------------------------------------------------------------------------

# 22. Model Training Design

Input:

``` text
pair features
```

Target:

``` text
1 = true match
0 = non-match
```

Use:

``` text
class imbalance handling
early stopping
validation monitoring
```

Potential starting configuration:

``` text
n_estimators: 1000
learning_rate: 0.03
max_depth: 6
subsample: 0.8
colsample_bytree: 0.8
min_child_weight: tuned
reg_alpha: tuned
reg_lambda: tuned
```

These are starting values, not final values.

Tune using validation F0.5, not generic accuracy.

------------------------------------------------------------------------

# 23. Validation Split

Use an S1-entity-level split.

Recommended starting point:

``` text
80% S1 entities → training
20% S1 entities → validation
```

Use a fixed random seed.

The split must be reproducible.

Important:

-   positives must come from the training portion,
-   negatives must be constructed consistently,
-   validation must be scored against all relevant candidates,
-   do not tune thresholds on the test set.

For a stronger internal estimate, later run several entity-level folds
if time permits.

------------------------------------------------------------------------

# 24. Implement the Exact Competition Metric

Create:

``` python
score_f05(predictions, ground_truth)
```

It must calculate:

``` text
precision
recall
F0.5
```

per Source 1 entity and then macro-average across Source 1 entities.

Do NOT calculate only global pair-level F0.5.

Singletons are explicitly part of the metric.

Test cases:

``` text
true=[]
pred=[]       -> 1.0

true=[]
pred=[x]      -> 0.0

true=[x]
pred=[]       -> 0.0

true=[x]
pred=[x]       -> 1.0
```

Also test multiple-match cases.

------------------------------------------------------------------------

# 25. Threshold Optimization

This is a major experiment.

Do not assume:

``` text
threshold = 0.5
```

For validation, sweep:

``` text
0.10
0.15
0.20
...
0.95
```

or a finer grid around the promising region.

For each threshold:

``` text
predict all candidates with score >= threshold
calculate macro F0.5
```

Choose the threshold based only on validation data.

------------------------------------------------------------------------

# 26. Why One Global Threshold May Not Be Enough

Test whether thresholds differ by:

``` text
country
candidate source
name/address availability
candidate-generation route
number of candidates
```

But only introduce segment-specific thresholds if validation proves they
generalize.

Examples:

``` text
threshold_high_confidence
threshold_normal
threshold_sparse_fields
```

Do not create many tiny segments; they overfit quickly.

------------------------------------------------------------------------

# 27. Singleton / Abstention Model

Because singleton mistakes are heavily punished, implement explicit
abstention.

For an S1 entity:

``` text
if no candidate has sufficiently strong evidence:
    return []
```

Do not force the top candidate.

This is especially important when:

``` text
max_score is low
```

or

``` text
top_score - second_score is small
```

or

``` text
only weak evidence exists
```

------------------------------------------------------------------------

# 28. Margin Features / Decision Rules

After the classifier scores candidates, calculate:

``` text
top_score
second_score
score_margin = top_score - second_score
candidate_count
```

Potential decision logic:

``` text
accept candidate if:
    score >= calibrated threshold
AND evidence is sufficiently strong
```

For singleton protection:

``` text
if max_score < singleton_threshold:
    output empty
```

Tune this threshold using validation.

------------------------------------------------------------------------

# 29. Multi-Match Handling

A Source 1 entity can have multiple true matches.

Therefore:

``` text
Do not simply select argmax.
```

Instead:

``` text
accept every candidate satisfying the final decision rule
```

while using precision-oriented safeguards.

Possible rule:

``` text
candidate_score >= T
```

plus optional:

``` text
candidate_score >= top_score - allowed_margin
```

Only introduce the margin rule if validation improves.

------------------------------------------------------------------------

# 30. S2 and S3 Handling

Treat S2 and S3 as separate candidate pools during inference.

For each S1:

``` text
score S2 candidates
score S3 candidates
```

Then combine accepted IDs.

Keep source provenance as a feature.

Validate that:

``` text
S1 never appears in output
only S2/S3 IDs appear
```

------------------------------------------------------------------------

# 31. Advanced Improvement --- Two-Stage Matcher

If time permits, implement:

## Stage 1

Fast classifier:

``` text
candidate -> probability
```

## Stage 2

Only for ambiguous/high-value candidates:

``` text
more expensive similarity features
```

This can include:

-   more RapidFuzz variants,
-   more detailed token analysis,
-   character TF-IDF similarity,
-   field agreement patterns.

Then final classifier or deterministic decision.

Do this only after the baseline is stable.

------------------------------------------------------------------------

# 32. Advanced Improvement --- Source-Aware Calibration

Check validation performance separately for:

``` text
S1 -> S2
S1 -> S3
```

If one source systematically behaves differently, test a source
indicator or source-specific calibration.

Do not assume the sources have identical noise characteristics.

------------------------------------------------------------------------

# 33. Advanced Improvement --- Graph Consistency

Only implement after the pair matcher is strong.

Construct a graph:

``` text
S1 -- candidate --> S2
S1 -- candidate --> S3
```

Use high-confidence matches to identify consistency patterns.

Example:

``` text
S1-A strongly matches S2-X
S1-A strongly matches S3-Y
```

This can increase confidence when both independent sources point to the
same S1.

However, do not allow graph propagation to create speculative matches.

Precision remains the priority.

------------------------------------------------------------------------

# 34. Advanced Improvement --- Pseudo-Labeling

Do NOT use pseudo-labeling initially.

If used at all:

1.  identify extremely high-confidence matches,
2.  verify that the rule has near-zero validation false positives,
3.  use only as a controlled experiment,
4.  compare validation F0.5.

Never let pseudo-labeling overwrite the ground truth.

------------------------------------------------------------------------

# 35. Experiment Matrix

Maintain:

``` text
experiments/experiment_log.csv
```

Columns:

``` text
experiment_id
date
blocking_version
feature_version
model_version
negative_ratio
threshold
candidate_recall
avg_candidates
validation_f05
precision
recall
singleton_accuracy
runtime
memory
notes
```

Run experiments in this order:

### E00

Rule-based exact normalized matching.

### E01

Exact + fuzzy features.

### E02

Multi-pass blocking + classifier.

### E03

Hard-negative mining.

### E04

Threshold optimization.

### E05

Singleton/abstention tuning.

### E06

Source-aware features.

### E07

TF-IDF candidate retrieval tuning.

### E08

Advanced two-stage matcher.

### E09

Optional graph consistency.

Stop when additional complexity no longer improves validation F0.5.

------------------------------------------------------------------------

# 36. Submission Strategy

The challenge permits a maximum of **5 submissions per day over 3
days**.

Treat these as controlled experiments.

Do NOT burn submissions on formatting mistakes.

Before every upload:

``` text
1. Freeze code version.
2. Generate output.
3. Run validator.
4. Inspect row count.
5. Inspect empty-match rate.
6. Inspect match-count distribution.
7. Confirm all matches are candidates.
8. Record configuration.
9. Upload.
10. Record leaderboard score.
```

------------------------------------------------------------------------

# 37. Submission Versions

Use:

``` text
submission_01_baseline.tsv
submission_02_blocking.tsv
submission_03_hard_negative.tsv
submission_04_threshold.tsv
submission_05_final_candidate.tsv
```

Keep an experiment log:

``` text
submission_id
git_commit
model
features
blocking
threshold
validation_f05
public_score
notes
```

Never overwrite historical results.

------------------------------------------------------------------------

# 38. What We Should NOT Do

## Absolutely prohibited by the challenge

Do not:

``` text
call Google Maps
call geocoding APIs
search business registrations
query commercial entity-resolution APIs
search the web for business identity information
use external business databases
perform internet-based entity augmentation
```

The supplied challenge explicitly states that external data lookup can
result in disqualification.

------------------------------------------------------------------------

# 39. What We Should NOT Trust From a Vibe-Coding Agent

Never accept an agent-generated implementation merely because it "looks
sophisticated."

Verify:

``` text
Does candidate recall actually increase?
Does validation F0.5 increase?
Does runtime remain feasible?
Does memory remain feasible?
Are singleton predictions sensible?
Are hard negatives represented?
Does the output validator pass?
Does candidate_pairs exactly correspond to model inputs?
```

A simpler measured pipeline is preferable to an impressive-looking
unmeasured pipeline.

------------------------------------------------------------------------

# 40. Memory / Scale Engineering

The challenge can contain very large test data.

Rules:

### Never

``` python
all_pairs = cartesian_product(S1, S2S3)
```

### Prefer

``` text
inverted indexes
sparse TF-IDF matrices
top-K retrieval
batch processing
compact dtypes
categorical/string compression where safe
streamed TSV processing
```

For candidate generation, process S1 in batches.

For pair features, process candidate batches.

Write intermediate artifacts in chunks.

------------------------------------------------------------------------

# 41. Large-Scale TF-IDF Retrieval

Do not calculate a giant dense similarity matrix.

Use:

``` text
scipy sparse
sklearn TfidfVectorizer
NearestNeighbors(metric="cosine")
```

or equivalent sparse top-K retrieval.

Conceptually:

``` python
vectorizer.fit(all_candidate_field_values)
matrix = vectorizer.transform(all_candidate_field_values)

for s1_batch:
    q = vectorizer.transform(s1_batch)
    similarities = q @ matrix.T
    retain only top-K
```

The implementation must be memory-profiled.

If full TF-IDF is too large:

``` text
use hashing
reduce ngram range
reduce vocabulary
retrieve separately by country/source
batch more aggressively
```

The agent must benchmark these alternatives.

------------------------------------------------------------------------

# 42. Feature Table Design

Candidate pair table:

``` text
s1_id
candidate_id
candidate_source
country_equal
name_exact
name_ratio
name_wratio
name_token_set
name_token_sort
name_jaccard
name_char_cosine
address_exact
address_ratio
address_wratio
address_token_set
address_token_sort
address_jaccard
address_char_cosine
postal_equal
number_equal
numeric_overlap
name_length_ratio
address_length_ratio
missing_name_flags
missing_address_flags
block_exact_name
block_exact_address
block_tfidf_name
block_tfidf_address
block_postal
block_token
block_route_count
candidate_count
```

Target only exists for training.

------------------------------------------------------------------------

# 43. Avoid Feature Leakage

Do not create features from:

``` text
ground-truth match IDs
```

or any information derived from labels for validation/test inference.

Allowed:

``` text
string similarity
blocking route
source
country
raw field content
derived text statistics
```

Not allowed:

``` text
"this candidate was seen as a positive for this entity"
```

during validation/test inference.

------------------------------------------------------------------------

# 44. Baseline Rule System

Before ML, build a deterministic baseline.

Examples:

``` text
if normalized_name == normalized_name_candidate
AND country compatible:
    high-confidence match

if normalized_address == normalized_address_candidate
AND name similarity high:
    high-confidence match
```

Measure its F0.5.

This gives us:

``` text
sanity check
error examples
feature expectations
```

Do not assume it will be the final system.

------------------------------------------------------------------------

# 45. Error Analysis Dashboard

Generate:

``` text
artifacts/diagnostics/
```

Reports:

### False positives

Show:

``` text
S1
candidate
predicted score
name values
address values
country
similarity features
why accepted
```

### False negatives

Show:

``` text
S1
true candidate
highest predicted score
missing blocking routes
name similarity
address similarity
```

### Singletons

Show:

``` text
S1
top candidate
top score
second score
decision
```

This is one of the fastest ways to discover threshold problems.

------------------------------------------------------------------------

# 46. Critical Diagnostic: Blocking Recall

For every validation true match:

``` text
was it generated?
```

Categorize missed true pairs:

``` text
missed_by_name
missed_by_address
missed_by_postal
missed_by_tfidf
missed_by_token
missed_by_all
```

If many true matches are:

``` text
missed_by_all
```

the normalization/blocking layer needs improvement.

Do not spend time tuning the classifier when the true pair is not even
present.

------------------------------------------------------------------------

# 47. Critical Diagnostic: Classifier Errors

If true pairs are candidates but classified incorrectly:

``` text
blocking is sufficient
feature/model layer needs improvement
```

Then investigate:

``` text
name features
address features
cross-field interaction
country
hard negatives
threshold
```

------------------------------------------------------------------------

# 48. Critical Diagnostic: Singleton Errors

If validation shows many:

``` text
true singleton -> predicted match
```

raise precision by:

``` text
higher threshold
stronger margin
stronger combined name+address evidence
better hard negatives
```

Do not blindly increase threshold if it destroys many true multi-match
entities.

------------------------------------------------------------------------

# 49. Critical Diagnostic: Over-Merging

If a single S1 receives many predictions:

``` text
S1 -> 20 candidates
```

inspect whether the model is learning common business names.

Potential solution:

``` text
increase importance of address evidence
down-weight generic name matches
add token rarity
add address-number agreement
add postal agreement
```

------------------------------------------------------------------------

# 50. Critical Diagnostic: Under-Matching

If true entities with multiple matches receive only one:

``` text
do not use argmax
```

Ensure inference accepts multiple candidates above the calibrated
decision boundary.

------------------------------------------------------------------------

# 51. Vibe Coding Prompt 01 --- Repository Setup

Paste this into the coding agent:

``` text
You are the implementation engineer for the Amazon ML Challenge 2026 Business Entity Resolution Challenge.

First inspect the entire repository and the supplied student_resource files. Do not invent columns, files, or challenge rules.

Create the production repository structure described in VIBE_IMPLEMENTATION_PLAN.md.

Requirements:
- Python 3.10+
- modular src/ architecture
- deterministic random seeds
- logging
- configuration through src/config.py
- requirements.txt
- unit-test skeleton
- no external business lookup
- no geocoding
- no internet entity augmentation
- no external entity-resolution API
- TSV input/output only
- memory-safe design for very large datasets

Before writing code, explain the proposed file structure.
Then implement it.
Then run import/static/smoke tests.
Report files changed and test results.
```

------------------------------------------------------------------------

# 52. Vibe Coding Prompt 02 --- Audit

``` text
Implement scripts/01_audit.py.

Read all train TSVs with sep='\\t'.

Produce:
- row counts
- unique IDs
- missingness
- field lengths
- country distribution
- duplicate values
- normalized exact overlap statistics
- ground-truth singleton distribution
- match-count distribution
- S2/S3 match distribution

Do not use external data.

Make the audit memory-conscious.

Save a human-readable report under artifacts/diagnostics/.

Run it and show the actual discovered statistics.

Do not proceed to model implementation until the audit completes successfully.
```

------------------------------------------------------------------------

# 53. Vibe Coding Prompt 03 --- Normalization

``` text
Implement src/normalization.py.

Create multiple representations for business_name and business_address:
- raw
- basic normalized
- alphanumeric normalized
- compact normalized
- token list
- sorted unique token representation

Implement conservative legal-suffix and abbreviation normalization using only transformations justified by the supplied challenge data.

Preserve raw fields.

Add extraction for:
- numeric tokens
- postal/PIN-like sequences
- address numbers

Do not use geocoding or external lookup.

Write unit tests for punctuation, whitespace, abbreviations, word order, missing values, and multilingual/unicode text.

Benchmark normalization speed on the supplied training data.
```

------------------------------------------------------------------------

# 54. Vibe Coding Prompt 04 --- Blocking

``` text
Implement src/blocking.py.

Build multi-pass candidate generation:
1. normalized name exact
2. normalized address exact
3. token signature
4. postal/numeric blocking
5. character TF-IDF name retrieval
6. character TF-IDF address retrieval
7. cross-field blocking

Return:
- candidate pairs
- blocking provenance
- candidate counts

The final candidate set must be exactly what the matcher will consume.

Implement batch processing and sparse retrieval. Never create an S1 x S2/S3 dense matrix.

Add validation code to measure candidate recall against training ground truth.

Report:
- candidate recall
- average candidates per S1
- p95 candidates
- max candidates
- reduction ratio
- runtime
- memory estimate

Do not optimize the classifier yet.
```

------------------------------------------------------------------------

# 55. Vibe Coding Prompt 05 --- Pair Features

``` text
Implement src/features.py.

For every candidate pair calculate:
- name exactness
- name edit/fuzzy similarities
- token Jaccard
- token overlap
- character TF-IDF similarity
- address equivalents
- postal equality
- numeric overlap
- country agreement
- missingness indicators
- field length ratios
- blocking provenance
- number of blocking routes

Use RapidFuzz/scikit-learn/scipy where appropriate.

Return a numeric feature matrix with stable column names.

No label-derived feature may enter inference.

Write unit tests and benchmark feature extraction.
```

------------------------------------------------------------------------

# 56. Vibe Coding Prompt 06 --- Hard Negatives

``` text
Implement hard-negative generation.

From the training candidate set, remove all true positive pairs.

Stratify negatives by:
- high name / low address
- low name / high address
- high name / high address but wrong entity
- same postal/numeric evidence
- exact address but wrong name
- exact/similar name but wrong address
- generic/common names
- candidate-generation route

Create a configurable negative sampling ratio.

Report the distribution of negative difficulty.

Do not use random negatives only.
```

------------------------------------------------------------------------

# 57. Vibe Coding Prompt 07 --- Model

``` text
Implement the first XGBoost pair classifier.

Use only training labels for the training split.

Use an S1-entity-level validation split.

Implement:
- reproducible split
- class imbalance handling
- early stopping
- model persistence
- feature importance
- validation prediction export

Do not optimize thresholds yet.

Return:
- validation probabilities
- feature importance
- training time
- validation runtime
```

------------------------------------------------------------------------

# 58. Vibe Coding Prompt 08 --- Exact F0.5

``` text
Implement the exact competition metric.

Requirements:
- per-S1 precision
- per-S1 recall
- per-S1 F0.5
- macro average
- singleton handling
- multiple-match handling

Explicitly test:
true empty / predicted empty
true empty / predicted nonempty
true nonempty / predicted empty
exact multi-match
partial multi-match
extra false positives

Compare implementation against manually computed examples.

Do not use micro F0.5 as the primary metric.
```

------------------------------------------------------------------------

# 59. Vibe Coding Prompt 09 --- Threshold Search

``` text
Implement threshold optimization against validation macro F0.5.

Evaluate a broad threshold grid.

For every threshold record:
- precision
- recall
- macro F0.5
- singleton correctness
- average predicted matches
- empty prediction rate

Then test score-margin and abstention rules.

Do not inspect the hidden test labels.
Do not optimize using leaderboard results as if they were validation labels.

Persist the selected configuration.
```

------------------------------------------------------------------------

# 60. Vibe Coding Prompt 10 --- Test Inference

``` text
Implement scripts/06_test_inference.py.

Load:
- trained model
- persisted normalization artifacts
- blocking artifacts/configuration
- threshold configuration

Run the full test pipeline.

Produce:
output/matching_results.tsv
output/candidate_pairs.tsv

Requirements:
- every test S1 exactly once
- empty list for no accepted matches
- no duplicate IDs
- only S2/S3 IDs
- every final match must be in candidate_pairs.tsv
- deterministic ordering
- exact tab separators

Do not add any external data.
```

------------------------------------------------------------------------

# 61. Vibe Coding Prompt 11 --- Validator and Packaging

``` text
Implement the final packaging workflow.

Run the supplied:
utils/validate_submission.py

Then verify:
- matching row count equals test S1 row count
- candidate row count equals test S1 row count
- every final match is in candidates
- no S1 ID appears as a match
- no duplicate IDs
- exact headers
- UTF-8
- tab separators

Create the final archive structure required by the challenge:
output/
code/business_entity_resolution/src/
code/business_entity_resolution/README.md
code/business_entity_resolution/requirements.txt
Documentation_template.md

Do not include secrets, caches, huge unnecessary temporary files, or external data.
```

------------------------------------------------------------------------

# 62. Vibe Coding Prompt 12 --- Final Code Review

``` text
Perform a hostile code review of the entire solution.

Pretend the submission will be audited.

Check:
- no external lookup
- no hidden data leakage
- no test-label access
- no ground-truth use during test inference
- no hard-coded US/India restriction
- no accidental S1 self-matches
- no candidate mismatch
- no non-reproducible randomness
- no invalid TSV formatting
- no dependency with an incompatible model license
- model <= 8B parameters
- requirements are pinned
- README reproduces the pipeline
- validator passes

Do not rewrite working code unnecessarily.

Produce a risk report with:
CRITICAL
HIGH
MEDIUM
LOW

Fix only real issues and rerun all tests.
```

------------------------------------------------------------------------

# 63. Final Methodology Document Structure

Fill the supplied `Documentation_template.md`.

Use this structure:

``` text
1. Executive Summary

2. Methodology
   2.1 Problem Analysis
   2.2 Solution Strategy

3. Candidate Generation
   3.1 Blocking methods
   3.2 Candidate recall
   3.3 Reduction ratio
   3.4 Candidate provenance

4. Matching Model
   4.1 Features
   4.2 Model
   4.3 Hard negatives
   4.4 Threshold selection
   4.5 Singleton handling

5. Results and Error Analysis
   5.1 Validation F0.5
   5.2 Precision
   5.3 Recall
   5.4 False positives
   5.5 False negatives
   5.6 Singleton errors

6. Conclusion

Appendix
```

Only report numbers that were actually measured.

------------------------------------------------------------------------

# 64. 72-Hour Execution Schedule

## DAY 1 --- Understand + Baseline + Candidate Recall

### Hour 0--2

Both:

``` text
download/unpack data
inspect files
run audit
verify columns
verify row counts
```

### Hour 2--5

Member 1:

``` text
normalization
blocking v1
candidate recall
```

Member 2:

``` text
metric
pair features
baseline model
```

### Hour 5--9

Integrate:

``` text
candidate generation
features
classifier
validation
```

### Hour 9--12

Run:

``` text
baseline F0.5
candidate recall
error analysis
```

### Hour 12--18

Implement:

``` text
hard negatives
TF-IDF retrieval
better address features
```

### Hour 18--24

Produce:

``` text
Experiment E00-E04
first leaderboard submission if stable
```

------------------------------------------------------------------------

# 65. DAY 2 --- Optimization Day

Primary goal:

``` text
increase validation F0.5
```

Priority order:

1.  blocking recall
2.  hard negatives
3.  name features
4.  address features
5.  threshold
6.  singleton logic
7.  source-aware features
8.  advanced matcher

Do not jump directly to deep learning.

------------------------------------------------------------------------

# 66. DAY 2 Submission Discipline

Before each submission:

``` text
validation improved?
YES -> candidate submission
NO  -> do not submit unless it tests a distinct hypothesis
```

Keep at most the challenge's allowed submission count.

Record every submission.

Do not chase leaderboard noise without a corresponding validation
hypothesis.

------------------------------------------------------------------------

# 67. DAY 3 --- Stabilize + Audit + Final

### First phase

Freeze the strongest validated architecture.

### Second phase

Run full test inference.

### Third phase

Run validator.

### Fourth phase

Inspect:

``` text
match count distribution
empty percentage
candidate count distribution
extreme candidate counts
```

### Fifth phase

Package.

### Final phase

Do not make speculative last-minute architectural changes.

------------------------------------------------------------------------

# 68. Final Pre-Submission Checklist

## Data

``` text
[ ] All test S1 records loaded
[ ] France works without special-case code
[ ] No hard-coded country list
[ ] Raw data untouched
```

## Blocking

``` text
[ ] Candidate recall measured
[ ] Candidate set is final model input
[ ] No all-pairs explosion
[ ] Candidate provenance retained
```

## Model

``` text
[ ] Model license compliant
[ ] <= 8B parameters
[ ] No external data
[ ] Hard negatives used
[ ] Threshold selected using validation
```

## Metric

``` text
[ ] Exact macro F0.5
[ ] Singleton handling
[ ] Multiple matches
[ ] Empty predictions
```

## Output

``` text
[ ] matching_results.tsv
[ ] candidate_pairs.tsv
[ ] exact headers
[ ] tabs, not commas
[ ] every S1 exactly once
[ ] no duplicate matches
[ ] only S2/S3 match IDs
[ ] final matches subset of candidates
```

## Package

``` text
[ ] output/
[ ] code/business_entity_resolution/src/
[ ] README.md
[ ] requirements.txt
[ ] Documentation_template.md
[ ] reproducible pipeline
[ ] no secrets
[ ] no external datasets
```

------------------------------------------------------------------------

# 69. Final Architecture

The final system should look like:

``` text
                    TRAIN
                      |
       +--------------+---------------+
       |                              |
       v                              v
  Ground Truth                    Raw Records
       |                              |
       |                              v
       |                       Normalization
       |                              |
       |                              v
       |                       Multi-Pass Blocking
       |                              |
       |                              v
       |                       Candidate Pairs
       |                              |
       +---------------> Hard Negative Mining
                                      |
                                      v
                              Feature Engineering
                                      |
                                      v
                                XGBoost Model
                                      |
                                      v
                              Validation Scores
                                      |
                         +------------+-------------+
                         |                          |
                         v                          v
                  Threshold Search          Error Analysis
                         |                          |
                         +------------+-------------+
                                      |
                                      v
                               Frozen Pipeline
                                      |
                                      v
                                    TEST
                                      |
                                      v
                             Normalization
                                      |
                                      v
                              Multi-Pass Blocking
                                      |
                                      v
                              Candidate Pairs
                                      |
                                      v
                              Pair Features
                                      |
                                      v
                                XGBoost Scores
                                      |
                                      v
                         Threshold + Abstention
                                      |
                                      v
                            Multi-Match Output
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
          matching_results.tsv               candidate_pairs.tsv
                     |                                 |
                     +----------------+----------------+
                                      |
                                      v
                              Official Validator
                                      |
                                      v
                                  Submission
```

------------------------------------------------------------------------

# 70. The Main Optimization Loop

This is the loop the team should follow throughout the hackathon:

``` text
OBSERVE
   ↓
Measure candidate recall
   ↓
Measure validation F0.5
   ↓
Inspect false positives / negatives
   ↓
Identify ONE bottleneck
   ↓
Implement ONE change
   ↓
Run validation
   ↓
Keep only if evidence improves
   ↓
Record experiment
   ↓
Repeat
```

Never:

``` text
change 10 things
    ↓
score changes
    ↓
we don't know why
```

------------------------------------------------------------------------

# 71. Priority Ladder

If time becomes limited, use this exact priority:

``` text
P0  Data correctness
P0  Submission validator
P0  Exact macro F0.5
P0  Candidate recall

P1  Strong normalization
P1  Multi-pass blocking
P1  Hard negatives
P1  Name + address features
P1  Threshold optimization
P1  Singleton protection

P2  Source-aware calibration
P2  More sophisticated retrieval
P2  Two-stage matcher

P3  Graph consistency
P3  Experimental pseudo-labeling
P3  Anything that does not show validation improvement
```

------------------------------------------------------------------------

# 72. The Most Important Technical Principle

The winning-oriented engineering mindset should be:

> **Do not optimize the model before proving that the true match reaches
> the model. Do not optimize recall after candidate generation has
> already saturated. Do not optimize generic accuracy when the
> competition metric is macro F0.5. Do not force a match when the
> evidence is weak.**

In practical terms:

``` text
Blocking determines the recall ceiling.
Features determine how well ambiguous candidates are separated.
Hard negatives teach the model what false merges look like.
Thresholding determines the precision/recall operating point.
Singleton handling protects the macro score.
Validation determines whether an idea is actually useful.
```

------------------------------------------------------------------------

# 73. First Things To Do Immediately

Run these in order:

``` bash
# 1. inspect environment
python --version

# 2. create virtual environment
python -m venv .venv

# 3. activate
# Windows:
.venv\Scripts\activate

# 4. install baseline dependencies
pip install pandas numpy scipy scikit-learn rapidfuzz xgboost joblib psutil

# 5. run data audit
python scripts/01_audit.py

# 6. build normalization
python scripts/02_build_features.py

# 7. build candidate sets
python scripts/03_make_candidates.py

# 8. train baseline
python scripts/04_train.py

# 9. validate
python scripts/05_validate.py

# 10. only after validation is stable:
python scripts/06_test_inference.py
```

The exact dependency versions should be pinned after the environment is
confirmed working.

------------------------------------------------------------------------

# 74. Final Principle

The objective is not to build the most complicated ML system.

The objective is to build the **highest-scoring reproducible system that
can be demonstrated from the supplied training data without violating
the rules**.

A strong final submission should be:

``` text
fast
memory-conscious
candidate-recall-aware
precision-aware
hard-negative-trained
threshold-calibrated
singleton-aware
reproducible
auditable
license-compliant
```

and every important design choice should have an experiment behind it.

# Engineering Plan: 0.99+ Local Entity-Level Macro F0.5

## Current State
- **Holdout Macro F0.5**: 0.5044
- **Candidate Recall**: 73.72% (high_recall profile)
- **Singleton F0.5**: 0.1184 (CATASTROPHIC — was 0.9551 in M7)
- **Multi-Match F0.5**: 0.5390
- **Precision**: 0.5381, **Recall**: 0.4797
- **Leaderboard Control**: 0.690088

## Root Cause Analysis

### 1. Singleton Detection Collapse (Priority 1 — Easy Win ~+0.04)
- M7 had singleton_f05 = 0.9551 → M10 has 0.1184
- ~245 singletons in holdout × (1.0 - 0.0) = 245 points lost
- The model is PREDICTING matches for almost all singletons
- Fix: Better singleton abstention using model calibration + confidence gating

### 2. Low Candidate Recall (Priority 2 — Medium effort ~+0.10–0.15)  
- Only 73.72% of true matches are in candidate set
- **35.8% of missed pairs are cross-script** (Latin S1 vs Indic target)
- **34.7% are spelling corruptions** 
- Need: TF-IDF retrieval, phonetic blocking, transliteration

### 3. Weak Classifier (Priority 3 — High effort ~+0.20)
- Model only achieves 0.54 F0.5 at matched candidates
- Need: Better features, LightGBM/XGBoost ensemble, calibration
- Need: Aggressive hard-negative mining at scale

## Action Plan

### Phase 1: Fix Singleton Detection (Expected: 0.50→0.56+)
- Restore proper singleton abstention from M7 logic
- Use max prediction score per entity as gating signal
- If max_score < threshold → predict empty (singleton)
- Sweep threshold on DEV to maximize F0.5

### Phase 2: Ultra-High Recall Blocking (Expected: 73%→95%+)
- Add TF-IDF cosine similarity retrieval (character 3-6 gram, top-K=50 per query)
- Add phonetic/Soundex blocking keys
- Add word n-gram overlap blocking (2-gram containment)
- Add transliteration-aware blocking (Indic→Latin mapping)
- Increase max_total_candidates to 500+
- Add address TF-IDF retrieval route

### Phase 3: Feature Engineering (Expected: +0.15)
- Add TF-IDF cosine similarity scores as continuous features
- Add Jaro-Winkler similarity
- Add normalized Damerau-Levenshtein
- Add containment ratio (longer name contains shorter)
- Add first-word match feature  
- Add numeric token set overlap features
- Add country-specific normalization features

### Phase 4: Model Training (Expected: +0.10)
- Train XGBoost with proper class weights
- Train LightGBM as secondary model
- Use stacking ensemble
- Calibrate probabilities with isotonic regression
- F0.5-aware threshold optimization

### Phase 5: Decision Engine Tuning (Expected: +0.05)
- Per-entity confidence calibration
- Optimal F0.5 threshold sweep per country
- Multi-match vs singleton decision boundary

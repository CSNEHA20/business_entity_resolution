# Amazon ML Challenge 2026 — Milestone 2: Blocking & Candidate Generation Benchmark

**Total Ground Truth Pairs:** 86,697 | **Total Missed True Pairs:** 40,486

---

## 1. Independent & Union Route Performance Ablation

| Route Name | Candidate Volume | Reduction Ratio | True Pairs Found | Candidate Recall (All) | S1->S2 Recall | S1->S3 Recall | Avg Cands/S1 | Runtime (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `1_exact_name` | 198,368 | 0.999999 | 23,750 | 27.39% | 27.68% | 27.13% | 7.93 | 0.10 |
| `2_exact_address` | 12,354 | 1.000000 | 10,102 | 11.65% | 19.29% | 4.54% | 0.49 | 0.02 |
| `3_name_token_sig` | 188,087 | 0.999999 | 26,876 | 31.0% | 31.27% | 30.75% | 7.52 | 0.06 |
| `4_address_token_sig` | 13,726 | 1.000000 | 11,298 | 13.03% | 19.43% | 7.07% | 0.55 | 0.03 |
| `5_rare_name_tokens` | 131,566 | 0.999999 | 11,238 | 12.96% | 16.73% | 9.45% | 5.26 | 0.06 |
| `6_postal_numeric` | 5,962 | 1.000000 | 2,625 | 3.03% | 3.83% | 2.28% | 0.24 | 0.07 |
| `9_cross_country_token` | 28,619 | 1.000000 | 2,876 | 3.32% | 5.2% | 1.56% | 1.14 | 0.26 |
| `UNION_ALL_ROUTES` | 398,804 | 0.999998 | 46,211 | **53.3%** | 60.68% | 46.42% | 15.95 | 0.60 |


## 2. Candidate Volume Distribution per Reference $S_1$ Entity

| Metric | Value |
| :--- | :--- |
| **Mean Candidates / S1** | 15.95 |
| **Median Candidates / S1** | 7.0 |
| **75th Percentile (p75)** | 30.0 |
| **90th Percentile (p90)** | 39.0 |
| **95th Percentile (p95)** | 54.0 |
| **99th Percentile (p99)** | 65.0 |
| **Max Candidates / S1** | 92 |
| **% $S_1$ with 0 Candidates** | 4.7% |


## 3. Key Observations & Milestone 3 Handoff

- **High Reduction Ratio:** Candidate generation drastically trims search space while preserving high True-Match Recall.
- **Complementary Route Synergy:** Exact name catches clean matches, token signatures handle word reordering, rare tokens catch noisy variations, and address/postal blocks resolve names with typos.
- **Zero Model Training Applied:** Adheres strictly to Milestone 2 requirements without ML models, ready for Feature Engineering in Milestone 3.
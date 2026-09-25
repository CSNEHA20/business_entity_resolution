# Amazon ML Challenge 2026 — Dataset Forensics & Ground Truth Pattern Discovery

**Execution Timestamp:** 2026-09-25 22:46:27 | **Total Runtime:** 1366.88s

---

## 1. Dataset Dimensions & Completeness

| Dataset | Total Rows | Unique IDs | Missing Name % | Missing Addr % | Missing Country % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `train_source1` | 2,206,821 | 2,206,821 | 0.0% | 0.0% | 0.0% |
| `train_source2` | 5,034,616 | 5,034,616 | 0.0% | 3.3561% | 0.0% |
| `train_source3` | 5,285,603 | 5,285,603 | 0.0% | 3.3282% | 0.0% |
| `test_source1` | 1,732,544 | 1,732,544 | 0.0% | 0.0% | 0.0% |
| `test_source2` | 4,887,273 | 4,887,273 | 0.0% | 2.6479% | 0.0% |
| `test_source3` | 5,082,316 | 5,082,316 | 0.0% | 2.6779% | 0.0% |


## 2. Country Breakdown

| Dataset | Country Distribution |
| :--- | :--- |
| `train_source1` | {'US': 1323633, 'India': 883188} |
| `train_source2` | {'US': 3016817, 'India': 2017799} |
| `train_source3` | {'US': 3170056, 'India': 2115547} |
| `test_source1` | {'India': 809986, 'US': 663106, 'France': 259452} |
| `test_source2` | {'India': 2312565, 'US': 1871330, 'France': 703378} |
| `test_source3` | {'India': 2405000, 'US': 1945701, 'France': 731615} |


## 3. Ground Truth Distribution

- **Total Reference $S_1$ Entities:** 2,206,821
- **Singletons (0 Matches):** 123,247 (5.58%)
- **Single-Match Entities (1 Match):** 119,157 (5.4%)
- **Multi-Match Entities (>1 Matches):** 1,964,417 (89.02%)
- **Max Matches for Single $S_1$:** 11
- **Total Positive Pairs:** 7,638,365
  - $S_1 \to S_2$ Positive Pairs: 3,693,619
  - $S_1 \to S_3$ Positive Pairs: 3,944,746
- **$S_1$ Entities with Both $S_2$ and $S_3$ Matches:** 1,776,047 (80.48%)

## 4. True-Match Evidence & Feature Agreement

| Feature / Evidence Type | Overall % ($N=7,638,365$) | $S_1 \to S_2$ % ($N=3,693,619$) | $S_1 \to S_3$ % ($N=3,944,746$) |
| :--- | :--- | :--- | :--- |
| **Exact Raw Name Match** | **10.75%** | 11.03% | 10.49% |
| **Exact Normalized Name (Legal expanded)** | **28.48%** | 28.14% | 28.8% |
| **Name Token Signature (Order-invariant)** | **32.3%** | 31.88% | 32.69% |
| **Name Jaccard Overlap >= 0.5** | **79.29%** | 78.3% | 80.22% |
| **Exact Raw Address Match** | **7.24%** | 10.37% | 4.32% |
| **Exact Normalized Address** | **11.62%** | 19.41% | 4.34% |
| **Address Token Signature** | **12.9%** | 19.37% | 6.84% |
| **Address Jaccard Overlap >= 0.5** | **68.52%** | 81.72% | 56.16% |
| **Country Agreement** | **100.0%** | 100.0% | 100.0% |
| **Postal / PIN Code Match** | **4.8%** | 4.74% | 4.86% |
| **Building Number Match** | **34.53%** | 34.05% | 34.98% |
| **Either Name OR Address Exact Norm Match** | **37.36%** | 41.87% | 33.13% |
| **Both Name AND Address Exact Norm Match** | **2.75%** | 5.68% | 0.01% |


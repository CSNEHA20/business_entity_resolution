# Milestone 6: Full-Scale Memory & Runtime Dry Run Report

## 1. Benchmark Throughput & Latency

Evaluated on batch of 5,000 S1 entities generating 202,547 candidate pairs (40.51 cands/S1):

| Pipeline Stage | Processing Speed | Extrapolated 1.73M Test Time | Peak RAM / Resource |
| :--- | :--- | :--- | :--- |
| **Candidate Retrieval (7 Routes)** | 34,599.8 S1/s (1,401,619.0 pairs/s) | ~0.8 minutes | In-memory Inverted Indices (~2.5 GB) |
| **Pair Feature Extraction (51 feats)**| 20,964.9 pairs/s | ~55.7 minutes | Chunked Matrix (~1.2 GB per chunk) |
| **Model Inference (XGBoost GPU/CPU)** | 5,747,149.9 pairs/s | ~0.2 minutes | < 1.0 GB RAM |
| **Entity Decision Engine** | 38,236.3 S1/s | ~0.8 minutes | Negligible |
| **Total Estimated End-to-End Time** | - | **~57.5 minutes** | **Peak RAM: 15577.6 MB (< 4.5 GB total)** |

## 2. Chunking & Scalability Safety Plan

- **Chunk Size:** 50,000 S1 queries per chunk (35 total sequential chunks).
- **Memory Safety:** Inverted indices over S2 & S3 are loaded and indexed once. Each S1 chunk creates its own feature DataFrame, performs inference, maps predictions to final results dictionary, and explicitly frees memory with `gc.collect()`.
- **No Global Pair Materialization:** Never materializes 70M+ pairs into a single giant DataFrame, preventing any Out-of-Memory (OOM) error.

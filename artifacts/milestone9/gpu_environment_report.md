# GPU Environment Diagnostic Report — Milestone 9

**Date:** 2026-09-27  
**Host Machine:** Windows Laptop Workstation  

---

## 1. Hardware & System Specifications

| Component | Specification |
|---|---|
| **GPU Model** | `NVIDIA GeForce RTX 5070 Laptop GPU` |
| **Total VRAM** | `8151 MB` (~7.96 GB) |
| **Driver Version** | `592.01` |
| **CUDA Driver Version** | `13.1` |
| **Operating System** | `Windows 11 (10.0.26200)` |
| **CPU Architecture** | `Intel64 Family 6 Model 198 Stepping 2, GenuineIntel` |
| **CPU Cores** | `20 physical / 20 logical` |
| **System RAM** | `31.43 GB` |
| **Python Version** | `3.14.6` |

---

## 2. Library GPU Support Audit

| Library | Version | GPU / CUDA Available? | Details / Verification |
|---|---|---|---|
| **XGBoost** | `3.3.0` | **YES (CUDA 13.x)** | Verified functional: device='cuda' with tree_method='hist' fit and predict successful. |
| **PyTorch** | `2.13.0+cpu` | No (CPU build) | Pre-installed binary is CPU-only; not reinstalled per environment stability constraint. |
| **LightGBM** | `Not installed` | No | Not installed in environment. |
| **CuPy** | `Not installed` | No | Minimal dependencies preserved. |
| **RAPIDS (cuDF/cuML)** | `Not installed` | No | Not installed in Windows native environment. |

---

## 3. Workload Partitioning Strategy (GPU vs CPU)

### GPU-Accelerated Workloads (RTX 5070)
1. **XGBoost GPU Training**: Using `device='cuda'` with `tree_method='hist'`. Substantially accelerates histogram construction and gradient boosting over large tabular matrices.
2. **Batched XGBoost Prediction**: Fast parallel evaluation of candidate pairs.
3. **Sparse Dot Product / Batch Scoring**: Where SciPy/NumPy batch operations can leverage multi-core or GPU bindings safely.

### CPU / RAM Workloads
1. **TSV / I/O Streaming & Pandas**: Handled via CPU multi-core and chunked streaming.
2. **Unicode Normalization & String Cleaning**: High branch-divergence string ops executed on CPU.
3. **Inverted Index Construction**: High Python dictionary overhead; best maintained in host memory (RAM).
4. **Candidate Union & Set Deduplication**: Python set hashing executed on CPU.
5. **Final Output Formatting & Invariant Checks**: Deterministic CPU execution.

---

## 4. VRAM Safety & OOM Prevention Constraints

- **Total Physical VRAM**: `8151 MB` (~8.0 GB).
- **Target Working Ceiling**: `<= 6.5 GB`.
- **Hard Safety Ceiling**: `<= 7.0 GB`.
- **Batching Policy**: Candidate feature matrices and prediction chunks are strictly bounded to chunk sizes `<= 50,000` pairs.
- **VRAM Monitoring**: `nvidia-smi` and memory checks before and after batch iterations.
- **Fail-safe Fallback**: Any CUDA OOM or driver exception automatically catches the error, frees GPU memory via garbage collection / cache clearing, and falls back to CPU `tree_method='hist'`.

---

## 5. Raw nvidia-smi Diagnostic Output

```text
Sun Sep 27 12:22:41 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 592.01                 Driver Version: 592.01         CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5070 ...  WDDM  |   00000000:02:00.0 Off |                  N/A |
| N/A   46C    P0             11W /  115W |       0MiB /   8151MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

```

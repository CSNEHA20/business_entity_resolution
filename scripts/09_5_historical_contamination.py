from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
import os

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "train"
OUT_DIR = ROOT_DIR / "artifacts" / "milestone9_5"

print("Loading data for contamination audit...")
s1_path = DATA_DIR / "train_source1.tsv"
gt_path = DATA_DIR / "train_ground_truth.tsv"

s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
s1_ids = s1_df["entity_id"].values

gt_map = defaultdict(set)
for sid, m_str in zip(gt_df["source1_entity_id"].values, gt_df["matched_entity_ids"].values):
    sid = str(sid).strip()
    m_str = str(m_str).strip()
    if m_str:
        for mid in m_str.split(","):
            mid = mid.strip()
            if mid:
                gt_map[sid].add(mid)

s1_bins = np.array([0 if len(gt_map.get(sid, [])) == 0 else (1 if len(gt_map.get(sid, [])) == 1 else 2) for sid in s1_ids], dtype=np.int32)

# M9 splits:
rng9 = np.random.RandomState(42)
val_indices_9, mining_indices_9, train_indices_9 = [], [], []
for b in [0, 1, 2]:
    b_idx = np.where(s1_bins == b)[0]
    rng9.shuffle(b_idx)
    n_val = int(len(b_idx) * 0.20)
    n_mining = int(len(b_idx) * 0.10)
    val_indices_9.extend(b_idx[:n_val])
    mining_indices_9.extend(b_idx[n_val : n_val + n_mining])
    train_indices_9.extend(b_idx[n_val + n_mining :])
rng9.shuffle(val_indices_9)
rng9.shuffle(mining_indices_9)
rng9.shuffle(train_indices_9)

m9_dev_ids = set([str(s1_ids[i]).strip() for i in val_indices_9[:5000]])
m9_holdout_ids = set([str(s1_ids[i]).strip() for i in val_indices_9[5000:10000]])

# 09_blocking_reconstruction.py eval set:
rng_br = np.random.RandomState(42)
val_br = []
for b in [0, 1, 2]:
    b_idx = np.where(s1_bins == b)[0]
    rng_br.shuffle(b_idx)
    n_val = int(len(b_idx) * 0.20)
    val_br.extend(b_idx[:n_val])
rng_br.shuffle(val_br)
br_eval_ids = set([str(s1_ids[i]).strip() for i in val_br[:5000]])

# 09_fast_blocking_benchmark.py head(5000):
fast_head_ids = set([str(sid).strip() for sid in s1_ids[:5000]])

# M5 / M6 Dev and Holdout:
rng5 = np.random.RandomState(42)
val_indices_5, mining_indices_5, train_indices_5 = [], [], []
for b in [0, 1, 2]:
    b_idx = np.where(s1_bins == b)[0]
    rng5.shuffle(b_idx)
    n_val = int(len(b_idx) * 0.20)
    n_mining = int(len(b_idx) * 0.10)
    val_indices_5.extend(b_idx[:n_val])
    mining_indices_5.extend(b_idx[n_val:n_val + n_mining])
    train_indices_5.extend(b_idx[n_val + n_mining:])
rng5.shuffle(train_indices_5)
rng5.shuffle(mining_indices_5)
rng5.shuffle(val_indices_5)
m5_dev_ids = set([str(s1_ids[i]).strip() for i in val_indices_5[:10000]])
m5_holdout_ids = set([str(s1_ids[i]).strip() for i in val_indices_5[10000:20000]])

# M7 Dev and Holdout:
s1_all_ids_list = s1_df["entity_id"].astype(str).str.strip().tolist()
rng7 = np.random.RandomState(42)
shuffled_ids_7 = rng7.permutation(s1_all_ids_list)
m7_dev_ids = set([str(x).strip() for x in shuffled_ids_7[:10000]])
m7_holdout_ids = set([str(x).strip() for x in shuffled_ids_7[10000:20000]])

# Milestone 4 Val (441k entities):
# Notice 20% of 2.2M is 441,363 entities.
rng4 = np.random.RandomState(42)
val_indices_4 = []
for b in [0, 1, 2]:
    b_idx = np.where(s1_bins == b)[0]
    rng4.shuffle(b_idx)
    n_val = int(len(b_idx) * 0.20)
    val_indices_4.extend(b_idx[:n_val])
m4_val_ids = set([str(s1_ids[i]).strip() for i in val_indices_4])

print(f"M9 Holdout size: {len(m9_holdout_ids)}")
print(f"1. Overlap with 09_blocking_reconstruction eval set (n-gram K selection, blocking design): {len(m9_holdout_ids & br_eval_ids)}")
print(f"2. Overlap with 09_fast_blocking_benchmark head(5000): {len(m9_holdout_ids & fast_head_ids)}")
print(f"3. Overlap with M5 Dev (decision engine optimization, threshold selection): {len(m9_holdout_ids & m5_dev_ids)}")
print(f"4. Overlap with M6 Dev (audit and validation): {len(m9_holdout_ids & m5_dev_ids)}")
print(f"5. Overlap with M5/M6 Holdout: {len(m9_holdout_ids & m5_holdout_ids)}")
print(f"6. Overlap with M7 Dev: {len(m9_holdout_ids & m7_dev_ids)}")
print(f"7. Overlap with M7 Holdout: {len(m9_holdout_ids & m7_holdout_ids)}")
print(f"8. Overlap with M4 Val: {len(m9_holdout_ids & m4_val_ids)} (all {len(m9_holdout_ids)} came from the 20% validation pool)")

# Check if any M9 Holdout IDs are in any CSV files or logs in experiments/ or artifacts/
print("\nScanning CSV and log files in artifacts/ and experiments/...")
sample_m9_holdout = set(list(m9_holdout_ids)[:500])
found_in_files = defaultdict(list)

for folder in ["artifacts", "experiments"]:
    for root, dirs, files in os.walk(ROOT_DIR / folder):
        for f in files:
            if f.endswith((".csv", ".tsv", ".txt", ".json", ".log", ".md")):
                fpath = Path(root) / f
                try:
                    content = fpath.read_text(encoding="utf-8", errors="ignore")
                    matches = [sid for sid in list(m9_holdout_ids)[:200] if sid in content]
                    if matches:
                        found_in_files[str(fpath.relative_to(ROOT_DIR))].extend(matches)
                except Exception as e:
                    pass

for fpath, matches in found_in_files.items():
    print(f"Found {len(matches)} sample M9 holdout IDs in {fpath}: {matches[:5]}")

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "train"
OUT_DIR = ROOT_DIR / "artifacts" / "milestone9_5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading S1 and GT...")
s1_path = DATA_DIR / "train_source1.tsv"
gt_path = DATA_DIR / "train_ground_truth.tsv"

s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)

s1_ids = s1_df["entity_id"].values
total_s1 = len(s1_ids)
print(f"Total S1: {total_s1}")

# Build GT mapping
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

# 1. Milestone 4 Validation Split
rng4 = np.random.RandomState(42)
val_indices_4 = []
train_indices_4 = []
for b in [0, 1, 2]:
    b_idx = np.where(s1_bins == b)[0]
    rng4.shuffle(b_idx)
    n_val = int(len(b_idx) * 0.20)
    val_indices_4.extend(b_idx[:n_val])
    train_indices_4.extend(b_idx[n_val:])
val_indices_4 = np.array(val_indices_4, dtype=np.int32)
rng4.shuffle(val_indices_4)
m4_val_ids = [str(s1_ids[i]).strip() for i in val_indices_4]

# 2 & 3. Milestone 5 Dev-Val and Holdout-Val
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
val_indices_5 = np.array(val_indices_5, dtype=np.int32)
mining_indices_5 = np.array(mining_indices_5, dtype=np.int32)
train_indices_5 = np.array(train_indices_5, dtype=np.int32)
rng5.shuffle(train_indices_5)
rng5.shuffle(mining_indices_5)
rng5.shuffle(val_indices_5)

m5_dev_val_ids = [str(s1_ids[i]).strip() for i in val_indices_5[:10000]]
m5_holdout_val_ids = [str(s1_ids[i]).strip() for i in val_indices_5[10000:20000]]

# 4. Milestone 6 Holdout (Identical split logic to M5)
m6_holdout_ids = list(m5_holdout_val_ids)

# 5. Milestone 7 Holdout
s1_all_ids_list = s1_df["entity_id"].astype(str).str.strip().tolist()
rng7 = np.random.RandomState(42)
shuffled_ids_7 = rng7.permutation(s1_all_ids_list)
m7_holdout_ids = [str(x).strip() for x in shuffled_ids_7[10000:20000]]

# 6 & 7. Milestone 9 Dev and Holdout, plus Train and Mining
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

m9_dev_ids = [str(s1_ids[i]).strip() for i in val_indices_9[:5000]]
m9_holdout_ids = [str(s1_ids[i]).strip() for i in val_indices_9[5000:10000]]
m9_train_ids = [str(s1_ids[i]).strip() for i in train_indices_9[:12000]]
m9_mining_ids = [str(s1_ids[i]).strip() for i in mining_indices_9[:4000]]

def get_hashes_and_stats(id_list):
    first_id = id_list[0] if id_list else ""
    first_hash = hashlib.sha256(first_id.encode("utf-8")).hexdigest()
    sorted_ids = sorted(id_list)
    full_joined = "\n".join(sorted_ids)
    full_hash = hashlib.sha256(full_joined.encode("utf-8")).hexdigest()
    min_id = sorted_ids[0] if sorted_ids else ""
    max_id = sorted_ids[-1] if sorted_ids else ""
    return len(id_list), first_hash, full_hash, min_id, max_id

splits = [
    ("Milestone-4 validation", m4_val_ids),
    ("Milestone-5 Dev-Val", m5_dev_val_ids),
    ("Milestone-5 Holdout-Val", m5_holdout_val_ids),
    ("Milestone-6 Holdout", m6_holdout_ids),
    ("Milestone-7 Holdout", m7_holdout_ids),
    ("Milestone-9 Dev", m9_dev_ids),
    ("Milestone-9 Holdout", m9_holdout_ids),
]

manifest_rows = []
for name, ids in splits:
    cnt, f_h, full_h, min_i, max_i = get_hashes_and_stats(ids)
    manifest_rows.append({
        "split_name": name,
        "S1_count": cnt,
        "first_ID_hash": f_h,
        "full_ID_set_hash": full_h,
        "min_ID": min_i,
        "max_ID": max_i,
    })

manifest_df = pd.DataFrame(manifest_rows)
manifest_path = OUT_DIR / "validation_split_manifest.csv"
manifest_df.to_csv(manifest_path, index=False)
print(f"Saved manifest to {manifest_path}")
print(manifest_df.to_string())

# PART 2: SPLIT OVERLAP AUDIT
# Compute pairwise intersection between TRAIN, MINING, DEV, HOLDOUT (M9)
m9_splits = {
    "TRAIN": set(m9_train_ids),
    "MINING": set(m9_mining_ids),
    "DEV": set(m9_dev_ids),
    "HOLDOUT": set(m9_holdout_ids),
}

overlap_matrix_rows = []
split_names = ["TRAIN", "MINING", "DEV", "HOLDOUT"]
for s1_n in split_names:
    row = {"split": s1_n}
    for s2_n in split_names:
        intersect_cnt = len(m9_splits[s1_n] & m9_splits[s2_n])
        row[s2_n] = intersect_cnt
    overlap_matrix_rows.append(row)

overlap_df = pd.DataFrame(overlap_matrix_rows)
overlap_path = OUT_DIR / "split_overlap_matrix.csv"
overlap_df.to_csv(overlap_path, index=False)
print(f"\nSaved split overlap matrix to {overlap_path}")
print(overlap_df.to_string())

# Required specific pairs:
# TRAIN ∩ DEV, TRAIN ∩ HOLDOUT, TRAIN ∩ MINING, DEV ∩ MINING, DEV ∩ HOLDOUT, MINING ∩ HOLDOUT
req_pairs = [
    ("TRAIN", "DEV"),
    ("TRAIN", "HOLDOUT"),
    ("TRAIN", "MINING"),
    ("DEV", "MINING"),
    ("DEV", "HOLDOUT"),
    ("MINING", "HOLDOUT"),
]
print("\nRequired Pairwise Intersections:")
for a, b in req_pairs:
    inter = len(m9_splits[a] & m9_splits[b])
    print(f"{a} intersect {b} = {inter}")

# Check target-record leakage across M9 splits
print("\nChecking target record leakage across M9 splits...")
targets_per_split = {}
for s_n, id_set in m9_splits.items():
    t_set = set()
    for sid in id_set:
        t_set.update(gt_map.get(sid, set()))
    targets_per_split[s_n] = t_set
    print(f"Target count for {s_n}: {len(t_set)}")

for a, b in req_pairs:
    t_inter = len(targets_per_split[a] & targets_per_split[b])
    print(f"Target intersection {a} intersect {b} = {t_inter}")

# PART 3: COMPARE M9 HOLDOUT AGAINST M6 HOLDOUT
m9_h_set = set(m9_holdout_ids)
m6_h_set = set(m6_holdout_ids)
m5_dev_set = set(m5_dev_val_ids)

inter_m9_m6 = len(m9_h_set & m6_h_set)
m9_minus_m6 = len(m9_h_set - m6_h_set)
m6_minus_m9 = len(m6_h_set - m9_h_set)
pct_overlap = (inter_m9_m6 / len(m9_h_set)) * 100

print(f"\nPART 3 Results:")
print(f"M9_HOLDOUT count: {len(m9_h_set)}")
print(f"M6_HOLDOUT count: {len(m6_h_set)}")
print(f"M9_HOLDOUT intersect M6_HOLDOUT: {inter_m9_m6} ({pct_overlap:.2f}%)")
print(f"M9_HOLDOUT - M6_HOLDOUT: {m9_minus_m6}")
print(f"M6_HOLDOUT - M9_HOLDOUT: {m6_minus_m9}")

# Check where M9 HOLDOUT actually came from!
inter_m9h_m5dev = len(m9_h_set & m5_dev_set)
print(f"M9_HOLDOUT intersect M5/M6_DEV: {inter_m9h_m5dev} (out of {len(m9_h_set)} - {(inter_m9h_m5dev/len(m9_h_set))*100:.2f}%)")


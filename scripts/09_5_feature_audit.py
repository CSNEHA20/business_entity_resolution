import pandas as pd
from pathlib import Path

OUT_DIR = Path("artifacts/milestone9_5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 51 Baseline Features
baseline_features = [
    # Name
    ("name_exact_raw", "name", True, True, False, False, "pairwise", True, True),
    ("name_exact_norm", "name", True, True, False, False, "pairwise", True, True),
    ("name_fuzz_ratio", "name", True, True, False, False, "pairwise", True, True),
    ("name_fuzz_wratio", "name", True, True, False, False, "pairwise", True, True),
    ("name_fuzz_token_sort", "name", True, True, False, False, "pairwise", True, True),
    ("name_fuzz_token_set", "name", True, True, False, False, "pairwise", True, True),
    ("name_fuzz_partial_ratio", "name", True, True, False, False, "pairwise", True, True),
    ("name_levenshtein_sim", "name", True, True, False, False, "pairwise", True, True),
    ("name_jaccard_token", "name", True, True, False, False, "pairwise", True, True),
    ("name_overlap_token", "name", True, True, False, False, "pairwise", True, True),
    ("name_char_ngram_jaccard", "name", True, True, False, False, "pairwise", True, True),
    ("name_token_count_diff", "name", True, True, False, False, "pairwise", True, True),
    ("name_len_ratio", "name", True, True, False, False, "pairwise", True, True),

    # Address
    ("addr_exact_raw", "address", True, True, False, False, "pairwise", True, True),
    ("addr_exact_norm", "address", True, True, False, False, "pairwise", True, True),
    ("addr_fuzz_ratio", "address", True, True, False, False, "pairwise", True, True),
    ("addr_fuzz_wratio", "address", True, True, False, False, "pairwise", True, True),
    ("addr_fuzz_token_sort", "address", True, True, False, False, "pairwise", True, True),
    ("addr_fuzz_token_set", "address", True, True, False, False, "pairwise", True, True),
    ("addr_fuzz_partial_ratio", "address", True, True, False, False, "pairwise", True, True),
    ("addr_levenshtein_sim", "address", True, True, False, False, "pairwise", True, True),
    ("addr_jaccard_token", "address", True, True, False, False, "pairwise", True, True),
    ("addr_overlap_token", "address", True, True, False, False, "pairwise", True, True),
    ("addr_char_ngram_jaccard", "address", True, True, False, False, "pairwise", True, True),
    ("addr_token_count_diff", "address", True, True, False, False, "pairwise", True, True),
    ("addr_len_ratio", "address", True, True, False, False, "pairwise", True, True),

    # Structural
    ("country_exact_match", "structure", True, True, False, False, "pairwise", True, True),
    ("country_mismatch", "structure", True, True, False, False, "pairwise", True, True),
    ("postal_exact_match", "structure", True, True, False, False, "pairwise", True, True),
    ("building_exact_match", "structure", True, True, False, False, "pairwise", True, True),
    ("numeric_token_overlap", "structure", True, True, False, False, "pairwise", True, True),
    ("numeric_token_exact_match", "structure", True, True, False, False, "pairwise", True, True),
    ("s1_has_address", "structure", True, False, False, False, "query_only", True, True),
    ("target_has_address", "structure", False, True, False, False, "target_only", True, True),
    ("both_have_address", "structure", True, True, False, False, "pairwise", True, True),
    ("target_is_s2", "structure", False, True, False, False, "target_only", True, True),

    # Provenance
    ("exact_name_hit", "provenance", True, True, False, False, "blocking", True, True),
    ("exact_address_hit", "provenance", True, True, False, False, "blocking", True, True),
    ("name_token_hit", "provenance", True, True, False, False, "blocking", True, True),
    ("address_token_hit", "provenance", True, True, False, False, "blocking", True, True),
    ("rare_token_hit", "provenance", True, True, False, False, "blocking", True, True),
    ("postal_numeric_hit", "provenance", True, True, False, False, "blocking", True, True),
    ("cross_field_hit", "provenance", True, True, False, False, "blocking", True, True),
    ("name_tfidf_hit", "provenance", True, True, False, False, "blocking", True, True),
    ("address_tfidf_hit", "provenance", True, True, False, False, "blocking", True, True),
    ("route_hit_count", "provenance", True, True, False, False, "blocking", True, True),

    # Cross-field
    ("name_x_addr_wratio", "cross_field", True, True, False, False, "pairwise", True, True),
    ("name_high_addr_high", "cross_field", True, True, False, False, "pairwise", True, True),
    ("name_high_addr_low", "cross_field", True, True, False, False, "pairwise", True, True),
    ("name_low_addr_high", "cross_field", True, True, False, False, "pairwise", True, True),
    ("both_fields_present", "cross_field", True, True, False, False, "pairwise", True, True),
]

# 14 Contextual Features from M9
contextual_features = [
    ("context_cand_rank", "contextual", True, True, False, False, "per_query_candidates", True, True),
    ("context_score_gap_top", "contextual", True, True, False, False, "per_query_candidates", True, True),
    ("context_name_rank", "contextual", True, True, False, False, "pairwise", True, True),
    ("context_addr_rank", "contextual", True, True, False, False, "pairwise", True, True),
    ("context_route_count", "contextual", True, True, False, False, "blocking", True, True),
    ("context_rare_tok_overlap", "contextual", True, True, False, False, "pairwise", True, True),
    ("context_char_sim", "contextual", True, True, False, False, "pairwise", True, True),
    ("context_addr_char_sim", "contextual", True, True, False, False, "pairwise", True, True),
    ("context_postal_compat", "contextual", True, True, False, False, "pairwise", True, True),
    ("context_bldg_compat", "contextual", True, True, False, False, "pairwise", True, True),
    ("context_name_addr_interaction", "contextual", True, True, False, False, "pairwise", True, True),
    ("context_cand_density", "contextual", True, False, False, False, "per_query_candidates", True, True),
    ("context_target_freq", "contextual", False, True, False, False, "per_batch_global", True, True),
    ("context_target_side_rank", "contextual", True, True, False, False, "per_batch_global", True, True),
]

all_feats = baseline_features + contextual_features
rows = []
for name, src, s1_d, tgt_d, lbl, fld, scope, val_safe, test_safe in all_feats:
    rows.append({
        "feature_name": name,
        "source": src,
        "uses_S1_data": s1_d,
        "uses_target_data": tgt_d,
        "uses_labels": lbl,
        "uses_fold_information": fld,
        "fit_scope": scope,
        "safe_for_validation": val_safe,
        "safe_for_test": test_safe,
    })

df = pd.DataFrame(rows)
csv_path = OUT_DIR / "feature_leakage_audit.csv"
df.to_csv(csv_path, index=False)
print(f"Saved {len(df)} feature audit rows to {csv_path}")

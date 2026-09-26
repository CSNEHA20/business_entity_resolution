"""
Pair Feature Engineering Module for Amazon ML Challenge 2026.
Extracts rich pairwise string similarity, token overlap, structural,
blocking provenance, and cross-field interaction features.
"""

from collections import defaultdict
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz

from src.config import DEFAULT_CONFIG, PipelineConfig
from src.normalization import (
    clean_unicode_ascii,
    extract_building_number,
    extract_numeric_tokens,
    extract_postal_code,
    normalize_basic,
    normalize_country,
    tokenize_text,
)

logger = logging.getLogger("features")


FEATURE_COLUMNS = [
    # --- Name Features ---
    "name_exact_raw",
    "name_exact_norm",
    "name_fuzz_ratio",
    "name_fuzz_wratio",
    "name_fuzz_token_sort",
    "name_fuzz_token_set",
    "name_fuzz_partial_ratio",
    "name_levenshtein_sim",
    "name_jaccard_token",
    "name_overlap_token",
    "name_char_ngram_jaccard",
    "name_token_count_diff",
    "name_len_ratio",

    # --- Address Features ---
    "addr_exact_raw",
    "addr_exact_norm",
    "addr_fuzz_ratio",
    "addr_fuzz_wratio",
    "addr_fuzz_token_sort",
    "addr_fuzz_token_set",
    "addr_fuzz_partial_ratio",
    "addr_levenshtein_sim",
    "addr_jaccard_token",
    "addr_overlap_token",
    "addr_char_ngram_jaccard",
    "addr_token_count_diff",
    "addr_len_ratio",

    # --- Structural Features ---
    "country_exact_match",
    "country_mismatch",
    "postal_exact_match",
    "building_exact_match",
    "numeric_token_overlap",
    "numeric_token_exact_match",
    "s1_has_address",
    "target_has_address",
    "both_have_address",
    "target_is_s2",

    # --- Blocking Provenance Features ---
    "exact_name_hit",
    "exact_address_hit",
    "name_token_hit",
    "address_token_hit",
    "rare_token_hit",
    "postal_numeric_hit",
    "cross_field_hit",
    "name_tfidf_hit",
    "address_tfidf_hit",
    "route_hit_count",

    # --- Cross-Field Interactions ---
    "name_x_addr_wratio",
    "name_high_addr_high",
    "name_high_addr_low",
    "name_low_addr_high",
    "both_fields_present",
]


def _char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generates set of character n-grams from a string."""
    if not text or len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _jaccard(s1: Set[Any], s2: Set[Any]) -> float:
    if not s1 or not s2:
        return 0.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0


def _overlap_coeff(s1: Set[Any], s2: Set[Any]) -> float:
    if not s1 or not s2:
        return 0.0
    inter = len(s1 & s2)
    min_len = min(len(s1), len(s2))
    return inter / min_len if min_len > 0 else 0.0


class PairFeatureExtractor:
    """High-performance pairwise feature extractor for entity candidate pairs."""

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.config = config
        self.feature_names = list(FEATURE_COLUMNS)

    def extract_features_for_pair(
        self,
        s1_record: Dict[str, Any],
        target_record: Dict[str, Any],
        route_hits: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        """
        Extracts feature dictionary for a single entity candidate pair.
        s1_record and target_record are expected to contain raw and normalized fields.
        """
        # Raw fields
        s1_name_raw = s1_record.get("business_name_raw", "") or ""
        t_name_raw = target_record.get("business_name_raw", "") or ""
        s1_addr_raw = s1_record.get("business_address_raw", "") or ""
        t_addr_raw = target_record.get("business_address_raw", "") or ""

        # Normalized fields
        s1_name_norm = s1_record.get("business_name_norm", "") or ""
        t_name_norm = target_record.get("business_name_norm", "") or ""
        s1_addr_norm = s1_record.get("business_address_norm", "") or ""
        t_addr_norm = target_record.get("business_address_norm", "") or ""

        s1_country = s1_record.get("country_norm", "") or ""
        t_country = target_record.get("country_norm", "") or ""

        # Pre-computed tokens
        s1_name_toks = s1_record.get("name_tokens", set())
        t_name_toks = target_record.get("name_tokens", set())
        s1_addr_toks = s1_record.get("addr_tokens", set())
        t_addr_toks = target_record.get("addr_tokens", set())

        # Postal & numeric tokens
        s1_postals = s1_record.get("postal_codes", set())
        t_postals = target_record.get("postal_codes", set())
        s1_bldg = s1_record.get("building_number", "") or ""
        t_bldg = target_record.get("building_number", "") or ""

        s1_nums = s1_record.get("numeric_tokens", set())
        t_nums = target_record.get("numeric_tokens", set())

        tid = target_record.get("entity_id", "")

        # --- Name Features ---
        name_exact_raw = 1.0 if s1_name_raw and s1_name_raw == t_name_raw else 0.0
        name_exact_norm = 1.0 if s1_name_norm and s1_name_norm == t_name_norm else 0.0
        name_fuzz_ratio = fuzz.ratio(s1_name_norm, t_name_norm) / 100.0 if (s1_name_norm or t_name_norm) else 0.0
        name_fuzz_wratio = fuzz.WRatio(s1_name_norm, t_name_norm) / 100.0 if (s1_name_norm or t_name_norm) else 0.0
        name_fuzz_tsort = fuzz.token_sort_ratio(s1_name_norm, t_name_norm) / 100.0 if (s1_name_norm or t_name_norm) else 0.0
        name_fuzz_tset = fuzz.token_set_ratio(s1_name_norm, t_name_norm) / 100.0 if (s1_name_norm or t_name_norm) else 0.0
        name_fuzz_partial = fuzz.partial_ratio(s1_name_norm, t_name_norm) / 100.0 if (s1_name_norm or t_name_norm) else 0.0
        name_lev_sim = 1.0 - distance.Levenshtein.normalized_distance(s1_name_norm, t_name_norm) if (s1_name_norm or t_name_norm) else 0.0

        name_jaccard = _jaccard(s1_name_toks, t_name_toks)
        name_overlap = _overlap_coeff(s1_name_toks, t_name_toks)

        s1_ngrams_name = _char_ngrams(s1_name_norm)
        t_ngrams_name = _char_ngrams(t_name_norm)
        name_char_jaccard = _jaccard(s1_ngrams_name, t_ngrams_name)

        name_tok_diff = float(abs(len(s1_name_toks) - len(t_name_toks)))
        len_name_max = max(len(s1_name_norm), len(t_name_norm), 1)
        len_name_min = min(len(s1_name_norm), len(t_name_norm))
        name_len_ratio = len_name_min / len_name_max

        # --- Address Features ---
        addr_exact_raw = 1.0 if s1_addr_raw and s1_addr_raw == t_addr_raw else 0.0
        addr_exact_norm = 1.0 if s1_addr_norm and s1_addr_norm == t_addr_norm else 0.0
        addr_fuzz_ratio = fuzz.ratio(s1_addr_norm, t_addr_norm) / 100.0 if (s1_addr_norm or t_addr_norm) else 0.0
        addr_fuzz_wratio = fuzz.WRatio(s1_addr_norm, t_addr_norm) / 100.0 if (s1_addr_norm or t_addr_norm) else 0.0
        addr_fuzz_tsort = fuzz.token_sort_ratio(s1_addr_norm, t_addr_norm) / 100.0 if (s1_addr_norm or t_addr_norm) else 0.0
        addr_fuzz_tset = fuzz.token_set_ratio(s1_addr_norm, t_addr_norm) / 100.0 if (s1_addr_norm or t_addr_norm) else 0.0
        addr_fuzz_partial = fuzz.partial_ratio(s1_addr_norm, t_addr_norm) / 100.0 if (s1_addr_norm or t_addr_norm) else 0.0
        addr_lev_sim = 1.0 - distance.Levenshtein.normalized_distance(s1_addr_norm, t_addr_norm) if (s1_addr_norm or t_addr_norm) else 0.0

        addr_jaccard = _jaccard(s1_addr_toks, t_addr_toks)
        addr_overlap = _overlap_coeff(s1_addr_toks, t_addr_toks)

        s1_ngrams_addr = _char_ngrams(s1_addr_norm)
        t_ngrams_addr = _char_ngrams(t_addr_norm)
        addr_char_jaccard = _jaccard(s1_ngrams_addr, t_ngrams_addr)

        addr_tok_diff = float(abs(len(s1_addr_toks) - len(t_addr_toks)))
        len_addr_max = max(len(s1_addr_norm), len(t_addr_norm), 1)
        len_addr_min = min(len(s1_addr_norm), len(t_addr_norm))
        addr_len_ratio = len_addr_min / len_addr_max

        # --- Structural Features ---
        country_match = 1.0 if s1_country and t_country and s1_country == t_country else 0.0
        country_mismatch = 1.0 if s1_country and t_country and s1_country != t_country else 0.0
        postal_match = 1.0 if (s1_postals and t_postals and (s1_postals & t_postals)) else 0.0
        bldg_match = 1.0 if (s1_bldg and t_bldg and s1_bldg == t_bldg) else 0.0

        num_jaccard = _jaccard(s1_nums, t_nums)
        num_exact = 1.0 if (s1_nums and t_nums and s1_nums == t_nums) else 0.0

        s1_has_addr = 1.0 if bool(s1_addr_norm) else 0.0
        t_has_addr = 1.0 if bool(t_addr_norm) else 0.0
        both_have_addr = 1.0 if (s1_has_addr and t_has_addr) else 0.0
        target_is_s2 = 1.0 if tid.startswith("S2-") else 0.0

        # --- Blocking Provenance Features ---
        hits = route_hits or {}
        r_exact_n = float(hits.get("exact_name_hit", 0))
        r_exact_a = float(hits.get("exact_address_hit", 0))
        r_name_sig = float(hits.get("name_token_hit", 0))
        r_addr_sig = float(hits.get("address_token_hit", 0))
        r_token_v2 = float(hits.get("rare_token_hit", 0))
        r_postal_num = float(hits.get("postal_numeric_hit", 0))
        r_cross = float(hits.get("cross_field_hit", 0))
        r_tfidf_n = float(hits.get("name_tfidf_hit", 0))
        r_tfidf_a = float(hits.get("address_tfidf_hit", 0))
        r_count = float(hits.get("route_hit_count", sum([
            r_exact_n, r_exact_a, r_name_sig, r_addr_sig,
            r_token_v2, r_postal_num, r_cross, r_tfidf_n, r_tfidf_a
        ])))

        # --- Cross-Field Interactions ---
        name_x_addr = name_fuzz_wratio * addr_fuzz_wratio
        n_high_a_high = 1.0 if (name_fuzz_wratio >= 0.85 and addr_fuzz_wratio >= 0.85) else 0.0
        n_high_a_low = 1.0 if (name_fuzz_wratio >= 0.85 and addr_fuzz_wratio <= 0.40) else 0.0
        n_low_a_high = 1.0 if (name_fuzz_wratio <= 0.40 and addr_fuzz_wratio >= 0.85) else 0.0
        both_present = 1.0 if (s1_name_norm and s1_addr_norm and t_name_norm and t_addr_norm) else 0.0

        return {
            "name_exact_raw": name_exact_raw,
            "name_exact_norm": name_exact_norm,
            "name_fuzz_ratio": name_fuzz_ratio,
            "name_fuzz_wratio": name_fuzz_wratio,
            "name_fuzz_token_sort": name_fuzz_tsort,
            "name_fuzz_token_set": name_fuzz_tset,
            "name_fuzz_partial_ratio": name_fuzz_partial,
            "name_levenshtein_sim": name_lev_sim,
            "name_jaccard_token": name_jaccard,
            "name_overlap_token": name_overlap,
            "name_char_ngram_jaccard": name_char_jaccard,
            "name_token_count_diff": name_tok_diff,
            "name_len_ratio": name_len_ratio,

            "addr_exact_raw": addr_exact_raw,
            "addr_exact_norm": addr_exact_norm,
            "addr_fuzz_ratio": addr_fuzz_ratio,
            "addr_fuzz_wratio": addr_fuzz_wratio,
            "addr_fuzz_token_sort": addr_fuzz_tsort,
            "addr_fuzz_token_set": addr_fuzz_tset,
            "addr_fuzz_partial_ratio": addr_fuzz_partial,
            "addr_levenshtein_sim": addr_lev_sim,
            "addr_jaccard_token": addr_jaccard,
            "addr_overlap_token": addr_overlap,
            "addr_char_ngram_jaccard": addr_char_jaccard,
            "addr_token_count_diff": addr_tok_diff,
            "addr_len_ratio": addr_len_ratio,

            "country_exact_match": country_match,
            "country_mismatch": country_mismatch,
            "postal_exact_match": postal_match,
            "building_exact_match": bldg_match,
            "numeric_token_overlap": num_jaccard,
            "numeric_token_exact_match": num_exact,
            "s1_has_address": s1_has_addr,
            "target_has_address": t_has_addr,
            "both_have_address": both_have_addr,
            "target_is_s2": target_is_s2,

            "exact_name_hit": r_exact_n,
            "exact_address_hit": r_exact_a,
            "name_token_hit": r_name_sig,
            "address_token_hit": r_addr_sig,
            "rare_token_hit": r_token_v2,
            "postal_numeric_hit": r_postal_num,
            "cross_field_hit": r_cross,
            "name_tfidf_hit": r_tfidf_n,
            "address_tfidf_hit": r_tfidf_a,
            "route_hit_count": r_count,

            "name_x_addr_wratio": name_x_addr,
            "name_high_addr_high": n_high_a_high,
            "name_high_addr_low": n_high_a_low,
            "name_low_addr_high": n_low_a_high,
            "both_fields_present": both_present,
        }

    def extract_features_matrix(
        self,
        candidate_pairs: List[Tuple[str, str]],
        s1_lookup: Dict[str, Dict[str, Any]],
        target_lookup: Dict[str, Dict[str, Any]],
        provenance_lookup: Optional[Dict[Tuple[str, str], Dict[str, int]]] = None,
    ) -> pd.DataFrame:
        """
        Batch extracts features for a list of (s1_id, target_id) pairs into a DataFrame.
        """
        rows = []
        for s1_id, tid in candidate_pairs:
            s1_rec = s1_lookup.get(s1_id, {})
            t_rec = target_lookup.get(tid, {})
            route_hits = provenance_lookup.get((s1_id, tid)) if provenance_lookup else None
            feat_dict = self.extract_features_for_pair(s1_rec, t_rec, route_hits=route_hits)
            rows.append(feat_dict)

        df = pd.DataFrame(rows, columns=self.feature_names)
        return df

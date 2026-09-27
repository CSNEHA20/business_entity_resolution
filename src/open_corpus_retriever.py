"""
Pure Open-Corpus Retrieval API for Amazon ML Challenge 2026.
Milestone 10: Real Open-Corpus Entity Resolution Engine.

ABSOLUTE INVARIANT:
Candidate generation MUST NOT receive ground truth, labels, positive_target_ids, or true matches.
Candidate retrieval operates against the COMPLETE 10.3M target corpus (S2 + S3).
"""

from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import gc
import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import psutil

from src.normalization import (
    clean_unicode_text,
    extract_building_number,
    extract_numeric_tokens,
    extract_postal_code,
    get_token_signature,
    normalize_address_abbreviations,
    normalize_basic,
    normalize_business_name_suffixes,
    normalize_country,
    tokenize_text,
)

logger = logging.getLogger("open_corpus_retriever")

# Common business stopwords to strip when computing 'core_name'
BUSINESS_STOPWORDS = {
    "the", "and", "of", "in", "for", "at", "by", "from", "with",
    "co", "company", "corp", "corporation", "inc", "incorporated",
    "ltd", "limited", "pvt", "private", "llc", "llp", "plc",
    "gmbh", "sa", "sarl", "sas", "ste", "soc", "services", "enterprises",
    "solutions", "group", "holdings", "technologies", "international",
    "india", "usa", "us",
}

# Country normalization alias mapping
COUNTRY_MAP = {
    "US": "US",
    "USA": "US",
    "UNITED STATES": "US",
    "UNITED STATES OF AMERICA": "US",
    "IN": "IN",
    "IND": "IN",
    "INDIA": "IN",
    "FR": "FR",
    "FRA": "FR",
    "FRANCE": "FR",
    "DE": "DE",
    "DEU": "DE",
    "GERMANY": "DE",
    "GB": "GB",
    "GBR": "GB",
    "UK": "GB",
    "UNITED KINGDOM": "GB",
    "CA": "CA",
    "CAN": "CA",
    "CANADA": "CA",
}


def normalize_country_code(country: Optional[str]) -> str:
    """Normalizes country string to standard 2-letter uppercase or cleaned string."""
    if not country or not isinstance(country, str):
        return ""
    c = country.strip().upper()
    return COUNTRY_MAP.get(c, c)


def compute_core_name(clean_name: str) -> str:
    """Strips common legal suffixes and business stopwords to produce the core business name."""
    if not clean_name:
        return ""
    words = clean_name.lower().split()
    core_words = [w for w in words if w not in BUSINESS_STOPWORDS and len(w) > 1]
    if not core_words:
        return clean_name.lower()
    return " ".join(core_words)


def compute_compact_name(clean_name: str) -> str:
    """Alphanumeric-only representation without spaces or punctuation."""
    if not clean_name:
        return ""
    return re.sub(r"[^a-z0-9]", "", clean_name.lower())


def extract_char_ngrams(text: str, n: int = 3) -> List[str]:
    """Extracts character n-grams from text without spaces."""
    if not text:
        return []
    clean = re.sub(r"[^a-z0-9]", "", text.lower())
    if len(clean) < n:
        return [clean] if clean else []
    return [clean[i : i + n] for i in range(len(clean) - n + 1)]


def parse_entity_record(
    eid: str,
    raw_name: str,
    raw_addr: str,
    raw_country: str,
) -> Dict[str, Any]:
    """
    Parses and normalizes an entity record into standardized search keys.
    Completely label-free: takes only raw entity fields.
    """
    eid = str(eid).strip()
    n_raw = str(raw_name or "")
    a_raw = str(raw_addr or "")
    c_raw = str(raw_country or "")

    n_clean = clean_unicode_text(n_raw)
    a_clean = clean_unicode_text(a_raw)
    n_compact = compute_compact_name(n_clean)
    n_core = compute_core_name(n_clean)
    n_sig = get_token_signature(n_clean)
    a_sig = get_token_signature(a_clean)

    n_tokens = tokenize_text(n_clean)
    a_tokens = tokenize_text(a_clean)
    c_norm = normalize_country_code(c_raw)

    pins = set(extract_postal_code(a_raw))
    bldg = extract_building_number(a_raw) or ""
    nums = set(extract_numeric_tokens(a_raw))

    # Structural composites
    street_toks = [t for t in a_tokens if len(t) >= 3 and not t.isdigit() and t != bldg]
    first_street_tok = street_toks[0] if street_toks else ""
    num_street = f"{bldg}_{first_street_tok}" if (bldg and first_street_tok) else ""

    # Name + number composite
    core_head = n_core.split()[0] if n_core else (n_tokens[0] if n_tokens else "")
    name_num = f"{core_head}_{bldg}" if (core_head and bldg) else ""

    # Name + postal composite
    first_pin = sorted(pins)[0] if pins else ""
    name_postal = f"{core_head}_{first_pin}" if (core_head and first_pin) else ""

    # Character n-grams of core name
    char3 = extract_char_ngrams(n_core or n_clean, 3)
    char4 = extract_char_ngrams(n_core or n_clean, 4)

    return {
        "entity_id": eid,
        "business_name_raw": n_raw,
        "business_address_raw": a_raw,
        "country_raw": c_raw,
        "clean_name": n_clean,
        "clean_addr": a_clean,
        "compact_name": n_compact,
        "core_name": n_core,
        "name_sig": n_sig,
        "addr_sig": a_sig,
        "country_norm": c_norm,
        "name_tokens": frozenset(n_tokens),
        "name_toks": frozenset(n_tokens),
        "addr_tokens": frozenset(a_tokens),
        "addr_toks": frozenset(a_tokens),
        "postal": pins,
        "postal_codes": pins,
        "building": bldg,
        "building_number": bldg,
        "numeric_tokens": nums,
        "num_street": num_street,
        "name_num": name_num,
        "name_postal": name_postal,
        "char_3grams": char3,
        "char_4grams": char4,
        "has_addr": bool(a_clean),
    }


@dataclass
class RetrievalConfig:
    """Configuration governing retrieval route caps and frequency thresholds."""
    profile: str = "high_recall"  # 'high_recall', 'balanced', 'lean'

    # Route candidate caps
    max_exact_name: int = 500
    max_compact_name: int = 400
    max_core_name: int = 300
    max_name_sig: int = 300
    max_rare_token: int = 250
    max_sig_token: int = 150
    max_postal: int = 100
    max_building: int = 100
    max_num_street: int = 100
    max_name_num: int = 100
    max_name_postal: int = 100
    max_addr_sig: int = 200
    max_exact_addr: int = 200
    max_char_3gram: int = 100
    max_char_4gram: int = 100

    # Frequency filtering thresholds
    rare_token_max_df: int = 350
    sig_token_max_df: int = 3500
    char_ngram_max_df: int = 500

    # Overall per-query candidate limit
    max_total_candidates: int = 250

    def apply_profile(self, profile_name: str) -> None:
        self.profile = profile_name
        if profile_name == "high_recall":
            self.max_exact_name = 500
            self.max_compact_name = 400
            self.max_core_name = 300
            self.max_name_sig = 300
            self.max_rare_token = 250
            self.max_sig_token = 150
            self.max_postal = 100
            self.max_building = 100
            self.max_num_street = 100
            self.max_name_num = 100
            self.max_name_postal = 100
            self.max_char_3gram = 100
            self.max_char_4gram = 100
            self.max_total_candidates = 300
            self.rare_token_max_df = 400
            self.sig_token_max_df = 4000
            self.char_ngram_max_df = 600
        elif profile_name == "balanced":
            self.max_exact_name = 250
            self.max_compact_name = 200
            self.max_core_name = 150
            self.max_name_sig = 150
            self.max_rare_token = 100
            self.max_sig_token = 80
            self.max_postal = 50
            self.max_building = 50
            self.max_num_street = 50
            self.max_name_num = 50
            self.max_name_postal = 50
            self.max_char_3gram = 50
            self.max_char_4gram = 50
            self.max_total_candidates = 150
            self.rare_token_max_df = 250
            self.sig_token_max_df = 2500
            self.char_ngram_max_df = 300
        elif profile_name == "lean":
            self.max_exact_name = 100
            self.max_compact_name = 80
            self.max_core_name = 60
            self.max_name_sig = 60
            self.max_rare_token = 50
            self.max_sig_token = 30
            self.max_postal = 25
            self.max_building = 25
            self.max_num_street = 25
            self.max_name_num = 25
            self.max_name_postal = 25
            self.max_char_3gram = 25
            self.max_char_4gram = 25
            self.max_total_candidates = 80
            self.rare_token_max_df = 150
            self.sig_token_max_df = 1500
            self.char_ngram_max_df = 150


class TargetCorpusIndex:
    """
    In-memory multi-pass inverted index over the complete 10.3M target corpus (S2 + S3).
    Uses compact array('i') for RAM efficiency.
    """

    def __init__(self):
        self.target_ids: List[str] = []
        self.target_sources: List[str] = []  # "S2" or "S3"
        self.target_countries: List[str] = []

        # 14 Multi-pass Posting Lists
        self.idx_exact_name: Dict[str, array] = defaultdict(lambda: array("i"))
        self.idx_compact_name: Dict[str, array] = defaultdict(lambda: array("i"))
        self.idx_core_name: Dict[str, array] = defaultdict(lambda: array("i"))
        self.idx_name_sig: Dict[str, array] = defaultdict(lambda: array("i"))
        self.idx_token: Dict[str, array] = defaultdict(lambda: array("i"))
        self.idx_postal: Dict[str, array] = defaultdict(lambda: array("i"))
        self.idx_building: Dict[str, array] = defaultdict(lambda: array("i"))
        self.idx_num_street: Dict[str, array] = defaultdict(lambda: array("i"))
        self.idx_name_num: Dict[str, array] = defaultdict(lambda: array("i"))
        self.idx_name_postal: Dict[str, array] = defaultdict(lambda: array("i"))
        self.idx_char_3gram: Dict[str, array] = defaultdict(lambda: array("i"))
        self.idx_char_4gram: Dict[str, array] = defaultdict(lambda: array("i"))
        self.idx_exact_addr: Dict[str, array] = defaultdict(lambda: array("i"))
        self.idx_addr_sig: Dict[str, array] = defaultdict(lambda: array("i"))

        # Token document frequencies
        self.token_df: Counter = Counter()
        self.char_ngram_df: Counter = Counter()

        # Lookup caches for target raw records
        self.s2_lookup: Dict[str, Tuple[str, str, str]] = {}
        self.s3_lookup: Dict[str, Tuple[str, str, str]] = {}

        self.is_indexed: bool = False

    @property
    def total_records(self) -> int:
        return len(self.target_ids)

    def build_from_files(
        self,
        s2_path: Path,
        s3_path: Path,
        chunksize: int = 500000,
    ) -> None:
        """
        Builds the complete index from S2 and S3 TSV files.
        Guaranteed to index EVERY target record without label awareness.
        """
        t0 = time.time()
        logger.info(f"Building full open-corpus index from {s2_path.name} and {s3_path.name}...")

        target_idx = 0

        for file_path, src_tag, lookup_dict in [
            (s2_path, "S2", self.s2_lookup),
            (s3_path, "S3", self.s3_lookup),
        ]:
            logger.info(f"Indexing source: {src_tag} from {file_path}...")
            chunk_num = 0
            for chunk in pd.read_csv(
                file_path,
                sep="\t",
                dtype=str,
                keep_default_na=False,
                chunksize=chunksize,
            ):
                chunk_num += 1
                for eid, name, addr, country in zip(
                    chunk["entity_id"].values,
                    chunk["business_name"].values,
                    chunk["business_address"].values,
                    chunk["country"].values,
                ):
                    tid = str(eid).strip()
                    n_raw = str(name or "")
                    a_raw = str(addr or "")
                    c_raw = str(country or "")

                    lookup_dict[tid] = (n_raw, a_raw, c_raw)
                    self.target_ids.append(tid)
                    self.target_sources.append(src_tag)

                    c_norm = normalize_country_code(c_raw)
                    self.target_countries.append(c_norm)

                    # Extract keys
                    n_clean = clean_unicode_text(n_raw)
                    a_clean = clean_unicode_text(a_raw)
                    n_compact = compute_compact_name(n_clean)
                    n_core = compute_core_name(n_clean)
                    n_sig = get_token_signature(n_clean)
                    a_sig = get_token_signature(a_clean)

                    pins = extract_postal_code(a_raw)
                    bldg = extract_building_number(a_raw) or ""
                    a_tokens = tokenize_text(a_clean)
                    street_toks = [t for t in a_tokens if len(t) >= 3 and not t.isdigit() and t != bldg]
                    first_street_tok = street_toks[0] if street_toks else ""
                    num_street = f"{bldg}_{first_street_tok}" if (bldg and first_street_tok) else ""

                    core_head = n_core.split()[0] if n_core else ""
                    name_num = f"{core_head}_{bldg}" if (core_head and bldg) else ""
                    first_pin = sorted(pins)[0] if pins else ""
                    name_postal = f"{core_head}_{first_pin}" if (core_head and first_pin) else ""

                    # 1. Exact Name
                    if n_clean:
                        self.idx_exact_name[n_clean].append(target_idx)
                    # 2. Compact Name
                    if n_compact:
                        self.idx_compact_name[n_compact].append(target_idx)
                    # 3. Core Name
                    if n_core and n_core != n_clean:
                        self.idx_core_name[n_core].append(target_idx)
                    # 4. Token Signature
                    if n_sig:
                        self.idx_name_sig[n_sig].append(target_idx)
                    # 5. Tokens
                    toks = tokenize_text(n_clean)
                    for tok in set(toks):
                        if len(tok) >= 3:
                            self.idx_token[tok].append(target_idx)
                            self.token_df[tok] += 1
                    # 6. Postal
                    for pin in set(pins):
                        self.idx_postal[pin].append(target_idx)
                    # 7. Building
                    if bldg:
                        self.idx_building[bldg].append(target_idx)
                    # 8. Number + Street
                    if num_street:
                        self.idx_num_street[num_street].append(target_idx)
                    # 9. Name + Number
                    if name_num:
                        self.idx_name_num[name_num].append(target_idx)
                    # 10. Name + Postal
                    if name_postal:
                        self.idx_name_postal[name_postal].append(target_idx)
                    # 11. Exact Address (Bidirectional)
                    if a_clean:
                        self.idx_exact_addr[a_clean].append(target_idx)
                    # 12. Address Signature (Bidirectional)
                    if a_sig:
                        self.idx_addr_sig[a_sig].append(target_idx)
                    # 13 & 14. Character n-grams of core name
                    for g3 in set(extract_char_ngrams(n_core or n_clean, 3)):
                        self.idx_char_3gram[g3].append(target_idx)
                        self.char_ngram_df[g3] += 1
                    for g4 in set(extract_char_ngrams(n_core or n_clean, 4)):
                        self.idx_char_4gram[g4].append(target_idx)
                        self.char_ngram_df[g4] += 1

                    target_idx += 1

                logger.info(
                    f"Processed {src_tag} chunk {chunk_num} ({len(chunk):,} rows). "
                    f"Cumulative records: {target_idx:,}. RAM: {psutil.Process().memory_info().rss / 1e6:.1f} MB"
                )

        self.is_indexed = True
        logger.info(
            f"Successfully built complete open-corpus index in {time.time() - t0:.1f}s. "
            f"Total indexed target records: {self.total_records:,}. "
            f"RAM: {psutil.Process().memory_info().rss / 1e6:.1f} MB"
        )

    def get_target_record_raw(self, target_id: str) -> Tuple[str, str, str]:
        """Fetches raw (name, address, country) for a given target entity ID."""
        if target_id.startswith("S2-"):
            return self.s2_lookup.get(target_id, ("", "", ""))
        return self.s3_lookup.get(target_id, ("", "", ""))

    def get_parsed_target_record(self, target_id: str) -> Dict[str, Any]:
        """Returns parsed entity record dictionary for feature extraction."""
        n_raw, a_raw, c_raw = self.get_target_record_raw(target_id)
        return parse_entity_record(target_id, n_raw, a_raw, c_raw)


def retrieve_candidates(
    query_record: Dict[str, Any],
    target_indices: TargetCorpusIndex,
    config: Optional[RetrievalConfig] = None,
) -> List[Dict[str, Any]]:
    """
    CRITICAL ARCHITECTURAL INVARIANT:
    Retrieves candidate target entities from the full open corpus (S2 + S3).
    MUST NOT accept ground_truth, gt_map, labels, or positive_target_ids.

    Args:
        query_record: Parsed or raw S1 query record dict.
        target_indices: TargetCorpusIndex over full 10.3M records.
        config: RetrievalConfig specifying route caps and profile.

    Returns:
        List of dicts:
        {
            "target_id": str,
            "source": str,  # "S2" or "S3"
            "routes": List[str],  # e.g. ["exact_name", "char_4gram", "postal"]
            "route_hit_count": int,
            "retrieval_score": float,
            "retrieval_rank": int
        }
    """
    if config is None:
        config = RetrievalConfig()

    # Ensure query record is parsed
    if "clean_name" not in query_record:
        q = parse_entity_record(
            query_record.get("entity_id", ""),
            query_record.get("business_name", ""),
            query_record.get("business_address", ""),
            query_record.get("country", ""),
        )
    else:
        q = query_record

    # Target int index -> set of routes hit
    cand_routes: Dict[int, Set[str]] = defaultdict(set)
    # Target int index -> route priority score
    cand_scores: Dict[int, float] = defaultdict(float)

    # Route 1: Exact Normalized Name (weight 5.0)
    if q["clean_name"] and q["clean_name"] in target_indices.idx_exact_name:
        for t_idx in target_indices.idx_exact_name[q["clean_name"]][: config.max_exact_name]:
            cand_routes[t_idx].add("exact_name")
            cand_scores[t_idx] += 5.0

    # Route 2: Compact Name (weight 4.0)
    if q["compact_name"] and q["compact_name"] in target_indices.idx_compact_name:
        for t_idx in target_indices.idx_compact_name[q["compact_name"]][: config.max_compact_name]:
            cand_routes[t_idx].add("compact_name")
            cand_scores[t_idx] += 4.0

    # Route 3: Core Name (weight 3.5)
    if q["core_name"] and q["core_name"] in target_indices.idx_core_name:
        for t_idx in target_indices.idx_core_name[q["core_name"]][: config.max_core_name]:
            cand_routes[t_idx].add("core_name")
            cand_scores[t_idx] += 3.5

    # Route 4: Name Token Signature (weight 3.5)
    if q["name_sig"] and q["name_sig"] in target_indices.idx_name_sig:
        for t_idx in target_indices.idx_name_sig[q["name_sig"]][: config.max_name_sig]:
            cand_routes[t_idx].add("name_sig")
            cand_scores[t_idx] += 3.5

    # Route 5: Rare Name Tokens (df <= rare_token_max_df, weight 3.0)
    for tok in q["name_toks"]:
        if len(tok) >= 3 and tok in target_indices.idx_token:
            df_val = target_indices.token_df.get(tok, 0)
            if 0 < df_val <= config.rare_token_max_df:
                for t_idx in target_indices.idx_token[tok][: config.max_rare_token]:
                    cand_routes[t_idx].add("rare_token")
                    cand_scores[t_idx] += 3.0

    # Route 6: Significant Name Tokens (df <= sig_token_max_df, weight 2.0)
    for tok in q["name_toks"]:
        if len(tok) >= 3 and tok in target_indices.idx_token:
            df_val = target_indices.token_df.get(tok, 0)
            if config.rare_token_max_df < df_val <= config.sig_token_max_df:
                for t_idx in target_indices.idx_token[tok][: config.max_sig_token]:
                    cand_routes[t_idx].add("sig_token")
                    cand_scores[t_idx] += 2.0

    # Route 7: Number + Street Token composite (weight 3.0)
    if q["num_street"] and q["num_street"] in target_indices.idx_num_street:
        for t_idx in target_indices.idx_num_street[q["num_street"]][: config.max_num_street]:
            cand_routes[t_idx].add("num_street")
            cand_scores[t_idx] += 3.0

    # Route 8: Name + Number composite (weight 3.0)
    if q["name_num"] and q["name_num"] in target_indices.idx_name_num:
        for t_idx in target_indices.idx_name_num[q["name_num"]][: config.max_name_num]:
            cand_routes[t_idx].add("name_num")
            cand_scores[t_idx] += 3.0

    # Route 9: Name + Postal composite (weight 3.0)
    if q["name_postal"] and q["name_postal"] in target_indices.idx_name_postal:
        for t_idx in target_indices.idx_name_postal[q["name_postal"]][: config.max_name_postal]:
            cand_routes[t_idx].add("name_postal")
            cand_scores[t_idx] += 3.0

    # Route 10: Postal Code (weight 1.5)
    for pin in q["postal"]:
        if pin in target_indices.idx_postal:
            for t_idx in target_indices.idx_postal[pin][: config.max_postal]:
                cand_routes[t_idx].add("postal")
                cand_scores[t_idx] += 1.5

    # Route 11: Building Number (weight 1.0)
    if q["building"] and q["building"] in target_indices.idx_building:
        for t_idx in target_indices.idx_building[q["building"]][: config.max_building]:
            cand_routes[t_idx].add("building")
            cand_scores[t_idx] += 1.0

    # Route 12: Character 4-gram retrieval of distinctive core n-grams (weight 2.5)
    for g4 in q["char_4grams"]:
        if g4 in target_indices.idx_char_4gram:
            df_g4 = target_indices.char_ngram_df.get(g4, 0)
            if 0 < df_g4 <= config.char_ngram_max_df:
                for t_idx in target_indices.idx_char_4gram[g4][: config.max_char_4gram]:
                    cand_routes[t_idx].add("char_4gram")
                    cand_scores[t_idx] += 2.5

    # Route 13: Character 3-gram retrieval of distinctive core n-grams (weight 2.0)
    for g3 in q["char_3grams"]:
        if g3 in target_indices.idx_char_3gram:
            df_g3 = target_indices.char_ngram_df.get(g3, 0)
            if 0 < df_g3 <= config.char_ngram_max_df:
                for t_idx in target_indices.idx_char_3gram[g3][: config.max_char_3gram]:
                    cand_routes[t_idx].add("char_3gram")
                    cand_scores[t_idx] += 2.0

    # Route 14: Exact Address (Bidirectional route, weight 3.0)
    if q["clean_addr"] and q["clean_addr"] in target_indices.idx_exact_addr:
        for t_idx in target_indices.idx_exact_addr[q["clean_addr"]][: config.max_exact_addr]:
            cand_routes[t_idx].add("exact_addr")
            cand_scores[t_idx] += 3.0

    # Route 15: Address Signature (Bidirectional route, weight 2.5)
    if q["addr_sig"] and q["addr_sig"] in target_indices.idx_addr_sig:
        for t_idx in target_indices.idx_addr_sig[q["addr_sig"]][: config.max_addr_sig]:
            cand_routes[t_idx].add("addr_sig")
            cand_scores[t_idx] += 2.5

    # Sort candidates by combined score & route count
    sorted_items = sorted(
        cand_scores.items(),
        key=lambda item: (len(cand_routes[item[0]]), item[1]),
        reverse=True,
    )

    # Country compatibility check (Route 16: country filter / penalty)
    # If query country is known and target country is known and different, penalize score
    q_country = q.get("country_norm", "")
    final_candidates: List[Dict[str, Any]] = []

    for rank, (t_idx, score) in enumerate(sorted_items[: config.max_total_candidates], 1):
        tid = target_indices.target_ids[t_idx]
        src = target_indices.target_sources[t_idx]
        t_country = target_indices.target_countries[t_idx]
        routes_list = sorted(cand_routes[t_idx])

        # Country agreement flag
        if q_country and t_country:
            if q_country == t_country:
                routes_list.append("country_match")
            else:
                routes_list.append("country_mismatch")

        final_candidates.append(
            {
                "target_id": tid,
                "source": src,
                "routes": routes_list,
                "route_hit_count": len(cand_routes[t_idx]),
                "retrieval_score": float(score),
                "retrieval_rank": rank,
            }
        )

    return final_candidates

"""
Multi-Pass Blocking and Candidate Generation Module for Amazon ML Challenge 2026.
Implements 9 independent blocking routes with strict candidate volume control,
sparse TF-IDF top-K retrieval, and full candidate provenance tracking.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import DEFAULT_CONFIG, PipelineConfig
from src.normalization import NormalizedEntityRecord, build_normalized_record

logger = logging.getLogger(__name__)


@dataclass
class CandidatePairProvenance:
    """Represents a candidate pair with the set of blocking routes that discovered it."""
    s1_id: str
    target_id: str
    target_source: str  # "S2" or "S3"
    routes: Set[str] = field(default_factory=set)
    route_count: int = 0
    tier: int = 2

    def __post_init__(self):
        self.route_count = len(self.routes)
        if any(r in {"exact_name", "exact_address", "name_token_sig", "address_token_sig"} for r in self.routes):
            self.tier = 1
        elif any(r in {"name_tfidf", "address_tfidf", "rare_name_tokens", "token_v2"} for r in self.routes):
            self.tier = 2
        else:
            self.tier = 3



class MultiPassCandidateGenerator:
    """
    Orchestrates multi-pass candidate generation across 9 independent blocking routes.
    Maintains inverted indices, sparse TF-IDF models, and provenance tracking.
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.config = config

        # Inverted Indices for Target Records (S2 + S3)
        self.target_records: Dict[str, NormalizedEntityRecord] = {}
        self.index_exact_name: Dict[str, List[str]] = defaultdict(list)
        self.index_exact_address: Dict[str, List[str]] = defaultdict(list)
        self.index_name_token_sig: Dict[str, List[str]] = defaultdict(list)
        self.index_address_token_sig: Dict[str, List[str]] = defaultdict(list)
        self.index_name_tokens: Dict[str, List[str]] = defaultdict(list)
        self.index_postal: Dict[str, List[str]] = defaultdict(list)
        self.index_building_number: Dict[str, List[str]] = defaultdict(list)

        # Token Frequencies for Rare-Token Blocking
        self.token_freq_name: Counter = Counter()

        # TF-IDF Retrieval Components
        self.name_vectorizer: Optional[TfidfVectorizer] = None
        self.target_name_matrix: Optional[csr_matrix] = None
        self.target_name_ids: List[str] = []

        self.address_vectorizer: Optional[TfidfVectorizer] = None
        self.target_address_matrix: Optional[csr_matrix] = None
        self.target_address_ids: List[str] = []

    def fit_targets(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> "MultiPassCandidateGenerator":
        """
        Builds all inverted indices and TF-IDF matrix indices over Source 2 and Source 3 records.
        """
        logger.info(f"Indexing target datasets: S2={len(s2_df)}, S3={len(s3_df)}")

        all_target_records: List[NormalizedEntityRecord] = []
        name_texts: List[str] = []
        address_texts: List[str] = []

        for df, src_name in [(s2_df, "S2"), (s3_df, "S3")]:
            for _, row in df.iterrows():
                rec = build_normalized_record(
                    entity_id=row["entity_id"],
                    business_name=row.get("business_name"),
                    business_address=row.get("business_address"),
                    country=row.get("country"),
                )
                self.target_records[rec.entity_id] = rec
                all_target_records.append(rec)

                tid = rec.entity_id

                # 1. Exact Name & Legal Index
                if rec.name_legal_normalized:
                    self.index_exact_name[rec.name_legal_normalized].append(tid)
                elif rec.name_basic:
                    self.index_exact_name[rec.name_basic].append(tid)

                # 2. Exact Address Index
                if rec.address_expanded:
                    self.index_exact_address[rec.address_expanded].append(tid)

                # 3. Name Token Signature Index
                if rec.name_token_signature:
                    self.index_name_token_sig[rec.name_token_signature].append(tid)

                # 4. Address Token Signature Index
                if rec.address_token_signature:
                    self.index_address_token_sig[rec.address_token_signature].append(tid)

                # 5. Name Tokens & Frequency
                for tok in set(rec.name_tokens):
                    if len(tok) >= 3:
                        self.index_name_tokens[tok].append(tid)
                        self.token_freq_name[tok] += 1

                # 6. Postal Index
                for pin in rec.postal_codes:
                    self.index_postal[pin].append(tid)

                # Building Number Index
                if rec.building_number:
                    self.index_building_number[rec.building_number].append(tid)

                # TF-IDF texts
                name_texts.append(rec.name_basic or "unknown")
                address_texts.append(rec.address_basic or "unknown")
                self.target_name_ids.append(tid)
                self.target_address_ids.append(tid)

        # Build Character TF-IDF Indices
        logger.info("Fitting sparse Character TF-IDF vectorizers...")
        self.name_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=self.config.char_ngram_range,
            max_features=self.config.tfidf_max_features,
            min_df=self.config.tfidf_min_df,
            sublinear_tf=self.config.tfidf_sublinear_tf,
            dtype=np.float32,
        )
        self.target_name_matrix = self.name_vectorizer.fit_transform(name_texts)

        self.address_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=self.config.char_ngram_range,
            max_features=self.config.tfidf_max_features,
            min_df=self.config.tfidf_min_df,
            sublinear_tf=self.config.tfidf_sublinear_tf,
            dtype=np.float32,
        )
        self.target_address_matrix = self.address_vectorizer.fit_transform(address_texts)

        logger.info(f"Target indexing complete. Total indexed records: {len(self.target_records)}")
        return self

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        top_k_tfidf: Optional[int] = None,
        max_total_candidates: Optional[int] = None,
        enabled_routes: Optional[Set[str]] = None,
    ) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], CandidatePairProvenance]]:
        """
        Executes multi-pass candidate generation for a given set of Source 1 records.
        
        Returns:
            candidates: mapping s1_id -> set of candidate target entity_ids
            provenance: mapping (s1_id, target_id) -> CandidatePairProvenance
        """
        k = top_k_tfidf or self.config.top_k_candidates_per_pass
        max_cands = max_total_candidates or self.config.max_total_candidates_per_s1

        all_routes = {
            "exact_name",
            "exact_address",
            "name_token_sig",
            "address_token_sig",
            "rare_name_tokens",
            "postal_numeric",
            "name_tfidf",
            "address_tfidf",
            "cross_country_token",
        }
        routes_to_run = enabled_routes if enabled_routes is not None else all_routes

        candidates: Dict[str, Set[str]] = {str(row["entity_id"]).strip(): set() for _, row in s1_df.iterrows()}
        provenance_map: Dict[Tuple[str, str], CandidatePairProvenance] = {}

        s1_records: List[NormalizedEntityRecord] = [
            build_normalized_record(
                entity_id=row["entity_id"],
                business_name=row.get("business_name"),
                business_address=row.get("business_address"),
                country=row.get("country"),
            )
            for _, row in s1_df.iterrows()
        ]

        def _add_pair(s1_id: str, target_id: str, route_name: str):
            if s1_id == target_id:
                return  # Prevent self-match
            if len(candidates[s1_id]) < max_cands:
                candidates[s1_id].add(target_id)
            key = (s1_id, target_id)
            if key not in provenance_map:
                src = "S2" if target_id.startswith("S2-") else "S3"
                provenance_map[key] = CandidatePairProvenance(
                    s1_id=s1_id,
                    target_id=target_id,
                    target_source=src,
                    routes={route_name},
                )
            else:
                provenance_map[key].routes.add(route_name)

        # -------------------------------------------------------------------
        # Block 1 to 6 & 9: Inverted Index Passes
        # -------------------------------------------------------------------
        logger.info(f"Running rule-based & inverted index blocking on {len(s1_records)} S1 queries...")

        for rec in s1_records:
            s1_id = rec.entity_id

            # Block 1: Exact Name
            if "exact_name" in routes_to_run:
                name_key = rec.name_legal_normalized or rec.name_basic
                if name_key and name_key in self.index_exact_name:
                    for tid in self.index_exact_name[name_key][:50]:
                        _add_pair(s1_id, tid, "exact_name")

            # Block 2: Exact Address
            if "exact_address" in routes_to_run:
                if rec.address_expanded and rec.address_expanded in self.index_exact_address:
                    for tid in self.index_exact_address[rec.address_expanded][:50]:
                        _add_pair(s1_id, tid, "exact_address")

            # Block 3: Name Token Signature
            if "name_token_sig" in routes_to_run:
                if rec.name_token_signature and rec.name_token_signature in self.index_name_token_sig:
                    for tid in self.index_name_token_sig[rec.name_token_signature][:50]:
                        _add_pair(s1_id, tid, "name_token_sig")

            # Block 4: Address Token Signature
            if "address_token_sig" in routes_to_run:
                if rec.address_token_signature and rec.address_token_signature in self.index_address_token_sig:
                    for tid in self.index_address_token_sig[rec.address_token_signature][:50]:
                        _add_pair(s1_id, tid, "address_token_sig")

            # Block 5: Rare Name Tokens
            if "rare_name_tokens" in routes_to_run:
                for tok in set(rec.name_tokens):
                    freq = self.token_freq_name.get(tok, 0)
                    # Use tokens that appear at least 1 time and at most 100 times (informative & selective)
                    if 1 <= freq <= 100:
                        for tid in self.index_name_tokens[tok][:30]:
                            _add_pair(s1_id, tid, "rare_name_tokens")

            # Block 6: Postal & Numeric Address
            if "postal_numeric" in routes_to_run:
                for pin in rec.postal_codes:
                    if pin in self.index_postal:
                        for tid in self.index_postal[pin][:30]:
                            t_rec = self.target_records.get(tid)
                            # Combine postal with at least 1 shared name token or building number
                            if t_rec and (set(rec.name_tokens) & set(t_rec.name_tokens)):
                                _add_pair(s1_id, tid, "postal_numeric")

            # Block 9: Cross-Field (Country + Building Number)
            if "cross_country_token" in routes_to_run and rec.building_number:
                if rec.building_number in self.index_building_number:
                    for tid in self.index_building_number[rec.building_number][:20]:
                        t_rec = self.target_records.get(tid)
                        if t_rec and t_rec.country_normalized == rec.country_normalized:
                            if set(rec.name_tokens) & set(t_rec.name_tokens):
                                _add_pair(s1_id, tid, "cross_country_token")

        # -------------------------------------------------------------------
        # Block 7 & 8: Batched Sparse TF-IDF Top-K Retrieval
        # -------------------------------------------------------------------
        if "name_tfidf" in routes_to_run and self.name_vectorizer is not None and self.target_name_matrix is not None:
            logger.info("Running sparse Name TF-IDF top-K retrieval...")
            batch_size = 5000
            for i in range(0, len(s1_records), batch_size):
                batch_records = s1_records[i : i + batch_size]
                s1_name_texts = [r.name_basic or "unknown" for r in batch_records]
                s1_matrix = self.name_vectorizer.transform(s1_name_texts)

                # Sparse matrix multiplication: (Batch x Vocabulary) * (Vocabulary x Targets)
                sim_matrix = s1_matrix.dot(self.target_name_matrix.T)

                for row_idx, r in enumerate(batch_records):
                    row_sims = sim_matrix.getrow(row_idx)
                    if row_sims.nnz == 0:
                        continue
                    # Extract top-K scores
                    col_indices = row_sims.indices
                    data = row_sims.data
                    if len(data) > k:
                        top_partition = np.argpartition(data, -k)[-k:]
                        top_sorted = top_partition[np.argsort(-data[top_partition])]
                        selected_target_indices = col_indices[top_sorted]
                    else:
                        top_sorted = np.argsort(-data)
                        selected_target_indices = col_indices[top_sorted]

                    for tidx in selected_target_indices:
                        tid = self.target_name_ids[tidx]
                        _add_pair(r.entity_id, tid, "name_tfidf")

        if "address_tfidf" in routes_to_run and self.address_vectorizer is not None and self.target_address_matrix is not None:
            logger.info("Running sparse Address TF-IDF top-K retrieval...")
            batch_size = 5000
            for i in range(0, len(s1_records), batch_size):
                batch_records = s1_records[i : i + batch_size]
                s1_addr_texts = [r.address_basic or "unknown" for r in batch_records]
                s1_matrix = self.address_vectorizer.transform(s1_addr_texts)

                sim_matrix = s1_matrix.dot(self.target_address_matrix.T)

                for row_idx, r in enumerate(batch_records):
                    row_sims = sim_matrix.getrow(row_idx)
                    if row_sims.nnz == 0:
                        continue
                    col_indices = row_sims.indices
                    data = row_sims.data
                    if len(data) > k:
                        top_partition = np.argpartition(data, -k)[-k:]
                        top_sorted = top_partition[np.argsort(-data[top_partition])]
                        selected_target_indices = col_indices[top_sorted]
                    else:
                        top_sorted = np.argsort(-data)
                        selected_target_indices = col_indices[top_sorted]

                    for tidx in selected_target_indices:
                        tid = self.target_address_ids[tidx]
                        _add_pair(r.entity_id, tid, "address_tfidf")

        logger.info(
            f"Candidate generation complete. Total unique candidate pairs: {len(provenance_map)}"
        )
        return candidates, provenance_map

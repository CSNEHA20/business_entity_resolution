"""
Comprehensive Multi-Representation Normalization Module for Amazon ML Challenge 2026.
Preserves original raw text while generating conservative, aggressive, token-sorted,
Unicode-preserving, and structural representations for robust candidate blocking and feature extraction.
"""

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

# Common legal business suffixes and abbreviation mappings (including Indic and European)
LEGAL_SUFFIX_MAP = {
    r"\bcorp\b": "corporation",
    r"\bcorporation\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bincorporated\b": "incorporated",
    r"\bltd\b": "limited",
    r"\blimited\b": "limited",
    r"\bpvt\b": "private",
    r"\bprivate\b": "private",
    r"\bllc\b": "limited liability company",
    r"\bllp\b": "limited liability partnership",
    r"\bplc\b": "public limited company",
    r"\bco\b": "company",
    r"\bcompany\b": "company",
    r"\bgmbh\b": "gmbh",
    r"\bsarl\b": "sarl",
    r"\bsa\b": "sa",
    r"\bsas\b": "sas",
    r"\bste\b": "societe",
    r"\bsoc\b": "societe",
    # Indic suffixes in Devanagari / transliterated
    r"प्राइवेट\s+लिमिटेड": "private limited",
    r"प्रा[\.\s]+लि[\.]?": "private limited",
    r"लिमिटेड": "limited",
    r"एलएलपी": "llp",
    r"कंपनी": "company",
}

# Common address street / unit abbreviation expansions
ADDRESS_ABBREVIATION_MAP = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bfl\b": "floor",
    r"\bno\b": "number",
    r"\bp\s*o\s*box\b": "pobox",
    r"\bpobox\b": "pobox",
    r"\brue\b": "rue",
    r"\bav\b": "avenue",
    r"\bbd\b": "boulevard",
}

PUNCT_CHARS = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
PUNCT_TABLE = str.maketrans({c: ' ' for c in PUNCT_CHARS})


def clean_unicode_ascii(text: Optional[str]) -> str:
    """Standardizes unicode characters to normalized ASCII representation."""
    if text is None or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("utf-8")


def clean_unicode_text(text: Optional[str]) -> str:
    """
    Normalizes text while strictly preserving non-Latin scripts (Devanagari, Tamil, Kannada, etc.)
    and accented European characters (French accents, German umlauts).
    Uses NFKC normalization and handles combining marks properly.
    """
    if text is None or not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", str(text)).casefold()
    t = t.replace("&", " and ").translate(PUNCT_TABLE)
    return " ".join(t.split())


def normalize_basic(text: Optional[str]) -> str:
    """
    Conservative basic normalization:
    - Unicode normalized (NFKC) & casefolded
    - '&' converted to ' and '
    - Punctuation replaced with space (preserving Unicode alphabets, Indic matras, and digits)
    - Multiple whitespace collapsed
    """
    if text is None or not isinstance(text, str):
        return ""
    return clean_unicode_text(text)


def normalize_alnum(text: Optional[str]) -> str:
    """Strict alphanumeric representation without spaces or punctuation, preserving Unicode letters/digits."""
    if text is None or not isinstance(text, str):
        return ""
    cleaned = clean_unicode_text(text)
    return "".join(c for c in cleaned if c.isalnum())


def normalize_compact(text: Optional[str]) -> str:
    """Alphanumeric words separated by single spaces, preserving Unicode characters."""
    if text is None or not isinstance(text, str):
        return ""
    cleaned = clean_unicode_text(text)
    words = ["".join(c for c in w if c.isalnum()) for w in cleaned.split()]
    return " ".join(w for w in words if w).strip()


def tokenize_text(text: Optional[str]) -> List[str]:
    """Extracts non-empty token list from normalized text."""
    clean = clean_unicode_text(text)
    return [t for t in clean.split(" ") if t]


def get_token_signature(text: Optional[str]) -> str:
    """
    Token signature: sorted, unique tokens joined by space.
    Extremely robust against word-order transpositions and duplicate tokens.
    """
    tokens = set(tokenize_text(text))
    return " ".join(sorted(tokens))


def get_char_ngrams(text: Optional[str], n_range: Tuple[int, int] = (3, 5)) -> List[str]:
    """Extracts character n-grams from normalized text."""
    if text is None:
        return []
    clean = clean_unicode_text(text).replace(" ", "")
    ngrams = []
    min_n, max_n = n_range
    for n in range(min_n, max_n + 1):
        if len(clean) >= n:
            ngrams.extend([clean[i : i + n] for i in range(len(clean) - n + 1)])
    return list(set(ngrams))


def normalize_business_name_suffixes(name: Optional[str]) -> str:
    """Expands/standardizes legal business suffixes."""
    clean = clean_unicode_text(name)
    for pattern, replacement in LEGAL_SUFFIX_MAP.items():
        clean = re.sub(pattern, replacement, clean)
    return re.sub(r"\s+", " ", clean).strip()


def normalize_address_abbreviations(address: Optional[str]) -> str:
    """Expands standard street and unit abbreviations."""
    clean = clean_unicode_text(address)
    for pattern, replacement in ADDRESS_ABBREVIATION_MAP.items():
        clean = re.sub(pattern, replacement, clean)
    return re.sub(r"\s+", " ", clean).strip()


def normalize_country(country: Optional[str]) -> str:
    """
    Normalizes country code/name string cleanly without restricting to fixed whitelist.
    Fully supports open-set countries (e.g. France, Germany, US, India).
    """
    if country is None or not isinstance(country, str):
        return ""
    return country.strip().upper()


# ---------------------------------------------------------------------------
# Structural Text Extraction
# ---------------------------------------------------------------------------

def extract_numeric_tokens(text: Optional[str]) -> List[str]:
    """Extracts all continuous digit sequences."""
    if text is None or not isinstance(text, str):
        return []
    return re.findall(r"\b\d+\b", str(text))


def extract_postal_code(address: Optional[str]) -> List[str]:
    """
    Extracts potential PIN/Postal codes (5 or 6 digit sequences).
    Works for US 5-digit ZIPs and India 6-digit PIN codes.
    """
    if address is None or not isinstance(address, str):
        return []
    return re.findall(r"\b\d{5,6}\b", str(address))


def extract_building_number(address: Optional[str]) -> Optional[str]:
    """Extracts leading house/building number if present."""
    if address is None or not isinstance(address, str):
        return None
    match = re.match(r"^\s*(\d+[a-zA-Z]?)\b", str(address))
    return match.group(1).lower() if match else None


# ---------------------------------------------------------------------------
# Multi-Representation Record Class
# ---------------------------------------------------------------------------

@dataclass
class NormalizedEntityRecord:
    """Holds all normalized representations of an entity record."""
    entity_id: str
    
    # Business Name Representations
    name_raw: str
    name_basic: str
    name_clean_unicode: str
    name_alnum: str
    name_compact: str
    name_tokens: List[str]
    name_sorted_tokens: List[str]
    name_token_signature: str
    name_legal_normalized: str
    
    # Address Representations
    address_raw: str
    address_basic: str
    address_clean_unicode: str
    address_alnum: str
    address_compact: str
    address_tokens: List[str]
    address_sorted_tokens: List[str]
    address_token_signature: str
    address_expanded: str
    
    # Country Representation (Open-Set)
    country_raw: str
    country_normalized: str
    
    # Structural Attributes
    postal_codes: List[str]
    numeric_tokens: List[str]
    building_number: Optional[str]
    name_token_count: int
    address_token_count: int


def build_normalized_record(
    entity_id: str,
    business_name: Optional[str],
    business_address: Optional[str],
    country: Optional[str]
) -> NormalizedEntityRecord:
    """Builds a full NormalizedEntityRecord from raw strings."""
    n_raw = str(business_name or "")
    a_raw = str(business_address or "")
    c_raw = str(country or "")

    n_u = clean_unicode_text(n_raw)
    n_basic = normalize_basic(n_raw)
    n_tokens = tokenize_text(n_raw)
    n_sorted = sorted(set(n_tokens))
    n_sig = " ".join(n_sorted)
    
    a_u = clean_unicode_text(a_raw)
    a_basic = normalize_basic(a_raw)
    a_tokens = tokenize_text(a_raw)
    a_sorted = sorted(set(a_tokens))
    a_sig = " ".join(a_sorted)

    return NormalizedEntityRecord(
        entity_id=str(entity_id).strip(),
        name_raw=n_raw,
        name_basic=n_basic,
        name_clean_unicode=n_u,
        name_alnum=normalize_alnum(n_raw),
        name_compact=normalize_compact(n_raw),
        name_tokens=n_tokens,
        name_sorted_tokens=n_sorted,
        name_token_signature=n_sig,
        name_legal_normalized=normalize_business_name_suffixes(n_raw),
        address_raw=a_raw,
        address_basic=a_basic,
        address_clean_unicode=a_u,
        address_alnum=normalize_alnum(a_raw),
        address_compact=normalize_compact(a_raw),
        address_tokens=a_tokens,
        address_sorted_tokens=a_sorted,
        address_token_signature=a_sig,
        address_expanded=normalize_address_abbreviations(a_raw),
        country_raw=c_raw,
        country_normalized=normalize_country(c_raw),
        postal_codes=extract_postal_code(a_raw),
        numeric_tokens=extract_numeric_tokens(a_raw),
        building_number=extract_building_number(a_raw),
        name_token_count=len(n_tokens),
        address_token_count=len(a_tokens),
    )

"""
Normalization module for Amazon ML Challenge 2026.
Provides deterministic, rule-based text cleaning for business names and addresses.
"""

import re
import unicodedata
from typing import Optional


# Common legal business suffixes and abbreviation mappings
LEGAL_SUFFIXES = {
    r"\bcorp\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bltd\b": "limited",
    r"\bpvt\b": "private",
    r"\bllc\b": "limited liability company",
    r"\bco\b": "company",
}

ADDRESS_ABBREVIATIONS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bfl\b": "floor",
    r"\bp\s*o\s*box\b": "pobox",
}


def clean_text_general(text: Optional[str]) -> str:
    """
    Base text normalization:
    - Unicode normalization (NFKD) and ASCII transliteration
    - Lowercase conversion
    - Replace punctuation with spaces
    - Collapse multiple whitespace
    """
    if text is None or not isinstance(text, str):
        return ""
    
    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    
    # Replace ampersands
    text = text.replace("&", " and ")
    
    # Replace punctuation with whitespace
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_business_name(name: Optional[str]) -> str:
    """
    Deterministic normalization for business names.
    Applies base cleaning and standardizes common legal suffixes.
    """
    cleaned = clean_text_general(name)
    for pattern, replacement in LEGAL_SUFFIXES.items():
        cleaned = re.sub(pattern, replacement, cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_business_address(address: Optional[str]) -> str:
    """
    Deterministic normalization for business addresses.
    Applies base cleaning and expands common street/address abbreviations.
    """
    cleaned = clean_text_general(address)
    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        cleaned = re.sub(pattern, replacement, cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_country(country: Optional[str]) -> str:
    """
    Normalizes country string without filtering or hardcoding allowable sets.
    """
    if country is None or not isinstance(country, str):
        return ""
    return country.strip().upper()

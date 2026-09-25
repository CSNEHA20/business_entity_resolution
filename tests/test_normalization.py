"""
Unit tests for deterministic normalization functions.
"""

from src.normalization import (
    clean_text_general,
    normalize_business_address,
    normalize_business_name,
    normalize_country,
)


def test_clean_text_general():
    assert clean_text_general("  Hello,  World! & Friends  ") == "hello world and friends"
    assert clean_text_general("Cafe\u0301 de Paris") == "cafe de paris"
    assert clean_text_general(None) == ""
    assert clean_text_general("") == ""


def test_normalize_business_name_suffixes():
    assert normalize_business_name("Acme Corp.") == "acme corporation"
    assert normalize_business_name("Tata Sons Pvt. Ltd.") == "tata sons private limited"
    assert normalize_business_name("Johnson & Johnson Inc") == "johnson and johnson incorporated"
    assert normalize_business_name("Apex Dynamics, LLC") == "apex dynamics limited liability company"


def test_normalize_business_address_abbreviations():
    assert normalize_business_address("123 Main Rd., Apt 4B") == "123 main road apartment 4b"
    assert normalize_business_address("45 Market St., Suite 200") == "45 market street suite 200"
    assert normalize_business_address("P.O. Box 789") == "pobox 789"
    assert normalize_business_address("99 Tech Blvd, San Jose") == "99 tech boulevard san jose"


def test_normalize_country_open_set():
    assert normalize_country("us") == "US"
    assert normalize_country("  India  ") == "INDIA"
    assert normalize_country("france") == "FRANCE"
    assert normalize_country("DE") == "DE"
    assert normalize_country(None) == ""

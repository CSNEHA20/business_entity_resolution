"""
Unit tests for deterministic normalization functions and multi-representation records.
"""

from src.normalization import (
    build_normalized_record,
    extract_building_number,
    extract_numeric_tokens,
    extract_postal_code,
    get_token_signature,
    normalize_address_abbreviations,
    normalize_alnum,
    normalize_basic,
    normalize_business_name_suffixes,
    normalize_compact,
    normalize_country,
    tokenize_text,
)


def test_normalize_basic():
    assert normalize_basic("  Hello,  World! & Friends  ") == "hello world and friends"
    assert normalize_basic("Cafe\u0301 de Paris") == "cafe de paris"
    assert normalize_basic(None) == ""
    assert normalize_basic("") == ""


def test_normalize_alnum_and_compact():
    assert normalize_alnum("Acme Corp., LLC 123") == "acmecorpllc123"
    assert normalize_compact("Acme   Corp.,  LLC 123") == "acme corp llc 123"


def test_token_signature():
    # Order independence & deduplication
    assert get_token_signature("Corporation Acme Corp") == "acme corp corporation"
    assert get_token_signature("Acme Corporation") == "acme corporation"


def test_normalize_business_name_suffixes():
    assert normalize_business_name_suffixes("Acme Corp.") == "acme corporation"
    assert normalize_business_name_suffixes("Tata Sons Pvt. Ltd.") == "tata sons private limited"
    assert normalize_business_name_suffixes("Johnson & Johnson Inc") == "johnson and johnson incorporated"
    assert normalize_business_name_suffixes("Apex Dynamics, LLC") == "apex dynamics limited liability company"


def test_normalize_address_abbreviations():
    assert normalize_address_abbreviations("123 Main Rd., Apt 4B") == "123 main road apartment 4b"
    assert normalize_address_abbreviations("45 Market St., Suite 200") == "45 market street suite 200"
    assert normalize_address_abbreviations("P.O. Box 789") == "pobox 789"
    assert normalize_address_abbreviations("99 Tech Blvd, San Jose") == "99 tech boulevard san jose"


def test_normalize_country_open_set():
    assert normalize_country("us") == "US"
    assert normalize_country("  India  ") == "INDIA"
    assert normalize_country("france") == "FRANCE"
    assert normalize_country("DE") == "DE"
    assert normalize_country(None) == ""


def test_structural_extraction():
    addr = "124-B Elm Street, Suite 400, Mumbai 400051"
    assert extract_building_number(addr) == "124"
    assert extract_postal_code(addr) == ["400051"]
    assert extract_numeric_tokens(addr) == ["124", "400", "400051"]


def test_build_normalized_record():
    rec = build_normalized_record(
        entity_id="S1-001",
        business_name="Acme Corp.",
        business_address="123 Main Rd., Apt 4B, Springfield 62701",
        country="US"
    )
    assert rec.entity_id == "S1-001"
    assert rec.name_basic == "acme corp"
    assert rec.name_legal_normalized == "acme corporation"
    assert rec.address_expanded == "123 main road apartment 4b springfield 62701"
    assert rec.country_normalized == "US"
    assert rec.postal_codes == ["62701"]
    assert rec.building_number == "123"

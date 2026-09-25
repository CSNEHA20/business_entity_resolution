"""
Pytest configuration and synthetic fixtures for Business Entity Resolution tests.
"""

from pathlib import Path
import pytest
import pandas as pd


@pytest.fixture
def sample_toy_datasets(tmp_path: Path):
    """
    Creates valid synthetic training and test TSV files for testing.
    """
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()

    # Train Source 1
    s1_data = [
        ["S1-0001", "Acme Corporation Inc", "123 Main Rd, Springfield", "US"],
        ["S1-0002", "Tata Consultancy Services Pvt Ltd", "Bandra Kurla Complex, Mumbai", "India"],
        ["S1-0003", "Apex Dynamics LLC", "456 Market St, Austin", "US"],
        ["S1-0004", "Lonely Singleton Enterprises", "789 Solo Way, Chicago", "US"],
    ]
    s1_df = pd.DataFrame(s1_data, columns=["entity_id", "business_name", "business_address", "country"])
    s1_df.to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)

    # Train Source 2
    s2_data = [
        ["S2-0001", "Acme Corp", "123 Main Road, Springfield", "US"],
        ["S2-0002", "Tata Consultancy", "BKC, Mumbai", "India"],
        ["S2-0003", "Random S2 Business", "100 Other Rd", "US"],
    ]
    s2_df = pd.DataFrame(s2_data, columns=["entity_id", "business_name", "business_address", "country"])
    s2_df.to_csv(train_dir / "train_source2.tsv", sep="\t", index=False)

    # Train Source 3
    s3_data = [
        ["S3-0001", "Acme Incorporated", "123 Main", "US"],
        ["S3-0002", "TCS Ltd", "Mumbai", "India"],
        ["S3-0003", "Apex Dynamics", "456 Market Street, Austin, TX", "US"],
    ]
    s3_df = pd.DataFrame(s3_data, columns=["entity_id", "business_name", "business_address", "country"])
    s3_df.to_csv(train_dir / "train_source3.tsv", sep="\t", index=False)

    # Train Ground Truth
    # S1-0001 -> S2-0001, S3-0001 (multi match)
    # S1-0002 -> S2-0002, S3-0002 (multi match)
    # S1-0003 -> S3-0003 (single match)
    # S1-0004 -> (singleton / empty)
    gt_data = [
        ["S1-0001", "S2-0001,S3-0001"],
        ["S1-0002", "S2-0002,S3-0002"],
        ["S1-0003", "S3-0003"],
        ["S1-0004", ""],
    ]
    gt_df = pd.DataFrame(gt_data, columns=["source1_entity_id", "matched_entity_ids"])
    gt_df.to_csv(train_dir / "train_ground_truth.tsv", sep="\t", index=False)

    # Test Source 1 (including France)
    ts1_data = [
        ["S1-1001", "Acme Paris SARL", "10 Rue de Rivoli, Paris", "France"],
        ["S1-1002", "Tech Innovations", "99 Tech Blvd, San Jose", "US"],
    ]
    ts1_df = pd.DataFrame(ts1_data, columns=["entity_id", "business_name", "business_address", "country"])
    ts1_df.to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)

    # Test Source 2
    ts2_data = [
        ["S2-1001", "Acme Paris", "10 Rue Rivoli", "France"],
    ]
    ts2_df = pd.DataFrame(ts2_data, columns=["entity_id", "business_name", "business_address", "country"])
    ts2_df.to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)

    # Test Source 3
    ts3_data = [
        ["S3-1001", "Tech Innovations LLC", "99 Tech Blvd", "US"],
    ]
    ts3_df = pd.DataFrame(ts3_data, columns=["entity_id", "business_name", "business_address", "country"])
    ts3_df.to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)

    return tmp_path

#!/usr/bin/env python3
"""
Script 01: Dataset Audit and Schema Verification.
Inspects train and test datasets, verifies schema compliance, and logs dataset statistics.
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import DIAGNOSTICS_DIR, get_data_dir
from src.data_io import (
    audit_dataframe,
    load_ground_truth,
    load_test_source1,
    load_test_source2,
    load_test_source3,
    load_train_source1,
    load_train_source2,
    load_train_source3,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("01_audit")


def run_audit(data_dir: Path = None):
    active_dir = data_dir or get_data_dir()
    logger.info(f"Starting dataset audit using data directory: {active_dir.resolve()}")
    
    report = {"datasets": {}, "status": "PENDING"}
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Audit Training Data
        logger.info("Loading and validating train_source1...")
        s1 = load_train_source1(active_dir)
        report["datasets"]["train_source1"] = audit_dataframe(s1, "train_source1")

        logger.info("Loading and validating train_source2...")
        s2 = load_train_source2(active_dir)
        report["datasets"]["train_source2"] = audit_dataframe(s2, "train_source2")

        logger.info("Loading and validating train_source3...")
        s3 = load_train_source3(active_dir)
        report["datasets"]["train_source3"] = audit_dataframe(s3, "train_source3")

        logger.info("Loading and validating train_ground_truth...")
        gt = load_ground_truth(active_dir, valid_source1_ids=set(s1["entity_id"]))
        report["datasets"]["train_ground_truth"] = audit_dataframe(gt, "train_ground_truth")

        # 2. Audit Test Data
        logger.info("Loading and validating test_source1...")
        ts1 = load_test_source1(active_dir)
        report["datasets"]["test_source1"] = audit_dataframe(ts1, "test_source1")

        logger.info("Loading and validating test_source2...")
        ts2 = load_test_source2(active_dir)
        report["datasets"]["test_source2"] = audit_dataframe(ts2, "test_source2")

        logger.info("Loading and validating test_source3...")
        ts3 = load_test_source3(active_dir)
        report["datasets"]["test_source3"] = audit_dataframe(ts3, "test_source3")

        report["status"] = "SUCCESS"
        logger.info("All datasets successfully loaded and validated.")

    except FileNotFoundError as e:
        logger.warning(f"Dataset files not found: {e}")
        report["status"] = "FILES_NOT_FOUND"
        report["error"] = str(e)
    except Exception as e:
        logger.error(f"Audit failed with error: {e}", exc_info=True)
        report["status"] = "ERROR"
        report["error"] = str(e)

    # Save audit report
    out_file = DIAGNOSTICS_DIR / "data_audit_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Audit report saved to {out_file}")
    
    print("\n" + "=" * 50)
    print("DATASET AUDIT SUMMARY")
    print("=" * 50)
    print(f"Status: {report['status']}")
    for name, stats in report.get("datasets", {}).items():
        print(f"\n[{name}]")
        print(f"  Rows: {stats.get('num_rows')}")
        print(f"  Columns: {stats.get('columns')}")
        if "country_distribution" in stats:
            print(f"  Countries: {stats.get('country_distribution')}")
        if "num_singletons" in stats:
            print(f"  Singletons: {stats.get('num_singletons')}, Single-match: {stats.get('num_1_match')}, Multi-match: {stats.get('num_multi_match')}")
    print("=" * 50 + "\n")
    return report


if __name__ == "__main__":
    run_audit()

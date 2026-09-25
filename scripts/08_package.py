#!/usr/bin/env python3
"""
Script 08: Final Submission Packager.
Creates the compliant submission zip archive per official instructions.
"""

import argparse
import logging
import os
from pathlib import Path
import shutil
import sys
import zipfile

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import OUTPUT_DIR, PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("08_package")


def create_submission_package(team_name: str = "SubmissionTeam") -> Path:
    """
    Builds the final zip package containing:
    - output/matching_results.tsv
    - output/candidate_pairs.tsv
    - code/business_entity_resolution/src/
    - code/business_entity_resolution/README.md
    - code/business_entity_resolution/requirements.txt
    - Documentation_template.md
    """
    zip_filename = f"{team_name}_submission.zip"
    zip_path = PROJECT_ROOT / zip_filename

    matching_file = OUTPUT_DIR / "matching_results.tsv"
    candidate_file = OUTPUT_DIR / "candidate_pairs.tsv"
    doc_template = PROJECT_ROOT / "Documentation_template.md"

    logger.info(f"Building submission zip: {zip_path}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Output files
        if matching_file.exists():
            zf.write(matching_file, arcname="output/matching_results.tsv")
        if candidate_file.exists():
            zf.write(candidate_file, arcname="output/candidate_pairs.tsv")

        # 2. Code directory
        src_dir = PROJECT_ROOT / "src"
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith(".py"):
                    full_path = Path(root) / file
                    rel_path = full_path.relative_to(PROJECT_ROOT)
                    zf.write(full_path, arcname=f"code/business_entity_resolution/{rel_path.as_posix()}")

        # README and requirements
        readme_file = PROJECT_ROOT / "README.md"
        req_file = PROJECT_ROOT / "requirements.txt"
        if readme_file.exists():
            zf.write(readme_file, arcname="code/business_entity_resolution/README.md")
        if req_file.exists():
            zf.write(req_file, arcname="code/business_entity_resolution/requirements.txt")

        # 3. Documentation template
        if doc_template.exists():
            zf.write(doc_template, arcname="Documentation_template.md")

    logger.info(f"Successfully generated package: {zip_path} ({zip_path.stat().st_size} bytes)")
    return zip_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package submission zip archive.")
    parser.add_argument("--team-name", default="Business_Entity_Resolution_Team", help="Team name for zip archive")
    args = parser.parse_args()
    create_submission_package(args.team_name)

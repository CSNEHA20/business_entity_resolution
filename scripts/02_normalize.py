#!/usr/bin/env python3
"""
Script 02: Normalization Pipeline Runner.
(To be executed in Milestone 2).
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("02_normalize")

def main():
    logger.info("Normalization pipeline ready for execution.")

if __name__ == "__main__":
    main()

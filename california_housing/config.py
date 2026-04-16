"""
Central configuration — paths and constants used across the package.

Import from here rather than hard-coding paths in individual modules.
"""

from pathlib import Path

# Project root (two levels up from this file: california_housing/config.py → root)
ROOT_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
RAW_DATA_PATH       = ROOT_DIR / "data" / "raw" / "housing.csv"
INTERIM_DATA_DIR    = ROOT_DIR / "data" / "interim"
PROCESSED_DATA_DIR  = ROOT_DIR / "data" / "processed"
EXTERNAL_DATA_DIR   = ROOT_DIR / "data" / "external"

# ---------------------------------------------------------------------------
# Model artefacts
# ---------------------------------------------------------------------------
MODELS_DIR          = ROOT_DIR / "models"
MODEL_PATH          = MODELS_DIR / "best_model.pkl"
METADATA_PATH       = MODELS_DIR / "model_metadata.pkl"

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
REPORTS_DIR         = ROOT_DIR / "reports"
FIGURES_DIR         = ROOT_DIR / "reports" / "figures"
REPORT_PATH         = ROOT_DIR / "reports" / "model_report.md"

# ---------------------------------------------------------------------------
# Modelling constants
# ---------------------------------------------------------------------------
RANDOM_STATE        = 42
TEST_SIZE           = 0.20   # fraction of full dataset held out as test
VAL_SIZE            = 0.20   # fraction of full dataset used for validation
TARGET_COL          = "median_house_value"

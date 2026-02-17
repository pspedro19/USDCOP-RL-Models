#!/usr/bin/env python3
"""
Standalone macro pipeline update. No Docker/Airflow needed.
Requires: FRED_API_KEY in .env

Runs the 4-step macro pipeline + validation dataset generation:
  1. HPC Scraper  -> raw CSVs
  2. Fusion       -> DATASET_MACRO_*.csv
  3. Cleaning     -> MACRO_*_CLEAN.csv
  4. Resampling   -> MACRO_DAILY_CONSOLIDATED.csv
  5. Validation   -> generate_validation_datasets.py
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = PROJECT_ROOT / "data" / "pipeline"

STEPS = [
    {
        "name": "HPC Scraper",
        "script": PIPELINE_DIR / "02_scrapers" / "01_orchestrator" / "actualizador_hpc_v3.py",
    },
    {
        "name": "Fusion",
        "script": PIPELINE_DIR / "03_fusion" / "run_fusion.py",
    },
    {
        "name": "Cleaning",
        "script": PIPELINE_DIR / "04_cleaning" / "run_clean.py",
    },
    {
        "name": "Resampling",
        "script": PIPELINE_DIR / "05_resampling" / "run_resample.py",
    },
    {
        "name": "Validation Datasets",
        "script": PROJECT_ROOT / "scripts" / "generate_validation_datasets.py",
    },
]

CONSOLIDATED_CSV = PIPELINE_DIR / "05_resampling" / "output" / "MACRO_DAILY_CONSOLIDATED.csv"


def validate_env():
    """Check .env has FRED_API_KEY."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        print(f"ERROR: .env not found at {env_path}")
        sys.exit(1)

    content = env_path.read_text()
    if "FRED_API_KEY" not in content:
        print("ERROR: FRED_API_KEY not found in .env")
        sys.exit(1)

    print("OK: .env found with FRED_API_KEY")


def run_step(step):
    """Run a single pipeline step. Exit on failure."""
    script = step["script"]
    name = step["name"]

    if not script.exists():
        print(f"ERROR: Script not found: {script}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Step: {name}")
    print(f"  Script: {script.relative_to(PROJECT_ROOT)}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print(f"\nFAILED: {name} exited with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nOK: {name} completed successfully")


def print_summary():
    """Print summary of the consolidated CSV."""
    print(f"\n{'='*60}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*60}")

    if not CONSOLIDATED_CSV.exists():
        print(f"WARNING: {CONSOLIDATED_CSV} not found")
        return

    try:
        import pandas as pd
        df = pd.read_csv(CONSOLIDATED_CSV)
        date_col = "fecha" if "fecha" in df.columns else df.columns[0]
        print(f"  File:    {CONSOLIDATED_CSV.relative_to(PROJECT_ROOT)}")
        print(f"  Rows:    {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Range:   {df[date_col].min()} to {df[date_col].max()}")
    except ImportError:
        print("  (install pandas for detailed summary)")
    except Exception as e:
        print(f"  Error reading CSV: {e}")


def main():
    print(f"Macro Pipeline Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {PROJECT_ROOT}\n")

    validate_env()

    for step in STEPS:
        run_step(step)

    print_summary()
    print("\nDone.")


if __name__ == "__main__":
    main()

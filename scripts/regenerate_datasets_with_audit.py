#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate all 9 datasets with audit row containing source info and URLs.

Output structure:
- Row 1: Column headers (variable names)
- Row 2: Audit info (source | URL)
- Row 3+: Data

Datasets:
- MACRO_DAILY_MASTER (17 daily variables)
- MACRO_MONTHLY_MASTER (15 monthly variables)
- MACRO_QUARTERLY_MASTER (4 quarterly variables)
- Same for CLEAN and FUSION outputs
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_MASTER = PROJECT_ROOT / "data" / "pipeline" / "01_sources" / "consolidated"
OUTPUT_CLEAN = PROJECT_ROOT / "data" / "pipeline" / "04_cleaning" / "output"
OUTPUT_FUSION = PROJECT_ROOT / "data" / "pipeline" / "03_fusion" / "output"

# Load diccionario for audit info
DICCIONARIO = pd.read_csv(
    PROJECT_ROOT / "data" / "pipeline" / "00_config" / "DICCIONARIO_MACROECONOMICOS_FINAL.csv",
    sep=";",
    encoding="latin1"
)

# Create lookup dict: variable -> (source, url)
AUDIT_INFO = {}
for _, row in DICCIONARIO.iterrows():
    var = row["VARIABLE_NUEVA_ESTANDAR"]
    source = row["FUENTE_SECUNDARIA"]
    url = row["URL_DESCARGA"]
    AUDIT_INFO[var] = f"{source} | {url}"

# Define variables by frequency
DAILY_VARS = [
    "FINC_BOND_YIELD5Y_COL_D_COL5Y",
    "FINC_BOND_YIELD10Y_COL_D_COL10Y",
    "FINC_BOND_YIELD10Y_USA_D_UST10Y",
    "FINC_RATE_IBR_OVERNIGHT_COL_D_IBR",
    "POLR_PRIME_RATE_USA_D_PRIME",
    "COMM_OIL_WTI_GLB_D_WTI",
    "COMM_OIL_BRENT_GLB_D_BRENT",
    "COMM_METAL_GOLD_GLB_D_GOLD",
    "COMM_AGRI_COFFEE_GLB_D_COFFEE",
    "FXRT_SPOT_USDCLP_CHL_D_USDCLP",
    "FXRT_SPOT_USDMXN_MEX_D_USDMXN",
    "FXRT_INDEX_DXY_USA_D_DXY",
    "CRSK_SPREAD_EMBI_COL_D_EMBI",
    "EQTY_INDEX_COLCAP_COL_D_COLCAP",
    "FINC_BOND_YIELD2Y_USA_D_DGS2",
    "POLR_POLICY_RATE_COL_M_TPM",  # Daily per diccionario
    "VOLT_VIX_USA_D_VIX",
]

MONTHLY_VARS = [
    "POLR_FED_FUNDS_USA_M_FEDFUNDS",
    "INFL_CPI_ALL_USA_M_CPIAUCSL",
    "INFL_PCE_USA_M_PCEPI",
    "LABR_UNEMPLOYMENT_USA_M_UNRATE",
    "PROD_INDUSTRIAL_USA_M_INDPRO",
    "MNYS_M2_SUPPLY_USA_M_M2SL",
    "SENT_CONSUMER_USA_M_UMCSENT",
    "INFL_CPI_TOTAL_COL_M_IPCCOL",
    "FXRT_REER_BILATERAL_COL_M_ITCR",
    "RSBP_RESERVES_INTERNATIONAL_COL_M_RESINT",
    "FTRD_TERMS_TRADE_COL_M_TOT",
    "FTRD_EXPORTS_TOTAL_COL_M_EXPUSD",
    "FTRD_IMPORTS_TOTAL_COL_M_IMPUSD",
    "CRSK_SENTIMENT_CCI_COL_M_CCI",
    "CRSK_SENTIMENT_ICI_COL_M_ICI",
]

QUARTERLY_VARS = [
    "GDPP_REAL_GDP_USA_Q_GDP",
    "RSBP_FDI_INFLOW_COL_Q_FDIIN",
    "RSBP_FDI_OUTFLOW_COL_Q_FDIOUT",
    "RSBP_CURRENT_ACCOUNT_COL_Q_CACCT",
]


def get_audit_row(columns):
    """Create audit row with source info for each column."""
    audit = {}
    for col in columns:
        if col in AUDIT_INFO:
            audit[col] = AUDIT_INFO[col]
        else:
            audit[col] = "N/A"
    return audit


def save_with_audit(df, output_path, freq_name):
    """Save DataFrame with audit row as second row."""
    # Create audit row
    audit_row = get_audit_row(df.columns)

    # Convert to list for insertion
    audit_df = pd.DataFrame([audit_row], index=["FUENTE_URL"])

    # Combine: audit row first, then data
    df_with_audit = pd.concat([audit_df, df])

    # Save CSV
    csv_path = output_path.with_suffix(".csv")
    df_with_audit.to_csv(csv_path, float_format="%.6f")

    # Save Excel
    xlsx_path = output_path.with_suffix(".xlsx")
    df_with_audit.to_excel(xlsx_path, freeze_panes=(2, 1))

    # Save Parquet (without audit row - pure data)
    parquet_path = output_path.with_suffix(".parquet")
    df.to_parquet(parquet_path, engine="pyarrow")

    print(f"  Saved {output_path.stem}: {len(df)} rows x {len(df.columns)} cols")


def main():
    print("=" * 70)
    print("REGENERATING DATASETS WITH AUDIT ROW")
    print("=" * 70)

    # Load existing data
    print("\n1. Loading source data...")

    # Load from current master (has all the data we need)
    df_daily_src = pd.read_csv(
        OUTPUT_MASTER / "MACRO_DAILY_MASTER.csv",
        index_col=0,
        parse_dates=True
    )
    df_monthly_src = pd.read_csv(
        OUTPUT_MASTER / "MACRO_MONTHLY_MASTER.csv",
        index_col=0,
        parse_dates=True
    )
    df_quarterly_src = pd.read_csv(
        OUTPUT_MASTER / "MACRO_QUARTERLY_MASTER.csv",
        index_col=0,
        parse_dates=True
    )

    # ========================================
    # CREATE DAILY DATASET
    # ========================================
    print("\n2. Creating DAILY dataset (17 vars)...")
    daily_cols = [c for c in DAILY_VARS if c in df_daily_src.columns]
    df_daily = df_daily_src[daily_cols].copy()
    df_daily = df_daily.sort_index()
    df_daily = df_daily[df_daily.index >= "2016-01-01"]
    df_daily = df_daily[df_daily.index.dayofweek < 5]

    print(f"   Columns: {len(df_daily.columns)}")
    print(f"   Rows: {len(df_daily)}")

    # ========================================
    # CREATE MONTHLY DATASET
    # ========================================
    print("\n3. Creating MONTHLY dataset (15 vars)...")
    monthly_cols = [c for c in MONTHLY_VARS if c in df_monthly_src.columns]
    df_monthly = df_monthly_src[monthly_cols].copy()
    df_monthly = df_monthly.sort_index()
    df_monthly = df_monthly[df_monthly.index >= "2016-01-01"]
    df_monthly = df_monthly.dropna(how="all")

    print(f"   Columns: {len(df_monthly.columns)}")
    print(f"   Rows: {len(df_monthly)}")

    # ========================================
    # CREATE QUARTERLY DATASET
    # ========================================
    print("\n4. Creating QUARTERLY dataset (4 vars)...")
    quarterly_cols = [c for c in QUARTERLY_VARS if c in df_quarterly_src.columns]
    df_quarterly = df_quarterly_src[quarterly_cols].copy()
    df_quarterly = df_quarterly.sort_index()
    df_quarterly = df_quarterly[df_quarterly.index >= "2016-01-01"]
    df_quarterly = df_quarterly.dropna(how="all")

    print(f"   Columns: {len(df_quarterly.columns)}")
    print(f"   Rows: {len(df_quarterly)}")

    # ========================================
    # SAVE ALL 9 DATASETS
    # ========================================
    print("\n5. Saving all 9 datasets with audit rows...")

    # Ensure output directories exist
    for out_dir in [OUTPUT_MASTER, OUTPUT_CLEAN, OUTPUT_FUSION]:
        out_dir.mkdir(parents=True, exist_ok=True)

    # MASTER
    print("\n  MASTER:")
    save_with_audit(df_daily, OUTPUT_MASTER / "MACRO_DAILY_MASTER", "DAILY")
    save_with_audit(df_monthly, OUTPUT_MASTER / "MACRO_MONTHLY_MASTER", "MONTHLY")
    save_with_audit(df_quarterly, OUTPUT_MASTER / "MACRO_QUARTERLY_MASTER", "QUARTERLY")

    # CLEAN
    print("\n  CLEAN:")
    save_with_audit(df_daily, OUTPUT_CLEAN / "MACRO_DAILY_CLEAN", "DAILY")
    save_with_audit(df_monthly, OUTPUT_CLEAN / "MACRO_MONTHLY_CLEAN", "MONTHLY")
    save_with_audit(df_quarterly, OUTPUT_CLEAN / "MACRO_QUARTERLY_CLEAN", "QUARTERLY")

    # FUSION
    print("\n  FUSION:")
    save_with_audit(df_daily, OUTPUT_FUSION / "DATASET_MACRO_DAILY", "DAILY")
    save_with_audit(df_monthly, OUTPUT_FUSION / "DATASET_MACRO_MONTHLY", "MONTHLY")
    save_with_audit(df_quarterly, OUTPUT_FUSION / "DATASET_MACRO_QUARTERLY", "QUARTERLY")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY - ALL 9 DATASETS GENERATED WITH AUDIT ROW")
    print("=" * 70)

    print("""
CSV/Excel Structure:
  Row 1: Column headers (variable names)
  Row 2: FUENTE_URL (Source | URL for verification)
  Row 3+: Data values

Parquet: Pure data without audit row (for ML pipelines)
""")

    print("DAILY (17 variables):")
    for col in sorted(df_daily.columns):
        s = df_daily[col].dropna()
        source = AUDIT_INFO.get(col, "N/A").split(" | ")[0]
        print(f"  {col}")
        print(f"    Source: {source}")
        print(f"    Range: {s.index.min().strftime('%Y-%m-%d')} -> {s.index.max().strftime('%Y-%m-%d')} ({len(s)} rows)")

    print("\nMONTHLY (15 variables):")
    for col in sorted(df_monthly.columns):
        s = df_monthly[col].dropna()
        source = AUDIT_INFO.get(col, "N/A").split(" | ")[0]
        if len(s) > 0:
            print(f"  {col}")
            print(f"    Source: {source}")
            print(f"    Range: {s.index.min().strftime('%Y-%m')} -> {s.index.max().strftime('%Y-%m')} ({len(s)} rows)")

    print("\nQUARTERLY (4 variables):")
    for col in sorted(df_quarterly.columns):
        s = df_quarterly[col].dropna()
        source = AUDIT_INFO.get(col, "N/A").split(" | ")[0]
        if len(s) > 0:
            print(f"  {col}")
            print(f"    Source: {source}")
            print(f"    Range: {s.index.min().strftime('%Y-%m')} -> {s.index.max().strftime('%Y-%m')} ({len(s)} rows)")

    print("\n" + "=" * 70)
    print("Output locations:")
    print(f"  MASTER: {OUTPUT_MASTER}")
    print(f"  CLEAN:  {OUTPUT_CLEAN}")
    print(f"  FUSION: {OUTPUT_FUSION}")
    print("=" * 70)


if __name__ == "__main__":
    main()

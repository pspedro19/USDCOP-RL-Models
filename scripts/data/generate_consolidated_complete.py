#!/usr/bin/env python3
"""
Generate Complete Consolidated Datasets from SSOT
==================================================
Contract: CTR-L0-SSOT-001

This script is the AUTHORITATIVE generator for consolidated macro datasets.
It reads variable definitions EXCLUSIVELY from the SSOT (Single Source of Truth).

Generates 9 files (3 datasets x 3 formats):
- MACRO_DAILY_MASTER (.csv, .parquet, .xlsx)
- MACRO_MONTHLY_MASTER (.csv, .parquet, .xlsx)
- MACRO_QUARTERLY_MASTER (.csv, .parquet, .xlsx)

Data flow:
1. Read variable definitions from SSOT (39 variables)
2. Load existing HPC datasets
3. Normalize column names to SSOT canonical names
4. Extract missing variables via Airflow extractors
5. Filter to date range (2015-01-01 to present)
6. Save in 3 formats

DRY Principle: ALL variable definitions come from SSOT - NO DUPLICATION.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from .env (CRITICAL: required for FRED API key)
load_dotenv(PROJECT_ROOT / '.env')

sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'airflow' / 'dags'))

from data.macro_ssot import MacroSSOT

# Paths
HPC_DATASETS = PROJECT_ROOT / 'data' / 'pipeline' / '02_scrapers' / 'storage' / 'datasets'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'pipeline' / '01_sources' / 'consolidated'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Backup CSV sources for variables that can't be extracted via API
BACKUP_SOURCES = {
    'crsk_spread_embi_col_d_embi': PROJECT_ROOT / 'data' / 'pipeline' / '01_sources' / '06_country_risk' / 'embi_COL_d.csv',
}

FORMATS = ['csv', 'parquet', 'xlsx']
START_DATE = '2015-01-01'

# ==============================================================================
# FORCE FRESH EXTRACTION MODE
# Set to True to skip HPC datasets and extract ALL variables from their sources
# This ensures clean, authoritative data directly from APIs
# ==============================================================================
FORCE_FRESH_EXTRACTION = True  # Set to False to use HPC datasets as base

# Enable HPC fallback when fresh extraction fails
# DISABLED: All extractors should now be functional with Selenium bypass for SUAMECA
HPC_FALLBACK_ENABLED = False  # Force fresh extraction from all sources

# Variables that previously required HPC fallback (now have working Selenium extractors)
# - infl_cpi_total_col_m_ipccol: SUAMECA serie 100002 via Selenium
# - ftrd_terms_trade_col_m_tot: SUAMECA serie 4180 via Selenium
# - rsbp_current_account_col_q_cacct: SUAMECA serie 414001 via Selenium
HPC_FALLBACK_REQUIRED = set()  # Empty - all extractors should work

# Variables to force re-extract (sparse data in HPC, need daily extraction)
# These variables exist in HPC but only have change-point data, not daily data
FORCE_EXTRACT_VARS = {
    'polr_prime_rate_usa_d_prime',  # HPC only has rate changes, need daily data from SUAMECA
}

# Variables to forward-fill (business day data needs to fill weekends/holidays)
# These variables only have values on business days but should fill to all days
FFILL_VARS = {
    'polr_prime_rate_usa_d_prime',  # Fill weekends/holidays with last business day value
    'polr_policy_rate_col_d_tpm',   # Same logic
    'finc_rate_ibr_overnight_col_d_ibr',  # Same logic
}

# Source URL templates
SOURCE_URLS = {
    'fred': 'https://fred.stlouisfed.org/series/{series_id}',
    'suameca': 'https://suameca.banrep.gov.co/estadisticas-economicas-back/rest/estadisticaEconomicaRestService/consultaInformacionSerie?idSerie={serie_id}',
    'bcrp': 'https://estadisticas.bcrp.gob.pe/estadisticas/series/diarias/resultados/{serie_code}/html',
    'investing': 'https://www.investing.com (instrument_id={instrument_id})',
}


def get_source_info(var_name: str) -> str:
    """Get source URL/info for a variable from SSOT."""
    ssot = MacroSSOT()
    var = ssot.get_variable(var_name)

    if not var:
        return "Unknown source"

    source = var.extraction.primary_source
    source_cfg = var.extraction.source_configs.get(source, {})

    # Check if there's a direct URL in config
    if 'url' in source_cfg:
        return f"{source.upper()}: {source_cfg['url']}"

    # Build URL from template
    if source == 'fred' and 'series_id' in source_cfg:
        return f"FRED: https://fred.stlouisfed.org/series/{source_cfg['series_id']}"

    if source == 'suameca' and 'serie_id' in source_cfg:
        return f"SUAMECA/BanRep: https://suameca.banrep.gov.co/graficador-interactivo/grafica/{source_cfg['serie_id']}"

    if source == 'bcrp' and 'serie_code' in source_cfg:
        return f"BCRP: https://estadisticas.bcrp.gob.pe/estadisticas/series/diarias/resultados/{source_cfg['serie_code']}/html"

    if source == 'investing':
        instrument_id = source_cfg.get('instrument_id') or source_cfg.get('pair_id', 'N/A')
        referer = source_cfg.get('referer_url') or source_cfg.get('url', '')
        if referer:
            return f"Investing.com: {referer}"
        return f"Investing.com (instrument_id={instrument_id})"

    return f"{source.upper()}: {source_cfg}"


def build_sources_row(columns: List[str]) -> Dict[str, str]:
    """Build a dictionary with source info for each column."""
    sources = {'fecha': 'Date column'}

    for col in columns:
        if col == 'fecha':
            continue
        sources[col] = get_source_info(col)

    return sources


def load_ssot_variables() -> Dict[str, List[str]]:
    """Load variable definitions from SSOT.

    Returns:
        Dict with keys 'daily', 'monthly', 'quarterly', each containing
        a list of canonical variable names.
    """
    ssot = MacroSSOT()

    # Get variables by frequency from SSOT groups
    result = {
        'daily': ssot.get_variables_by_group('daily'),
        'monthly': ssot.get_variables_by_group('monthly'),
        'quarterly': ssot.get_variables_by_group('quarterly'),
    }

    print(f"[SSOT] Loaded variable definitions:")
    print(f"  Daily: {len(result['daily'])} variables")
    print(f"  Monthly: {len(result['monthly'])} variables")
    print(f"  Quarterly: {len(result['quarterly'])} variables")
    print(f"  TOTAL: {sum(len(v) for v in result.values())} variables")

    return result


def load_hpc_datasets() -> Dict[str, pd.DataFrame]:
    """Load existing HPC datasets (or return empty if FORCE_FRESH_EXTRACTION)."""
    datasets = {}

    # Skip HPC loading if forcing fresh extraction
    if FORCE_FRESH_EXTRACTION:
        print("[HPC] SKIPPED - FORCE_FRESH_EXTRACTION=True")
        print("[HPC] All variables will be extracted fresh from their sources")
        return {
            'daily': pd.DataFrame(),
            'monthly': pd.DataFrame(),
            'quarterly': pd.DataFrame(),
        }

    freq_map = {
        'daily': 'diarios',
        'monthly': 'mensuales',
        'quarterly': 'trimestrales',
    }

    for freq, name in freq_map.items():
        path = HPC_DATASETS / f'datos_{name}_hpc.csv'
        if path.exists():
            df = pd.read_csv(path, encoding='utf-8-sig')
            if 'fecha' in df.columns:
                df['fecha'] = pd.to_datetime(df['fecha'])
                datasets[freq] = df
                print(f"[HPC] Loaded {name}: {len(df)} rows, {len(df.columns)-1} vars")
            else:
                print(f"[HPC] WARNING: {path.name} missing 'fecha' column")
        else:
            print(f"[HPC] File not found: {path}")

    return datasets


def normalize_columns(df: pd.DataFrame, ssot_vars: List[str], freq: str) -> pd.DataFrame:
    """Normalize DataFrame columns to match SSOT canonical names.

    Handles case differences and minor naming variations.
    Also removes columns that don't belong to this frequency.
    """
    if df.empty:
        return df

    # Build mapping: lowercase -> SSOT canonical name
    ssot_lower = {v.lower(): v for v in ssot_vars}

    # Known column name variations (HPC name -> SSOT name)
    known_mappings = {
        'gdpp_real_gdp_usa_q_gdp': 'gdpp_real_gdp_usa_q_gdp_q',  # HPC uses shorter name
    }

    # Columns to remove (wrong frequency in dataset)
    cols_to_remove = []

    # Build rename mapping for this DataFrame
    rename_map = {}
    for col in df.columns:
        if col == 'fecha':
            continue

        col_lower = col.lower()

        # Check if column is in wrong dataset (e.g., _M_ in daily dataset)
        if freq == 'daily' and '_m_' in col_lower:
            cols_to_remove.append(col)
            continue
        if freq == 'monthly' and '_d_' in col_lower:
            cols_to_remove.append(col)
            continue

        # Check known mappings first
        if col_lower in known_mappings:
            rename_map[col] = known_mappings[col_lower]
        elif col_lower in ssot_lower:
            canonical = ssot_lower[col_lower]
            if col != canonical:
                rename_map[col] = canonical

    # Remove columns in wrong frequency
    if cols_to_remove:
        df = df.drop(columns=cols_to_remove)
        print(f"  Removed {len(cols_to_remove)} columns (wrong frequency): {cols_to_remove}")

    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  Renamed {len(rename_map)} columns to SSOT canonical names")

    return df


def get_missing_variables(df: pd.DataFrame, ssot_vars: List[str]) -> Set[str]:
    """Identify SSOT variables missing from the DataFrame."""
    if df.empty:
        return set(ssot_vars)

    existing = {c.lower() for c in df.columns if c != 'fecha'}
    expected = {v.lower() for v in ssot_vars}
    missing_lower = expected - existing

    # Map back to canonical names
    missing = {v for v in ssot_vars if v.lower() in missing_lower}
    return missing


def load_hpc_variable(var: str, freq: str) -> Optional[pd.DataFrame]:
    """Load a single variable from HPC dataset as fallback.

    Args:
        var: Variable name to load
        freq: Frequency ('daily', 'monthly', 'quarterly')

    Returns:
        DataFrame with fecha and variable column, or None if not found
    """
    freq_map = {
        'daily': 'diarios',
        'monthly': 'mensuales',
        'quarterly': 'trimestrales',
    }

    hpc_file = HPC_DATASETS / f'datos_{freq_map[freq]}_hpc.csv'
    if not hpc_file.exists():
        return None

    try:
        df = pd.read_csv(hpc_file, encoding='utf-8-sig')
        df['fecha'] = pd.to_datetime(df['fecha'])

        # Find column (case-insensitive)
        var_col = None
        for col in df.columns:
            if col.lower() == var.lower():
                var_col = col
                break

        if var_col is None:
            return None

        result = df[['fecha', var_col]].copy()
        result = result.rename(columns={var_col: var})
        result = result.dropna(subset=[var])

        if result.empty:
            return None

        return result
    except Exception:
        return None


def extract_missing_variables(missing_vars: Set[str], force_vars: Set[str] = None, freq: str = 'daily') -> Dict[str, pd.DataFrame]:
    """Extract missing variables using Airflow extractors and custom scrapers.

    Args:
        missing_vars: Variables not present in HPC
        force_vars: Variables to force re-extract even if present (to fill gaps)
        freq: Frequency of variables being extracted (for HPC fallback)
    """
    all_vars = missing_vars | (force_vars or set())

    if not all_vars:
        return {}

    print(f"\n[EXTRACT] Attempting to extract {len(all_vars)} variables...")
    if force_vars:
        print(f"  (Force re-extract to fill gaps: {force_vars})")

    # Fedesarrollo variables mapping
    FEDESARROLLO_VARS = {
        'crsk_sentiment_cci_col_m_cci': 'obtener_cci',
        'crsk_sentiment_ici_col_m_ici': 'obtener_ici',
        'infl_exp_eof_col_m_infexp': 'obtener_eof_inflacion',
    }

    try:
        from extractors.registry import ExtractorRegistry
        # Reset singleton to ensure fresh config
        ExtractorRegistry._instance = None
        ExtractorRegistry._initialized = False
        registry = ExtractorRegistry()
    except Exception as e:
        print(f"[EXTRACT] ERROR: Could not load extractors: {e}")
        registry = None

    extracted = {}
    from datetime import datetime
    start = datetime(2015, 1, 1)
    end = datetime.now()

    for var in all_vars:
        try:
            # Check if this is a Fedesarrollo variable
            if var in FEDESARROLLO_VARS:
                func_name = FEDESARROLLO_VARS[var]
                try:
                    # Import Fedesarrollo scraper
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(
                        "scraper_fedesarrollo",
                        PROJECT_ROOT / 'data' / 'pipeline' / '02_scrapers' / '02_custom' / 'scraper_fedesarrollo.py'
                    )
                    fedesarrollo = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(fedesarrollo)

                    # Get the function
                    func = getattr(fedesarrollo, func_name)
                    df = func(n=500)  # Get all historical data

                    if df is not None and not df.empty:
                        df['fecha'] = pd.to_datetime(df['fecha'])
                        df = df.rename(columns={'valor': var})
                        df = df[df['fecha'] >= start]
                        extracted[var] = df[['fecha', var]]
                        print(f"  [OK] {var}: {len(df)} rows (Fedesarrollo)")
                    else:
                        print(f"  [WARN] {var}: Fedesarrollo returned empty")
                except Exception as e:
                    print(f"  [FAIL] {var}: Fedesarrollo error - {e}")
                continue

            # Use ExtractorRegistry for other variables
            if registry is None:
                print(f"  [SKIP] {var}: no registry available")
                continue

            result = registry.extract_variable(var, start_date=start, end_date=end)
            if result.success and result.data is not None and not result.data.empty:
                df = result.data
                if 'fecha' not in df.columns and df.index.name == 'fecha':
                    df = df.reset_index()
                df['fecha'] = pd.to_datetime(df['fecha'])

                # Ensure we have the variable column
                if var not in df.columns and var.lower() in [c.lower() for c in df.columns]:
                    # Find and rename
                    for c in df.columns:
                        if c.lower() == var.lower() and c != 'fecha':
                            df = df.rename(columns={c: var})
                            break

                if var in df.columns:
                    extracted[var] = df[['fecha', var]]
                    print(f"  [OK] {var}: {len(df)} rows")
                else:
                    print(f"  [WARN] {var}: extracted but column not found")
                    # Try HPC fallback
                    if HPC_FALLBACK_ENABLED:
                        hpc_df = load_hpc_variable(var, freq)
                        if hpc_df is not None:
                            extracted[var] = hpc_df
                            print(f"  [HPC-FALLBACK] {var}: {len(hpc_df)} rows from HPC")
            else:
                # Extraction failed - try HPC fallback
                if HPC_FALLBACK_ENABLED:
                    hpc_df = load_hpc_variable(var, freq)
                    if hpc_df is not None:
                        extracted[var] = hpc_df
                        print(f"  [HPC-FALLBACK] {var}: {len(hpc_df)} rows from HPC")
                    else:
                        print(f"  [FAIL] {var}: extraction failed, no HPC fallback available")
                else:
                    print(f"  [FAIL] {var}: {result.error if result else 'unknown error'}")
        except Exception as e:
            print(f"  [FAIL] {var}: {e}")
            # Try HPC fallback on exception
            if HPC_FALLBACK_ENABLED:
                hpc_df = load_hpc_variable(var, freq)
                if hpc_df is not None:
                    extracted[var] = hpc_df
                    print(f"  [HPC-FALLBACK] {var}: {len(hpc_df)} rows from HPC")

    return extracted


def merge_and_fill(
    hpc_df: pd.DataFrame,
    extracted: Dict[str, pd.DataFrame],
    ssot_vars: List[str],
    force_vars: Set[str] = None
) -> pd.DataFrame:
    """Merge HPC data with extracted variables.

    For force_vars, the extracted data REPLACES HPC data (fills gaps).
    """
    force_vars = force_vars or set()

    if hpc_df.empty:
        # Start with first extracted variable
        if not extracted:
            return pd.DataFrame({'fecha': []})

        first_var = list(extracted.keys())[0]
        result = extracted[first_var].copy()
        for var, df in list(extracted.items())[1:]:
            result = result.merge(df, on='fecha', how='outer')
    else:
        result = hpc_df.copy()

        for var, df in extracted.items():
            if var in result.columns:
                if var in force_vars:
                    # Force-extracted: drop HPC column and use extracted data
                    result = result.drop(columns=[var])
                    result = result.merge(df, on='fecha', how='outer')
                    print(f"  [REPLACE] {var}: HPC sparse data replaced with extracted data")
                else:
                    continue  # Skip if already present and not forced
            else:
                result = result.merge(df, on='fecha', how='outer')

    # Sort by date
    result = result.sort_values('fecha').reset_index(drop=True)

    return result


def filter_date_range(df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """Filter DataFrame to start from start_date."""
    if df.empty:
        return df

    df = df[df['fecha'] >= start_date].reset_index(drop=True)
    return df


def apply_ffill(df: pd.DataFrame, ffill_vars: Set[str], max_days: int = 5) -> pd.DataFrame:
    """Apply forward-fill to specified variables (for weekends/holidays).

    Args:
        df: DataFrame with fecha column
        ffill_vars: Variables to forward-fill
        max_days: Maximum number of days to forward-fill (default 5 for weekends + holidays)

    Returns:
        DataFrame with ffill applied to specified columns
    """
    if df.empty or not ffill_vars:
        return df

    df = df.copy()
    filled_count = 0

    for var in ffill_vars:
        if var in df.columns:
            before_nulls = df[var].isna().sum()
            df[var] = df[var].ffill(limit=max_days)
            after_nulls = df[var].isna().sum()
            filled = before_nulls - after_nulls
            if filled > 0:
                filled_count += filled
                print(f"  [FFILL] {var}: filled {filled} weekend/holiday gaps")

    return df


def ensure_column_order(df: pd.DataFrame, ssot_vars: List[str]) -> pd.DataFrame:
    """Ensure columns are ordered: fecha first, then SSOT vars in order."""
    if df.empty:
        return df

    # Start with fecha
    cols = ['fecha']

    # Add SSOT vars that exist in df
    existing_lower = {c.lower(): c for c in df.columns if c != 'fecha'}
    for var in ssot_vars:
        if var.lower() in existing_lower:
            actual_col = existing_lower[var.lower()]
            if actual_col not in cols:
                cols.append(actual_col)

    # Add any remaining columns not in SSOT (shouldn't happen but safe)
    for col in df.columns:
        if col not in cols:
            cols.append(col)

    return df[cols]


def save_datasets(datasets: Dict[str, pd.DataFrame], ssot_vars: Dict[str, List[str]]):
    """Save datasets in 3 formats with source info in second row."""
    freq_names = {
        'daily': 'DAILY',
        'monthly': 'MONTHLY',
        'quarterly': 'QUARTERLY',
    }

    files_created = []

    for freq, df in datasets.items():
        if df.empty:
            print(f"  [SKIP] {freq}: empty DataFrame")
            continue

        name = f"MACRO_{freq_names[freq]}_MASTER"

        # Count variables (excluding fecha)
        var_count = len([c for c in df.columns if c != 'fecha'])
        expected_count = len(ssot_vars[freq])

        print(f"\n  [{freq.upper()}] {var_count}/{expected_count} SSOT variables")

        # List present and missing
        present = [c for c in df.columns if c != 'fecha']
        expected = [v for v in ssot_vars[freq]]
        missing = [v for v in expected if v not in present and v.upper() not in [p.upper() for p in present]]

        if missing:
            print(f"    Missing: {missing}")

        # Build sources row
        sources_row = build_sources_row(df.columns.tolist())

        for fmt in FORMATS:
            path = OUTPUT_DIR / f"{name}.{fmt}"

            try:
                if fmt == 'csv':
                    # Create DataFrame with sources as first data row
                    sources_df = pd.DataFrame([sources_row])
                    combined_df = pd.concat([sources_df, df], ignore_index=True)
                    combined_df.to_csv(path, index=False, encoding='utf-8')

                elif fmt == 'parquet':
                    # Parquet: save data only (sources in separate metadata or skip)
                    df.to_parquet(path, index=False, engine='pyarrow')

                elif fmt == 'xlsx':
                    # Excel: add sources as first row after header
                    sources_df = pd.DataFrame([sources_row])
                    combined_df = pd.concat([sources_df, df], ignore_index=True)
                    combined_df.to_excel(path, index=False, engine='openpyxl')

                size_kb = path.stat().st_size / 1024
                files_created.append({
                    'name': path.name,
                    'size_kb': round(size_kb, 1),
                    'rows': len(df),
                    'vars': var_count,
                    'expected': expected_count,
                })
                print(f"    [OK] {path.name} ({size_kb:.1f} KB)")
            except Exception as e:
                print(f"    [ERROR] {path.name}: {e}")

    return files_created


def main():
    import time
    start_time = time.time()

    print("=" * 70)
    print("GENERATE CONSOLIDATED DATASETS FROM SSOT")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Start date: {START_DATE}")
    print(f"Extraction mode: {'FRESH (all from sources)' if FORCE_FRESH_EXTRACTION else 'INCREMENTAL (HPC + missing)'}")
    print("=" * 70)

    # Step 1: Load SSOT variable definitions
    print("\n[STEP 1] Loading SSOT variable definitions...")
    ssot_vars = load_ssot_variables()

    # Step 2: Load HPC datasets
    print("\n[STEP 2] Loading HPC datasets...")
    hpc_datasets = load_hpc_datasets()

    # Step 3: Process each frequency
    print("\n[STEP 3] Processing datasets...")

    final_datasets = {}

    for freq in ['daily', 'monthly', 'quarterly']:
        print(f"\n--- {freq.upper()} ---")

        # Get HPC data
        hpc_df = hpc_datasets.get(freq, pd.DataFrame())

        # Normalize column names
        hpc_df = normalize_columns(hpc_df, ssot_vars[freq], freq)

        # Find missing variables
        missing = get_missing_variables(hpc_df, ssot_vars[freq])
        if missing:
            print(f"  Missing from HPC: {len(missing)} variables")
            print(f"    {list(missing)}")

        # Determine force-extract variables for this frequency
        force_vars = FORCE_EXTRACT_VARS & set(ssot_vars[freq])
        if force_vars:
            print(f"  Force re-extract (sparse HPC data): {list(force_vars)}")

        # Extract missing variables + force vars
        extracted = extract_missing_variables(missing, force_vars, freq=freq)

        # Merge (with force_vars to replace HPC sparse data)
        merged_df = merge_and_fill(hpc_df, extracted, ssot_vars[freq], force_vars)

        # Filter date range
        merged_df = filter_date_range(merged_df, START_DATE)

        # Apply forward-fill to fill weekends/holidays
        ffill_vars_freq = FFILL_VARS & set(ssot_vars[freq])
        if ffill_vars_freq:
            merged_df = apply_ffill(merged_df, ffill_vars_freq, max_days=5)

        # Ensure column order
        merged_df = ensure_column_order(merged_df, ssot_vars[freq])

        final_datasets[freq] = merged_df

        print(f"  Final: {len(merged_df)} rows, {len(merged_df.columns)-1} variables")

    # Step 4: Save
    print("\n[STEP 4] Saving consolidated files...")
    files = save_datasets(final_datasets, ssot_vars)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files created: {len(files)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nExpected from SSOT:")
    print(f"  Daily: {len(ssot_vars['daily'])} variables")
    print(f"  Monthly: {len(ssot_vars['monthly'])} variables")
    print(f"  Quarterly: {len(ssot_vars['quarterly'])} variables")

    print(f"\nGenerated files:")
    for f in files:
        status = "OK" if f['vars'] == f['expected'] else f"({f['vars']}/{f['expected']})"
        print(f"  {f['name']}: {f['rows']} rows, {f['vars']} vars {status}")

    total_expected = sum(len(v) for v in ssot_vars.values())
    total_generated = sum(f['vars'] for f in files) // 3  # Divide by 3 formats

    print(f"\nCoverage: {total_generated}/{total_expected} SSOT variables")

    if total_generated < total_expected:
        print("\n[WARNING] Some variables could not be extracted.")
        print("  Run Airflow extractors or HPC with updated config to fill gaps.")

    # Timing summary
    elapsed = time.time() - start_time
    print(f"\n[TIMING] Total execution: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("\n[DONE]")


if __name__ == '__main__':
    main()

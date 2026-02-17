#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update ALL monthly macro variables to latest available data.

This script uses the BEST available extraction method for each source:
- SUAMECA (IPCCOL, ITCR, TOT): Selenium with undetected-chromedriver
- DANE (EXPUSD, IMPUSD): Direct Excel download
- Fedesarrollo (CCI, ICI): PDF scraping

The script updates MACRO_MONTHLY_MASTER.csv with all new data.

Usage:
    python scripts/update_all_monthly_macro.py
    python scripts/update_all_monthly_macro.py --source suameca  # Only SUAMECA
    python scripts/update_all_monthly_macro.py --source dane     # Only DANE
    python scripts/update_all_monthly_macro.py --source fedes    # Only Fedesarrollo
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import time

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRAPERS_DIR = PROJECT_ROOT / "data" / "pipeline" / "02_scrapers" / "02_custom"
UTILS_DIR = PROJECT_ROOT / "airflow" / "dags" / "utils"
OUTPUT_DIR = PROJECT_ROOT / "data" / "pipeline" / "01_sources" / "consolidated"

sys.path.insert(0, str(SCRAPERS_DIR))
sys.path.insert(0, str(UTILS_DIR))

# Column name mappings
SUAMECA_COLUMNS = {
    'IPCCOL': 'INFL_CPI_TOTAL_COL_M_IPCCOL',
    'ITCR': 'FXRT_REER_BILATERAL_COL_M_ITCR',
    'TOT': 'FTRD_TERMS_TRADE_COL_M_TOT',
    'RESINT': 'RSBP_RESERVES_INTERNATIONAL_COL_M_RESINT',
}

DANE_COLUMNS = {
    'EXPUSD': 'FTRD_EXPORTS_TOTAL_COL_M_EXPUSD',
    'IMPUSD': 'FTRD_IMPORTS_TOTAL_COL_M_IMPUSD',
}

FEDESARROLLO_COLUMNS = {
    'CCI': 'CRSK_SENTIMENT_CCI_COL_M_CCI',
    'ICI': 'CRSK_SENTIMENT_ICI_COL_M_ICI',
}


def load_monthly_master():
    """Load current monthly master CSV."""
    csv_path = OUTPUT_DIR / "MACRO_MONTHLY_MASTER.csv"
    print(f"\n[LOAD] Loading {csv_path}...")

    df = pd.read_csv(csv_path, index_col=0)

    # Separate audit row
    audit_row = df.loc["FUENTE_URL"] if "FUENTE_URL" in df.index else None
    df_data = df.drop("FUENTE_URL", errors="ignore")
    df_data.index = pd.to_datetime(df_data.index)

    print(f"  Rows: {len(df_data)}")
    print(f"  Date range: {df_data.index.min()} to {df_data.index.max()}")

    return df_data, audit_row


def save_monthly_master(df_data, audit_row):
    """Save updated monthly master CSV."""
    csv_path = OUTPUT_DIR / "MACRO_MONTHLY_MASTER.csv"

    # Sort by index
    df_data = df_data.sort_index()

    # Reconstruct with audit row
    if audit_row is not None:
        df_final = pd.concat([pd.DataFrame([audit_row], index=["FUENTE_URL"]), df_data])
    else:
        df_final = df_data

    # Save CSV
    df_final.to_csv(csv_path)
    print(f"\n[SAVED] {csv_path}")

    # Save parquet (data only)
    parquet_path = csv_path.with_suffix(".parquet")
    try:
        df_data.to_parquet(parquet_path, engine="pyarrow")
        print(f"[SAVED] {parquet_path}")
    except Exception as e:
        print(f"[WARN] Could not save parquet: {e}")


def update_dataframe(df_data, col_name, new_data, use_end_of_month=True):
    """
    Update DataFrame with new data.

    Args:
        df_data: Main DataFrame
        col_name: Column name to update
        new_data: List of dicts with 'fecha' and 'valor'
        use_end_of_month: If True, convert dates to end of month

    Returns:
        Number of updates made
    """
    updates = 0

    for item in new_data:
        fecha = item['fecha']
        valor = item['valor']

        # Handle date types
        if isinstance(fecha, str):
            fecha = pd.to_datetime(fecha)
        elif hasattr(fecha, 'date'):
            # Convert date to datetime
            fecha = pd.to_datetime(fecha)

        # Convert to end of month for Colombian variables
        if use_end_of_month:
            fecha = fecha + pd.offsets.MonthEnd(0)
        else:
            # Use start of month for FRED-style variables
            fecha = fecha.replace(day=1)

        # Normalize time
        fecha = fecha.normalize()

        # Update or create row
        if fecha in df_data.index:
            old_val = df_data.loc[fecha, col_name]
            if pd.isna(old_val) or old_val != valor:
                df_data.loc[fecha, col_name] = valor
                updates += 1
        else:
            # Create new row
            df_data.loc[fecha, col_name] = valor
            updates += 1

    return updates


# =============================================================================
# SUAMECA EXTRACTION (Selenium)
# =============================================================================

def extract_suameca_selenium():
    """
    Extract SUAMECA variables using Selenium with undetected-chromedriver.

    Returns dict with variable -> list of {fecha, valor}
    """
    print("\n" + "=" * 70)
    print("SUAMECA - SELENIUM EXTRACTION")
    print("=" * 70)

    try:
        import undetected_chromedriver as uc
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
    except ImportError:
        print("[ERROR] undetected-chromedriver not installed")
        print("  Install with: pip install undetected-chromedriver selenium")
        return {}

    # Series configuration
    SERIES = {
        'IPCCOL': {
            'serie_id': '100002',
            'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/100002/ipc',
            'desc': 'IPC Colombia'
        },
        'ITCR': {
            'serie_id': '4170',
            'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/4170/indice_tasa_cambio_real_itcr',
            'desc': 'Indice Tasa de Cambio Real'
        },
        'TOT': {
            'serie_id': '4180',
            'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/4180/indice_terminos_intercambio_bienes',
            'desc': 'Terminos de Intercambio'
        },
    }

    results = {}

    # Setup Chrome driver
    print("\n[SELENIUM] Setting up undetected Chrome driver...")
    try:
        options = uc.ChromeOptions()
        options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--lang=es-CO')

        driver = uc.Chrome(options=options, version_main=None)
        print("  Driver initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to setup Chrome: {e}")
        return {}

    try:
        for var_name, config in SERIES.items():
            print(f"\n[{var_name}] {config['desc']}")
            print(f"  URL: {config['url']}")

            data = []

            try:
                driver.get(config['url'])
                time.sleep(5)  # Initial page load

                # Check for captcha
                page_source = driver.page_source.lower()
                if 'captcha' in page_source or 'radware' in page_source:
                    print("  Captcha detected, waiting for bypass...")
                    time.sleep(15)

                # Wait for page content
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )

                # Look for table view button
                buttons = driver.find_elements(By.TAG_NAME, "button")
                for btn in buttons:
                    btn_text = btn.text.lower()
                    btn_title = (btn.get_attribute('title') or '').lower()
                    if any(x in btn_text + btn_title for x in ['tabla', 'table', 'datos']):
                        try:
                            btn.click()
                            time.sleep(3)
                            break
                        except:
                            pass

                # Find data table
                time.sleep(2)
                tables = driver.find_elements(By.TAG_NAME, "table")

                data_table = None
                max_rows = 0
                for table in tables:
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    if len(rows) > max_rows:
                        max_rows = len(rows)
                        data_table = table

                if data_table and max_rows > 10:
                    print(f"  Found table with {max_rows} rows")
                    rows = data_table.find_elements(By.TAG_NAME, "tr")

                    for row in rows[1:]:  # Skip header
                        try:
                            th_elements = row.find_elements(By.TAG_NAME, "th")
                            td_elements = row.find_elements(By.TAG_NAME, "td")

                            if not th_elements or not td_elements:
                                continue

                            fecha_str = th_elements[0].text.strip()
                            valor_str = td_elements[0].text.strip()

                            # Clean value
                            valor_str = valor_str.replace(',', '.').replace('%', '').replace(' ', '')

                            if not fecha_str or not valor_str:
                                continue

                            try:
                                valor = float(valor_str)
                            except:
                                continue

                            # Parse date
                            fecha = None
                            for fmt in ['%Y/%m/%d', '%Y-%m-%d', '%d/%m/%Y']:
                                try:
                                    fecha = datetime.strptime(fecha_str, fmt).date()
                                    break
                                except:
                                    continue

                            if fecha is None:
                                try:
                                    fecha = pd.to_datetime(fecha_str).date()
                                except:
                                    continue

                            if fecha:
                                data.append({'fecha': fecha, 'valor': valor})

                        except Exception:
                            continue

                    # Sort and filter to recent data
                    if data:
                        data = sorted(data, key=lambda x: x['fecha'], reverse=True)
                        # Keep last 24 months
                        data = data[:24]
                        print(f"  Extracted {len(data)} records")
                        print(f"  Latest: {data[0]['fecha']} = {data[0]['valor']}")
                        results[var_name] = data
                else:
                    print(f"  No data table found")

            except Exception as e:
                print(f"  Error: {e}")
                continue

            time.sleep(5)  # Rate limiting between requests

    finally:
        print("\n[SELENIUM] Closing browser...")
        driver.quit()

    return results


# =============================================================================
# DANE EXTRACTION
# =============================================================================

def extract_dane():
    """
    Extract DANE trade balance data (EXPUSD, IMPUSD).

    Returns dict with variable -> list of {fecha, valor}
    """
    print("\n" + "=" * 70)
    print("DANE - TRADE BALANCE EXTRACTION")
    print("=" * 70)

    try:
        from scraper_dane_balanza import obtener_balanza_completa
    except ImportError:
        print("[ERROR] Could not import DANE scraper")
        print(f"  Check: {SCRAPERS_DIR / 'scraper_dane_balanza.py'}")
        return {}

    results = {}

    try:
        # Get last 24 months of data
        df = obtener_balanza_completa(n=24)

        if df is not None and not df.empty:
            print(f"\n[DANE] Retrieved {len(df)} records")

            # EXPUSD
            expusd_data = []
            for _, row in df.iterrows():
                expusd_data.append({
                    'fecha': row['fecha'],
                    'valor': row['exportaciones_usd_millones']
                })
            results['EXPUSD'] = expusd_data
            print(f"  EXPUSD: {len(expusd_data)} records")
            if expusd_data:
                print(f"    Latest: {expusd_data[-1]['fecha']} = {expusd_data[-1]['valor']:.2f}")

            # IMPUSD
            impusd_data = []
            for _, row in df.iterrows():
                impusd_data.append({
                    'fecha': row['fecha'],
                    'valor': row['importaciones_usd_millones']
                })
            results['IMPUSD'] = impusd_data
            print(f"  IMPUSD: {len(impusd_data)} records")
            if impusd_data:
                print(f"    Latest: {impusd_data[-1]['fecha']} = {impusd_data[-1]['valor']:.2f}")
        else:
            print("[DANE] No data retrieved")

    except Exception as e:
        print(f"[ERROR] DANE extraction failed: {e}")

    return results


# =============================================================================
# FEDESARROLLO EXTRACTION
# =============================================================================

def extract_fedesarrollo():
    """
    Extract Fedesarrollo confidence indices (CCI, ICI).

    Returns dict with variable -> list of {fecha, valor}
    """
    print("\n" + "=" * 70)
    print("FEDESARROLLO - CONFIDENCE INDICES")
    print("=" * 70)

    try:
        from scraper_fedesarrollo import obtener_cci, obtener_ici
    except ImportError:
        print("[ERROR] Could not import Fedesarrollo scraper")
        print(f"  Check: {SCRAPERS_DIR / 'scraper_fedesarrollo.py'}")
        return {}

    results = {}

    # CCI - Consumer Confidence Index
    try:
        print("\n[CCI] Extracting Consumer Confidence Index...")
        df_cci = obtener_cci(n=24)

        if df_cci is not None and not df_cci.empty:
            cci_data = []
            for _, row in df_cci.iterrows():
                cci_data.append({
                    'fecha': row['fecha'],
                    'valor': row['valor']
                })
            results['CCI'] = cci_data
            print(f"  CCI: {len(cci_data)} records")
            if cci_data:
                print(f"    Latest: {cci_data[0]['fecha']} = {cci_data[0]['valor']:.2f}")
    except Exception as e:
        print(f"[ERROR] CCI extraction failed: {e}")

    # ICI - Industrial Confidence Index
    try:
        print("\n[ICI] Extracting Industrial Confidence Index...")
        df_ici = obtener_ici(n=24)

        if df_ici is not None and not df_ici.empty:
            ici_data = []
            for _, row in df_ici.iterrows():
                ici_data.append({
                    'fecha': row['fecha'],
                    'valor': row['valor']
                })
            results['ICI'] = ici_data
            print(f"  ICI: {len(ici_data)} records")
            if ici_data:
                print(f"    Latest: {ici_data[0]['fecha']} = {ici_data[0]['valor']:.2f}")
    except Exception as e:
        print(f"[ERROR] ICI extraction failed: {e}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Update all monthly macro variables')
    parser.add_argument('--source', choices=['suameca', 'dane', 'fedes', 'all'],
                       default='all', help='Source to extract from')
    args = parser.parse_args()

    print("=" * 70)
    print("UPDATE ALL MONTHLY MACRO VARIABLES")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load current data
    df_data, audit_row = load_monthly_master()

    # Track updates
    all_updates = {}

    # Extract from each source
    if args.source in ['suameca', 'all']:
        suameca_data = extract_suameca_selenium()
        for var, data in suameca_data.items():
            col_name = SUAMECA_COLUMNS.get(var)
            if col_name and data:
                updates = update_dataframe(df_data, col_name, data, use_end_of_month=True)
                all_updates[var] = updates

    if args.source in ['dane', 'all']:
        dane_data = extract_dane()
        for var, data in dane_data.items():
            col_name = DANE_COLUMNS.get(var)
            if col_name and data:
                # DANE uses start of month dates
                updates = update_dataframe(df_data, col_name, data, use_end_of_month=False)
                all_updates[var] = updates

    if args.source in ['fedes', 'all']:
        fedes_data = extract_fedesarrollo()
        for var, data in fedes_data.items():
            col_name = FEDESARROLLO_COLUMNS.get(var)
            if col_name and data:
                # Fedesarrollo uses start of month dates
                updates = update_dataframe(df_data, col_name, data, use_end_of_month=False)
                all_updates[var] = updates

    # Summary
    print("\n" + "=" * 70)
    print("UPDATE SUMMARY")
    print("=" * 70)

    total_updates = 0
    for var, count in all_updates.items():
        print(f"  {var}: {count} updates")
        total_updates += count

    print(f"\n  TOTAL: {total_updates} updates")

    # Save if there were updates
    if total_updates > 0:
        save_monthly_master(df_data, audit_row)

        # Also update downstream files
        clean_path = PROJECT_ROOT / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_MONTHLY_CLEAN.csv"
        fusion_path = PROJECT_ROOT / "data" / "pipeline" / "03_fusion" / "output" / "DATASET_MACRO_MONTHLY.csv"

        for path in [clean_path, fusion_path]:
            if path.parent.exists():
                if audit_row is not None:
                    df_final = pd.concat([pd.DataFrame([audit_row], index=["FUENTE_URL"]), df_data])
                else:
                    df_final = df_data
                df_final.to_csv(path)
                print(f"[SAVED] {path}")
    else:
        print("\n[INFO] No updates needed - data is already current")

    # Final status check
    print("\n" + "=" * 70)
    print("CURRENT DATA STATUS")
    print("=" * 70)

    for var, col_name in {**SUAMECA_COLUMNS, **DANE_COLUMNS, **FEDESARROLLO_COLUMNS}.items():
        if col_name in df_data.columns:
            data = df_data[col_name].dropna()
            if len(data) > 0:
                last_date = data.index.max()
                last_val = data.iloc[-1]
                try:
                    print(f"  {var}: {last_date.strftime('%Y-%m-%d')} = {float(last_val):.2f}")
                except (ValueError, TypeError):
                    print(f"  {var}: {last_date.strftime('%Y-%m-%d')} = {last_val}")
            else:
                print(f"  {var}: NO DATA")


if __name__ == "__main__":
    main()

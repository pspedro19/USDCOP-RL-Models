#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update monthly variables with missing recent data from SUAMECA REST API.

Variables to update:
- INFL_CPI_TOTAL_COL_M_IPCCOL (serie 100002)
- FXRT_REER_BILATERAL_COL_M_ITCR (serie 4170)
- RSBP_RESERVES_INTERNATIONAL_COL_M_RESINT (serie 15051)
- FTRD_TERMS_TRADE_COL_M_TOT (serie 4180)

DANE and Fedesarrollo require separate handling.
"""

import pandas as pd
import requests
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "pipeline" / "01_sources" / "consolidated"

# SUAMECA API
API_BASE = "https://suameca.banrep.gov.co/estadisticas-economicas-back/rest/estadisticaEconomicaRestService"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Accept-Language": "es-CO,es;q=0.9,en;q=0.8",
}

# Variables to update from SUAMECA
SUAMECA_VARS = {
    "INFL_CPI_TOTAL_COL_M_IPCCOL": {"serie_id": 100002, "desc": "IPC Colombia"},
    "FXRT_REER_BILATERAL_COL_M_ITCR": {"serie_id": 4170, "desc": "ITCR"},
    "RSBP_RESERVES_INTERNATIONAL_COL_M_RESINT": {"serie_id": 15051, "desc": "Reservas Internacionales"},
    "FTRD_TERMS_TRADE_COL_M_TOT": {"serie_id": 4180, "desc": "Terminos de Intercambio"},
}


def fetch_suameca_serie(serie_id: int, start_year: int = 2025) -> pd.DataFrame:
    """Fetch serie data from SUAMECA REST API."""
    url = f"{API_BASE}/consultaInformacionSerie"
    params = {"idSerie": serie_id}

    print(f"  Fetching serie {serie_id}...")
    resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Debug: show structure
    print(f"    Response type: {type(data)}")
    if isinstance(data, list):
        print(f"    List length: {len(data)}")
        if len(data) > 0:
            print(f"    First element type: {type(data[0])}")
            if isinstance(data[0], dict):
                print(f"    First element keys: {data[0].keys()}")

    # Handle various response formats
    serie_data = []
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            # List of dicts with 'data' key
            serie_data = data[0].get('data', [])
        elif len(data) > 0 and isinstance(data[0], list):
            # Direct list of [timestamp, value] pairs
            serie_data = data
    elif isinstance(data, dict):
        serie_data = data.get('data', [])

    if not serie_data:
        print(f"  WARNING: No data returned for serie {serie_id}")
        return pd.DataFrame()

    records = []
    for item in serie_data:
        if isinstance(item, list) and len(item) >= 2:
            timestamp_ms = item[0]
            value = item[1]

            dt = pd.to_datetime(timestamp_ms, unit='ms')

            # Filter to recent data only
            if dt.year >= start_year:
                records.append({
                    "fecha": dt.normalize(),
                    "value": float(value) if value is not None else None
                })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.dropna(subset=['value'])
        df = df.drop_duplicates(subset=['fecha'])
        df = df.sort_values("fecha").reset_index(drop=True)

    return df


def main():
    print("=" * 70)
    print("UPDATING MONTHLY VARIABLES FROM SUAMECA")
    print("=" * 70)

    # Load current monthly dataset
    csv_path = OUTPUT_DIR / "MACRO_MONTHLY_MASTER.csv"
    print(f"\nLoading {csv_path}...")

    df_monthly = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Skip the FUENTE_URL row for data operations
    audit_row = df_monthly.loc["FUENTE_URL"] if "FUENTE_URL" in df_monthly.index else None
    df_data = df_monthly.drop("FUENTE_URL", errors="ignore")
    df_data.index = pd.to_datetime(df_data.index)

    print(f"Current data: {len(df_data)} rows")

    # Track changes
    updates = []

    # Fetch and update each SUAMECA variable
    for var_name, config in SUAMECA_VARS.items():
        print(f"\n{var_name} ({config['desc']}):")

        # Check current last date
        if var_name in df_data.columns:
            current_data = df_data[var_name].dropna()
            if len(current_data) > 0:
                last_date = current_data.index.max()
                print(f"  Current last date: {last_date.strftime('%Y-%m-%d')}")
            else:
                last_date = None
                print(f"  No existing data")
        else:
            print(f"  Column not found in dataset")
            continue

        # Fetch from API
        df_new = fetch_suameca_serie(config['serie_id'], start_year=2025)

        if df_new.empty:
            print(f"  No new data from API")
            continue

        print(f"  API returned {len(df_new)} records from 2025+")
        print(f"  API last date: {df_new['fecha'].max().strftime('%Y-%m-%d')}")

        # Update existing data
        for _, row in df_new.iterrows():
            fecha = row['fecha']
            value = row['value']

            # Find or create the row for this date
            if fecha in df_data.index:
                old_val = df_data.loc[fecha, var_name]
                if pd.isna(old_val) or old_val != value:
                    df_data.loc[fecha, var_name] = value
                    updates.append((fecha, var_name, old_val, value))
            else:
                # Create new row if needed
                df_data.loc[fecha, var_name] = value
                updates.append((fecha, var_name, None, value))

    # Sort by index
    df_data = df_data.sort_index()

    # Show updates
    print("\n" + "=" * 70)
    print(f"UPDATES MADE: {len(updates)}")
    print("=" * 70)

    if updates:
        for fecha, var, old, new in updates[-20:]:  # Show last 20
            try:
                old_str = f"{float(old):.2f}" if pd.notna(old) else "NaN"
                new_str = f"{float(new):.2f}" if pd.notna(new) else "NaN"
            except (ValueError, TypeError):
                old_str = str(old) if pd.notna(old) else "NaN"
                new_str = str(new) if pd.notna(new) else "NaN"
            print(f"  {fecha.strftime('%Y-%m-%d')} | {var[-25:]:<25} | {old_str} -> {new_str}")

    # Save updated dataset
    if updates:
        # Reconstruct with audit row
        if audit_row is not None:
            df_final = pd.concat([pd.DataFrame([audit_row], index=["FUENTE_URL"]), df_data])
        else:
            df_final = df_data

        # Save CSV
        df_final.to_csv(csv_path)
        print(f"\nSaved: {csv_path}")

        # Save parquet (data only)
        parquet_path = csv_path.with_suffix(".parquet")
        df_data.to_parquet(parquet_path, engine="pyarrow")
        print(f"Saved: {parquet_path}")

        # Also update CLEAN and FUSION outputs
        clean_path = PROJECT_ROOT / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_MONTHLY_CLEAN.csv"
        fusion_path = PROJECT_ROOT / "data" / "pipeline" / "03_fusion" / "output" / "DATASET_MACRO_MONTHLY.csv"

        for path in [clean_path, fusion_path]:
            if path.parent.exists():
                df_final.to_csv(path)
                print(f"Saved: {path}")
                df_data.to_parquet(path.with_suffix(".parquet"), engine="pyarrow")
    else:
        print("\nNo updates needed - data is already current.")

    # Summary of each variable's status
    print("\n" + "=" * 70)
    print("VARIABLE STATUS")
    print("=" * 70)

    for var_name, config in SUAMECA_VARS.items():
        if var_name in df_data.columns:
            data = df_data[var_name].dropna()
            if len(data) > 0:
                last_date = data.index.max()
                last_val = data.iloc[-1]
                print(f"{var_name}:")
                print(f"  Last date: {last_date.strftime('%Y-%m-%d')}")
                print(f"  Last value: {last_val:.2f}")


if __name__ == "__main__":
    main()

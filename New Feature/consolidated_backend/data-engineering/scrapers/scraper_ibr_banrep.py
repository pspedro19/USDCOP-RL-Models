#!/usr/bin/env python3
"""
Scraper para IBR (Indicador Bancario de Referencia) desde datos.gov.co
======================================================================

Fuente: Datos Abiertos Colombia (Socrata API)
Dataset: IBR Plazo Overnight
URL: https://www.datos.gov.co/d/b8fs-cx24
API: https://www.datos.gov.co/resource/b8fs-cx24.json

Datos disponibles desde: 2008-01-02
Actualización: Diaria a las 11:00am COT
"""
import requests
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# API Configuration
DATOS_GOV_CO_API = "https://www.datos.gov.co/resource/b8fs-cx24.json"


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'usdcop-postgres-timescale'),
        port=int(os.environ.get('POSTGRES_PORT', 5432)),
        database=os.environ.get('POSTGRES_DB', 'usdcop_trading'),
        user=os.environ.get('POSTGRES_USER', 'admin'),
        password=os.environ.get('POSTGRES_PASSWORD', 'admin123')
    )


def fetch_ibr_from_datos_gov(days: int = 60) -> list:
    """
    Fetch IBR Overnight data from datos.gov.co Socrata API.

    Args:
        days: Number of days of historical data to fetch

    Returns:
        List of dicts with 'fecha' and 'valor' keys
    """
    logger.info(f"Fetching IBR from datos.gov.co (last {days} days)...")

    try:
        # Calculate date filter
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        # Socrata API query with SoQL
        # Filter for overnight (plazo=1), effective rate, and recent dates
        params = {
            '$select': 'fecha, valor_tasa',
            '$where': f"plazo='1' AND tipo_tasa='Efectiva' AND fecha >= '{start_date}'",
            '$order': 'fecha DESC',
            '$limit': days * 2  # Extra margin for missing days
        }

        headers = {
            'Accept': 'application/json',
            'User-Agent': 'USDCOP-Trading-System/1.0'
        }

        response = requests.get(DATOS_GOV_CO_API, params=params, headers=headers, timeout=30)

        if response.status_code != 200:
            logger.error(f"API returned HTTP {response.status_code}")
            logger.error(f"Response: {response.text[:500]}")
            return []

        data = response.json()

        if not data:
            logger.warning("No data returned from API")
            return []

        # Process and aggregate by date (average of all banks for each day)
        df = pd.DataFrame(data)

        # Convert types
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['valor_tasa'] = pd.to_numeric(df['valor_tasa'], errors='coerce')

        # Group by date and get average rate
        df_daily = df.groupby(df['fecha'].dt.date)['valor_tasa'].mean().reset_index()
        df_daily.columns = ['fecha', 'valor']

        # Sort by date descending
        df_daily = df_daily.sort_values('fecha', ascending=False)

        # Convert to list of dicts
        result = [{'fecha': row['fecha'], 'valor': round(row['valor'], 4)}
                  for _, row in df_daily.iterrows()]

        logger.info(f"Fetched {len(result)} days of IBR data")
        if result:
            logger.info(f"Latest: {result[0]['fecha']} = {result[0]['valor']}%")

        return result

    except requests.exceptions.Timeout:
        logger.error("API request timed out")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error fetching IBR: {e}")
        return []


def fetch_ibr_from_suameca(days: int = 60) -> list:
    """
    Fallback: Fetch IBR from SUAMECA BanRep portal via web scraping.

    Args:
        days: Number of days of historical data to fetch

    Returns:
        List of dicts with 'fecha' and 'valor' keys
    """
    logger.info("Trying SUAMECA BanRep fallback...")

    try:
        import cloudscraper
        from bs4 import BeautifulSoup

        url = "https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/241/tasas_interes_indicador_bancario_referencia_ibr"

        scraper = cloudscraper.create_scraper()
        response = scraper.get(url, timeout=30)

        if response.status_code != 200:
            logger.warning(f"SUAMECA returned HTTP {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, 'html.parser')

        # Look for data tables
        tables = soup.find_all('table')

        for table in tables:
            rows = table.find_all('tr')
            data = []

            for row in rows[1:]:  # Skip header
                cols = row.find_all(['td', 'th'])
                if len(cols) >= 2:
                    try:
                        fecha_str = cols[0].get_text(strip=True)
                        valor_str = cols[1].get_text(strip=True).replace(',', '.')

                        fecha = pd.to_datetime(fecha_str, dayfirst=True)
                        valor = float(valor_str)

                        if 0 < valor < 30:  # Reasonable IBR range
                            data.append({'fecha': fecha.date(), 'valor': valor})
                    except:
                        continue

            if data:
                logger.info(f"Found {len(data)} records from SUAMECA")
                return sorted(data, key=lambda x: x['fecha'], reverse=True)[:days]

        logger.warning("No data found in SUAMECA page")
        return []

    except Exception as e:
        logger.error(f"SUAMECA scraping error: {e}")
        return []


def update_database(data: list) -> int:
    """
    Update database with IBR data.

    Args:
        data: List of dicts with 'fecha' and 'valor' keys

    Returns:
        Number of rows updated
    """
    if not data:
        logger.warning("No IBR data to update")
        return 0

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        updated = 0
        column = 'finc_rate_ibr_overnight_col_d_ibr'

        for item in data:
            fecha = item['fecha']
            valor = item['valor']

            cur.execute(f"""
                INSERT INTO macro_indicators_daily (fecha, {column})
                VALUES (%s, %s)
                ON CONFLICT (fecha) DO UPDATE SET
                    {column} = EXCLUDED.{column},
                    updated_at = NOW()
                WHERE macro_indicators_daily.{column} IS NULL
                   OR macro_indicators_daily.{column} != EXCLUDED.{column}
            """, [fecha, valor])

            if cur.rowcount > 0:
                updated += 1

        conn.commit()
        logger.info(f"Updated {updated} rows for IBR")
        return updated

    except Exception as e:
        logger.error(f"Database error: {e}")
        conn.rollback()
        return 0

    finally:
        cur.close()
        conn.close()


def main():
    """Main function to fetch and update IBR data."""
    print("=" * 70)
    print("IBR SCRAPER - Banco de la República Colombia")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Try datos.gov.co first (recommended)
    print("\n[1] Trying datos.gov.co Socrata API...")
    data = fetch_ibr_from_datos_gov(days=60)

    # Fallback to SUAMECA if needed
    if not data:
        print("\n[2] Trying SUAMECA BanRep fallback...")
        data = fetch_ibr_from_suameca(days=60)

    if data:
        print(f"\n[SUCCESS] Fetched {len(data)} days of IBR data")
        print(f"Latest: {data[0]['fecha']} = {data[0]['valor']}%")
        print(f"Oldest: {data[-1]['fecha']} = {data[-1]['valor']}%")

        # Update database
        print("\n[3] Updating database...")
        updated = update_database(data)
        print(f"Updated {updated} rows")
    else:
        print("\n[ERROR] Could not fetch IBR data from any source")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()

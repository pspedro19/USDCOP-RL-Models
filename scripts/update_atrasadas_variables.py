#!/usr/bin/env python3
"""
Script to directly update the "atrasadas" variables from Investing.com and FRED.
This fills in NULL values for existing dates.
"""
import requests
import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import psycopg2
from datetime import datetime
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Variables to update with their Investing.com URLs
INVESTING_URLS = {
    'comm_metal_gold_glb_d_gold': 'https://www.investing.com/commodities/gold-historical-data',
    'comm_agri_coffee_glb_d_coffee': 'https://www.investing.com/commodities/us-coffee-c-historical-data',
    'eqty_index_colcap_col_d_colcap': 'https://www.investing.com/indices/colcap-historical-data',
    'finc_bond_yield10y_col_d_col10y': 'https://www.investing.com/rates-bonds/colombia-10-year-bond-yield-historical-data',
    'finc_bond_yield5y_col_d_col5y': 'https://www.investing.com/rates-bonds/colombia-5-year-bond-yield-historical-data',
}


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'usdcop-postgres-timescale'),
        port=int(os.environ.get('POSTGRES_PORT', 5432)),
        database=os.environ.get('POSTGRES_DB', 'usdcop_trading'),
        user=os.environ.get('POSTGRES_USER', 'admin'),
        password=os.environ.get('POSTGRES_PASSWORD', 'admin123')
    )


def fetch_investing_data(url: str, days: int = 60) -> list:
    """Fetch historical data from Investing.com."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }

        scraper = cloudscraper.create_scraper()
        response = scraper.get(url, headers=headers, timeout=20)

        if response.status_code != 200:
            logger.error(f"HTTP {response.status_code} for {url}")
            return []

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find historical data table
        table = soup.find('table', class_='freeze-column-w-1')
        if not table:
            table = soup.find('table', {'data-test': 'historical-data-table'})
        if not table:
            tables = soup.find_all('table')
            if tables:
                table = max(tables, key=lambda t: len(str(t)))

        if not table:
            logger.error("No table found")
            return []

        rows = table.find_all('tr')[1:days+10]
        data = []

        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 2:
                fecha_str = cols[0].get_text(strip=True)
                valor_str = cols[1].get_text(strip=True).replace(',', '')

                try:
                    fecha = pd.to_datetime(fecha_str)
                    valor = float(valor_str)
                    data.append({'fecha': fecha.date(), 'valor': valor})
                except:
                    continue

        return data

    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return []


def fetch_fred_prime_rate() -> list:
    """Fetch Prime Rate from FRED."""
    try:
        from fredapi import Fred

        api_key = os.environ.get('FRED_API_KEY')
        if not api_key:
            logger.error("FRED_API_KEY not set")
            return []

        fred = Fred(api_key=api_key)
        series = fred.get_series('DPRIME', observation_start='2025-01-01')

        if series is not None and not series.empty:
            data = []
            for date, value in series.items():
                if pd.notna(value):
                    data.append({'fecha': date.date(), 'valor': float(value)})
            return data

        return []

    except Exception as e:
        logger.error(f"Error fetching FRED Prime Rate: {e}")
        return []


def update_database(column: str, data: list):
    """Update database with fetched data."""
    if not data:
        logger.warning(f"No data to update for {column}")
        return 0

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        updated = 0
        for item in data:
            fecha = item['fecha']
            valor = item['valor']

            # Update existing row or insert new one
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
        logger.info(f"Updated {updated} rows for {column}")
        return updated

    except Exception as e:
        logger.error(f"Error updating {column}: {e}")
        conn.rollback()
        return 0

    finally:
        cur.close()
        conn.close()


def main():
    print("=" * 80)
    print("UPDATING ATRASADAS VARIABLES")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    total_updated = 0

    # 1. Update Investing.com variables
    for column, url in INVESTING_URLS.items():
        print(f"\n[{column}]")
        print(f"  Source: Investing.com")

        data = fetch_investing_data(url, days=60)
        if data:
            print(f"  Fetched: {len(data)} records")
            print(f"  Latest: {data[0]['fecha']} = {data[0]['valor']}")
            updated = update_database(column, data)
            total_updated += updated
        else:
            print(f"  ERROR: No data fetched")

    # 2. Update Prime Rate from FRED
    print(f"\n[polr_prime_rate_usa_d_prime]")
    print(f"  Source: FRED API")

    data = fetch_fred_prime_rate()
    if data:
        print(f"  Fetched: {len(data)} records")
        print(f"  Latest: {data[-1]['fecha']} = {data[-1]['valor']}")
        updated = update_database('polr_prime_rate_usa_d_prime', data)
        total_updated += updated
    else:
        print(f"  ERROR: No data fetched")

    print("\n" + "=" * 80)
    print(f"TOTAL ROWS UPDATED: {total_updated}")
    print("=" * 80)


if __name__ == '__main__':
    main()

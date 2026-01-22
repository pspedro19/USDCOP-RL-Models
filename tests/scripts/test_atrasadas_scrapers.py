#!/usr/bin/env python3
"""
Test script to verify scraping sources for the 7 "atrasadas" variables.
Run this inside the Airflow container to test data fetching.
"""
import requests
import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Variables atrasadas with their sources
ATRASADAS_CONFIG = {
    'Gold': {
        'db_column': 'comm_metal_gold_glb_d_gold',
        'investing_url': 'https://www.investing.com/commodities/gold-historical-data',
        'yahoo_symbol': 'GC=F',
        'source_note': 'Investing.com/Yahoo Finance (International market)'
    },
    'Coffee': {
        'db_column': 'comm_agri_coffee_glb_d_coffee',
        'investing_url': 'https://www.investing.com/commodities/us-coffee-c-historical-data',
        'yahoo_symbol': 'KC=F',
        'source_note': 'Investing.com/Yahoo Finance (International market)'
    },
    'COLCAP': {
        'db_column': 'eqty_index_colcap_col_d_colcap',
        'investing_url': 'https://www.investing.com/indices/colcap-historical-data',
        'yahoo_symbol': 'ICOLCAP.CL',
        'source_note': 'Investing.com (BVC Colombia data)'
    },
    'TES_10Y': {
        'db_column': 'finc_bond_yield10y_col_d_col10y',
        'investing_url': 'https://www.investing.com/rates-bonds/colombia-10-year-bond-yield-historical-data',
        'yahoo_symbol': None,
        'source_note': 'Investing.com (BanRep/Colombia bond market)'
    },
    'TES_5Y': {
        'db_column': 'finc_bond_yield5y_col_d_col5y',
        'investing_url': 'https://www.investing.com/rates-bonds/colombia-5-year-bond-yield-historical-data',
        'yahoo_symbol': None,
        'source_note': 'Investing.com (BanRep/Colombia bond market)'
    },
    'IBR': {
        'db_column': 'finc_rate_ibr_overnight_col_d_ibr',
        'investing_url': 'https://www.investing.com/rates-bonds/colombia-overnight-interbank-rate-historical-data',
        'yahoo_symbol': None,
        'banrep_api': True,
        'source_note': 'BanRep API / Investing.com (Colombia overnight rate)'
    },
    'Prime_Rate': {
        'db_column': 'polr_prime_rate_usa_d_prime',
        'fred_series': 'DPRIME',
        'investing_url': 'https://www.investing.com/economic-calendar/prime-rate-1057',
        'source_note': 'FRED API (US Prime Rate)'
    }
}


def test_investing_com(url: str, variable: str) -> dict:
    """Test Investing.com scraper"""
    result = {'source': 'Investing.com', 'status': 'FAILED', 'data': None, 'error': None}

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }

        scraper = cloudscraper.create_scraper()
        response = scraper.get(url, headers=headers, timeout=20)

        if response.status_code != 200:
            result['error'] = f"HTTP {response.status_code}"
            return result

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
            result['error'] = "No table found"
            return result

        rows = table.find_all('tr')[1:6]  # Get first 5 data rows

        data = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 2:
                fecha_str = cols[0].get_text(strip=True)
                valor_str = cols[1].get_text(strip=True).replace(',', '')

                try:
                    fecha = pd.to_datetime(fecha_str)
                    valor = float(valor_str)
                    data.append({'fecha': fecha, 'valor': valor})
                except:
                    continue

        if data:
            result['status'] = 'SUCCESS'
            result['data'] = data
            result['latest_date'] = data[0]['fecha'].strftime('%Y-%m-%d')
            result['latest_value'] = data[0]['valor']
        else:
            result['error'] = "Could not parse data from table"

    except Exception as e:
        result['error'] = str(e)

    return result


def test_yahoo_finance(symbol: str, variable: str) -> dict:
    """Test Yahoo Finance"""
    result = {'source': 'Yahoo Finance', 'status': 'FAILED', 'data': None, 'error': None}

    if not symbol:
        result['error'] = "No Yahoo symbol configured"
        return result

    try:
        data = yf.download(symbol, period='7d', progress=False)

        if data is not None and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                close_data = data['Close'].iloc[:, 0]
            else:
                close_data = data['Close']

            result['status'] = 'SUCCESS'
            result['latest_date'] = close_data.index[-1].strftime('%Y-%m-%d')
            result['latest_value'] = float(close_data.iloc[-1])
            result['rows'] = len(close_data)
        else:
            result['error'] = "No data returned"

    except Exception as e:
        result['error'] = str(e)

    return result


def test_fred_api(series_id: str) -> dict:
    """Test FRED API"""
    result = {'source': 'FRED API', 'status': 'FAILED', 'data': None, 'error': None}

    import os
    api_key = os.environ.get('FRED_API_KEY')

    if not api_key:
        result['error'] = "FRED_API_KEY not set"
        return result

    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        data = fred.get_series(series_id)

        if data is not None and not data.empty:
            latest = data.dropna()
            result['status'] = 'SUCCESS'
            result['latest_date'] = latest.index[-1].strftime('%Y-%m-%d')
            result['latest_value'] = float(latest.iloc[-1])
        else:
            result['error'] = "No data returned"

    except Exception as e:
        result['error'] = str(e)

    return result


def test_banrep_ibr() -> dict:
    """Test BanRep API for IBR (Overnight Interbank Rate)"""
    result = {'source': 'BanRep API', 'status': 'FAILED', 'data': None, 'error': None}

    try:
        # BanRep SUAMECA API - Series IBR Overnight
        # Series ID: 4120 (IBR Overnight)
        url = "https://www.banrep.gov.co/series-estadisticas/estadisticas-financieras/4120"

        # Alternative: Direct SUAMECA API endpoint
        api_url = "https://www.banrep.gov.co/estadisticas/serieEstadistica?serie=4120"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept': 'application/json, text/html'
        }

        response = requests.get(api_url, headers=headers, timeout=15)

        if response.status_code == 200:
            # Parse response (JSON or HTML)
            result['status'] = 'PARTIAL'
            result['error'] = f"Response received but parsing needed. Status: {response.status_code}"
        else:
            result['error'] = f"HTTP {response.status_code}"

    except Exception as e:
        result['error'] = str(e)

    return result


def main():
    print("=" * 80)
    print("TEST DE SCRAPERS PARA VARIABLES ATRASADAS")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    results = {}

    for var_name, config in ATRASADAS_CONFIG.items():
        print(f"\n{'='*40}")
        print(f"Variable: {var_name}")
        print(f"DB Column: {config['db_column']}")
        print(f"Source: {config['source_note']}")
        print(f"{'='*40}")

        var_results = []

        # Test 1: Investing.com
        if 'investing_url' in config:
            print(f"\n  [1] Testing Investing.com...")
            inv_result = test_investing_com(config['investing_url'], var_name)
            var_results.append(inv_result)

            if inv_result['status'] == 'SUCCESS':
                print(f"      ✅ SUCCESS - Latest: {inv_result['latest_date']} = {inv_result['latest_value']}")
            else:
                print(f"      ❌ FAILED - {inv_result['error']}")

        # Test 2: Yahoo Finance
        if config.get('yahoo_symbol'):
            print(f"\n  [2] Testing Yahoo Finance ({config['yahoo_symbol']})...")
            yf_result = test_yahoo_finance(config['yahoo_symbol'], var_name)
            var_results.append(yf_result)

            if yf_result['status'] == 'SUCCESS':
                print(f"      ✅ SUCCESS - Latest: {yf_result['latest_date']} = {yf_result['latest_value']}")
            else:
                print(f"      ❌ FAILED - {yf_result['error']}")

        # Test 3: FRED API (for Prime Rate)
        if config.get('fred_series'):
            print(f"\n  [3] Testing FRED API ({config['fred_series']})...")
            fred_result = test_fred_api(config['fred_series'])
            var_results.append(fred_result)

            if fred_result['status'] == 'SUCCESS':
                print(f"      ✅ SUCCESS - Latest: {fred_result['latest_date']} = {fred_result['latest_value']}")
            else:
                print(f"      ❌ FAILED - {fred_result['error']}")

        # Test 4: BanRep API (for IBR)
        if config.get('banrep_api'):
            print(f"\n  [4] Testing BanRep API...")
            banrep_result = test_banrep_ibr()
            var_results.append(banrep_result)

            if banrep_result['status'] == 'SUCCESS':
                print(f"      ✅ SUCCESS - {banrep_result.get('latest_date', 'N/A')}")
            else:
                print(f"      ⚠️ {banrep_result['status']} - {banrep_result['error']}")

        results[var_name] = var_results

    # Summary
    print("\n" + "=" * 80)
    print("RESUMEN DE FUENTES")
    print("=" * 80)

    print("""
Las 7 variables atrasadas NO vienen de BanRep ni Fedesarrollo directamente.
Las fuentes reales son:

| Variable   | Fuente Primaria     | Fuente Backup      | Nota                     |
|------------|---------------------|-------------------|--------------------------|
| Gold       | Investing.com       | Yahoo (GC=F)      | Mercado internacional    |
| Coffee     | Investing.com       | Yahoo (KC=F)      | Mercado internacional    |
| COLCAP     | Investing.com       | Yahoo (ICOLCAP)   | BVC Colombia via Investing|
| TES_10Y    | Investing.com       | N/A               | Bonos Colombia          |
| TES_5Y     | Investing.com       | N/A               | Bonos Colombia          |
| IBR        | BanRep SUAMECA      | Investing.com     | Tasa interbancaria COL  |
| Prime_Rate | FRED (DPRIME)       | N/A               | Tasa USA                |

NOTA IMPORTANTE:
- Los datos de bonos colombianos (TES) en Investing.com PROVIENEN de BanRep/BVC
- El IBR viene directamente de la API SUAMECA de BanRep
- Investing.com es un AGREGADOR que recibe datos de BanRep/BVC/Bloomberg
""")

    print("=" * 80)


if __name__ == '__main__':
    main()

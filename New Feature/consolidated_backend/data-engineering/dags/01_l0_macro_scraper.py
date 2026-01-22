"""
L0 - Macro Data Update (Scrapers + FRED) - INCREMENTAL
=======================================================
Layer 0: Actualización INCREMENTAL semanal de datos macro.

FLUJO INCREMENTAL:
1. Consulta la última fecha por indicador en core.macro_indicators
2. Scrape/API solo datos NUEVOS desde esa fecha
3. UPSERT incremental (acumula semana a semana)

FUENTES:
1. SCRAPERS (datos de mercado):
   - Investing.com: DXY, VIX, UST10Y, UST2Y, WTI, Brent, USD/MXN, USD/CLP, Gold, Coffee, COLCAP
   - BanRep Selenium: IBR, TPM, ITCR, TOT, Reservas, IPC Colombia
   - EMBI BCRP: EMBI Colombia
   - DANE: Exports, Imports

2. FRED API (datos macro oficiales USA):
   - Fed Funds, CPI, Core CPI, PCE, Unemployment, Industrial Prod, M2, Consumer Sentiment, GDP, Prime

LAGS DE PUBLICACIÓN (para evitar data leakage):
- Datos diarios mercado: T+0
- FRED mensual: aplicar lag según variable (5-30 días)
- Colombia macro: exports(45d), reserves(5d), itcr(15d)

Schedule: Lunes 7:00 AM COT (12:00 UTC) - Semanal antes de inferencia
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
from airflow.exceptions import AirflowException

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import requests
import os
import time
import logging
import sys

# Agregar paths
sys.path.insert(0, '/opt/airflow/scrapers')
sys.path.insert(0, '/opt/airflow/backend')

DAG_ID = "01_l0_macro_scraper"

# =============================================================================
# CONFIGURATION
# =============================================================================

FRED_API_KEY = os.getenv('FRED_API_KEY', '752b8fee0c781f117dc0ba1d71bdd0f2')

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database=os.getenv('POSTGRES_DB', 'pipeline_db'),
        user=os.getenv('POSTGRES_USER', 'pipeline'),
        password=os.getenv('POSTGRES_PASSWORD', 'pipeline_secret')
    )


def get_last_macro_dates():
    """
    Obtiene la última fecha por indicador en core.macro_indicators.
    Returns: dict {indicator_code: last_date}
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT indicator_code, MAX(date) as last_date
            FROM core.macro_indicators
            GROUP BY indicator_code
        """)
        result = {row[0]: row[1] for row in cursor.fetchall()}
        logging.info(f"[INCREMENTAL] Encontradas {len(result)} series con datos previos")
        return result
    except Exception as e:
        logging.warning(f"Error getting last dates: {e}")
        return {}
    finally:
        cursor.close()
        conn.close()


def get_global_last_date():
    """Obtiene la fecha más antigua entre todos los indicadores (para scraping batch)."""
    last_dates = get_last_macro_dates()
    if last_dates:
        # Usar la fecha más antigua para asegurar que capturamos todo
        min_date = min(last_dates.values())
        logging.info(f"[INCREMENTAL] Fecha mínima global: {min_date}")
        return min_date
    return None


# =============================================================================
# PUBLICATION LAGS - Para evitar Data Leakage
# =============================================================================

# Lags en días desde la fecha del dato hasta disponibilidad real
PUBLICATION_LAGS = {
    # Datos diarios de mercado - disponibles mismo día o T+1
    'dxy': 0,
    'vix': 0,
    'ust10y': 0,
    'ust2y': 0,
    'wti': 0,
    'brent': 0,
    'gold': 0,
    'coffee': 0,
    'usdmxn': 0,
    'usdclp': 0,
    'colcap': 0,
    'embi': 1,

    # Colombia - BanRep
    'ibr': 1,
    'tpm': 0,  # Cambios anunciados inmediatamente
    'itcr': 15,  # Mensual con lag
    'terms_trade': 60,  # Trimestral con lag
    'reserves': 5,  # Semanal
    'ipc_col': 10,  # Mensual

    # Colombia - DANE
    'exports': 45,
    'imports': 45,

    # FRED - USA Macro (mensual/trimestral)
    'fedfunds': 5,  # Publicado ~5 días después del mes
    'cpi_usa': 15,  # Publicado ~15 días después del mes
    'core_cpi': 15,
    'pce': 30,  # Publicado ~30 días después del mes
    'unemployment': 5,  # Primer viernes del mes siguiente
    'indpro': 15,
    'm2': 20,
    'consumer_sentiment': 15,
    'gdp': 30,  # Publicado ~30 días después del trimestre
    'prime': 1,  # Diario, T+1
}

# Frecuencia de cada variable (para ffill)
VARIABLE_FREQUENCY = {
    # Diario
    'dxy': 'daily', 'vix': 'daily', 'ust10y': 'daily', 'ust2y': 'daily',
    'wti': 'daily', 'brent': 'daily', 'gold': 'daily', 'coffee': 'daily',
    'usdmxn': 'daily', 'usdclp': 'daily', 'colcap': 'daily', 'embi': 'daily',
    'ibr': 'daily', 'tpm': 'daily', 'prime': 'daily',

    # Semanal
    'reserves': 'weekly',

    # Mensual
    'itcr': 'monthly', 'ipc_col': 'monthly', 'exports': 'monthly', 'imports': 'monthly',
    'fedfunds': 'monthly', 'cpi_usa': 'monthly', 'core_cpi': 'monthly',
    'pce': 'monthly', 'unemployment': 'monthly', 'indpro': 'monthly',
    'm2': 'monthly', 'consumer_sentiment': 'monthly',

    # Trimestral
    'terms_trade': 'quarterly', 'gdp': 'quarterly',
}

# Máximo días de ffill por frecuencia
MAX_FFILL = {
    'daily': 5,      # Max 5 días (fines de semana + feriados)
    'weekly': 10,    # Max 10 días
    'monthly': 40,   # Max 40 días
    'quarterly': 100 # Max 100 días
}


# =============================================================================
# INVESTING.COM SCRAPER - Reemplaza TwelveData + parte de FRED
# =============================================================================

INVESTING_URLS = {
    # Indices
    'dxy': 'https://www.investing.com/indices/usdollar-historical-data',
    'vix': 'https://www.investing.com/indices/volatility-s-p-500-historical-data',
    'colcap': 'https://www.investing.com/indices/colcap-historical-data',

    # Bonds
    'ust10y': 'https://www.investing.com/rates-bonds/u.s.-10-year-bond-yield-historical-data',
    'ust2y': 'https://www.investing.com/rates-bonds/u.s.-2-year-bond-yield-historical-data',
    'col10y': 'https://www.investing.com/rates-bonds/colombia-10-year-bond-yield-historical-data',
    'col5y': 'https://www.investing.com/rates-bonds/colombia-5-year-bond-yield-historical-data',

    # Commodities
    'wti': 'https://www.investing.com/commodities/crude-oil-historical-data',
    'brent': 'https://www.investing.com/commodities/brent-oil-historical-data',
    'gold': 'https://www.investing.com/commodities/gold-historical-data',
    'coffee': 'https://www.investing.com/commodities/us-coffee-c-historical-data',

    # Forex (reemplaza TwelveData)
    'usdmxn': 'https://www.investing.com/currencies/usd-mxn-historical-data',
    'usdclp': 'https://www.investing.com/currencies/usd-clp-historical-data',
    'usdcop': 'https://www.investing.com/currencies/usd-cop-historical-data',
}


def scrape_investing_batch(**context):
    """
    Scrape múltiples variables de Investing.com usando cloudscraper.
    """
    try:
        import cloudscraper
        from bs4 import BeautifulSoup
    except ImportError:
        logging.error("cloudscraper/beautifulsoup4 not installed")
        return {'status': 'error', 'message': 'Missing packages'}

    scraper = cloudscraper.create_scraper()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    all_data = {}
    errors = []
    variables_captured = set()

    for var_name, url in INVESTING_URLS.items():
        try:
            logging.info(f"[Investing] Scraping {var_name}...")

            resp = scraper.get(url, headers=headers, timeout=30)

            if resp.status_code != 200:
                errors.append(f"{var_name}: HTTP {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Buscar tabla de datos históricos
            table = soup.find('table', class_='freeze-column-w-1')
            if not table:
                tables = soup.find_all('table')
                if tables:
                    table = max(tables, key=lambda t: len(str(t)))

            if not table:
                errors.append(f"{var_name}: No table found")
                continue

            # Extraer últimos 30 días
            rows = table.find_all('tr')[1:35]
            count = 0

            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    try:
                        date_str = cols[0].get_text(strip=True)
                        value_str = cols[1].get_text(strip=True).replace(',', '')

                        # Parsear fecha
                        date = pd.to_datetime(date_str).strftime('%Y-%m-%d')
                        value = float(value_str)

                        if date not in all_data:
                            all_data[date] = {}
                        all_data[date][var_name] = value
                        variables_captured.add(var_name)
                        count += 1

                    except Exception as e:
                        continue

            logging.info(f"  -> {count} records")
            time.sleep(2)  # Rate limit

        except Exception as e:
            errors.append(f"{var_name}: {str(e)}")
            logging.error(f"[Investing] Error {var_name}: {e}")

    context['ti'].xcom_push(key='investing_data', value=all_data)
    context['ti'].xcom_push(key='investing_variables', value=list(variables_captured))

    logging.info(f"[Investing] Captured {len(variables_captured)} variables, {len(all_data)} dates")

    return {
        'status': 'success',
        'variables': len(variables_captured),
        'dates': len(all_data),
        'errors': errors
    }


# =============================================================================
# BANREP SELENIUM SCRAPER
# =============================================================================

BANREP_SERIES = {
    'ibr': {
        'serie_id': '241',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/241/tasas_interes_indicador_bancario_referencia_ibr',
    },
    'tpm': {
        'serie_id': '59',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/59/tasas_interes_politica_monetaria',
    },
    'itcr': {
        'serie_id': '4170',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/4170/indice_tasa_cambio_real_itcr',
    },
    'terms_trade': {
        'serie_id': '4180',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/4180/indice_terminos_intercambio_bienes',
    },
    'reserves': {
        'serie_id': '15051',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/15051/reservas_internacionales',
    },
    'ipc_col': {
        'serie_id': '100002',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/100002/ipc',
    },
}


def scrape_banrep(**context):
    """Scrape BanRep data via Selenium."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.service import Service
    except ImportError:
        logging.warning("Selenium not available")
        return {'status': 'skip', 'message': 'Selenium not installed'}

    all_data = {}
    errors = []
    variables_captured = set()

    # Setup Chrome headless
    try:
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')

        driver = webdriver.Chrome(options=chrome_options)
        logging.info("[BanRep] Chrome initialized")

    except Exception as e:
        logging.error(f"[BanRep] Chrome init failed: {e}")
        return {'status': 'error', 'message': str(e)}

    try:
        for var_name, config in BANREP_SERIES.items():
            try:
                logging.info(f"[BanRep] Scraping {var_name}...")

                driver.get(config['url'])
                time.sleep(5)

                # Click "Vista tabla" button
                try:
                    buttons = driver.find_elements(By.TAG_NAME, "button")
                    for btn in buttons:
                        if 'tabla' in btn.text.lower():
                            btn.click()
                            time.sleep(3)
                            break
                except:
                    pass

                # Find data table
                tables = driver.find_elements(By.TAG_NAME, "table")
                data_table = None
                max_rows = 0

                for table in tables:
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    if len(rows) > max_rows:
                        max_rows = len(rows)
                        data_table = table

                if not data_table or max_rows < 5:
                    errors.append(f"{var_name}: No table found")
                    continue

                # Extract data
                rows = data_table.find_elements(By.TAG_NAME, "tr")
                count = 0

                for row in rows[1:100]:  # Últimos 100 registros
                    try:
                        th = row.find_elements(By.TAG_NAME, "th")
                        td = row.find_elements(By.TAG_NAME, "td")

                        if not th or not td:
                            continue

                        fecha_str = th[0].text.strip()
                        valor_str = td[0].text.strip().replace(',', '.').replace('%', '')

                        if not fecha_str or not valor_str:
                            continue

                        # Parse date (YYYY/MM/DD format from BanRep)
                        try:
                            fecha = datetime.strptime(fecha_str, '%Y/%m/%d')
                        except:
                            fecha = pd.to_datetime(fecha_str)

                        date_str = fecha.strftime('%Y-%m-%d')
                        valor = float(valor_str)

                        if date_str not in all_data:
                            all_data[date_str] = {}
                        all_data[date_str][var_name] = valor
                        variables_captured.add(var_name)
                        count += 1

                    except:
                        continue

                logging.info(f"  -> {count} records")

            except Exception as e:
                errors.append(f"{var_name}: {str(e)}")
                logging.error(f"[BanRep] Error {var_name}: {e}")

    finally:
        driver.quit()
        logging.info("[BanRep] Chrome closed")

    context['ti'].xcom_push(key='banrep_data', value=all_data)
    context['ti'].xcom_push(key='banrep_variables', value=list(variables_captured))

    return {
        'status': 'success',
        'variables': len(variables_captured),
        'dates': len(all_data),
        'errors': errors
    }


# =============================================================================
# EMBI SCRAPER (BCRP)
# =============================================================================

# =============================================================================
# DANE SCRAPER (Trade Balance)
# =============================================================================

def scrape_dane_balanza(**context):
    """
    Scrape exports/imports from DANE Colombia.
    Monthly data with ~45 days publication lag.
    """
    try:
        from scrapers.scraper_dane_balanza import obtener_balanza_completa
    except ImportError:
        logging.warning("[DANE] scraper_dane_balanza not available")
        return {'status': 'skip', 'message': 'DANE scraper not installed'}

    all_data = {}
    variables_captured = set()

    try:
        logging.info("[DANE] Fetching trade balance data...")
        df = obtener_balanza_completa(n=15)

        if df is not None and not df.empty:
            for _, row in df.iterrows():
                date_str = row['fecha']
                if date_str not in all_data:
                    all_data[date_str] = {}

                # Exportaciones en millones USD
                if 'exportaciones_usd_millones' in row:
                    all_data[date_str]['exports'] = row['exportaciones_usd_millones']
                    variables_captured.add('exports')

                # Importaciones en millones USD
                if 'importaciones_usd_millones' in row:
                    all_data[date_str]['imports'] = row['importaciones_usd_millones']
                    variables_captured.add('imports')

            logging.info(f"[DANE] Captured {len(df)} months of trade data")
        else:
            logging.warning("[DANE] No data returned")

    except Exception as e:
        logging.error(f"[DANE] Error: {e}")

    context['ti'].xcom_push(key='dane_data', value=all_data)
    context['ti'].xcom_push(key='dane_variables', value=list(variables_captured))

    return {'status': 'success', 'captured': len(variables_captured)}


def scrape_embi(**context):
    """Scrape EMBI Colombia from BCRP or fallback sources."""
    try:
        import cloudscraper
        from bs4 import BeautifulSoup
    except ImportError:
        return {'status': 'error', 'message': 'Missing packages'}

    all_data = {}
    variables_captured = set()

    # Intentar múltiples fuentes para EMBI
    sources = [
        # Ambito.com (Argentina, tiene EMBI latam)
        'https://www.ambito.com/contenidos/riesgo-pais-colombia.html',
    ]

    scraper = cloudscraper.create_scraper()

    for source_url in sources:
        try:
            logging.info(f"[EMBI] Trying {source_url}")
            resp = scraper.get(source_url, timeout=20)

            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')

                # Buscar el valor del EMBI
                # La estructura varía por fuente, intentar patrones comunes
                value_elem = soup.find('span', class_='data-valor')
                if value_elem:
                    try:
                        embi_value = float(value_elem.get_text(strip=True).replace(',', ''))
                        today = datetime.now().strftime('%Y-%m-%d')
                        all_data[today] = {'embi': embi_value}
                        variables_captured.add('embi')
                        logging.info(f"  -> EMBI = {embi_value}")
                        break
                    except:
                        pass

        except Exception as e:
            logging.warning(f"[EMBI] Error with {source_url}: {e}")

    context['ti'].xcom_push(key='embi_data', value=all_data)
    context['ti'].xcom_push(key='embi_variables', value=list(variables_captured))

    return {'status': 'success', 'captured': len(variables_captured)}


# =============================================================================
# FRED API - Solo datos macro oficiales USA
# =============================================================================

FRED_SERIES = {
    # Mensual - política monetaria y inflación
    'fedfunds': 'FEDFUNDS',      # Federal Funds Rate
    'cpi_usa': 'CPIAUCSL',       # CPI All Items
    'core_cpi': 'CPILFESL',      # Core CPI (less food & energy)
    'pce': 'PCEPI',              # PCE Price Index
    'unemployment': 'UNRATE',     # Unemployment Rate
    'indpro': 'INDPRO',          # Industrial Production
    'm2': 'M2SL',                # M2 Money Supply
    'consumer_sentiment': 'UMCSENT',  # U of Michigan Consumer Sentiment

    # Trimestral
    'gdp': 'GDP',                # Gross Domestic Product

    # Diario
    'prime': 'DPRIME',           # Prime Rate
}


def fetch_fred_data(**context):
    """
    Fetch data from FRED API for official US macro data.
    Applies publication lags to avoid data leakage.
    """
    try:
        from fredapi import Fred
    except ImportError:
        logging.warning("fredapi not installed, trying requests fallback")
        return fetch_fred_requests(**context)

    fred = Fred(api_key=FRED_API_KEY)

    all_data = {}
    errors = []
    variables_captured = set()

    today = datetime.now().date()
    start_date = today - timedelta(days=180)  # Últimos 6 meses

    for var_name, series_id in FRED_SERIES.items():
        try:
            logging.info(f"[FRED] Fetching {var_name} ({series_id})...")

            data = fred.get_series(series_id, observation_start=start_date)

            if data is not None and not data.empty:
                # Obtener lag de publicación para esta variable
                pub_lag = PUBLICATION_LAGS.get(var_name, 0)

                for date, value in data.items():
                    if pd.notna(value):
                        # Fecha original del dato
                        data_date = date.date()

                        # Fecha en que el dato estuvo disponible (para evitar leakage)
                        available_date = data_date + timedelta(days=pub_lag)

                        # Solo incluir si ya está disponible
                        if available_date <= today:
                            date_str = data_date.strftime('%Y-%m-%d')

                            if date_str not in all_data:
                                all_data[date_str] = {}
                            all_data[date_str][var_name] = float(value)
                            all_data[date_str][f'{var_name}_available_date'] = available_date.strftime('%Y-%m-%d')
                            variables_captured.add(var_name)

                logging.info(f"  -> {len([d for d in all_data.values() if var_name in d])} records")
            else:
                errors.append(f"{var_name}: No data")

        except Exception as e:
            errors.append(f"{var_name}: {str(e)}")
            logging.error(f"[FRED] Error {var_name}: {e}")

    context['ti'].xcom_push(key='fred_data', value=all_data)
    context['ti'].xcom_push(key='fred_variables', value=list(variables_captured))

    logging.info(f"[FRED] Captured {len(variables_captured)} variables")

    return {
        'status': 'success',
        'variables': len(variables_captured),
        'dates': len(all_data),
        'errors': errors
    }


def fetch_fred_requests(**context):
    """Fallback: fetch FRED data using requests if fredapi not available."""
    base_url = "https://api.stlouisfed.org/fred/series/observations"

    all_data = {}
    errors = []
    variables_captured = set()

    today = datetime.now()
    start_date = (today - timedelta(days=180)).strftime('%Y-%m-%d')

    for var_name, series_id in FRED_SERIES.items():
        try:
            params = {
                'series_id': series_id,
                'api_key': FRED_API_KEY,
                'file_type': 'json',
                'observation_start': start_date,
                'sort_order': 'desc',
                'limit': 100
            }

            resp = requests.get(base_url, params=params, timeout=30)
            data = resp.json()

            if 'observations' in data:
                pub_lag = PUBLICATION_LAGS.get(var_name, 0)

                for obs in data['observations']:
                    if obs['value'] != '.':
                        try:
                            date_str = obs['date']
                            value = float(obs['value'])

                            data_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                            available_date = data_date + timedelta(days=pub_lag)

                            if available_date <= today.date():
                                if date_str not in all_data:
                                    all_data[date_str] = {}
                                all_data[date_str][var_name] = value
                                variables_captured.add(var_name)
                        except:
                            continue

                logging.info(f"[FRED] {var_name}: OK")

            time.sleep(0.5)  # Rate limit

        except Exception as e:
            errors.append(f"{var_name}: {str(e)}")
            logging.error(f"[FRED] Error {var_name}: {e}")

    context['ti'].xcom_push(key='fred_data', value=all_data)
    context['ti'].xcom_push(key='fred_variables', value=list(variables_captured))

    return {'status': 'success', 'variables': len(variables_captured), 'errors': errors}


# =============================================================================
# MERGE AND UPSERT
# =============================================================================

def merge_and_upsert(**context):
    """
    Merge all data sources and UPSERT INCREMENTAL to PostgreSQL.

    Solo inserta datos NUEVOS (fechas posteriores a las existentes).
    Acumula semana a semana sin recargar todo el histórico.
    """
    ti = context['ti']

    # Obtener últimas fechas por indicador
    last_dates = get_last_macro_dates()

    # Obtener datos de todas las fuentes
    investing_data = ti.xcom_pull(key='investing_data', task_ids='scrape_investing') or {}
    banrep_data = ti.xcom_pull(key='banrep_data', task_ids='scrape_banrep') or {}
    embi_data = ti.xcom_pull(key='embi_data', task_ids='scrape_embi') or {}
    dane_data = ti.xcom_pull(key='dane_data', task_ids='scrape_dane') or {}
    fred_data = ti.xcom_pull(key='fred_data', task_ids='fetch_fred') or {}

    # Merge all data by date
    all_data = {}

    for source_data in [investing_data, banrep_data, embi_data, dane_data, fred_data]:
        for date_str, columns in source_data.items():
            if date_str not in all_data:
                all_data[date_str] = {}
            for col, val in columns.items():
                if not col.endswith('_available_date'):
                    all_data[date_str][col] = val

    if not all_data:
        logging.warning("No data to upsert")
        return {'status': 'no_data'}

    logging.info(f"[MERGE] {len(all_data)} fechas de todas las fuentes")

    conn = get_db_connection()
    cursor = conn.cursor()

    total_inserted = 0
    skipped = 0

    try:
        for date_str, columns in all_data.items():
            fecha = datetime.strptime(date_str, '%Y-%m-%d').date()

            for indicator, value in columns.items():
                if value is None:
                    continue

                # FILTRO INCREMENTAL: Solo fechas nuevas por indicador
                indicator_last_date = last_dates.get(indicator)
                if indicator_last_date and fecha <= indicator_last_date:
                    skipped += 1
                    continue

                frequency = VARIABLE_FREQUENCY.get(indicator, 'daily')
                source = 'fred' if indicator in FRED_SERIES else 'scraper'

                try:
                    cursor.execute("""
                        INSERT INTO core.macro_indicators
                        (date, indicator_code, value, source, frequency)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (date, indicator_code) DO UPDATE SET
                            value = EXCLUDED.value,
                            source = EXCLUDED.source,
                            updated_at = CURRENT_TIMESTAMP
                    """, [fecha, indicator, float(value), source, frequency])
                    total_inserted += 1

                except Exception as e:
                    logging.warning(f"Error inserting {indicator} for {date_str}: {e}")

        conn.commit()
        logging.info(f"[INCREMENTAL] Inserted: {total_inserted}, Skipped (ya existentes): {skipped}")

    except Exception as e:
        conn.rollback()
        logging.error(f"Database error: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

    context['ti'].xcom_push(key='total_inserted', value=total_inserted)
    context['ti'].xcom_push(key='skipped', value=skipped)

    return {'status': 'success', 'inserted': total_inserted, 'skipped': skipped}


# =============================================================================
# SMART FORWARD FILL
# =============================================================================

def apply_publication_aware_ffill(**context):
    """
    Apply forward fill considering publication frequencies.

    Rules:
    - Daily data: max 5 days ffill
    - Weekly data: max 10 days ffill
    - Monthly data: max 40 days ffill
    - Quarterly data: max 100 days ffill
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    total_filled = 0

    try:
        # Get all unique indicators
        cursor.execute("""
            SELECT DISTINCT indicator_code, frequency
            FROM core.macro_indicators
        """)
        indicators = cursor.fetchall()

        for indicator, frequency in indicators:
            if frequency is None:
                frequency = VARIABLE_FREQUENCY.get(indicator, 'daily')

            max_ffill_days = MAX_FFILL.get(frequency, 5)

            # Get last known value
            cursor.execute("""
                SELECT date, value
                FROM core.macro_indicators
                WHERE indicator_code = %s AND value IS NOT NULL
                ORDER BY date DESC
                LIMIT 1
            """, [indicator])

            result = cursor.fetchone()
            if not result:
                continue

            last_date, last_value = result
            today = datetime.now().date()
            days_since = (today - last_date).days

            # Only fill if within allowed window
            if days_since > 0 and days_since <= max_ffill_days:
                # Get trading days that need filling
                cursor.execute("""
                    SELECT DISTINCT h.date
                    FROM core.usdcop_historical h
                    LEFT JOIN core.macro_indicators m
                        ON m.date = h.date AND m.indicator_code = %s
                    WHERE h.date > %s
                      AND h.date <= CURRENT_DATE
                      AND m.value IS NULL
                    ORDER BY h.date
                """, [indicator, last_date])

                dates_to_fill = [row[0] for row in cursor.fetchall()]

                for fill_date in dates_to_fill:
                    cursor.execute("""
                        INSERT INTO core.macro_indicators
                        (date, indicator_code, value, source, frequency)
                        VALUES (%s, %s, %s, 'ffill', %s)
                        ON CONFLICT (date, indicator_code) DO UPDATE SET
                            value = EXCLUDED.value,
                            source = 'ffill',
                            updated_at = CURRENT_TIMESTAMP
                    """, [fill_date, indicator, last_value, frequency])
                    total_filled += 1

                if dates_to_fill:
                    logging.info(f"[FFILL] {indicator}: filled {len(dates_to_fill)} days")

        conn.commit()
        logging.info(f"[FFILL] Total: {total_filled} records filled")

    except Exception as e:
        conn.rollback()
        logging.error(f"[FFILL] Error: {e}")
    finally:
        cursor.close()
        conn.close()

    return {'status': 'success', 'filled': total_filled}


# =============================================================================
# VALIDATION
# =============================================================================

def validate_coverage(**context):
    """Validate data coverage and freshness."""
    ti = context['ti']

    investing_vars = ti.xcom_pull(key='investing_variables', task_ids='scrape_investing') or []
    banrep_vars = ti.xcom_pull(key='banrep_variables', task_ids='scrape_banrep') or []
    embi_vars = ti.xcom_pull(key='embi_variables', task_ids='scrape_embi') or []
    dane_vars = ti.xcom_pull(key='dane_variables', task_ids='scrape_dane') or []
    fred_vars = ti.xcom_pull(key='fred_variables', task_ids='fetch_fred') or []

    all_captured = set(investing_vars + banrep_vars + embi_vars + dane_vars + fred_vars)

    # Expected variables
    expected = set(INVESTING_URLS.keys()) | set(BANREP_SERIES.keys()) | set(FRED_SERIES.keys()) | {'embi', 'exports', 'imports'}

    coverage = len(all_captured) / len(expected) if expected else 0
    missing = expected - all_captured

    logging.info("=" * 60)
    logging.info("COVERAGE REPORT")
    logging.info("=" * 60)
    logging.info(f"Expected: {len(expected)} variables")
    logging.info(f"Captured: {len(all_captured)} variables")
    logging.info(f"Coverage: {coverage:.1%}")

    if missing:
        logging.warning(f"Missing: {missing}")

    logging.info("=" * 60)

    # Minimum 60% coverage required
    if coverage < 0.60:
        raise AirflowException(f"Coverage too low: {coverage:.1%} < 60%")

    return {
        'coverage': coverage,
        'captured': len(all_captured),
        'expected': len(expected),
        'missing': list(missing)
    }


def generate_report(**context):
    """Generate summary report - INCREMENTAL."""
    ti = context['ti']

    total_inserted = ti.xcom_pull(key='total_inserted', task_ids='merge_upsert') or 0
    skipped = ti.xcom_pull(key='skipped', task_ids='merge_upsert') or 0

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Total records and date range
        cursor.execute("""
            SELECT
                COUNT(DISTINCT indicator_code) as indicators,
                COUNT(*) as total_records,
                MIN(date) as min_date,
                MAX(date) as max_date
            FROM core.macro_indicators
        """)
        row = cursor.fetchone()
        total_indicators, total_records, min_date, max_date = row

        logging.info("=" * 60)
        logging.info("L0 MACRO DATA UPDATE - INCREMENTAL REPORT")
        logging.info(f"Timestamp: {datetime.now().isoformat()}")
        logging.info("=" * 60)
        logging.info(f"INCREMENTAL: {total_inserted} nuevos, {skipped} ya existentes")
        logging.info(f"TOTAL en DB: {total_indicators} indicadores, {total_records} registros")
        logging.info(f"Rango fechas: {min_date} a {max_date}")
        logging.info("=" * 60)

        # Recent data by indicator
        cursor.execute("""
            SELECT
                indicator_code,
                COUNT(*) as records,
                MAX(date) as last_date
            FROM core.macro_indicators
            WHERE date >= CURRENT_DATE - 7
            GROUP BY indicator_code
            ORDER BY last_date DESC
            LIMIT 15
        """)

        logging.info("Últimos datos por indicador (top 15):")
        for row in cursor.fetchall():
            indicator, records, last_date = row
            logging.info(f"  {indicator}: {records} rec, last={last_date}")

        logging.info("=" * 60)

    finally:
        cursor.close()
        conn.close()

    return {'status': 'completed', 'inserted': total_inserted, 'skipped': skipped}


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='L0: Macro data update using Scrapers + FRED API',
    schedule_interval='0 12 * * 1',  # Lunes 7:00 AM COT (12:00 UTC) - Semanal antes de inferencia
    catchup=False,
    max_active_runs=1,
    tags=['layer0', 'macro', 'scrapers', 'fred', 'weekly'],
)

# Tasks
start = EmptyOperator(task_id='start', dag=dag)

# Parallel scraping tasks
t_investing = PythonOperator(
    task_id='scrape_investing',
    python_callable=scrape_investing_batch,
    provide_context=True,
    execution_timeout=timedelta(minutes=15),
    dag=dag,
)

t_banrep = PythonOperator(
    task_id='scrape_banrep',
    python_callable=scrape_banrep,
    provide_context=True,
    execution_timeout=timedelta(minutes=20),
    dag=dag,
)

t_embi = PythonOperator(
    task_id='scrape_embi',
    python_callable=scrape_embi,
    provide_context=True,
    execution_timeout=timedelta(minutes=5),
    dag=dag,
)

t_dane = PythonOperator(
    task_id='scrape_dane',
    python_callable=scrape_dane_balanza,
    provide_context=True,
    execution_timeout=timedelta(minutes=10),
    dag=dag,
)

t_fred = PythonOperator(
    task_id='fetch_fred',
    python_callable=fetch_fred_data,
    provide_context=True,
    execution_timeout=timedelta(minutes=10),
    dag=dag,
)

# Sequential processing
t_merge = PythonOperator(
    task_id='merge_upsert',
    python_callable=merge_and_upsert,
    provide_context=True,
    dag=dag,
)

t_ffill = PythonOperator(
    task_id='apply_ffill',
    python_callable=apply_publication_aware_ffill,
    provide_context=True,
    dag=dag,
)

t_validate = PythonOperator(
    task_id='validate_coverage',
    python_callable=validate_coverage,
    provide_context=True,
    dag=dag,
)

t_report = PythonOperator(
    task_id='generate_report',
    python_callable=generate_report,
    provide_context=True,
    dag=dag,
)

end = EmptyOperator(task_id='end', dag=dag)

# Dependencies
# Parallel: investing, banrep, embi, dane, fred -> merge -> ffill -> validate -> report
start >> [t_investing, t_banrep, t_embi, t_dane, t_fred]
[t_investing, t_banrep, t_embi, t_dane, t_fred] >> t_merge
t_merge >> t_ffill >> t_validate >> t_report >> end

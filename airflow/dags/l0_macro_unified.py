"""
DAG: v3.l0_macro_unified
=========================
Unified Daily Macro Indicators DAG - ALL 37 Variables

Purpose:
    Single DAG that fetches ALL 37 macro variables from multiple sources:
    - FRED API (12 variables)
    - TwelveData API (4 variables)
    - BanRep SUAMECA via Selenium (6 variables)
    - Investing.com via Cloudscraper (7 variables)
    - Forward-fill for remaining/stale variables

Schedule:
    50 12 * * 1-5 (7:50am COT, Mon-Fri, pre-session)

Data Flow:
    Multiple Sources -> macro_indicators_daily (UPSERT) -> inference_features
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
import pandas as pd
import psycopg2
import os
import time
import logging
import requests

# Shared utilities
from utils.dag_common import get_db_connection

# Contract imports
from contracts.l0_data_contracts import (
    MACRO_INDICATOR_REGISTRY,
    PublicationSchedule,
    FFillConfig,
    FFillApplicationResult,
    IndicatorReadiness,
    IndicatorReadinessStatus,
    DailyDataReadinessReport,
    DEFAULT_FFILL_CONFIG,
)

DAG_ID = "v3.l0_macro_unified"

# =============================================================================
# SOURCE CONFIGURATION - ALL 37 VARIABLES
# =============================================================================

# FRED API - 12 Variables (Daily + Monthly + Quarterly)
FRED_MAPPING = {
    # Daily
    "DTWEXBGS": "fxrt_index_dxy_usa_d_dxy",
    "VIXCLS": "volt_vix_usa_d_vix",
    "DGS10": "finc_bond_yield10y_usa_d_ust10y",
    "DGS2": "finc_bond_yield2y_usa_d_dgs2",
    "DPRIME": "polr_prime_rate_usa_d_prime",
    # Monthly
    "FEDFUNDS": "polr_fed_funds_usa_m_fedfunds",
    "CPIAUCSL": "infl_cpi_all_usa_m_cpiaucsl",
    "CPILFESL": "infl_cpi_core_usa_m_cpilfesl",
    "PCEPI": "infl_pce_usa_m_pcepi",
    "UNRATE": "labr_unemployment_usa_m_unrate",
    "INDPRO": "prod_industrial_usa_m_indpro",
    "M2SL": "mnys_m2_supply_usa_m_m2sl",
    "UMCSENT": "sent_consumer_usa_m_umcsent",
    # Quarterly
    "GDP": "gdpp_real_gdp_usa_q_gdp_q",
}

# TwelveData API - 4 Variables (Daily)
TWELVEDATA_MAPPING = {
    "USD/MXN": "fxrt_spot_usdmxn_mex_d_usdmxn",
    "USD/CLP": "fxrt_spot_usdclp_chl_d_usdclp",
    "CL": "comm_oil_wti_glb_d_wti",
    "BZ": "comm_oil_brent_glb_d_brent",
}

# BanRep SUAMECA via Selenium - 6 Variables
BANREP_MAPPING = {
    '241': {
        'column': 'finc_rate_ibr_overnight_col_d_ibr',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/241/tasas_interes_indicador_bancario_referencia_ibr',
        'name': 'IBR Overnight'
    },
    '59': {
        'column': 'polr_policy_rate_col_d_tpm',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/59/tasas_interes_politica_monetaria',
        'name': 'TPM'
    },
    '4170': {
        'column': 'fxrt_reer_bilateral_col_m_itcr',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/4170/indice_tasa_cambio_real_itcr',
        'name': 'ITCR'
    },
    '4180': {
        'column': 'ftrd_terms_trade_col_m_tot',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/4180/indice_terminos_intercambio_bienes',
        'name': 'TOT'
    },
    '15051': {
        'column': 'rsbp_reserves_international_col_m_resint',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/15051/reservas_internacionales',
        'name': 'Reservas Int.'
    },
    '100002': {
        'column': 'infl_cpi_total_col_m_ipccol',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/100002/ipc',
        'name': 'IPC Colombia'
    },
}

# Investing.com via Cloudscraper - 7 Variables
INVESTING_MAPPING = {
    'https://www.investing.com/indices/colcap-historical-data': 'eqty_index_colcap_col_d_colcap',
    'https://www.investing.com/rates-bonds/colombia-10-year-bond-yield-historical-data': 'finc_bond_yield10y_col_d_col10y',
    'https://www.investing.com/rates-bonds/colombia-5-year-bond-yield-historical-data': 'finc_bond_yield5y_col_d_col5y',
    'https://www.investing.com/commodities/gold-historical-data': 'comm_metal_gold_glb_d_gold',
    'https://www.investing.com/commodities/us-coffee-c-historical-data': 'comm_agri_coffee_glb_d_coffee',
}

# EMBI Colombia via BCRP Peru - 1 Variable
EMBI_URL = 'https://estadisticas.bcrp.gob.pe/estadisticas/series/diarias/resultados/PD04715XD/html/'
EMBI_COLUMN = 'crsk_spread_embi_col_d_embi'  # Uses existing column name

# =============================================================================
# FETCH FUNCTIONS
# =============================================================================

def fetch_fred_data(**context):
    """Fetch all FRED series."""
    from fredapi import Fred

    api_key = os.environ.get('FRED_API_KEY')
    if not api_key:
        logging.error("FRED_API_KEY not set")
        return {'status': 'error', 'message': 'No API key'}

    fred = Fred(api_key=api_key)
    results = {}
    errors = []

    today = datetime.now().date()
    start_date = today - timedelta(days=90)

    for series_id, column in FRED_MAPPING.items():
        try:
            logging.info(f"[FRED] Fetching {series_id} -> {column}")
            data = fred.get_series(series_id, observation_start=start_date)

            if data is not None and not data.empty:
                for date, value in data.items():
                    if pd.notna(value):
                        date_str = date.strftime('%Y-%m-%d')
                        if date_str not in results:
                            results[date_str] = {}
                        results[date_str][column] = float(value)

                logging.info(f"  -> {len(data)} records")
            else:
                errors.append(f"{series_id}: No data")

        except Exception as e:
            errors.append(f"{series_id}: {str(e)}")
            logging.error(f"[FRED] Error {series_id}: {e}")

    context['ti'].xcom_push(key='fred_data', value=results)
    return {'status': 'success', 'records': len(results), 'errors': errors}


def fetch_twelvedata(**context):
    """Fetch TwelveData series."""
    api_key = os.environ.get('TWELVEDATA_API_KEY_1')
    if not api_key:
        logging.error("TWELVEDATA_API_KEY_1 not set")
        return {'status': 'error', 'message': 'No API key'}

    results = {}
    errors = []

    for symbol, column in TWELVEDATA_MAPPING.items():
        try:
            logging.info(f"[TwelveData] Fetching {symbol} -> {column}")

            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': symbol,
                'interval': '1day',
                'outputsize': 60,
                'apikey': api_key
            }

            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()

            if 'values' in data:
                for item in data['values']:
                    date_str = item['datetime']
                    value = float(item['close'])
                    if date_str not in results:
                        results[date_str] = {}
                    results[date_str][column] = value

                logging.info(f"  -> {len(data['values'])} records")
            else:
                errors.append(f"{symbol}: {data.get('message', 'No data')}")

            time.sleep(1)  # Rate limit

        except Exception as e:
            errors.append(f"{symbol}: {str(e)}")
            logging.error(f"[TwelveData] Error {symbol}: {e}")

    context['ti'].xcom_push(key='twelvedata_data', value=results)
    return {'status': 'success', 'records': len(results), 'errors': errors}


def fetch_investing(**context):
    """Fetch Investing.com data via cloudscraper."""
    try:
        import cloudscraper
        from bs4 import BeautifulSoup
    except ImportError:
        logging.error("cloudscraper/beautifulsoup4 not installed")
        return {'status': 'error', 'message': 'Missing packages'}

    scraper = cloudscraper.create_scraper()
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    results = {}
    errors = []

    for url, column in INVESTING_MAPPING.items():
        try:
            logging.info(f"[Investing] Fetching {column}")

            resp = scraper.get(url, headers=headers, timeout=25)

            if resp.status_code != 200:
                errors.append(f"{column}: HTTP {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')
            table = soup.find('table', class_='freeze-column-w-1')
            if not table:
                tables = soup.find_all('table')
                if tables:
                    table = max(tables, key=lambda t: len(str(t)))

            if not table:
                errors.append(f"{column}: No table")
                continue

            rows = table.find_all('tr')[1:65]
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    try:
                        date_str = pd.to_datetime(cols[0].get_text(strip=True)).strftime('%Y-%m-%d')
                        value = float(cols[1].get_text(strip=True).replace(',', ''))

                        if date_str not in results:
                            results[date_str] = {}
                        results[date_str][column] = value
                    except:
                        continue

            logging.info(f"  -> Data extracted")
            time.sleep(2)  # Rate limit

        except Exception as e:
            errors.append(f"{column}: {str(e)}")
            logging.error(f"[Investing] Error {column}: {e}")

    context['ti'].xcom_push(key='investing_data', value=results)
    return {'status': 'success', 'records': len(results), 'errors': errors}


def fetch_embi(**context):
    """Fetch EMBI Colombia spread from BCRP Peru."""
    import re
    try:
        import cloudscraper
        from bs4 import BeautifulSoup
    except ImportError:
        logging.error("cloudscraper/beautifulsoup4 not installed")
        return {'status': 'error', 'message': 'Missing packages'}

    results = {}
    errors = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }

    scraper = cloudscraper.create_scraper()

    try:
        logging.info(f"[EMBI] Fetching from BCRP Peru: {EMBI_URL}")
        response = scraper.get(EMBI_URL, headers=headers, timeout=20)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            tables = soup.find_all('table')

            # Find largest table
            main_table = max(tables, key=lambda t: len(str(t)), default=None)

            if main_table:
                rows = main_table.find_all('tr')

                # Spanish month mapping
                meses = {
                    'Ene': '01', 'Feb': '02', 'Mar': '03', 'Abr': '04',
                    'May': '05', 'Jun': '06', 'Jul': '07', 'Ago': '08',
                    'Set': '09', 'Oct': '10', 'Nov': '11', 'Dic': '12'
                }

                count = 0
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        fecha_str = cells[0].get_text(strip=True)
                        valor_str = cells[1].get_text(strip=True)

                        # Parse date like '06Ene26'
                        match = re.match(r'(\d{2})([A-Za-z]{3})(\d{2})', fecha_str)
                        if match and valor_str.isdigit():
                            dia, mes_txt, anio = match.groups()
                            mes = meses.get(mes_txt.capitalize())
                            if mes:
                                anio_full = f"20{anio}" if int(anio) <= 49 else f"19{anio}"
                                date_str = f"{anio_full}-{mes}-{dia}"

                                if date_str not in results:
                                    results[date_str] = {}
                                results[date_str][EMBI_COLUMN] = int(valor_str)
                                count += 1

                logging.info(f"  -> {count} EMBI records extracted")
            else:
                errors.append("EMBI: No table found")
        else:
            errors.append(f"EMBI: HTTP {response.status_code}")

    except Exception as e:
        errors.append(f"EMBI: {str(e)}")
        logging.error(f"[EMBI] Error: {e}")

    context['ti'].xcom_push(key='embi_data', value=results)
    return {'status': 'success', 'records': len(results), 'errors': errors}


def fetch_banrep_selenium(**context):
    """Fetch BanRep data via Selenium."""
    try:
        import undetected_chromedriver as uc
        from selenium.webdriver.common.by import By
    except ImportError:
        logging.error("undetected-chromedriver not installed")
        return {'status': 'error', 'message': 'Missing Selenium'}

    results = {}
    errors = []

    # Setup Chrome
    try:
        options = uc.ChromeOptions()
        options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = uc.Chrome(options=options)
        logging.info("[BanRep] Chrome initialized")
    except Exception as e:
        logging.error(f"[BanRep] Chrome init failed: {e}")
        return {'status': 'error', 'message': str(e)}

    try:
        for serie_id, config in BANREP_MAPPING.items():
            try:
                logging.info(f"[BanRep] Scraping {config['name']} (Serie {serie_id})")

                driver.get(config['url'])
                time.sleep(5)

                # Click "Vista tabla"
                buttons = driver.find_elements(By.TAG_NAME, "button")
                for btn in buttons:
                    if 'tabla' in btn.text.lower():
                        btn.click()
                        time.sleep(5)
                        break

                # Find data table
                tables = driver.find_elements(By.TAG_NAME, "table")
                data_table = None
                max_rows = 0
                for table in tables:
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    if len(rows) > max_rows:
                        max_rows = len(rows)
                        data_table = table

                if not data_table or max_rows < 10:
                    errors.append(f"{config['name']}: No table")
                    continue

                # Extract data (date in <th>, value in first <td>)
                rows = data_table.find_elements(By.TAG_NAME, "tr")
                count = 0

                for row in rows[1:]:
                    try:
                        th = row.find_elements(By.TAG_NAME, "th")
                        td = row.find_elements(By.TAG_NAME, "td")

                        if not th or not td:
                            continue

                        fecha_str = th[0].text.strip()
                        valor_str = td[0].text.strip().replace(',', '.').replace('%', '')

                        if not fecha_str or not valor_str:
                            continue

                        # Parse date
                        try:
                            fecha = datetime.strptime(fecha_str, '%Y/%m/%d').date()
                        except:
                            fecha = pd.to_datetime(fecha_str).date()

                        valor = float(valor_str)
                        date_str = fecha.strftime('%Y-%m-%d')

                        if date_str not in results:
                            results[date_str] = {}
                        results[date_str][config['column']] = valor
                        count += 1

                    except:
                        continue

                logging.info(f"  -> {count} records")

            except Exception as e:
                errors.append(f"{config['name']}: {str(e)}")
                logging.error(f"[BanRep] Error {config['name']}: {e}")

    finally:
        driver.quit()
        logging.info("[BanRep] Chrome closed")

    context['ti'].xcom_push(key='banrep_data', value=results)
    return {'status': 'success', 'records': len(results), 'errors': errors}


def merge_and_upsert(**context):
    """Merge all sources and upsert to database."""
    ti = context['ti']

    # Get data from all sources
    fred_data = ti.xcom_pull(key='fred_data', task_ids='fetch_fred') or {}
    twelvedata_data = ti.xcom_pull(key='twelvedata_data', task_ids='fetch_twelvedata') or {}
    investing_data = ti.xcom_pull(key='investing_data', task_ids='fetch_investing') or {}
    banrep_data = ti.xcom_pull(key='banrep_data', task_ids='fetch_banrep') or {}
    embi_data = ti.xcom_pull(key='embi_data', task_ids='fetch_embi') or {}

    # Merge all data
    all_data = {}

    for source_data in [fred_data, twelvedata_data, investing_data, banrep_data, embi_data]:
        for date_str, columns in source_data.items():
            if date_str not in all_data:
                all_data[date_str] = {}
            all_data[date_str].update(columns)

    if not all_data:
        logging.warning("No data to upsert")
        return {'status': 'no_data', 'inserted': 0, 'updated': 0}

    logging.info(f"Merging {len(all_data)} dates from all sources")

    # Upsert to database
    conn = get_db_connection()
    cur = conn.cursor()

    total_inserted = 0
    total_updated = 0

    today = datetime.now().date()

    try:
        for date_str, columns in all_data.items():
            if not columns:
                continue

            fecha = datetime.strptime(date_str, '%Y-%m-%d').date()

            # Filter out future dates (API timezone issues)
            if fecha > today:
                logging.warning(f"Skipping future date from API: {fecha}")
                continue

            for column, value in columns.items():
                try:
                    cur.execute(f"""
                        INSERT INTO macro_indicators_daily (fecha, {column})
                        VALUES (%s, %s)
                        ON CONFLICT (fecha) DO UPDATE SET
                            {column} = EXCLUDED.{column},
                            updated_at = NOW()
                        WHERE macro_indicators_daily.{column} IS NULL
                           OR macro_indicators_daily.{column} != EXCLUDED.{column}
                    """, [fecha, value])

                    if cur.rowcount > 0:
                        total_updated += 1

                except Exception as e:
                    logging.warning(f"Error upserting {column} for {fecha}: {e}")

        conn.commit()
        logging.info(f"Database updated: {total_updated} values changed")

    except Exception as e:
        conn.rollback()
        logging.error(f"Database error: {e}")
        raise
    finally:
        cur.close()
        conn.close()

    return {'status': 'success', 'dates': len(all_data), 'updated': total_updated}


def apply_forward_fill(**context) -> dict:
    """Apply bounded forward fill respecting per-indicator limits.

    Uses MACRO_INDICATOR_REGISTRY to determine max FFILL days per indicator.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    config = DEFAULT_FFILL_CONFIG
    total_filled = 0
    exceeded_columns = []
    fill_details = {}

    try:
        for column, metadata in MACRO_INDICATOR_REGISTRY.items():
            # Get max FFILL days based on publication schedule
            max_days = config.get_max_days_for_schedule(metadata.schedule)

            # Get last known value with its date
            cur.execute(f"""
                SELECT fecha, {column}
                FROM macro_indicators_daily
                WHERE {column} IS NOT NULL
                ORDER BY fecha DESC
                LIMIT 1
            """)

            result = cur.fetchone()
            if not result:
                fill_details[column] = {"filled": 0, "exceeded": 0, "max_days": max_days, "status": "no_data"}
                continue

            last_date, last_value = result

            # Fill only up to max_days from last known value
            cur.execute(f"""
                UPDATE macro_indicators_daily
                SET {column} = %s, updated_at = NOW()
                WHERE fecha > %s
                  AND fecha <= %s + INTERVAL '%s days'
                  AND fecha <= CURRENT_DATE
                  AND {column} IS NULL
            """, [last_value, last_date, last_date, max_days])

            filled = cur.rowcount
            total_filled += filled

            # Count rows that exceed the limit
            cur.execute(f"""
                SELECT COUNT(*) FROM macro_indicators_daily
                WHERE fecha > %s + INTERVAL '%s days'
                  AND fecha <= CURRENT_DATE
                  AND {column} IS NULL
            """, [last_date, max_days])
            exceeded = cur.fetchone()[0]

            if exceeded > 0:
                exceeded_columns.append(column)
                if config.on_limit_exceeded == "warn":
                    logging.warning(f"[FFILL] {column}: {exceeded} rows exceed {max_days}-day limit (schedule: {metadata.schedule.value})")

            fill_details[column] = {
                "filled": filled,
                "exceeded": exceeded,
                "max_days": max_days,
                "last_value": float(last_value) if last_value else None,
                "last_date": str(last_date),
            }

            if filled > 0:
                logging.info(f"[FFILL] {metadata.display_name}: {filled} rows filled (max {max_days} days)")

        conn.commit()
        logging.info(f"Bounded forward fill completed: {total_filled} total rows filled")
        if exceeded_columns:
            logging.warning(f"Columns exceeding FFILL limits: {exceeded_columns}")

    except Exception as e:
        conn.rollback()
        logging.error(f"Forward fill error: {e}")
        raise
    finally:
        cur.close()
        conn.close()

    # Push result to XCom
    result = {
        'status': 'success',
        'filled': total_filled,
        'exceeded_columns': exceeded_columns,
        'fill_details': fill_details,
    }
    context['ti'].xcom_push(key='ffill_result', value=result)

    return result


def generate_report(**context):
    """Generate quality report."""
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Check coverage for today
        cur.execute("""
            SELECT
                fecha,
                COUNT(*) FILTER (WHERE fxrt_index_dxy_usa_d_dxy IS NOT NULL) as dxy,
                COUNT(*) FILTER (WHERE volt_vix_usa_d_vix IS NOT NULL) as vix,
                COUNT(*) FILTER (WHERE finc_rate_ibr_overnight_col_d_ibr IS NOT NULL) as ibr,
                COUNT(*) FILTER (WHERE polr_policy_rate_col_d_tpm IS NOT NULL) as tpm,
                COUNT(*) FILTER (WHERE eqty_index_colcap_col_d_colcap IS NOT NULL) as colcap
            FROM macro_indicators_daily
            WHERE fecha >= CURRENT_DATE - 7
            GROUP BY fecha
            ORDER BY fecha DESC
        """)

        report = cur.fetchall()

        logging.info("=" * 60)
        logging.info("MACRO DATA QUALITY REPORT")
        logging.info("=" * 60)

        for row in report:
            fecha, dxy, vix, ibr, tpm, colcap = row
            logging.info(f"{fecha}: DXY={dxy} VIX={vix} IBR={ibr} TPM={tpm} COLCAP={colcap}")

        return {'status': 'success', 'report': report}

    except Exception as e:
        logging.error(f"Report error: {e}")
        return {'status': 'error', 'message': str(e)}
    finally:
        cur.close()
        conn.close()


def generate_readiness_report(**context) -> dict:
    """Generate comprehensive data readiness report.

    Creates DailyDataReadinessReport as the SINGLE SOURCE OF TRUTH
    for whether data is ready for inference.
    """
    from datetime import date
    ti = context['ti']

    # Get FFILL result from previous task
    ffill_result = ti.xcom_pull(task_ids='apply_forward_fill', key='ffill_result') or {}

    conn = get_db_connection()
    cur = conn.cursor()

    indicator_details = []
    today = date.today()

    try:
        for column, metadata in MACRO_INDICATOR_REGISTRY.items():
            # Get current value for this indicator
            cur.execute(f"""
                SELECT {column}, fecha
                FROM macro_indicators_daily
                WHERE fecha = CURRENT_DATE
            """)
            row = cur.fetchone()

            if row and row[0] is not None:
                value, fecha = row

                # Check if this was FFILLed based on fill_details
                fill_info = ffill_result.get('fill_details', {}).get(column, {})
                is_ffilled = fill_info.get('filled', 0) > 0
                ffill_days = (today - fecha).days if fecha else 0

                if not is_ffilled and ffill_days == 0:
                    status = IndicatorReadinessStatus.FRESH
                elif ffill_days <= metadata.max_ffill_days:
                    status = IndicatorReadinessStatus.FFILLED
                else:
                    status = IndicatorReadinessStatus.STALE

                indicator_details.append(IndicatorReadiness(
                    column_name=column,
                    display_name=metadata.display_name,
                    status=status,
                    latest_value=float(value),
                    latest_observation_date=fecha,
                    age_days=ffill_days,
                    is_ffilled=is_ffilled,
                    ffill_days=ffill_days,
                    ffill_within_limit=ffill_days <= metadata.max_ffill_days
                ))
            else:
                indicator_details.append(IndicatorReadiness(
                    column_name=column,
                    display_name=metadata.display_name,
                    status=IndicatorReadinessStatus.MISSING
                ))

        # Build readiness report
        report = DailyDataReadinessReport(
            indicator_details=indicator_details,
            indicators_fresh=sum(1 for d in indicator_details if d.status == IndicatorReadinessStatus.FRESH),
            indicators_ffilled=sum(1 for d in indicator_details if d.status == IndicatorReadinessStatus.FFILLED),
            indicators_stale=sum(1 for d in indicator_details if d.status == IndicatorReadinessStatus.STALE),
            indicators_missing=sum(1 for d in indicator_details if d.status == IndicatorReadinessStatus.MISSING),
            indicators_error=sum(1 for d in indicator_details if d.status == IndicatorReadinessStatus.ERROR),
            ffill_applied=ffill_result.get('filled', 0) > 0,
            ffill_total_rows=ffill_result.get('filled', 0),
            ffill_exceeded_limit=len(ffill_result.get('exceeded_columns', [])),
        )

        # Log comprehensive summary
        summary = report.to_summary_dict()
        logging.info("=" * 60)
        logging.info("DAILY DATA READINESS REPORT")
        logging.info("=" * 60)
        logging.info(f"Date: {summary['date']}")
        logging.info(f"Ready for Inference: {'YES' if summary['ready'] else 'NO'}")
        logging.info(f"Readiness Score: {summary['score']}")
        logging.info(f"Indicators: Fresh={summary['fresh']}, FFilled={summary['ffilled']}, "
                     f"Stale={summary['stale']}, Missing={summary['missing']}, Errors={summary['errors']}")

        if report.blocking_issues:
            logging.error("BLOCKING ISSUES:")
            for issue in report.blocking_issues:
                logging.error(f"  - {issue}")

        if report.warnings:
            logging.warning("WARNINGS:")
            for warn in report.warnings:
                logging.warning(f"  - {warn}")

        # Push to XCom for downstream DAGs (L5 inference can check this)
        ti.xcom_push(key='readiness_report', value=summary)
        ti.xcom_push(key='is_ready_for_inference', value=report.is_ready_for_inference)

        return {
            'status': 'success',
            'is_ready': report.is_ready_for_inference,
            'score': report.readiness_score,
            'summary': summary,
        }

    except Exception as e:
        logging.error(f"Readiness report error: {e}")
        return {'status': 'error', 'message': str(e), 'is_ready': False}
    finally:
        cur.close()
        conn.close()


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'usdcop-trading',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Unified macro DAG - ALL 37 variables from all sources',
    schedule_interval='50 12 * * 1-5',  # 7:50am COT Mon-Fri
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['l0', 'macro', 'unified', 'daily'],
) as dag:

    start = EmptyOperator(task_id='start')

    # Parallel fetch tasks
    fetch_fred_task = PythonOperator(
        task_id='fetch_fred',
        python_callable=fetch_fred_data,
        execution_timeout=timedelta(minutes=5),
    )

    fetch_twelvedata_task = PythonOperator(
        task_id='fetch_twelvedata',
        python_callable=fetch_twelvedata,
        execution_timeout=timedelta(minutes=5),
    )

    fetch_investing_task = PythonOperator(
        task_id='fetch_investing',
        python_callable=fetch_investing,
        execution_timeout=timedelta(minutes=10),
    )

    fetch_banrep_task = PythonOperator(
        task_id='fetch_banrep',
        python_callable=fetch_banrep_selenium,
        execution_timeout=timedelta(minutes=20),
    )

    fetch_embi_task = PythonOperator(
        task_id='fetch_embi',
        python_callable=fetch_embi,
        execution_timeout=timedelta(minutes=5),
    )

    # Merge and upsert
    merge_task = PythonOperator(
        task_id='merge_and_upsert',
        python_callable=merge_and_upsert,
        execution_timeout=timedelta(minutes=5),
    )

    # Forward fill
    ffill_task = PythonOperator(
        task_id='apply_forward_fill',
        python_callable=apply_forward_fill,
        execution_timeout=timedelta(minutes=5),
    )

    # Report
    report_task = PythonOperator(
        task_id='generate_report',
        python_callable=generate_report,
        execution_timeout=timedelta(minutes=2),
    )

    # Readiness Report - Single source of truth for inference readiness
    readiness_task = PythonOperator(
        task_id='generate_readiness_report',
        python_callable=generate_readiness_report,
        execution_timeout=timedelta(minutes=2),
    )

    end = EmptyOperator(task_id='end')

    # Task dependencies
    # Parallel fetch -> merge -> ffill -> report -> readiness -> end
    start >> [fetch_fred_task, fetch_twelvedata_task, fetch_investing_task, fetch_banrep_task, fetch_embi_task]
    [fetch_fred_task, fetch_twelvedata_task, fetch_investing_task, fetch_banrep_task, fetch_embi_task] >> merge_task
    merge_task >> ffill_task >> report_task >> readiness_task >> end

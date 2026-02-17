#!/usr/bin/env python3
"""
Simple SUAMECA scraper for IPCCOL, ITCR, TOT.
Uses Selenium with longer waits and explicit table extraction.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "pipeline" / "01_sources" / "consolidated"

# Series to scrape
SERIES = {
    'IPCCOL': {
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/100002/ipc',
        'col': 'INFL_CPI_TOTAL_COL_M_IPCCOL',
    },
    'ITCR': {
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/4170/indice_tasa_cambio_real_itcr',
        'col': 'FXRT_REER_BILATERAL_COL_M_ITCR',
    },
    'TOT': {
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/4180/indice_terminos_intercambio_bienes',
        'col': 'FTRD_TERMS_TRADE_COL_M_TOT',
    },
}


def setup_driver(headless=True):
    """Setup Chrome driver."""
    try:
        import undetected_chromedriver as uc

        options = uc.ChromeOptions()
        if headless:
            options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--lang=es-CO')

        driver = uc.Chrome(options=options, version_main=None)
        print("[OK] Chrome driver initialized")
        return driver
    except Exception as e:
        print(f"[ERROR] Failed to setup Chrome: {e}")
        return None


def scrape_serie(driver, name, config):
    """Scrape a single SUAMECA serie."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.action_chains import ActionChains

    url = config['url']
    print(f"\n[{name}] Loading {url}")

    data = []

    try:
        driver.get(url)
        print("  Waiting for page load...")
        time.sleep(8)

        # Check for captcha
        page_source = driver.page_source.lower()
        if 'captcha' in page_source or 'radware' in page_source or 'challenge' in page_source:
            print("  [WARN] Captcha detected, waiting longer...")
            time.sleep(20)

        # Wait for Angular app to load
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, "app-root"))
        )
        print("  Angular app loaded")

        # Look for table view button and click it
        time.sleep(3)

        # Try different selectors for the table button
        table_clicked = False

        # Method 1: Find button with table/list icon
        try:
            buttons = driver.find_elements(By.TAG_NAME, "button")
            for btn in buttons:
                try:
                    # Check for mat-icon inside button
                    icons = btn.find_elements(By.TAG_NAME, "mat-icon")
                    for icon in icons:
                        icon_text = icon.text.lower()
                        if any(x in icon_text for x in ['table', 'list', 'grid', 'view_list', 'table_chart', 'format_list']):
                            print(f"  Clicking table button (icon: {icon_text})")
                            ActionChains(driver).move_to_element(btn).click().perform()
                            table_clicked = True
                            time.sleep(3)
                            break
                except:
                    pass
                if table_clicked:
                    break
        except Exception as e:
            print(f"  Method 1 failed: {e}")

        # Method 2: Try clicking any button that might show table
        if not table_clicked:
            try:
                # Look for buttons with tooltip/title containing tabla/table
                buttons = driver.find_elements(By.CSS_SELECTOR, "button[mattooltip], button[title]")
                for btn in buttons:
                    tooltip = btn.get_attribute('mattooltip') or btn.get_attribute('title') or ''
                    if 'tabla' in tooltip.lower() or 'table' in tooltip.lower():
                        print(f"  Clicking table button (tooltip: {tooltip})")
                        btn.click()
                        table_clicked = True
                        time.sleep(3)
                        break
            except Exception as e:
                print(f"  Method 2 failed: {e}")

        # Wait for table to render
        time.sleep(5)

        # Find and extract table data
        tables = driver.find_elements(By.TAG_NAME, "table")
        print(f"  Found {len(tables)} tables")

        # Find the data table (largest one)
        data_table = None
        max_rows = 0
        for table in tables:
            rows = table.find_elements(By.TAG_NAME, "tr")
            if len(rows) > max_rows:
                max_rows = len(rows)
                data_table = table

        if data_table and max_rows > 5:
            print(f"  Processing table with {max_rows} rows")
            rows = data_table.find_elements(By.TAG_NAME, "tr")

            for row in rows[1:]:  # Skip header
                try:
                    # SUAMECA structure: <th>date</th><td>value</td>
                    th_elements = row.find_elements(By.TAG_NAME, "th")
                    td_elements = row.find_elements(By.TAG_NAME, "td")

                    if not th_elements or not td_elements:
                        continue

                    fecha_str = th_elements[0].text.strip()
                    valor_str = td_elements[0].text.strip()

                    # Clean value
                    valor_str = valor_str.replace(',', '.').replace('%', '').replace(' ', '').replace('\n', '')

                    if not fecha_str or not valor_str:
                        continue

                    try:
                        valor = float(valor_str)
                    except:
                        continue

                    # Parse date (YYYY/MM/DD format)
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

            if data:
                data = sorted(data, key=lambda x: x['fecha'], reverse=True)
                print(f"  Extracted {len(data)} records")
                print(f"  Latest: {data[0]['fecha']} = {data[0]['valor']}")
                print(f"  Oldest: {data[-1]['fecha']} = {data[-1]['valor']}")
        else:
            print(f"  No suitable table found")
            # Save screenshot for debugging
            try:
                screenshot_path = f"/tmp/suameca_{name}_debug.png"
                driver.save_screenshot(screenshot_path)
                print(f"  Screenshot saved: {screenshot_path}")
            except:
                pass

    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()

    return data


def update_csv(results):
    """Update MACRO_MONTHLY_MASTER.csv with new data."""
    csv_path = OUTPUT_DIR / "MACRO_MONTHLY_MASTER.csv"

    print(f"\n[UPDATE] Loading {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)

    # Separate audit row
    audit_row = df.loc["FUENTE_URL"] if "FUENTE_URL" in df.index else None
    df_data = df.drop("FUENTE_URL", errors="ignore")
    df_data.index = pd.to_datetime(df_data.index)

    updates = 0

    for name, data in results.items():
        if not data:
            continue

        col_name = SERIES[name]['col']
        print(f"\n  [{name}] Updating {col_name}")

        for item in data:
            fecha = item['fecha']
            valor = item['valor']

            # Convert to end of month
            fecha_dt = pd.to_datetime(fecha)
            fecha_eom = fecha_dt + pd.offsets.MonthEnd(0)
            fecha_eom = fecha_eom.normalize()

            # Update or create row
            if fecha_eom in df_data.index:
                old_val = df_data.loc[fecha_eom, col_name]
                if pd.isna(old_val) or abs(old_val - valor) > 0.001:
                    df_data.loc[fecha_eom, col_name] = valor
                    updates += 1
            else:
                df_data.loc[fecha_eom, col_name] = valor
                updates += 1

    print(f"\n  Total updates: {updates}")

    if updates > 0:
        # Sort and save
        df_data = df_data.sort_index()

        if audit_row is not None:
            df_final = pd.concat([pd.DataFrame([audit_row], index=["FUENTE_URL"]), df_data])
        else:
            df_final = df_data

        df_final.to_csv(csv_path)
        print(f"[SAVED] {csv_path}")

        # Update downstream files
        for path in [
            PROJECT_ROOT / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_MONTHLY_CLEAN.csv",
            PROJECT_ROOT / "data" / "pipeline" / "03_fusion" / "output" / "DATASET_MACRO_MONTHLY.csv",
        ]:
            if path.parent.exists():
                df_final.to_csv(path)
                print(f"[SAVED] {path}")

    return updates


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true', default=True,
                       help='Run in headless mode (default: True)')
    parser.add_argument('--no-headless', action='store_false', dest='headless',
                       help='Run with visible browser window')
    parser.add_argument('--series', nargs='+', choices=list(SERIES.keys()) + ['all'],
                       default=['all'], help='Series to scrape')
    args = parser.parse_args()

    print("=" * 70)
    print("SUAMECA SCRAPER - IPCCOL, ITCR, TOT")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Headless: {args.headless}")
    print("=" * 70)

    # Determine which series to scrape
    if 'all' in args.series:
        series_to_scrape = list(SERIES.keys())
    else:
        series_to_scrape = args.series

    # Setup driver
    driver = setup_driver(headless=args.headless)
    if driver is None:
        print("[ERROR] Could not setup Chrome driver")
        sys.exit(1)

    results = {}

    try:
        for name in series_to_scrape:
            config = SERIES[name]
            data = scrape_serie(driver, name, config)
            results[name] = data
            time.sleep(5)  # Rate limiting
    finally:
        print("\n[CLOSE] Closing browser...")
        driver.quit()

    # Update CSV
    if any(results.values()):
        updates = update_csv(results)
        print(f"\n[DONE] Total updates: {updates}")
    else:
        print("\n[WARN] No data extracted from any series")

    # Final status
    print("\n" + "=" * 70)
    print("FINAL STATUS")
    print("=" * 70)

    csv_path = OUTPUT_DIR / "MACRO_MONTHLY_MASTER.csv"
    df = pd.read_csv(csv_path, index_col=0)
    df_data = df.drop("FUENTE_URL", errors="ignore")
    df_data.index = pd.to_datetime(df_data.index)

    for name, config in SERIES.items():
        col = config['col']
        if col in df_data.columns:
            data = df_data[col].dropna()
            if len(data) > 0:
                last_date = data.index.max()
                last_val = data.iloc[-1]
                print(f"  {name}: {last_date.strftime('%Y-%m-%d')} = {last_val:.2f}")


if __name__ == "__main__":
    main()

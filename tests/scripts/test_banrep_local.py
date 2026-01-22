#!/usr/bin/env python3
"""
Test script para BanRep SUAMECA - Ejecutar localmente en Windows
================================================================

Este script prueba la conexión a SUAMECA y la extracción de datos usando Selenium.
Requiere Chrome instalado en tu sistema.

Instalación (en terminal de Windows):
    pip install undetected-chromedriver selenium pandas

Uso:
    python scripts/test_banrep_local.py
"""
import os
import sys
import time

# Test URLs - verified 2025-12-18
TEST_URLS = {
    'IBR': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/241/tasas_interes_indicador_bancario_referencia_ibr',
    'TPM': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/59/tasas_interes_politica_monetaria',
}

def test_with_undetected_chrome():
    """Test using undetected-chromedriver."""
    print("\n[TEST] undetected-chromedriver")
    print("-" * 50)

    try:
        import undetected_chromedriver as uc
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        print("Initializing Chrome driver...")

        options = uc.ChromeOptions()
        options.add_argument('--headless=new')  # Headless mode
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--lang=es-CO')

        driver = uc.Chrome(options=options)
        print("Chrome initialized successfully!")

        for name, url in TEST_URLS.items():
            print(f"\n[{name}] Testing URL...")
            print(f"  URL: {url}")

            try:
                driver.get(url)
                time.sleep(8)  # Wait for page to fully load

                # Check for captcha
                page_source = driver.page_source.lower()
                if 'captcha' in page_source or 'radware' in page_source:
                    print(f"  ⚠️ CAPTCHA detected - waiting for bypass...")
                    time.sleep(15)
                    page_source = driver.page_source.lower()

                    if 'captcha' in page_source:
                        print(f"  ❌ Could not bypass captcha")
                        continue

                print(f"  Page loaded! Looking for data...")

                # Find buttons
                buttons = driver.find_elements(By.TAG_NAME, "button")
                print(f"  Found {len(buttons)} buttons")

                # Look for table/download buttons
                for btn in buttons:
                    btn_text = btn.text.strip()
                    btn_title = btn.get_attribute('title') or ''
                    if btn_text or btn_title:
                        print(f"    Button: '{btn_text}' title='{btn_title}'")

                # Find tables
                tables = driver.find_elements(By.TAG_NAME, "table")
                print(f"  Found {len(tables)} tables")

                for i, table in enumerate(tables):
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    print(f"    Table {i+1}: {len(rows)} rows")

                    # Print first few rows
                    for row in rows[:5]:
                        cols = row.find_elements(By.TAG_NAME, "td")
                        if cols:
                            data = [col.text.strip()[:20] for col in cols[:3]]
                            print(f"      {' | '.join(data)}")

                # Take screenshot for debugging
                screenshot_path = f"banrep_{name}_screenshot.png"
                driver.save_screenshot(screenshot_path)
                print(f"  Screenshot saved: {screenshot_path}")

            except Exception as e:
                print(f"  ❌ Error: {e}")

        driver.quit()
        print("\n✅ Test completed!")

    except ImportError:
        print("❌ undetected-chromedriver not installed")
        print("   Run: pip install undetected-chromedriver")
    except Exception as e:
        print(f"❌ Error: {e}")


def test_with_requests():
    """Test basic connectivity with requests."""
    print("\n[TEST] Basic HTTP Request")
    print("-" * 50)

    try:
        import requests

        for name, url in TEST_URLS.items():
            print(f"\n[{name}]")
            resp = requests.get(url, timeout=20, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            print(f"  HTTP Status: {resp.status_code}")

            if resp.status_code == 200:
                content = resp.text.lower()
                if 'captcha' in content or 'radware' in content:
                    print(f"  ⚠️ Page has CAPTCHA/Radware protection")
                else:
                    print(f"  ✅ Page accessible (no captcha detected)")
                    print(f"  Content length: {len(resp.text)} chars")

    except Exception as e:
        print(f"❌ Error: {e}")


def test_with_cloudscraper():
    """Test with cloudscraper (can bypass some protections)."""
    print("\n[TEST] Cloudscraper")
    print("-" * 50)

    try:
        import cloudscraper
        from bs4 import BeautifulSoup

        scraper = cloudscraper.create_scraper()

        for name, url in TEST_URLS.items():
            print(f"\n[{name}]")
            resp = scraper.get(url, timeout=20)
            print(f"  HTTP Status: {resp.status_code}")

            if resp.status_code == 200:
                content = resp.text.lower()
                if 'captcha' in content or 'radware' in content:
                    print(f"  ⚠️ CAPTCHA/Radware detected")
                else:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    tables = soup.find_all('table')
                    buttons = soup.find_all('button')
                    print(f"  ✅ Page loaded: {len(tables)} tables, {len(buttons)} buttons")

    except ImportError:
        print("❌ cloudscraper not installed")
        print("   Run: pip install cloudscraper beautifulsoup4")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    print("=" * 60)
    print("BANREP SUAMECA - LOCAL TEST")
    print("=" * 60)

    # Test 1: Basic HTTP
    test_with_requests()

    # Test 2: Cloudscraper
    test_with_cloudscraper()

    # Test 3: Selenium (most robust)
    test_with_undetected_chrome()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()

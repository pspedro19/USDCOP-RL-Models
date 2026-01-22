#!/usr/bin/env python3
"""
Explorar series de Inversión Extranjera Directa en BanRep SUAMECA
"""
import subprocess
import json

# Series candidatas para IED (Inversión Extranjera Directa EN Colombia)
# Basado en la estructura del catálogo de BanRep
CANDIDATE_SERIES = [
    # Flujo de inversión extranjera directa
    ("4189", "Flujo neto de IED"),
    ("4191", "IED en Colombia - Total"),
    ("4192", "IED por sector económico"),
    ("4193", "IED por país de origen"),
    ("414002", "IED trimestral - Cuenta Financiera"),
    ("414003", "IED - Inflows"),
    ("4188", "Inversión directa - Pasivos"),
    ("4187", "Inversión directa - Total"),
    ("4186", "Cuenta Financiera - Inversión Directa"),
    # Variantes adicionales
    ("15016", "IED acumulada"),
    ("15017", "IED flujos"),
    ("100010", "Inversión extranjera"),
    ("100011", "IED Colombia"),
]

def test_series(series_id, description):
    """Test if a series exists and has data"""
    url = f"https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/{series_id}"

    cmd = f'''
docker exec -i usdcop-postgres-timescale bash -c "
python3 << 'PYEOF'
import subprocess
import sys
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    import time

    chrome_options = Options()
    chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(options=chrome_options)
    driver.set_page_load_timeout(15)

    url = '{url}'
    print(f'Testing: {series_id} - {description}')
    driver.get(url)
    time.sleep(3)

    # Check if page has data table
    page_source = driver.page_source
    if 'No se encontró' in page_source or 'error' in page_source.lower():
        print('  -> NO DATA')
    elif '<table' in page_source.lower() or 'valor' in page_source.lower():
        # Try to find the series title
        try:
            title = driver.find_element(By.CSS_SELECTOR, 'h1, h2, .title, .serie-nombre').text
            print(f'  -> FOUND: {{title[:80]}}')
        except:
            print('  -> HAS DATA (title not found)')
    else:
        print('  -> UNKNOWN')

    driver.quit()
except Exception as e:
    print(f'  -> ERROR: {{str(e)[:50]}}')
PYEOF
"
'''
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

if __name__ == "__main__":
    print("="*70)
    print("Explorando series de IED en BanRep SUAMECA")
    print("="*70)

    for series_id, desc in CANDIDATE_SERIES:
        output = test_series(series_id, desc)
        print(output)
        print("-"*50)

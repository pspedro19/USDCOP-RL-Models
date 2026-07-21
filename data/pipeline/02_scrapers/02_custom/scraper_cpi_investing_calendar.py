"""
Scraper para CPI MoM desde Economic Calendar de Investing.com
Formato diferente a historical data - usa economic calendar
"""

import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime

# URL del CPI MoM en Economic Calendar
CPI_MOM_URL = 'https://www.investing.com/economic-calendar/cpi-69'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}

def parse_calendar_date(date_str):
    """
    Convierte fecha del calendar (ej: 'Oct 24, 2025  (Sep)') a formato estándar

    Args:
        date_str: Fecha en formato calendar

    Returns:
        Tupla (fecha_formateada, periodo) o (None, None) si falla
    """
    try:
        # Extraer fecha de publicación y periodo
        # Formato: "Oct 24, 2025  (Sep)" -> fecha: Oct 24, 2025, periodo: Sep 2025
        match = re.match(r'([A-Za-z]+\s+\d+,\s+\d{4})\s+\(([A-Za-z]+)\)', date_str)

        if match:
            release_date_str, period_month = match.groups()

            # Parsear fecha de publicación
            release_date = datetime.strptime(release_date_str, '%b %d, %Y')

            # El periodo es el mes anterior al release
            # Por ejemplo, si se publica en Oct 24, 2025, el periodo es Sep 2025
            year = release_date.year
            month_map = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }

            period_month_num = month_map.get(period_month)
            if period_month_num:
                # Formato: YYYY-MM (para el periodo del dato)
                period_date = f"{year}-{period_month_num:02d}"
                release_date_fmt = release_date.strftime('%Y-%m-%d')

                return release_date_fmt, period_date

        return None, None

    except Exception as e:
        print(f"  [ERROR] Parseando fecha '{date_str}': {e}")
        return None, None


def parse_percentage(value_str):
    """
    Convierte string de porcentaje a float

    Args:
        value_str: String como "0.3%" o "3.0%"

    Returns:
        Float del valor (ej: 0.3) o None si falla
    """
    try:
        # Remover '%' y convertir
        clean_value = value_str.strip().replace('%', '')
        return float(clean_value)
    except:
        return None


def get_cpi_mom_last_n(n=5):
    """
    Obtiene los últimos N valores de CPI MoM desde Economic Calendar

    Args:
        n: Número de datos a obtener (default: 5)

    Returns:
        DataFrame con últimos N datos o None si falla
    """
    print(f"\n[CPI MoM] Obteniendo últimos {n} datos desde Economic Calendar...")
    print(f"  URL: {CPI_MOM_URL}")

    scraper = cloudscraper.create_scraper()

    try:
        response = scraper.get(CPI_MOM_URL, headers=HEADERS, timeout=15)

        print(f"  Status: {response.status_code}")

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Buscar tabla del economic calendar
            table = soup.find('table', {'id': 'eventHistoryTable'})

            if not table:
                table = soup.find('table', class_='genTbl')

            if table:
                rows = table.find_all('tr')

                data_rows = []

                for row in rows[1:]:  # Skip header
                    cells = row.find_all('td')

                    if len(cells) >= 5:  # Date, Time, Actual, Forecast, Previous
                        date_str = cells[0].get_text(strip=True)
                        # cells[1] es Time (07:30), lo saltamos
                        actual_str = cells[2].get_text(strip=True)  # Actual está en cells[2]
                        forecast_str = cells[3].get_text(strip=True)
                        previous_str = cells[4].get_text(strip=True)

                        # Parsear fecha
                        release_date, period_date = parse_calendar_date(date_str)

                        if release_date and period_date:
                            # Parsear valores
                            actual = parse_percentage(actual_str)
                            forecast = parse_percentage(forecast_str)
                            previous = parse_percentage(previous_str)

                            data_rows.append({
                                'Period': period_date,  # YYYY-MM del periodo del dato
                                'Release_Date': release_date,  # Fecha de publicación
                                'Actual': actual,  # Valor real publicado
                                'Forecast': forecast,  # Pronóstico
                                'Previous': previous  # Valor anterior
                            })

                if data_rows:
                    # Tomar los últimos N
                    last_n = data_rows[:n] if len(data_rows) >= n else data_rows

                    df = pd.DataFrame(last_n)

                    print(f"  [OK] {len(last_n)} registros extraídos")
                    print(f"\n  Últimos {len(last_n)} valores:")
                    for row in last_n:
                        print(f"    {row['Period']} (pub: {row['Release_Date']}) = {row['Actual']}% MoM")

                    return df
                else:
                    print(f"  [ERROR] No se encontraron datos válidos")
                    return None
            else:
                print(f"  [ERROR] No se encontró tabla de datos")
                return None

        else:
            print(f"  [ERROR] Status {response.status_code}")
            return None

    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {str(e)[:80]}")
        return None


def save_cpi_mom_data(df, filepath=None):
    """Guarda datos de CPI MoM"""
    if df is None or df.empty:
        return False

    if filepath is None:
        from pathlib import Path
        base_path = Path(__file__).parent / 'data' / 'Macros' / 'INFLATION'
        base_path.mkdir(parents=True, exist_ok=True)
        filepath = base_path / 'CPIUSMOM_last5.csv'

    df.to_csv(filepath, index=False)
    print(f"\n  [SAVED] {filepath}")
    return True


# ============================================================================
# TEST
# ============================================================================

def test_cpi_mom():
    """Test del scraper CPI MoM"""
    print("=" * 80)
    print("TEST: CPI MoM USA - INVESTING.COM ECONOMIC CALENDAR")
    print("=" * 80)

    # Test: Obtener últimos 5 datos
    df = get_cpi_mom_last_n(n=5)

    if df is not None and not df.empty:
        print("\n[TEST] ÉXITO - Datos obtenidos")

        # Guardar
        save_cpi_mom_data(df)

        # Mostrar resumen
        print("\n" + "=" * 80)
        print("RESUMEN")
        print("=" * 80)

        print(f"\nÚltimo dato:")
        last_row = df.iloc[0]
        print(f"  Periodo: {last_row['Period']}")
        print(f"  Publicado: {last_row['Release_Date']}")
        print(f"  Actual: {last_row['Actual']}% MoM")
        print(f"  Forecast: {last_row['Forecast']}%")
        print(f"  Previous: {last_row['Previous']}%")

        print("\n\nPróximo paso:")
        print("  Integrar en actualizar_diario_v2.py")
    else:
        print("\n[TEST] FALLO - No se pudieron obtener datos")


if __name__ == "__main__":
    test_cpi_mom()

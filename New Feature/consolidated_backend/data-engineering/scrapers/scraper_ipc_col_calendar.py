"""
Scraper para IPC Colombia desde Economic Calendar de Investing.com
Similar a CPI USA MoM pero para Colombia
"""

import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime

# URL del IPC Colombia en Economic Calendar
IPC_COL_URL = 'https://www.investing.com/economic-calendar/colombian-cpi-1502'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9,es;q=0.8',
}

def parse_calendar_date(date_str):
    """
    Convierte fecha del calendar (ej: 'Oct 16, 2025  (Oct)') a formato estándar

    Args:
        date_str: Fecha en formato calendar

    Returns:
        Tupla (fecha_publicacion, periodo) o (None, None) si falla
    """
    try:
        # Formato: "Oct 16, 2025  (Oct)" -> fecha: Oct 16, 2025, periodo: Oct 2025
        match = re.match(r'([A-Za-z]+\s+\d+,\s+\d{4})\s+\(([A-Za-z]+)\)', date_str)

        if match:
            release_date_str, period_month = match.groups()

            # Parsear fecha de publicación
            release_date = datetime.strptime(release_date_str, '%b %d, %Y')

            # El periodo es el mes del dato
            year = release_date.year
            month_map = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }

            period_month_num = month_map.get(period_month)
            if period_month_num:
                # Formato: YYYY-MM-01 (primer día del mes del periodo)
                period_date = f"{year}-{period_month_num:02d}-01"
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
        value_str: String como "0.18%" o "5.2%"

    Returns:
        Float del valor (ej: 0.18) o None si falla
    """
    try:
        clean_value = value_str.strip().replace('%', '')
        return float(clean_value)
    except:
        return None


def get_ipc_col_last_n(n=5):
    """
    Obtiene los últimos N valores de IPC Colombia desde Economic Calendar

    Args:
        n: Número de datos a obtener (default: 5)

    Returns:
        DataFrame con últimos N datos o None si falla
    """
    print(f"\n[IPC Colombia] Obteniendo últimos {n} datos desde Economic Calendar...")
    print(f"  URL: {IPC_COL_URL}")

    scraper = cloudscraper.create_scraper()

    try:
        response = scraper.get(IPC_COL_URL, headers=HEADERS, timeout=15)

        print(f"  Status: {response.status_code}")

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Buscar tabla de datos históricos
            table = soup.find('table', class_='genTbl')

            if not table:
                print(f"  [ERROR] No se encontró tabla de datos")
                return None

            rows = table.find_all('tr')

            data = []
            for row in rows:
                cols = row.find_all('td')

                # Formato: fecha, hora, actual, forecast, previous, cambio
                if len(cols) >= 6:
                    fecha_str = cols[0].get_text(strip=True)
                    hora_str = cols[1].get_text(strip=True)
                    actual_str = cols[2].get_text(strip=True)  # Columna 2 = Actual

                    # Parsear fecha y periodo
                    release_date, period_date = parse_calendar_date(fecha_str)

                    if release_date and period_date and actual_str:
                        # Parsear valor (puede tener % o no)
                        actual_val = parse_percentage(actual_str)

                        if actual_val is not None:
                            data.append({
                                'Period': period_date,
                                'Release_Date': release_date,
                                'Actual': actual_val,
                                'Original_Date': fecha_str
                            })

            if data:
                # Tomar los últimos N
                last_n = data[:n]

                df = pd.DataFrame(last_n)

                print(f"  [OK] {len(last_n)} registros extraídos")
                print(f"\n  Últimos {len(last_n)} valores:")
                for row in last_n:
                    print(f"    {row['Period'][:7]} (pub: {row['Release_Date']}) = {row['Actual']:.2f}%")

                return df
            else:
                print(f"  [ERROR] No se encontraron datos válidos")
                return None

        else:
            print(f"  [ERROR] Status {response.status_code}")
            return None

    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {str(e)[:80]}")
        return None


def save_ipc_col_data(df, filepath=None):
    """Guarda datos del IPC Colombia"""
    if df is None or df.empty:
        return False

    if filepath is None:
        from pathlib import Path
        base_path = Path(__file__).parent / 'data' / 'Macros' / 'INFLATION'
        base_path.mkdir(parents=True, exist_ok=True)
        filepath = base_path / 'IPC_COL_last5.csv'

    df.to_csv(filepath, index=False)
    print(f"\n  [SAVED] {filepath}")
    return True


# ============================================================================
# TEST
# ============================================================================

def test_ipc_col():
    """Test del scraper IPC Colombia"""
    print("=" * 80)
    print("TEST: IPC COLOMBIA - INVESTING.COM ECONOMIC CALENDAR")
    print("=" * 80)

    df = get_ipc_col_last_n(n=5)

    if df is not None and not df.empty:
        print("\n[TEST] ÉXITO - Últimos 5 datos obtenidos")

        # Guardar
        save_ipc_col_data(df)

        # Resumen
        print("\n" + "=" * 80)
        print("RESUMEN")
        print("=" * 80)
        print(f"\n[ÉXITO] Scraper IPC Colombia funciona correctamente")
        print(f"\nÚltimo dato:")
        print(f"  Periodo: {df['Period'].iloc[0]}")
        print(f"  Valor: {df['Actual'].iloc[0]:.2f}%")
        print(f"  Fecha publicación: {df['Release_Date'].iloc[0]}")

        print("\n\nPróximo paso:")
        print("  Integrar en actualizar_diario_completo.py")
    else:
        print("\n[FALLO] Scraper IPC Colombia no funciona")
        print("\nProblema posible:")
        print("  - Investing.com cambió formato de página")
        print("  - Cloudflare bloqueó")
        print("  - URL incorrecta")


if __name__ == "__main__":
    test_ipc_col()

"""
Scraper para EMBI Colombia desde BCRP Peru
Obtiene el ultimo dato disponible del spread EMBI de Colombia
"""

import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import re

# URL del EMBI Colombia en BCRP Peru
EMBI_URL = 'https://estadisticas.bcrp.gob.pe/estadisticas/series/diarias/resultados/PD04715XD/html/'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9,es;q=0.8',
}

def parse_bcrp_date(date_str):
    """
    Convierte fecha BCRP (ej: '17Nov25') a formato estandar

    Args:
        date_str: Fecha en formato BCRP (ddMMMaa)

    Returns:
        Fecha en formato YYYY-MM-DD o None si falla
    """
    try:
        # Diccionario de meses en español
        meses = {
            'Ene': '01', 'Feb': '02', 'Mar': '03', 'Abr': '04',
            'May': '05', 'Jun': '06', 'Jul': '07', 'Ago': '08',
            'Set': '09', 'Oct': '10', 'Nov': '11', 'Dic': '12'
        }

        # Extraer dia, mes, año (ej: 17Nov25)
        match = re.match(r'(\d{2})([A-Za-z]{3})(\d{2})', date_str)
        if match:
            dia, mes_txt, anio = match.groups()

            mes = meses.get(mes_txt.capitalize())
            if not mes:
                return None

            # Convertir año de 2 digitos a 4
            # Asumimos: 00-49 = 2000-2049, 50-99 = 1950-1999
            anio_int = int(anio)
            if anio_int <= 49:
                anio_full = f"20{anio}"
            else:
                anio_full = f"19{anio}"

            return f"{anio_full}-{mes}-{dia}"

        return None

    except Exception as e:
        print(f"  [ERROR] Parseando fecha '{date_str}': {e}")
        return None


def get_embi_ultimo():
    """
    Obtiene el ultimo valor del EMBI Colombia desde BCRP

    Returns:
        dict con 'date', 'value', 'source' o None si falla
    """
    print("\n[EMBI] Obteniendo ultimo dato desde BCRP Peru...")
    print(f"  URL: {EMBI_URL}")

    scraper = cloudscraper.create_scraper()

    try:
        response = scraper.get(EMBI_URL, headers=HEADERS, timeout=15)

        print(f"  Status: {response.status_code}")

        if response.status_code == 200:
            # Parsear HTML con BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Buscar la tabla de datos
            tables = soup.find_all('table')

            # La tabla principal es la que tiene mas filas
            main_table = None
            max_rows = 0
            for table in tables:
                rows = table.find_all('tr')
                if len(rows) > max_rows:
                    max_rows = len(rows)
                    main_table = table

            if main_table:
                rows = main_table.find_all('tr')

                # La ultima fila tiene el ultimo dato
                # Empezar desde el final y buscar la primera fila valida
                last_date = None
                last_value = None

                for row in reversed(rows):
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        fecha_str = cells[0].get_text(strip=True)
                        valor_str = cells[1].get_text(strip=True)

                        # Verificar que la fecha tiene el formato correcto
                        if re.match(r'\d{2}[A-Za-z]{3}\d{2}', fecha_str) and valor_str.isdigit():
                            last_date = fecha_str
                            last_value = valor_str
                            break

                if last_date and last_value:
                    # Convertir fecha a formato estandar
                    fecha_formateada = parse_bcrp_date(last_date)

                    print(f"  [OK] Ultimo dato encontrado")
                    print(f"    Fecha: {last_date} ({fecha_formateada})")
                    print(f"    Valor: {last_value} pbs")

                    return {
                        'date': fecha_formateada,
                        'date_original': last_date,
                        'value': int(last_value),
                        'source': 'BCRP Peru',
                        'url': EMBI_URL
                    }
                else:
                    print(f"  [ERROR] No se encontraron datos validos en la tabla")
                    return None
            else:
                print(f"  [ERROR] No se encontro tabla de datos")
                return None

        elif response.status_code == 403:
            print(f"  [ERROR] 403 - Acceso denegado")
            return None

        else:
            print(f"  [ERROR] Status code inesperado")
            return None

    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {str(e)[:80]}")
        return None


def get_embi_last_n(n=2):
    """
    Obtiene los ultimos N valores del EMBI Colombia

    Args:
        n: Numero de datos a obtener (default: 2)

    Returns:
        DataFrame con ultimos N datos o None si falla
    """
    print(f"\n[EMBI] Obteniendo ultimos {n} datos desde BCRP Peru...")
    print(f"  URL: {EMBI_URL}")

    scraper = cloudscraper.create_scraper()

    try:
        response = scraper.get(EMBI_URL, headers=HEADERS, timeout=15)

        print(f"  Status: {response.status_code}")

        if response.status_code == 200:
            # Parsear HTML con BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Buscar la tabla de datos
            tables = soup.find_all('table')

            # La tabla principal es la que tiene mas filas
            main_table = None
            max_rows = 0
            for table in tables:
                rows = table.find_all('tr')
                if len(rows) > max_rows:
                    max_rows = len(rows)
                    main_table = table

            if main_table:
                rows = main_table.find_all('tr')

                # Extraer todas las filas validas
                data_rows = []

                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        fecha_str = cells[0].get_text(strip=True)
                        valor_str = cells[1].get_text(strip=True)

                        # Verificar que la fecha tiene el formato correcto
                        if re.match(r'\d{2}[A-Za-z]{3}\d{2}', fecha_str) and valor_str.isdigit():
                            fecha_formateada = parse_bcrp_date(fecha_str)

                            if fecha_formateada:
                                data_rows.append({
                                    'Date': fecha_formateada,
                                    'Date_Original': fecha_str,
                                    'Value': int(valor_str)
                                })

                if data_rows:
                    # Tomar los ultimos N
                    last_n = data_rows[-n:] if len(data_rows) >= n else data_rows

                    df = pd.DataFrame(last_n)

                    print(f"  [OK] {len(last_n)} registros extraidos")
                    print(f"\n  Ultimos {len(last_n)} valores:")
                    for row in last_n:
                        print(f"    {row['Date_Original']:10} ({row['Date']}) = {row['Value']:4} pbs")

                    return df
                else:
                    print(f"  [ERROR] No se encontraron datos validos")
                    return None
            else:
                print(f"  [ERROR] No se encontro tabla de datos")
                return None

        else:
            print(f"  [ERROR] Status {response.status_code}")
            return None

    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {str(e)[:80]}")
        return None


def save_embi_data(df, filepath=None):
    """Guarda datos del EMBI"""
    if df is None or df.empty:
        return False

    if filepath is None:
        from pathlib import Path
        base_path = Path(__file__).parent / 'data' / 'Macros' / 'COUNTRY_RISK'
        base_path.mkdir(parents=True, exist_ok=True)
        filepath = base_path / 'EMBI_last2.csv'

    df.to_csv(filepath, index=False)
    print(f"\n  [SAVED] {filepath}")
    return True


# ============================================================================
# TEST
# ============================================================================

def test_embi():
    """Test del scraper EMBI"""
    print("=" * 80)
    print("TEST: EMBI COLOMBIA - BCRP PERU")
    print("=" * 80)

    # Test 1: Obtener ultimo dato
    ultimo = get_embi_ultimo()

    if ultimo:
        print("\n[TEST 1] EXITO - Ultimo dato obtenido")
    else:
        print("\n[TEST 1] FALLO - No se pudo obtener ultimo dato")

    # Test 2: Obtener ultimos 2 datos
    print("\n" + "-" * 80)
    df = get_embi_last_n(n=2)

    if df is not None and not df.empty:
        print("\n[TEST 2] EXITO - Ultimos 2 datos obtenidos")

        # Guardar
        save_embi_data(df)
    else:
        print("\n[TEST 2] FALLO - No se pudieron obtener ultimos 2 datos")

    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN")
    print("=" * 80)

    if ultimo and df is not None:
        print("\n[EXITO] Scraper EMBI funciona correctamente")
        print(f"\nUltimo dato:")
        print(f"  Fecha: {ultimo['date']}")
        print(f"  Valor: {ultimo['value']} puntos basicos")
        print(f"  Fuente: {ultimo['source']}")

        print("\n\nProximo paso:")
        print("  Integrar en actualizar_diario_v2.py")
    else:
        print("\n[FALLO] Scraper EMBI no funciona")
        print("\nProblema posible:")
        print("  - BCRP cambio formato de pagina")
        print("  - Cloudflare bloqueo")
        print("  - URL incorrecta")


if __name__ == "__main__":
    test_embi()

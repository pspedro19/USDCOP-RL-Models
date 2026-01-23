# -*- coding: utf-8 -*-
"""
Scraper genérico para Investing.com - Commodities y Forex
==========================================================

Extrae datos históricos de Investing.com usando cloudscraper para evitar anti-bot.
"""

import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# Mapeo de URLs de Investing.com
INVESTING_URLS = {
    # Commodities
    'WTI': 'https://www.investing.com/commodities/crude-oil-historical-data',
    'BRENT': 'https://www.investing.com/commodities/brent-oil-historical-data',
    'COAL': 'https://www.investing.com/commodities/newcastle-coal-futures-historical-data',
    'GOLD': 'https://www.investing.com/commodities/gold-historical-data',
    'COFFEE': 'https://www.investing.com/commodities/us-coffee-c-historical-data',

    # Forex - USDCOP is PRIMARY for Forecasting pipeline
    'USDCOP': 'https://www.investing.com/currencies/usd-cop-historical-data',  # OFFICIAL source
    'USDCLP': 'https://www.investing.com/currencies/usd-clp-historical-data',
    'USDMXN': 'https://www.investing.com/currencies/usd-mxn-historical-data',

    # Indices
    'DXY': 'https://www.investing.com/indices/usdollar-historical-data',
    'COLCAP': 'https://www.investing.com/indices/colcap-historical-data',

    # Bonos
    'UST10Y': 'https://www.investing.com/rates-bonds/u.s.-10-year-bond-yield-historical-data',
}


def obtener_investing_com(url: str, n: int = 20) -> Optional[pd.DataFrame]:
    """
    Obtener datos históricos de Investing.com

    Args:
        url: URL de la página de datos históricos
        n: Número de registros más recientes

    Returns:
        DataFrame con columnas [fecha, valor] ordenado descendente
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }

        # Usar cloudscraper para evitar anti-bot
        scraper = cloudscraper.create_scraper()
        response = scraper.get(url, headers=headers, timeout=15)

        if response.status_code != 200:
            logger.debug(f"Error HTTP {response.status_code} para {url}")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Buscar tabla de datos históricos
        table = soup.find('table', class_='freeze-column-w-1')
        if not table:
            table = soup.find('table', {'data-test': 'historical-data-table'})
        if not table:
            # Buscar cualquier tabla grande
            tables = soup.find_all('table')
            if tables:
                table = max(tables, key=lambda t: len(str(t)))

        if not table:
            logger.debug("No se encontró tabla de datos")
            return None

        # Extraer filas
        rows = table.find_all('tr')[1:]  # Saltar encabezado

        data = []
        for row in rows[:n*2]:  # Pedir más por si hay filas vacías
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

        if not data:
            logger.debug("No se pudieron extraer datos de la tabla")
            return None

        # Crear DataFrame
        df = pd.DataFrame(data)
        df = df.sort_values('fecha', ascending=False).reset_index(drop=True)

        # Retornar últimos N
        return df.head(n)

    except Exception as e:
        logger.debug(f"Error obteniendo datos de Investing.com: {str(e)}")
        return None


# Funciones específicas para cada variable
def obtener_wti(n: int = 20) -> Optional[pd.DataFrame]:
    """Obtener precio WTI Oil"""
    return obtener_investing_com(INVESTING_URLS['WTI'], n)


def obtener_brent(n: int = 20) -> Optional[pd.DataFrame]:
    """Obtener precio Brent Oil"""
    return obtener_investing_com(INVESTING_URLS['BRENT'], n)


def obtener_coal(n: int = 20) -> Optional[pd.DataFrame]:
    """Obtener precio Coal"""
    return obtener_investing_com(INVESTING_URLS['COAL'], n)


def obtener_gold(n: int = 20) -> Optional[pd.DataFrame]:
    """Obtener precio Gold"""
    return obtener_investing_com(INVESTING_URLS['GOLD'], n)


def obtener_coffee(n: int = 20) -> Optional[pd.DataFrame]:
    """Obtener precio Coffee"""
    return obtener_investing_com(INVESTING_URLS['COFFEE'], n)


def obtener_usdclp(n: int = 20) -> Optional[pd.DataFrame]:
    """Obtener tipo de cambio USD/CLP"""
    return obtener_investing_com(INVESTING_URLS['USDCLP'], n)


def obtener_usdmxn(n: int = 20) -> Optional[pd.DataFrame]:
    """Obtener tipo de cambio USD/MXN"""
    return obtener_investing_com(INVESTING_URLS['USDMXN'], n)


def obtener_dxy(n: int = 20) -> Optional[pd.DataFrame]:
    """Obtener índice DXY (Dollar Index)"""
    return obtener_investing_com(INVESTING_URLS['DXY'], n)


def obtener_ust10y(n: int = 20) -> Optional[pd.DataFrame]:
    """Obtener rendimiento bono USA 10Y"""
    return obtener_investing_com(INVESTING_URLS['UST10Y'], n)


def obtener_usdcop(n: int = 20) -> Optional[pd.DataFrame]:
    """
    Obtener tipo de cambio USD/COP (OFICIAL para Forecasting).

    IMPORTANTE: Para datos OHLCV completos, usar USDCOPInvestingScraper
    de scraper_usdcop_investing.py que retorna open, high, low, close.

    Esta función solo retorna fecha y precio de cierre.
    """
    return obtener_investing_com(INVESTING_URLS['USDCOP'], n)


# Test
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("\n=== TEST: WTI ===")
    df = obtener_wti(5)
    if df is not None:
        print(df)
    else:
        print("FAILED")

    print("\n=== TEST: DXY ===")
    df = obtener_dxy(5)
    if df is not None:
        print(df)
    else:
        print("FAILED")

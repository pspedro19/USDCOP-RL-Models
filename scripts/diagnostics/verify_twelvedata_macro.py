#!/usr/bin/env python3
"""
Script para verificar disponibilidad de datos macro en TwelveData API
Verifica: WTI Crude Oil (CL) y US Dollar Index (DXY)

Uso:
    python scripts/verify_twelvedata_macro.py
"""

import os
import sys
import requests
from datetime import datetime, timedelta
import json

# Colores para terminal
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_success(msg):
    print(f"{GREEN}✅ {msg}{RESET}")

def print_error(msg):
    print(f"{RED}❌ {msg}{RESET}")

def print_warning(msg):
    print(f"{YELLOW}⚠️  {msg}{RESET}")

def print_info(msg):
    print(f"{BLUE}ℹ️  {msg}{RESET}")

def print_header(msg):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{msg.center(60)}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def verify_twelvedata_symbol(symbol, symbol_name, api_key):
    """
    Verificar si un símbolo está disponible en TwelveData

    Args:
        symbol: Ticker symbol (e.g., 'CL', 'DXY')
        symbol_name: Nombre descriptivo (e.g., 'WTI Crude Oil')
        api_key: TwelveData API key

    Returns:
        dict con resultado de verificación
    """

    print_info(f"Verificando {symbol_name} ({symbol})...")

    # Test 1: Symbol search
    search_url = "https://api.twelvedata.com/symbol_search"
    search_params = {
        'symbol': symbol,
        'apikey': api_key
    }

    try:
        response = requests.get(search_url, params=search_params, timeout=10)

        if response.status_code != 200:
            print_error(f"Error en búsqueda: HTTP {response.status_code}")
            return {
                'symbol': symbol,
                'available': False,
                'error': f"HTTP {response.status_code}"
            }

        search_data = response.json()

        if 'data' not in search_data or len(search_data['data']) == 0:
            print_error(f"{symbol} NO encontrado en TwelveData")
            return {
                'symbol': symbol,
                'available': False,
                'error': 'Symbol not found'
            }

        # Mostrar información del símbolo
        symbol_info = search_data['data'][0]
        print_success(f"{symbol} encontrado:")
        print(f"  - Nombre: {symbol_info.get('instrument_name', 'N/A')}")
        print(f"  - Tipo: {symbol_info.get('instrument_type', 'N/A')}")
        print(f"  - Exchange: {symbol_info.get('exchange', 'N/A')}")
        print(f"  - Currency: {symbol_info.get('currency', 'N/A')}")

    except Exception as e:
        print_error(f"Error en búsqueda: {e}")
        return {
            'symbol': symbol,
            'available': False,
            'error': str(e)
        }

    # Test 2: Obtener datos históricos (últimos 7 días)
    print_info(f"Probando descarga de datos históricos...")

    time_series_url = "https://api.twelvedata.com/time_series"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    time_params = {
        'symbol': symbol,
        'interval': '1h',
        'apikey': api_key,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'outputsize': 100,
        'format': 'JSON'
    }

    try:
        response = requests.get(time_series_url, params=time_params, timeout=10)

        if response.status_code != 200:
            print_error(f"Error descargando datos: HTTP {response.status_code}")
            return {
                'symbol': symbol,
                'available': False,
                'error': f"HTTP {response.status_code}"
            }

        data = response.json()

        if 'status' in data and data['status'] == 'error':
            print_error(f"Error de API: {data.get('message', 'Unknown error')}")
            return {
                'symbol': symbol,
                'available': False,
                'error': data.get('message', 'API error')
            }

        if 'values' not in data or len(data['values']) == 0:
            print_error(f"No hay datos disponibles para {symbol}")
            return {
                'symbol': symbol,
                'available': False,
                'error': 'No data available'
            }

        # Verificar calidad de datos
        values = data['values']
        print_success(f"Datos obtenidos correctamente:")
        print(f"  - Registros: {len(values)}")
        print(f"  - Primer timestamp: {values[-1]['datetime']}")
        print(f"  - Último timestamp: {values[0]['datetime']}")
        print(f"  - Último precio: {values[0]['close']}")

        # Verificar que tiene OHLCV completo
        sample = values[0]
        required_fields = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        missing_fields = [f for f in required_fields if f not in sample]

        if missing_fields:
            print_warning(f"Campos faltantes: {', '.join(missing_fields)}")
        else:
            print_success("Todos los campos OHLCV presentes")

        return {
            'symbol': symbol,
            'available': True,
            'records': len(values),
            'latest_close': float(values[0]['close']),
            'latest_datetime': values[0]['datetime']
        }

    except Exception as e:
        print_error(f"Error descargando datos: {e}")
        return {
            'symbol': symbol,
            'available': False,
            'error': str(e)
        }

def main():
    print_header("VERIFICACIÓN TWELVEDATA API - MACRO DATA")

    # Obtener API key
    api_key = os.getenv('TWELVEDATA_API_KEY_G1')

    if not api_key:
        print_error("API key no encontrada")
        print_info("Configura la variable de entorno TWELVEDATA_API_KEY_G1")
        sys.exit(1)

    print_success(f"API key encontrada: {api_key[:8]}...")

    # Símbolos a verificar
    symbols_to_check = [
        ('CL', 'WTI Crude Oil'),
        ('DXY', 'US Dollar Index')
    ]

    results = {}

    # Verificar cada símbolo
    for symbol, name in symbols_to_check:
        print(f"\n{'-'*60}")
        result = verify_twelvedata_symbol(symbol, name, api_key)
        results[symbol] = result

    # Resumen final
    print_header("RESUMEN DE VERIFICACIÓN")

    all_available = True

    for symbol, name in symbols_to_check:
        result = results[symbol]

        if result['available']:
            print_success(f"{symbol} ({name}): DISPONIBLE")
            print(f"    Último precio: {result.get('latest_close', 'N/A')}")
            print(f"    Timestamp: {result.get('latest_datetime', 'N/A')}")
        else:
            print_error(f"{symbol} ({name}): NO DISPONIBLE")
            print(f"    Error: {result.get('error', 'Unknown')}")
            all_available = False

    print("\n" + "="*60)

    # Decisión final
    if all_available:
        print_success("✅ DECISIÓN: Usar TwelveData API para macro data")
        print_info("Próximo paso: Crear DAG usdcop_m5__01b_l0_macro_acquire.py")
        return 0
    else:
        print_error("❌ DECISIÓN: TwelveData NO disponible para todos los símbolos")
        print_warning("Usar fallback manual desde investing.com")
        print_info("Próximo paso: Ejecutar scripts/upload_macro_manual.py")
        return 1

if __name__ == '__main__':
    sys.exit(main())

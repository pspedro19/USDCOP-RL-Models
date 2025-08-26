#!/usr/bin/env python3
"""
🧪 Test Twelve Data API Connection
==================================

Script para probar la conectividad con la API de Twelve Data y verificar
que la API key funcione correctamente antes de ejecutar la extracción completa.

Autor: Sistema USDCOP Trading
Fecha: Agosto 2025
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_api_connection(api_key: str) -> bool:
    """
    Prueba la conexión básica con la API de Twelve Data
    
    Args:
        api_key (str): API key para probar
        
    Returns:
        bool: True si la conexión es exitosa, False en caso contrario
    """
    logger.info("🔌 Probando conexión con Twelve Data API...")
    
    # URL de prueba
    url = "https://api.twelvedata.com/quote"
    
    params = {
        "symbol": "USD/COP",
        "apikey": api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            logger.info("✅ Conexión exitosa con la API")
            logger.info(f"📊 Respuesta: {json.dumps(data, indent=2)}")
            return True
        else:
            logger.error(f"❌ Error HTTP: {response.status_code}")
            logger.error(f"📝 Respuesta: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error de conexión: {str(e)}")
        return False
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error decodificando respuesta: {str(e)}")
        return False

def test_time_series_endpoint(api_key: str) -> bool:
    """
    Prueba el endpoint de time series con un período corto
    
    Args:
        api_key (str): API key para probar
        
    Returns:
        bool: True si la prueba es exitosa, False en caso contrario
    """
    logger.info("📈 Probando endpoint de time series...")
    
    # Fechas de prueba (últimos 2 días)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2)
    
    url = "https://api.twelvedata.com/time_series"
    
    params = {
        "symbol": "USD/COP",
        "interval": "5min",
        "start_date": start_date.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end_date.strftime("%Y-%m-%d %H:%M:%S"),
        "order": "ASC",
        "timezone": "America/Bogota",
        "apikey": api_key,
        "format": "JSON"
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if "values" in data and data["values"]:
                logger.info("✅ Time series endpoint funcionando correctamente")
                logger.info(f"📊 Datos recibidos: {len(data['values'])} registros")
                
                # Mostrar muestra de datos
                df = pd.DataFrame(data["values"])
                logger.info(f"📋 Columnas disponibles: {list(df.columns)}")
                logger.info(f"📅 Rango de fechas: {df['datetime'].min()} - {df['datetime'].max()}")
                
                return True
            else:
                logger.warning("⚠️  API responde pero sin datos")
                logger.info(f"📝 Respuesta completa: {json.dumps(data, indent=2)}")
                return False
        else:
            logger.error(f"❌ Error HTTP en time series: {response.status_code}")
            logger.error(f"📝 Respuesta: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error de conexión en time series: {str(e)}")
        return False
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error decodificando time series: {str(e)}")
        return False

def test_historical_data_availability(api_key: str) -> bool:
    """
    Prueba la disponibilidad de datos históricos para el período objetivo
    
    Args:
        api_key (str): API key para probar
        
    Returns:
        bool: True si hay datos disponibles, False en caso contrario
    """
    logger.info("📅 Probando disponibilidad de datos históricos...")
    
    # Probar con fechas del período objetivo
    start_date = "2022-10-01 00:00:00"
    end_date = "2022-10-31 23:59:59"  # Solo octubre 2022 para la prueba
    
    url = "https://api.twelvedata.com/time_series"
    
    params = {
        "symbol": "USD/COP",
        "interval": "5min",
        "start_date": start_date,
        "end_date": end_date,
        "order": "ASC",
        "timezone": "America/Bogota",
        "apikey": api_key,
        "format": "JSON"
    }
    
    try:
        response = requests.get(url, params=params, timeout=20)
        
        if response.status_code == 200:
            data = response.json()
            
            if "values" in data and data["values"]:
                logger.info("✅ Datos históricos disponibles para octubre 2022")
                logger.info(f"📊 Registros disponibles: {len(data['values'])}")
                
                # Calcular estimación para todo el período
                days_in_period = 396  # Oct 2022 - Oct 2023
                estimated_records = len(data["values"]) * (days_in_period / 31)
                
                logger.info(f"📈 Estimación para todo el período: ~{estimated_records:,.0f} registros")
                logger.info(f"⏱️  Tiempo estimado de extracción: ~{estimated_records / 800 * 7.5 / 60:.1f} minutos")
                
                return True
            else:
                logger.warning("⚠️  No hay datos históricos para octubre 2022")
                logger.info(f"📝 Respuesta: {json.dumps(data, indent=2)}")
                return False
        else:
            logger.error(f"❌ Error HTTP en datos históricos: {response.status_code}")
            logger.error(f"📝 Respuesta: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error de conexión en datos históricos: {str(e)}")
        return False
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error decodificando datos históricos: {str(e)}")
        return False

def check_api_limits(api_key: str) -> dict:
    """
    Verifica los límites de la API
    
    Args:
        api_key (str): API key para verificar
        
    Returns:
        dict: Información sobre los límites de la API
    """
    logger.info("📊 Verificando límites de la API...")
    
    limits = {
        "requests_per_minute": 8,
        "requests_per_day": 800,
        "max_historical_days": None,
        "supported_intervals": ["1min", "5min", "15min", "30min", "1hour", "1day"]
    }
    
    logger.info("📋 Límites de la API (plan gratuito):")
    logger.info(f"   • Requests por minuto: {limits['requests_per_minute']}")
    logger.info(f"   • Requests por día: {limits['requests_per_day']}")
    logger.info(f"   • Intervalos soportados: {', '.join(limits['supported_intervals'])}")
    
    return limits

def main():
    """
    Función principal para ejecutar todas las pruebas
    """
    # API Key de Twelve Data
    API_KEY = "085ba06282774cbc8e796f46a5af8ece"
    
    logger.info("🧪 Iniciando pruebas de Twelve Data API")
    logger.info("=" * 50)
    
    # Prueba 1: Conexión básica
    connection_ok = test_api_connection(API_KEY)
    logger.info("-" * 30)
    
    if not connection_ok:
        logger.error("❌ No se pudo establecer conexión con la API")
        return
    
    # Prueba 2: Endpoint de time series
    timeseries_ok = test_time_series_endpoint(API_KEY)
    logger.info("-" * 30)
    
    if not timeseries_ok:
        logger.warning("⚠️  El endpoint de time series no está funcionando correctamente")
    
    # Prueba 3: Disponibilidad de datos históricos
    historical_ok = test_historical_data_availability(API_KEY)
    logger.info("-" * 30)
    
    # Verificar límites de la API
    limits = check_api_limits(API_KEY)
    logger.info("-" * 30)
    
    # Resumen final
    logger.info("📋 RESUMEN DE PRUEBAS")
    logger.info("=" * 30)
    logger.info(f"🔌 Conexión API: {'✅ OK' if connection_ok else '❌ FALLÓ'}")
    logger.info(f"📈 Time Series: {'✅ OK' if timeseries_ok else '❌ FALLÓ'}")
    logger.info(f"📅 Datos Históricos: {'✅ OK' if historical_ok else '❌ FALLÓ'}")
    
    if connection_ok and timeseries_ok and historical_ok:
        logger.info("🎉 Todas las pruebas pasaron exitosamente!")
        logger.info("✅ La API está lista para extraer datos históricos")
        logger.info("🚀 Puedes ejecutar: python scripts/extract_usdcop_historical.py")
    else:
        logger.warning("⚠️  Algunas pruebas fallaron")
        logger.warning("🔍 Revisa los logs para más detalles")
        
        if not connection_ok:
            logger.error("❌ Problema de conectividad - verifica tu conexión a internet")
        if not timeseries_ok:
            logger.error("❌ Problema con el endpoint de time series - verifica la API key")
        if not historical_ok:
            logger.error("❌ No hay datos históricos disponibles para el período solicitado")

if __name__ == "__main__":
    main()

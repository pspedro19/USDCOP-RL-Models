#!/usr/bin/env python3
"""
📊 USDCOP Historical Data Extractor
====================================

Script para extraer datos históricos de USD/COP desde octubre 2022 hasta octubre 2023
usando la API de Twelve Data con intervalos de 5 minutos.

Autor: Sistema USDCOP Trading
Fecha: Agosto 2025
"""

import requests
import pandas as pd
import time
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/usdcop_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class USDCOPDataExtractor:
    """
    Extractor de datos históricos de USD/COP usando Twelve Data API
    """
    
    def __init__(self, api_key: str):
        """
        Inicializa el extractor con la API key
        
        Args:
            api_key (str): API key de Twelve Data
        """
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com/time_series"
        self.symbol = "USD/COP"
        self.interval = "5min"
        self.timezone = "America/Bogota"
        
        # Crear directorio de logs si no existe
        os.makedirs('logs', exist_ok=True)
        
        # Crear directorio de datos si no existe
        os.makedirs('data/raw/twelve_data', exist_ok=True)
        
        logger.info(f"🚀 Inicializando USDCOP Data Extractor")
        logger.info(f"📊 Símbolo: {self.symbol}")
        logger.info(f"⏱️  Intervalo: {self.interval}")
        logger.info(f"🌍 Zona horaria: {self.timezone}")
    
    def extract_historical_data(self, 
                               start_date: str = "2022-10-01 00:00:00",
                               end_date: str = "2023-10-31 23:59:59") -> pd.DataFrame:
        """
        Extrae datos históricos completos de USD/COP
        
        Args:
            start_date (str): Fecha de inicio en formato YYYY-MM-DD HH:MM:SS
            end_date (str): Fecha de fin en formato YYYY-MM-DD HH:MM:SS
            
        Returns:
            pd.DataFrame: DataFrame con datos históricos
        """
        logger.info(f"📅 Extrayendo datos desde {start_date} hasta {end_date}")
        
        # Convertir fechas a datetime
        start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
        
        # Dividir en períodos de 30 días para evitar límites de API
        periods = self._create_date_periods(start_dt, end_dt, days=30)
        
        all_data = []
        total_periods = len(periods)
        
        for i, (period_start, period_end) in enumerate(periods, 1):
            logger.info(f"📊 Procesando período {i}/{total_periods}: {period_start} - {period_end}")
            
            try:
                period_data = self._extract_period_data(period_start, period_end)
                if period_data is not None and not period_data.empty:
                    all_data.append(period_data)
                    logger.info(f"✅ Período {i} extraído: {len(period_data)} registros")
                else:
                    logger.warning(f"⚠️  Período {i} sin datos")
                
                # Pausa para respetar límites de API (8 requests/min)
                time.sleep(7.5)  # 7.5 segundos entre períodos
                
            except Exception as e:
                logger.error(f"❌ Error en período {i}: {str(e)}")
                continue
        
        if not all_data:
            logger.error("❌ No se pudieron extraer datos de ningún período")
            return pd.DataFrame()
        
        # Combinar todos los datos
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Limpiar y ordenar datos
        final_df = self._clean_and_process_data(combined_df)
        
        logger.info(f"🎉 Extracción completada: {len(final_df)} registros totales")
        return final_df
    
    def _create_date_periods(self, start_dt: datetime, end_dt: datetime, days: int = 30) -> List[tuple]:
        """
        Crea períodos de fechas para dividir la extracción
        
        Args:
            start_dt (datetime): Fecha de inicio
            end_dt (datetime): Fecha de fin
            days (int): Días por período
            
        Returns:
            List[tuple]: Lista de tuplas (inicio, fin) para cada período
        """
        periods = []
        current_start = start_dt
        
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=days), end_dt)
            periods.append((current_start, current_end))
            current_start = current_end
        
        return periods
    
    def _extract_period_data(self, start_dt: datetime, end_dt: datetime) -> Optional[pd.DataFrame]:
        """
        Extrae datos para un período específico
        
        Args:
            start_dt (datetime): Inicio del período
            end_dt (datetime): Fin del período
            
        Returns:
            Optional[pd.DataFrame]: DataFrame con datos del período o None si hay error
        """
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "start_date": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "order": "ASC",
            "timezone": self.timezone,
            "apikey": self.api_key,
            "format": "JSON"
        }
        
        try:
            logger.info(f"🌐 Solicitando datos: {start_dt} - {end_dt}")
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if "values" not in data or not data["values"]:
                logger.warning(f"⚠️  No hay datos para el período: {start_dt} - {end_dt}")
                return None
            
            # Crear DataFrame
            df = pd.DataFrame(data["values"])
            
            # Convertir columnas numéricas
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convertir datetime
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error de red: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error decodificando JSON: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"❌ Error inesperado: {str(e)}")
            return None
    
    def _clean_and_process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y procesa los datos extraídos
        
        Args:
            df (pd.DataFrame): DataFrame con datos crudos
            
        Returns:
            pd.DataFrame: DataFrame limpio y procesado
        """
        logger.info("🧹 Limpiando y procesando datos...")
        
        # Eliminar duplicados
        initial_rows = len(df)
        df = df.drop_duplicates(subset=['datetime'], keep='first')
        logger.info(f"🔄 Duplicados eliminados: {initial_rows - len(df)}")
        
        # Ordenar por datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Renombrar columnas para consistencia
        column_mapping = {
            'datetime': 'time',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Seleccionar solo columnas necesarias
        required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_columns]
        
        # Convertir time a string para consistencia
        df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Eliminar filas con valores nulos
        df = df.dropna()
        
        logger.info(f"✅ Datos limpios: {len(df)} registros finales")
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Guarda los datos extraídos en archivo CSV
        
        Args:
            df (pd.DataFrame): DataFrame con datos
            filename (str): Nombre del archivo (opcional)
            
        Returns:
            str: Ruta del archivo guardado
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"USDCOP_5min_{timestamp}.csv"
        
        filepath = f"data/raw/twelve_data/{filename}"
        
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"💾 Datos guardados en: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"❌ Error guardando datos: {str(e)}")
            raise
    
    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """
        Obtiene información estadística de los datos
        
        Args:
            df (pd.DataFrame): DataFrame con datos
            
        Returns:
            Dict: Diccionario con información estadística
        """
        if df.empty:
            return {}
        
        info = {
            "total_records": len(df),
            "date_range": {
                "start": df['time'].min(),
                "end": df['time'].max()
            },
            "columns": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "price_stats": {
                "open": {
                    "min": df['open'].min(),
                    "max": df['open'].max(),
                    "mean": df['open'].mean(),
                    "std": df['open'].std()
                },
                "close": {
                    "min": df['close'].min(),
                    "max": df['close'].max(),
                    "mean": df['close'].mean(),
                    "std": df['close'].std()
                }
            }
        }
        
        return info


def main():
    """
    Función principal para ejecutar la extracción
    """
    # API Key de Twelve Data
    API_KEY = "085ba06282774cbc8e796f46a5af8ece"
    
    # Fechas de extracción
    START_DATE = "2022-10-01 00:00:00"
    END_DATE = "2023-10-31 23:59:59"
    
    try:
        logger.info("🚀 Iniciando extracción de datos históricos USD/COP")
        logger.info(f"📅 Período: {START_DATE} - {END_DATE}")
        
        # Crear extractor
        extractor = USDCOPDataExtractor(API_KEY)
        
        # Extraer datos
        df = extractor.extract_historical_data(START_DATE, END_DATE)
        
        if df.empty:
            logger.error("❌ No se pudieron extraer datos")
            return
        
        # Mostrar información de los datos
        info = extractor.get_data_info(df)
        logger.info("📊 Información de los datos extraídos:")
        logger.info(f"   Total registros: {info.get('total_records', 0):,}")
        logger.info(f"   Rango de fechas: {info.get('date_range', {}).get('start', 'N/A')} - {info.get('date_range', {}).get('end', 'N/A')}")
        logger.info(f"   Columnas: {', '.join(info.get('columns', []))}")
        
        # Guardar datos
        filename = f"USDCOP_5min_2022_10_2023_10.csv"
        filepath = extractor.save_data(df, filename)
        
        logger.info("🎉 Extracción completada exitosamente!")
        logger.info(f"📁 Archivo guardado: {filepath}")
        logger.info(f"📊 Total de registros: {len(df):,}")
        
        # Mostrar primeras y últimas filas
        logger.info("\n📋 Primeras 5 filas:")
        logger.info(df.head().to_string())
        
        logger.info("\n📋 Últimas 5 filas:")
        logger.info(df.tail().to_string())
        
    except Exception as e:
        logger.error(f"❌ Error en la extracción principal: {str(e)}")
        raise


if __name__ == "__main__":
    main()

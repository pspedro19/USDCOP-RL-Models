#!/usr/bin/env python3
"""
📊 USDCOP Historical Data Extractor - 2023-2025
===============================================

Script para extraer datos históricos de USD/COP desde octubre 2023 hasta la fecha más reciente
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

def extract_usdcop_2023_2025():
    """
    Función principal para extraer datos de USD/COP 2023-2025
    """
    print("Iniciando extraccion de datos historicos USD/COP 2023-2025")
    print("=" * 70)
    
    # API Key de Twelve Data
    API_KEY = "085ba06282774cbc8e796f46a5af8ece"
    
    # Crear directorios si no existen
    os.makedirs('data/raw/twelve_data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Fechas de extracción para 2023-2025
    start_date = "2023-11-01 00:00:00"
    end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Hasta hoy
    
    print(f"Periodo: {start_date} - {end_date}")
    print(f"Duracion: Desde noviembre 2023 hasta la fecha actual")
    print("-" * 50)
    
    # Dividir en períodos de 30 días para evitar límites de API
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.now()
    
    all_data = []
    current_start = start_dt
    period_count = 0
    total_records = 0
    
    while current_start < end_dt:
        period_count += 1
        current_end = min(current_start + timedelta(days=30), end_dt)
        
        print(f"Procesando periodo {period_count}: {current_start.strftime('%Y-%m-%d')} - {current_end.strftime('%Y-%m-%d')}")
        
        try:
            # Extraer datos del período
            period_data = extract_period_data(API_KEY, current_start, current_end)
            
            if period_data is not None and not period_data.empty:
                all_data.append(period_data)
                period_records = len(period_data)
                total_records += period_records
                print(f"  -> Extraidos {period_records:,} registros (Total: {total_records:,})")
            else:
                print(f"  -> Sin datos para este periodo")
            
            # Pausa para respetar límites de API (8 requests/min)
            time.sleep(7.5)
            
        except Exception as e:
            print(f"  -> Error: {str(e)}")
            continue
        
        current_start = current_end
    
    if not all_data:
        print("No se pudieron extraer datos de ningun periodo")
        return
    
    # Combinar todos los datos
    print("\nCombinando datos...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Limpiar y procesar datos
    print("Limpiando y procesando datos...")
    final_df = clean_data(combined_df)
    
    # Guardar datos con nombre específico para 2023-2025
    filename = f"USDCOP_5min_2023_11_2025_08.csv"
    filepath = f"data/raw/twelve_data/{filename}"
    
    try:
        final_df.to_csv(filepath, index=False)
        print(f"\nDatos guardados exitosamente en: {filepath}")
        print(f"Total de registros: {len(final_df):,}")
        
        # Mostrar información de los datos
        print("\nInformacion de los datos 2023-2025:")
        print(f"  Rango de fechas: {final_df['time'].min()} - {final_df['time'].max()}")
        print(f"  Columnas: {', '.join(final_df.columns)}")
        print(f"  Precio minimo: {final_df['low'].min():.2f} COP")
        print(f"  Precio maximo: {final_df['high'].max():.2f} COP")
        print(f"  Precio promedio: {final_df['close'].mean():.2f} COP")
        
        # Calcular estadísticas adicionales
        print(f"\nEstadisticas adicionales:")
        total_days = (end_dt - start_dt).days
        print(f"  Total de dias cubiertos: {total_days}")
        print(f"  Registros por dia promedio: {len(final_df) / total_days:.1f}")
        print(f"  Tamaño del archivo: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"Error guardando datos: {str(e)}")

def extract_period_data(api_key, start_dt, end_dt):
    """
    Extrae datos para un período específico
    """
    url = "https://api.twelvedata.com/time_series"
    
    params = {
        "symbol": "USD/COP",
        "interval": "5min",
        "start_date": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "order": "ASC",
        "timezone": "America/Bogota",
        "apikey": api_key,
        "format": "JSON"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if "values" not in data or not data["values"]:
            return None
        
        # Crear DataFrame
        df = pd.DataFrame(data["values"])
        
        # Mostrar columnas disponibles para debugging
        print(f"    Columnas disponibles: {list(df.columns)}")
        
        # Convertir columnas numéricas disponibles
        numeric_columns = ['open', 'high', 'low', 'close']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convertir datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        return df
        
    except Exception as e:
        print(f"    Error en request: {str(e)}")
        return None

def clean_data(df):
    """
    Limpia y procesa los datos extraídos
    """
    print(f"Columnas originales: {list(df.columns)}")
    
    # Eliminar duplicados
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['datetime'], keep='first')
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        print(f"  Duplicados eliminados: {duplicates_removed}")
    
    # Ordenar por datetime
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Renombrar columnas para consistencia
    column_mapping = {
        'datetime': 'time',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Seleccionar solo columnas necesarias (sin volume)
    required_columns = ['time', 'open', 'high', 'low', 'close']
    df = df[required_columns]
    
    # Convertir time a string para consistencia
    df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Eliminar filas con valores nulos
    initial_rows = len(df)
    df = df.dropna()
    nulls_removed = initial_rows - len(df)
    if nulls_removed > 0:
        print(f"  Filas con nulos eliminadas: {nulls_removed}")
    
    return df

def estimate_extraction_time():
    """
    Estima el tiempo total de extracción
    """
    start_date = "2023-11-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.now()
    
    total_days = (end_dt - start_dt).days
    periods = (total_days // 30) + 1
    
    # Estimación: 7.5 segundos por período
    total_time_seconds = periods * 7.5
    total_time_minutes = total_time_seconds / 60
    
    print(f"Estimacion de tiempo de extraccion:")
    print(f"  Total de dias: {total_days}")
    print(f"  Periodos estimados: {periods}")
    print(f"  Tiempo estimado: {total_time_minutes:.1f} minutos")
    print(f"  Tiempo estimado: {total_time_minutes/60:.1f} horas")
    print("-" * 50)

def check_current_data():
    """
    Verifica qué datos están disponibles actualmente
    """
    print("Verificando datos disponibles actualmente...")
    
    # Verificar archivos existentes
    existing_files = []
    data_dir = "data/raw/twelve_data"
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.startswith("USDCOP_5min_") and file.endswith(".csv"):
                existing_files.append(file)
    
    if existing_files:
        print("Archivos de datos existentes:")
        for file in existing_files:
            print(f"  - {file}")
    else:
        print("No hay archivos de datos existentes")
    
    print("-" * 50)

if __name__ == "__main__":
    # Mostrar archivos existentes
    check_current_data()
    
    # Mostrar estimación de tiempo antes de comenzar
    estimate_extraction_time()
    
    # Preguntar si continuar
    response = input("\n¿Deseas continuar con la extraccion? (s/n): ")
    if response.lower() in ['s', 'si', 'y', 'yes']:
        extract_usdcop_2023_2025()
    else:
        print("Extraccion cancelada por el usuario.")

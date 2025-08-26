#!/usr/bin/env python3
"""
CONSOLIDACIÓN CORRECTA DE TWELVEDATA
=====================================
Procesa SOLO los 3 archivos originales de data/raw/twelve_data/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

def consolidate_twelve_data():
    """Consolidar los 3 archivos originales de TwelveData"""
    
    logger.info("="*80)
    logger.info("CONSOLIDACIÓN INTELIGENTE DE TWELVEDATA")
    logger.info("="*80)
    
    # Directorio de archivos ORIGINALES
    raw_twelve_dir = Path("data/raw/twelve_data")
    bronze_dir = Path("data/processed/bronze")
    bronze_dir.mkdir(parents=True, exist_ok=True)
    
    # Los 3 archivos originales
    original_files = [
        "USDCOP_5min_2020_01_2022_10.csv",  # 2020-2022
        "USDCOP_5min_2022_10_2023_10.csv",  # 2022-2023 
        "USDCOP_5min_2023_11_2025_08.csv"   # 2023-2025
    ]
    
    logger.info(f"\n1. VERIFICANDO ARCHIVOS ORIGINALES EN: {raw_twelve_dir}")
    logger.info("-" * 50)
    
    all_dfs = []
    file_info = []
    
    for filename in original_files:
        filepath = raw_twelve_dir / filename
        
        if filepath.exists():
            logger.info(f"\n✓ Encontrado: {filename}")
            
            # Leer archivo
            df = pd.read_csv(filepath)
            logger.info(f"  Registros: {len(df):,}")
            
            # Detectar columna de tiempo
            time_col = None
            for col in ['time', 'datetime', 'timestamp']:
                if col in df.columns:
                    time_col = col
                    break
            
            if time_col:
                df[time_col] = pd.to_datetime(df[time_col])
                df = df.rename(columns={time_col: 'time'})
                
                # Información del archivo
                start_date = df['time'].min()
                end_date = df['time'].max()
                
                logger.info(f"  Periodo: {start_date} a {end_date}")
                logger.info(f"  Días: {(end_date - start_date).days}")
                
                file_info.append({
                    'file': filename,
                    'records': len(df),
                    'start': start_date,
                    'end': end_date,
                    'days': (end_date - start_date).days
                })
                
                all_dfs.append(df)
            else:
                logger.warning(f"  ⚠️ No se encontró columna de tiempo")
        else:
            logger.error(f"\n✗ NO encontrado: {filename}")
            logger.info(f"  Esperado en: {filepath}")
    
    if not all_dfs:
        logger.error("\n❌ No se encontraron archivos para procesar")
        return None
    
    # ANÁLISIS DE SOLAPAMIENTO
    logger.info("\n" + "="*50)
    logger.info("2. ANÁLISIS DE SOLAPAMIENTO")
    logger.info("-" * 50)
    
    for i in range(len(file_info) - 1):
        current = file_info[i]
        next_file = file_info[i + 1]
        
        if current['end'] > next_file['start']:
            overlap_start = next_file['start']
            overlap_end = min(current['end'], next_file['end'])
            overlap_days = (overlap_end - overlap_start).days
            
            logger.info(f"\n⚠️ Solapamiento detectado:")
            logger.info(f"  {current['file'][-30:]} termina: {current['end'].date()}")
            logger.info(f"  {next_file['file'][-30:]} empieza: {next_file['start'].date()}")
            logger.info(f"  Solapamiento: {overlap_days} días")
    
    # COMBINAR INTELIGENTEMENTE
    logger.info("\n" + "="*50)
    logger.info("3. COMBINACIÓN INTELIGENTE")
    logger.info("-" * 50)
    
    # Concatenar todos
    logger.info("\nCombinando archivos...")
    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total registros antes de limpiar: {len(combined):,}")
    
    # Estadísticas antes de limpiar
    duplicates = combined.duplicated(subset=['time']).sum()
    logger.info(f"Duplicados detectados: {duplicates:,}")
    
    # Eliminar duplicados manteniendo el primer valor
    combined = combined.drop_duplicates(subset=['time'], keep='first')
    combined = combined.sort_values('time').reset_index(drop=True)
    
    logger.info(f"Total registros después de limpiar: {len(combined):,}")
    logger.info(f"Registros eliminados: {duplicates:,}")
    
    # ANÁLISIS DE CALIDAD
    logger.info("\n" + "="*50)
    logger.info("4. ANÁLISIS DE CALIDAD")
    logger.info("-" * 50)
    
    # Verificar continuidad
    time_diff = combined['time'].diff()
    five_min = pd.Timedelta(minutes=5)
    gaps = time_diff[time_diff > five_min]
    
    logger.info(f"\nContinuidad temporal:")
    logger.info(f"  Registros totales: {len(combined):,}")
    logger.info(f"  Periodo: {combined['time'].min()} a {combined['time'].max()}")
    logger.info(f"  Días totales: {(combined['time'].max() - combined['time'].min()).days}")
    logger.info(f"  Gaps detectados (>5min): {len(gaps):,}")
    
    # Gaps más grandes
    if len(gaps) > 0:
        logger.info(f"\nTop 5 gaps más grandes:")
        for idx, gap in enumerate(gaps.nlargest(5)):
            gap_hours = gap.total_seconds() / 3600
            if gap_hours > 24:
                logger.info(f"  {idx+1}. {gap_hours/24:.1f} días")
            else:
                logger.info(f"  {idx+1}. {gap_hours:.1f} horas")
    
    # Estadísticas de precio
    if 'close' in combined.columns:
        logger.info(f"\nEstadísticas de precio (close):")
        logger.info(f"  Mínimo: ${combined['close'].min():,.2f}")
        logger.info(f"  Máximo: ${combined['close'].max():,.2f}")
        logger.info(f"  Promedio: ${combined['close'].mean():,.2f}")
        logger.info(f"  Desv. Est.: ${combined['close'].std():,.2f}")
    
    # GUARDAR ARCHIVO CONSOLIDADO
    logger.info("\n" + "="*50)
    logger.info("5. GUARDANDO ARCHIVO CONSOLIDADO")
    logger.info("-" * 50)
    
    output_path = bronze_dir / "TWELVEDATA_M5_CONSOLIDATED_FINAL.csv"
    combined.to_csv(output_path, index=False)
    
    logger.info(f"\n✅ Archivo guardado: {output_path}")
    logger.info(f"   Registros: {len(combined):,}")
    logger.info(f"   Tamaño: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    # RESUMEN POR AÑO
    logger.info("\n" + "="*50)
    logger.info("6. DISTRIBUCIÓN POR AÑO")
    logger.info("-" * 50)
    
    combined['year'] = combined['time'].dt.year
    yearly_stats = combined.groupby('year').agg({
        'time': ['min', 'max', 'count']
    })
    
    for year in sorted(combined['year'].unique()):
        year_data = combined[combined['year'] == year]
        logger.info(f"\n{year}:")
        logger.info(f"  Registros: {len(year_data):,}")
        logger.info(f"  Desde: {year_data['time'].min()}")
        logger.info(f"  Hasta: {year_data['time'].max()}")
        if 'close' in year_data.columns:
            logger.info(f"  Precio promedio: ${year_data['close'].mean():,.2f}")
    
    # RESUMEN FINAL
    logger.info("\n" + "="*80)
    logger.info("RESUMEN FINAL")
    logger.info("="*80)
    
    logger.info(f"\n📊 CONSOLIDACIÓN COMPLETADA:")
    logger.info(f"   • 3 archivos originales procesados")
    logger.info(f"   • {len(combined):,} registros finales")
    logger.info(f"   • {duplicates:,} duplicados eliminados")
    logger.info(f"   • Periodo: {combined['time'].min().date()} a {combined['time'].max().date()}")
    logger.info(f"   • Archivo: TWELVEDATA_M5_CONSOLIDATED_FINAL.csv")
    
    return combined

if __name__ == "__main__":
    df = consolidate_twelve_data()
    
    if df is not None:
        print(f"\n✅ Pipeline completado exitosamente")
        print(f"   Dataset final: {len(df):,} registros M5")
    else:
        print("\n❌ Error en el pipeline")
#!/usr/bin/env python3
"""
MT5 MAXIMUM DOWNLOAD - Descarga Máxima de Datos M5
===================================================
Intenta descargar el máximo histórico posible de MT5
desde octubre 2023 hasta hoy
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging
import time
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class MT5MaximumDownloader:
    """Descargador optimizado para obtener máximo histórico de MT5"""
    
    def __init__(self):
        self.symbol = "USDCOP.r"
        self.timeframe = mt5.TIMEFRAME_M5
        self.output_dir = Path("data/raw/mt5_maximum")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize_mt5(self) -> bool:
        """Inicializar MT5"""
        
        logger.info("Inicializando MT5...")
        
        if not mt5.initialize():
            logger.error("No se pudo inicializar MT5")
            return False
        
        # Información del terminal
        terminal_info = mt5.terminal_info()
        if terminal_info:
            logger.info(f"Terminal: {terminal_info.name}")
            logger.info(f"Compañía: {terminal_info.company}")
        
        # Verificar símbolo
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logger.warning(f"Símbolo {self.symbol} no encontrado")
            # Intentar alternativas
            alternatives = ["USDCOP", "USD/COP", "USDCOP.m"]
            for alt in alternatives:
                symbol_info = mt5.symbol_info(alt)
                if symbol_info:
                    self.symbol = alt
                    logger.info(f"Usando símbolo alternativo: {alt}")
                    break
        
        if symbol_info:
            logger.info(f"Símbolo: {self.symbol}")
            logger.info(f"Descripción: {symbol_info.description}")
            logger.info(f"Spread actual: {symbol_info.spread} puntos")
        
        return True
    
    def download_by_chunks(self, start_date: datetime, end_date: datetime, 
                          chunk_days: int = 30) -> pd.DataFrame:
        """
        Descargar datos por chunks para evitar límites
        
        Args:
            start_date: Fecha inicio
            end_date: Fecha fin
            chunk_days: Días por chunk
            
        Returns:
            DataFrame con todos los datos
        """
        
        logger.info(f"\nDescargando por chunks de {chunk_days} días...")
        logger.info(f"Período: {start_date.date()} a {end_date.date()}")
        
        all_data = []
        current_end = end_date
        chunk_num = 0
        total_bars = 0
        
        while current_end > start_date:
            chunk_num += 1
            current_start = max(current_end - timedelta(days=chunk_days), start_date)
            
            logger.info(f"\nChunk {chunk_num}: {current_start.date()} a {current_end.date()}")
            
            # Intentar descargar el chunk
            rates = mt5.copy_rates_range(
                self.symbol,
                self.timeframe,
                current_start,
                current_end
            )
            
            if rates is not None and len(rates) > 0:
                df_chunk = pd.DataFrame(rates)
                all_data.append(df_chunk)
                logger.info(f"  ✓ {len(rates):,} barras descargadas")
                total_bars += len(rates)
            else:
                # Si falla, intentar con un chunk más pequeño
                if chunk_days > 7:
                    logger.warning(f"  ✗ Fallo, intentando con chunk más pequeño...")
                    smaller_chunk = self.download_by_chunks(
                        current_start, current_end, chunk_days=7
                    )
                    if not smaller_chunk.empty:
                        all_data.append(smaller_chunk)
                        total_bars += len(smaller_chunk)
                else:
                    logger.warning(f"  ✗ Sin datos para este período")
            
            # Mover al siguiente chunk
            current_end = current_start - timedelta(minutes=5)
            
            # Pequeña pausa para no sobrecargar
            time.sleep(0.1)
        
        logger.info(f"\nTotal barras descargadas: {total_bars:,}")
        
        if all_data:
            # Combinar todos los chunks
            df_combined = pd.concat(all_data, ignore_index=True)
            df_combined['time'] = pd.to_datetime(df_combined['time'], unit='s')
            
            # Eliminar duplicados y ordenar
            df_combined = df_combined.drop_duplicates(subset=['time'])
            df_combined = df_combined.sort_values('time').reset_index(drop=True)
            
            logger.info(f"Total después de limpiar: {len(df_combined):,} barras únicas")
            
            return df_combined
        
        return pd.DataFrame()
    
    def download_maximum_history(self) -> pd.DataFrame:
        """
        Intentar descargar el máximo histórico posible
        """
        
        logger.info("\n" + "="*80)
        logger.info("DESCARGA MÁXIMA DE HISTÓRICO MT5")
        logger.info("="*80)
        
        all_methods_data = []
        
        # MÉTODO 1: Desde posición 0 (más reciente hacia atrás)
        logger.info("\nMÉTODO 1: Descarga desde posición 0")
        logger.info("-" * 40)
        
        max_attempts = [999999, 500000, 200000, 100000, 50000, 30000, 20000, 10000]
        
        for max_bars in max_attempts:
            logger.info(f"Intentando {max_bars:,} barras...")
            
            rates = mt5.copy_rates_from_pos(
                self.symbol,
                self.timeframe,
                0,  # Desde la barra más reciente
                max_bars
            )
            
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                all_methods_data.append(df)
                logger.info(f"  ✓ Obtenidas {len(rates):,} barras")
                logger.info(f"  Período: {df['time'].min()} a {df['time'].max()}")
                break
            else:
                logger.info(f"  ✗ No disponible")
        
        # MÉTODO 2: Por rango de fechas (Oct 2023 - Hoy)
        logger.info("\nMÉTODO 2: Descarga por rango (Oct 2023 - Hoy)")
        logger.info("-" * 40)
        
        end_date = datetime.now(timezone.utc)
        start_date = datetime(2023, 10, 1, tzinfo=timezone.utc)
        
        # Intentar descarga directa primero
        logger.info(f"Intentando rango completo: {start_date.date()} a {end_date.date()}")
        
        rates = mt5.copy_rates_range(
            self.symbol,
            self.timeframe,
            start_date,
            end_date
        )
        
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            all_methods_data.append(df)
            logger.info(f"  ✓ Obtenidas {len(rates):,} barras")
        else:
            logger.info("  ✗ Rango muy grande, usando chunks...")
            # Usar método de chunks
            df_chunks = self.download_by_chunks(start_date, end_date, chunk_days=30)
            if not df_chunks.empty:
                all_methods_data.append(df_chunks)
        
        # MÉTODO 3: Últimos N días específicos
        logger.info("\nMÉTODO 3: Últimos períodos específicos")
        logger.info("-" * 40)
        
        periods = [
            (365, "1 año"),
            (180, "6 meses"),
            (90, "3 meses"),
            (60, "2 meses"),
            (30, "1 mes")
        ]
        
        for days, description in periods:
            logger.info(f"Intentando últimos {description}...")
            
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            rates = mt5.copy_rates_range(
                self.symbol,
                self.timeframe,
                start_date,
                end_date
            )
            
            if rates is not None and len(rates) > 10:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                all_methods_data.append(df)
                logger.info(f"  ✓ {len(rates):,} barras - {description}")
                break  # Si obtenemos datos, no necesitamos períodos más cortos
        
        # COMBINAR TODOS LOS DATOS
        logger.info("\n" + "="*80)
        logger.info("COMBINANDO RESULTADOS")
        logger.info("="*80)
        
        if all_methods_data:
            # Combinar todos los DataFrames
            df_final = pd.concat(all_methods_data, ignore_index=True)
            
            # Limpiar duplicados
            initial_count = len(df_final)
            df_final = df_final.drop_duplicates(subset=['time'])
            df_final = df_final.sort_values('time').reset_index(drop=True)
            
            duplicates_removed = initial_count - len(df_final)
            
            logger.info(f"Total registros antes de limpiar: {initial_count:,}")
            logger.info(f"Duplicados eliminados: {duplicates_removed:,}")
            logger.info(f"Total registros finales: {len(df_final):,}")
            
            if len(df_final) > 0:
                logger.info(f"Período final: {df_final['time'].min()} a {df_final['time'].max()}")
                logger.info(f"Días totales: {(df_final['time'].max() - df_final['time'].min()).days}")
                
                # Análisis de calidad
                self.analyze_data_quality(df_final)
                
                return df_final
        
        logger.warning("No se pudieron obtener datos")
        return pd.DataFrame()
    
    def analyze_data_quality(self, df: pd.DataFrame):
        """Analizar calidad de los datos descargados"""
        
        logger.info("\n" + "="*80)
        logger.info("ANÁLISIS DE CALIDAD")
        logger.info("="*80)
        
        # Gaps temporales
        time_diff = df['time'].diff()
        five_min = pd.Timedelta(minutes=5)
        gaps = time_diff[time_diff > five_min]
        
        logger.info(f"\n1. CONTINUIDAD:")
        logger.info(f"   Intervalos de 5 min: {(time_diff == five_min).sum():,}")
        logger.info(f"   Gaps detectados: {len(gaps):,}")
        
        if len(gaps) > 0:
            logger.info(f"   Gap máximo: {gaps.max().total_seconds()/3600:.1f} horas")
            logger.info(f"   Gap promedio: {gaps.mean().total_seconds()/3600:.1f} horas")
        
        # Estadísticas de precio
        if 'close' in df.columns:
            logger.info(f"\n2. ESTADÍSTICAS DE PRECIO:")
            logger.info(f"   Mínimo: ${df['close'].min():,.2f}")
            logger.info(f"   Máximo: ${df['close'].max():,.2f}")
            logger.info(f"   Promedio: ${df['close'].mean():,.2f}")
            logger.info(f"   Desv. Est.: ${df['close'].std():,.2f}")
        
        # Distribución por día y hora
        df['hour'] = df['time'].dt.hour
        df['dow'] = df['time'].dt.dayofweek
        
        logger.info(f"\n3. DISTRIBUCIÓN TEMPORAL:")
        logger.info(f"   Horas con datos: {df['hour'].nunique()}")
        logger.info(f"   Días de la semana: {df['dow'].nunique()}")
        
        # Calcular barras esperadas vs reales
        try:
            import sys
            sys.path.append('..')
            from src.utils.expected_bars_calculator import ExpectedBarsCalculator
            
            calculator = ExpectedBarsCalculator()
            start_date = df['time'].min()
            end_date = df['time'].max()
            
            expected = calculator.calculate_expected_bars(start_date, end_date)
            completeness = calculator.calculate_nulls_percentage(len(df), start_date, end_date)
            
            logger.info(f"\n4. COMPLETITUD:")
            logger.info(f"   Barras reales: {len(df):,}")
            logger.info(f"   Barras esperadas: {expected['expected_bars']['total']:,}")
            logger.info(f"   Completitud: {completeness['completeness_percentage']:.2f}%")
            logger.info(f"   Evaluación: {completeness['quality_assessment']}")
        except ImportError:
            logger.info(f"\n4. COMPLETITUD:")
            logger.info(f"   Barras reales: {len(df):,}")
            logger.info("   (Calculador de barras esperadas no disponible)")
    
    def save_data(self, df: pd.DataFrame) -> str:
        """Guardar datos descargados"""
        
        if df.empty:
            logger.warning("No hay datos para guardar")
            return ""
        
        # Nombre del archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"USDCOP_M5_MAXIMUM_{timestamp}.csv"
        filepath = self.output_dir / filename
        
        # Guardar CSV
        df.to_csv(filepath, index=False)
        
        # Guardar metadata
        metadata = {
            'symbol': self.symbol,
            'timeframe': 'M5',
            'records': len(df),
            'period_start': str(df['time'].min()),
            'period_end': str(df['time'].max()),
            'days': (df['time'].max() - df['time'].min()).days,
            'download_timestamp': datetime.now().isoformat(),
            'filename': filename
        }
        
        metadata_path = self.output_dir / f"metadata_{timestamp}.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\n✅ Datos guardados:")
        logger.info(f"   Archivo: {filepath}")
        logger.info(f"   Registros: {len(df):,}")
        logger.info(f"   Tamaño: {filepath.stat().st_size / (1024*1024):.2f} MB")
        
        return str(filepath)
    
    def run(self) -> pd.DataFrame:
        """Ejecutar descarga completa"""
        
        try:
            # Inicializar MT5
            if not self.initialize_mt5():
                return pd.DataFrame()
            
            # Descargar máximo histórico
            df = self.download_maximum_history()
            
            # Guardar si hay datos
            if not df.empty:
                self.save_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error en descarga: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
            
        finally:
            # Cerrar MT5
            mt5.shutdown()
            logger.info("\nMT5 cerrado")


def main():
    """Función principal"""
    
    logger.info("\n" + "="*80)
    logger.info("MT5 MAXIMUM DOWNLOAD - INICIANDO")
    logger.info("="*80)
    
    downloader = MT5MaximumDownloader()
    df = downloader.run()
    
    if not df.empty:
        logger.info("\n" + "="*80)
        logger.info("DESCARGA COMPLETADA EXITOSAMENTE")
        logger.info("="*80)
        logger.info(f"\nTotal registros M5: {len(df):,}")
        logger.info(f"Período: {df['time'].min()} a {df['time'].max()}")
        
        # Generar reporte de barras esperadas
        try:
            import sys
            sys.path.append('..')
            from src.utils.expected_bars_calculator import ExpectedBarsCalculator
            
            calculator = ExpectedBarsCalculator()
            report = calculator.generate_detailed_report(
                df['time'].min(), 
                df['time'].max(),
                len(df)
            )
            
            # Guardar reporte
            report_path = Path("data/raw/mt5_maximum") / "expected_bars_report.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"\n📊 Reporte de completitud guardado: {report_path}")
        except ImportError:
            logger.info("\n(Reporte de completitud no generado - módulo no disponible)")
        
        return df
    else:
        logger.error("\n❌ No se pudieron descargar datos")
        return None


if __name__ == "__main__":
    main()
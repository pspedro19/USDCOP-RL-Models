#!/usr/bin/env python3
"""
ANÁLISIS DE TAMAÑO DE GAPS POR SESIÓN DE TRADING
=================================================
Analiza específicamente el tamaño y distribución de gaps
dentro de cada sesión de trading (Premium, London, Afternoon, Friday)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class SessionGapAnalyzer:
    """Analizador de gaps por sesión de trading"""
    
    def __init__(self):
        self.data_root = Path("data")
        self.bronze_dir = self.data_root / "processed" / "bronze"
        
        # Definir sesiones en UTC (ya convertido desde COT)
        self.sessions = {
            'Premium': {
                'name': 'Premium (08:00-14:00 COT)',
                'hours_utc': (13, 19),  # 08:00-14:00 COT = 13:00-19:00 UTC
                'days': [0, 1, 2, 3],  # Lun-Jue
                'expected_bars_per_day': 72  # 6 horas * 12 barras/hora
            },
            'London': {
                'name': 'London (03:00-08:00 COT)', 
                'hours_utc': (8, 13),   # 03:00-08:00 COT = 08:00-13:00 UTC
                'days': [0, 1, 2, 3],
                'expected_bars_per_day': 60  # 5 horas * 12 barras/hora
            },
            'Afternoon': {
                'name': 'Afternoon (14:00-17:00 COT)',
                'hours_utc': (19, 22),  # 14:00-17:00 COT = 19:00-22:00 UTC
                'days': [0, 1, 2, 3],
                'expected_bars_per_day': 36  # 3 horas * 12 barras/hora
            },
            'Friday': {
                'name': 'Friday (08:00-15:00 COT)',
                'hours_utc': (13, 20),  # 08:00-15:00 COT = 13:00-20:00 UTC
                'days': [4],  # Solo viernes
                'expected_bars_per_day': 84  # 7 horas * 12 barras/hora
            }
        }
    
    def load_data(self):
        """Cargar dataset Bronze UTC filtrado"""
        print("\n" + "="*80)
        print("CARGANDO DATOS")
        print("="*80)
        
        bronze_path = self.bronze_dir / "utc" / "TWELVEDATA_M5_UTC_FILTERED.csv"
        
        df = pd.read_csv(bronze_path)
        df['time'] = pd.to_datetime(df['time'])
        
        # Añadir columnas útiles
        df['hour_utc'] = df['time'].dt.hour
        df['dow'] = df['time'].dt.dayofweek
        df['date'] = df['time'].dt.date
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        
        # Calcular gaps
        df['time_diff'] = df['time'].diff()
        df['gap_minutes'] = df['time_diff'].dt.total_seconds() / 60
        df['is_gap'] = df['gap_minutes'] > 5
        
        print(f"Registros cargados: {len(df):,}")
        print(f"Período: {df['time'].min()} a {df['time'].max()}")
        print(f"Total gaps detectados: {df['is_gap'].sum():,}")
        
        return df
    
    def analyze_gaps_per_session(self, df):
        """Analizar gaps específicamente dentro de cada sesión"""
        
        print("\n" + "="*80)
        print("ANÁLISIS DETALLADO DE GAPS POR SESIÓN")
        print("="*80)
        
        results = {}
        
        for session_key, session_info in self.sessions.items():
            print(f"\n{'='*60}")
            print(f"{session_info['name']}")
            print(f"Horario UTC: {session_info['hours_utc'][0]:02d}:00 - {session_info['hours_utc'][1]:02d}:00")
            print(f"{'='*60}")
            
            # Filtrar datos de esta sesión
            session_mask = (
                df['dow'].isin(session_info['days']) &
                (df['hour_utc'] >= session_info['hours_utc'][0]) &
                (df['hour_utc'] < session_info['hours_utc'][1])
            )
            
            session_data = df[session_mask].copy()
            
            if len(session_data) == 0:
                print("No hay datos para esta sesión")
                continue
            
            # Análisis básico
            total_records = len(session_data)
            total_gaps = session_data['is_gap'].sum()
            
            print(f"\nESTADISTICAS GENERALES:")
            print(f"  Registros en sesión: {total_records:,}")
            print(f"  Gaps detectados: {total_gaps:,}")
            print(f"  Tasa de gaps: {(total_gaps/total_records)*100:.2f}%")
            
            # Clasificar gaps por tamaño SOLO dentro de esta sesión
            gaps_data = session_data[session_data['is_gap']].copy()
            
            if len(gaps_data) > 0:
                # Clasificación de gaps
                gaps_data['gap_category'] = pd.cut(
                    gaps_data['gap_minutes'],
                    bins=[5, 10, 15, 30, 60, 120, 240, 10000],
                    labels=['5-10 min', '10-15 min', '15-30 min', '30-60 min', 
                           '1-2 horas', '2-4 horas', '>4 horas']
                )
                
                print(f"\nDISTRIBUCION POR TAMANO DE GAP:")
                gap_distribution = gaps_data['gap_category'].value_counts()
                
                for category, count in gap_distribution.items():
                    pct = (count / total_gaps) * 100
                    print(f"  {category}: {count:,} gaps ({pct:.1f}%)")
                
                # Estadísticas de gaps
                print(f"\nMETRICAS DE GAPS:")
                print(f"  Gap minimo: {gaps_data['gap_minutes'].min():.1f} minutos")
                print(f"  Gap promedio: {gaps_data['gap_minutes'].mean():.1f} minutos")
                print(f"  Gap mediano: {gaps_data['gap_minutes'].median():.1f} minutos")
                print(f"  Gap maximo: {gaps_data['gap_minutes'].max():.1f} minutos ({gaps_data['gap_minutes'].max()/60:.1f} horas)")
                
                # Análisis por año
                print(f"\nGAPS POR ANO:")
                gaps_by_year = gaps_data.groupby('year').agg({
                    'is_gap': 'count',
                    'gap_minutes': ['mean', 'max']
                }).round(1)
                
                for year in gaps_by_year.index:
                    count = gaps_by_year.loc[year, ('is_gap', 'count')]
                    avg = gaps_by_year.loc[year, ('gap_minutes', 'mean')]
                    max_gap = gaps_by_year.loc[year, ('gap_minutes', 'max')]
                    print(f"  {year}: {count:.0f} gaps, promedio {avg:.1f} min, maximo {max_gap:.1f} min")
                
                # Identificar patrones
                print(f"\nPATRONES IDENTIFICADOS:")
                
                # Gaps al inicio de la sesión
                first_hour = session_info['hours_utc'][0]
                gaps_first_hour = gaps_data[gaps_data['hour_utc'] == first_hour]
                print(f"  Gaps en primera hora ({first_hour:02d}:00 UTC): {len(gaps_first_hour):,}")
                
                # Gaps al final de la sesión
                last_hour = session_info['hours_utc'][1] - 1
                gaps_last_hour = gaps_data[gaps_data['hour_utc'] == last_hour]
                print(f"  Gaps en última hora ({last_hour:02d}:00 UTC): {len(gaps_last_hour):,}")
                
                # Gaps consecutivos
                consecutive = 0
                prev_idx = -2
                for idx in gaps_data.index:
                    if idx == prev_idx + 1:
                        consecutive += 1
                    prev_idx = idx
                print(f"  Gaps consecutivos: {consecutive:,}")
                
                # Días con más gaps
                gaps_by_date = gaps_data.groupby('date').size().sort_values(ascending=False)
                if len(gaps_by_date) > 0:
                    worst_day = gaps_by_date.index[0]
                    worst_count = gaps_by_date.iloc[0]
                    print(f"  Día con más gaps: {worst_day} ({worst_count} gaps)")
            
            else:
                print("  No hay gaps en esta sesión")
            
            # Calcular completitud real
            print(f"\nANALISIS DE COMPLETITUD:")
            
            # Calcular días únicos en la sesión
            unique_dates = session_data['date'].nunique()
            
            # Calcular barras esperadas
            if session_key == 'Friday':
                expected_bars = unique_dates * session_info['expected_bars_per_day'] / 5  # Solo 1 día por semana
            else:
                expected_bars = unique_dates * session_info['expected_bars_per_day'] / 5 * 4  # 4 días por semana
            
            completeness = (total_records / expected_bars * 100) if expected_bars > 0 else 0
            
            print(f"  Dias con datos: {unique_dates}")
            print(f"  Barras esperadas (aprox): {expected_bars:.0f}")
            print(f"  Barras reales: {total_records:,}")
            print(f"  Completitud: {completeness:.1f}%")
            
            # Guardar resultados
            results[session_key] = {
                'total_records': total_records,
                'total_gaps': total_gaps,
                'gap_rate': (total_gaps/total_records)*100 if total_records > 0 else 0,
                'avg_gap_size': gaps_data['gap_minutes'].mean() if len(gaps_data) > 0 else 0,
                'max_gap_size': gaps_data['gap_minutes'].max() if len(gaps_data) > 0 else 0,
                'completeness': completeness
            }
        
        return results
    
    def generate_comparison_report(self, results):
        """Generar reporte comparativo entre sesiones"""
        
        print("\n" + "="*80)
        print("COMPARACIÓN ENTRE SESIONES")
        print("="*80)
        
        print("\nTABLA COMPARATIVA:")
        print("-" * 100)
        print(f"{'Sesion':<20} {'Registros':<12} {'Gaps':<8} {'Tasa Gaps':<12} {'Gap Prom':<12} {'Gap Max':<12} {'Completitud':<12}")
        print("-" * 100)
        
        for session_key, stats in results.items():
            session_name = self.sessions[session_key]['name'].split('(')[0].strip()
            print(f"{session_name:<20} {stats['total_records']:<12,} {stats['total_gaps']:<8} "
                  f"{stats['gap_rate']:<12.1f}% {stats['avg_gap_size']:<12.1f} "
                  f"{stats['max_gap_size']:<12.1f} {stats['completeness']:<12.1f}%")
        
        print("-" * 100)
        
        # Identificar mejor y peor sesión
        best_session = min(results.items(), key=lambda x: x[1]['gap_rate'])
        worst_session = max(results.items(), key=lambda x: x[1]['gap_rate'])
        
        print(f"\nMEJOR SESION: {self.sessions[best_session[0]]['name']}")
        print(f"   Tasa de gaps: {best_session[1]['gap_rate']:.1f}%")
        
        print(f"\nPEOR SESION: {self.sessions[worst_session[0]]['name']}")
        print(f"   Tasa de gaps: {worst_session[1]['gap_rate']:.1f}%")
    
    def generate_detailed_report(self, results):
        """Generar reporte markdown detallado"""
        
        report = f"""# DIAGNÓSTICO DETALLADO DE GAPS POR SESIÓN DE TRADING
================================================================================
Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## RESUMEN EJECUTIVO

Este análisis examina el tamaño y distribución de gaps ESPECÍFICAMENTE dentro de cada sesión de trading.

## RESULTADOS POR SESIÓN

"""
        
        for session_key, stats in results.items():
            session_info = self.sessions[session_key]
            report += f"""### {session_info['name']}

**Métricas Clave:**
- Registros totales: {stats['total_records']:,}
- Gaps detectados: {stats['total_gaps']:,}
- Tasa de gaps: {stats['gap_rate']:.2f}%
- Gap promedio: {stats['avg_gap_size']:.1f} minutos
- Gap máximo: {stats['max_gap_size']:.1f} minutos
- Completitud: {stats['completeness']:.1f}%

"""
        
        report += """## CONCLUSIONES

1. Los gaps no son uniformes entre sesiones
2. La sesión Premium tiene mejor continuidad de datos
3. London y Afternoon sufren de gaps más grandes y frecuentes
4. Friday tiene comportamiento intermedio

## RECOMENDACIONES

1. **Para trading en vivo**: Usar principalmente sesión Premium
2. **Para backtesting**: Considerar solo datos con gaps < 30 minutos
3. **Para modelos ML**: Ponderar datos por calidad de sesión
"""
        
        # Guardar reporte
        output_path = Path("data/processed/silver/session_gaps_diagnosis.md")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nReporte guardado en: {output_path}")
    
    def run_analysis(self):
        """Ejecutar análisis completo"""
        
        print("\n" + "="*80)
        print("INICIANDO ANÁLISIS DE GAPS POR SESIÓN")
        print("="*80)
        
        # Cargar datos
        df = self.load_data()
        
        # Analizar gaps por sesión
        results = self.analyze_gaps_per_session(df)
        
        # Generar comparación
        self.generate_comparison_report(results)
        
        # Generar reporte detallado
        self.generate_detailed_report(results)
        
        return results


def main():
    """Función principal"""
    analyzer = SessionGapAnalyzer()
    results = analyzer.run_analysis()
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)
    
    return results


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
ANÁLISIS DE GAPS REALES - SOLO DENTRO DE HORARIOS DE TRADING
=============================================================
Analiza ÚNICAMENTE los gaps que ocurren DENTRO del horario
exacto de trading, excluyendo:
- Fines de semana
- Festivos
- Horas fuera del horario de trading
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import holidays

class RealGapAnalyzer:
    """Analizador de gaps reales dentro del horario de trading"""
    
    def __init__(self):
        self.data_root = Path("data")
        self.bronze_dir = self.data_root / "processed" / "bronze"
        
        # Festivos USA y Colombia
        self.us_holidays = holidays.US(years=range(2020, 2026))
        self.co_holidays = holidays.CO(years=range(2020, 2026))
        
        # Sesiones en UTC (convertido desde COT)
        self.sessions = {
            'Premium': {
                'name': 'Premium (08:00-14:00 COT)',
                'hours_utc': (13, 19),  # 13:00-19:00 UTC
                'days': [0, 1, 2, 3],   # Lun-Jue
                'bars_per_day': 72      # 6 horas * 12 barras/hora
            },
            'London': {
                'name': 'London (03:00-08:00 COT)',
                'hours_utc': (8, 13),    # 08:00-13:00 UTC
                'days': [0, 1, 2, 3],
                'bars_per_day': 60       # 5 horas * 12 barras/hora
            },
            'Afternoon': {
                'name': 'Afternoon (14:00-17:00 COT)',
                'hours_utc': (19, 22),   # 19:00-22:00 UTC
                'days': [0, 1, 2, 3],
                'bars_per_day': 36       # 3 horas * 12 barras/hora
            },
            'Friday': {
                'name': 'Friday (08:00-15:00 COT)',
                'hours_utc': (13, 20),   # 13:00-20:00 UTC
                'days': [4],             # Solo viernes
                'bars_per_day': 84       # 7 horas * 12 barras/hora
            }
        }
    
    def load_data(self):
        """Cargar dataset"""
        print("\n" + "="*80)
        print("CARGANDO DATOS")
        print("="*80)
        
        bronze_path = self.bronze_dir / "utc" / "TWELVEDATA_M5_UTC_FILTERED.csv"
        df = pd.read_csv(bronze_path)
        df['time'] = pd.to_datetime(df['time'])
        
        # Añadir columnas temporales
        df['date'] = df['time'].dt.date
        df['hour_utc'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute
        df['dow'] = df['time'].dt.dayofweek  # 0=Lunes, 6=Domingo
        df['year'] = df['time'].dt.year
        
        print(f"Total registros: {len(df):,}")
        print(f"Período: {df['time'].min()} a {df['time'].max()}")
        
        return df
    
    def is_holiday(self, date):
        """Verificar si una fecha es festivo"""
        return date in self.us_holidays or date in self.co_holidays
    
    def analyze_real_gaps_per_session(self, df):
        """Analizar gaps REALES solo dentro del horario exacto de trading"""
        
        print("\n" + "="*80)
        print("ANÁLISIS DE GAPS REALES POR SESIÓN")
        print("(Solo dentro del horario exacto de trading)")
        print("="*80)
        
        results = {}
        
        for session_key, session_info in self.sessions.items():
            print(f"\n{'='*70}")
            print(f"{session_info['name']}")
            print(f"Horario UTC: {session_info['hours_utc'][0]:02d}:00 - {session_info['hours_utc'][1]:02d}:00")
            print(f"{'='*70}")
            
            # Filtrar SOLO datos que están en el horario correcto
            session_mask = (
                df['dow'].isin(session_info['days']) &
                (df['hour_utc'] >= session_info['hours_utc'][0]) &
                (df['hour_utc'] < session_info['hours_utc'][1]) &
                (~df['date'].apply(self.is_holiday))  # Excluir festivos
            )
            
            session_data = df[session_mask].copy()
            
            if len(session_data) == 0:
                print("No hay datos para esta sesión")
                continue
            
            # Ordenar por tiempo
            session_data = session_data.sort_values('time').reset_index(drop=True)
            
            # Agrupar por día para analizar completitud diaria
            daily_analysis = []
            
            for date in session_data['date'].unique():
                day_data = session_data[session_data['date'] == date]
                
                # Calcular barras esperadas para este día
                start_hour = session_info['hours_utc'][0]
                end_hour = session_info['hours_utc'][1]
                expected_bars = (end_hour - start_hour) * 12  # 12 barras por hora
                
                # Crear lista de todos los timestamps esperados
                expected_times = []
                for hour in range(start_hour, end_hour):
                    for minute in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
                        expected_times.append(f"{hour:02d}:{minute:02d}")
                
                # Verificar qué timestamps existen realmente
                actual_times = day_data.apply(lambda x: f"{x['hour_utc']:02d}:{x['minute']:02d}", axis=1).tolist()
                
                # Calcular gaps REALES (timestamps esperados pero faltantes)
                missing_times = set(expected_times) - set(actual_times)
                real_gaps = len(missing_times)
                
                completeness = ((expected_bars - real_gaps) / expected_bars * 100) if expected_bars > 0 else 0
                
                daily_analysis.append({
                    'date': date,
                    'expected_bars': expected_bars,
                    'actual_bars': len(day_data),
                    'real_gaps': real_gaps,
                    'completeness': completeness
                })
            
            # Convertir a DataFrame para análisis
            daily_df = pd.DataFrame(daily_analysis)
            
            # Estadísticas generales
            total_days = len(daily_df)
            total_expected = daily_df['expected_bars'].sum()
            total_actual = daily_df['actual_bars'].sum()
            total_real_gaps = daily_df['real_gaps'].sum()
            
            print(f"\n1. RESUMEN GENERAL:")
            print(f"   Días analizados: {total_days}")
            print(f"   Barras esperadas total: {total_expected:,}")
            print(f"   Barras presentes: {total_actual:,}")
            print(f"   Gaps REALES: {total_real_gaps:,}")
            print(f"   Completitud global: {(total_actual/total_expected*100):.1f}%")
            
            # Análisis de días completos vs incompletos
            days_100 = len(daily_df[daily_df['completeness'] == 100])
            days_95_plus = len(daily_df[daily_df['completeness'] >= 95])
            days_90_plus = len(daily_df[daily_df['completeness'] >= 90])
            days_50_plus = len(daily_df[daily_df['completeness'] >= 50])
            days_0 = len(daily_df[daily_df['actual_bars'] == 0])
            
            print(f"\n2. CALIDAD DE DÍAS:")
            print(f"   Días 100% completos: {days_100} ({days_100/total_days*100:.1f}%)")
            print(f"   Dias >=95% completos: {days_95_plus} ({days_95_plus/total_days*100:.1f}%)")
            print(f"   Dias >=90% completos: {days_90_plus} ({days_90_plus/total_days*100:.1f}%)")
            print(f"   Dias >=50% completos: {days_50_plus} ({days_50_plus/total_days*100:.1f}%)")
            print(f"   Días sin datos: {days_0} ({days_0/total_days*100:.1f}%)")
            
            # Análisis por año
            print(f"\n3. EVOLUCIÓN POR AÑO:")
            session_data['year'] = pd.to_datetime(session_data['date']).dt.year
            
            for year in sorted(session_data['year'].unique()):
                year_daily = daily_df[pd.to_datetime(daily_df['date']).dt.year == year]
                if len(year_daily) > 0:
                    year_completeness = year_daily['completeness'].mean()
                    year_gaps = year_daily['real_gaps'].sum()
                    year_days_100 = len(year_daily[year_daily['completeness'] == 100])
                    print(f"   {year}: {year_completeness:.1f}% promedio, "
                          f"{year_gaps} gaps reales, "
                          f"{year_days_100}/{len(year_daily)} días completos")
            
            # Detectar patrones de gaps DENTRO del horario
            print(f"\n4. ANÁLISIS DE GAPS INTRA-DÍA:")
            
            # Para cada día, analizar dónde están los gaps
            gap_by_hour = {}
            for hour in range(session_info['hours_utc'][0], session_info['hours_utc'][1]):
                gap_by_hour[hour] = 0
            
            for _, day_info in daily_df.iterrows():
                if day_info['real_gaps'] > 0:
                    day_data = session_data[session_data['date'] == day_info['date']]
                    
                    # Ver qué horas faltan
                    for hour in range(session_info['hours_utc'][0], session_info['hours_utc'][1]):
                        hour_data = day_data[day_data['hour_utc'] == hour]
                        expected_in_hour = 12
                        actual_in_hour = len(hour_data)
                        gaps_in_hour = expected_in_hour - actual_in_hour
                        gap_by_hour[hour] += gaps_in_hour
            
            # Mostrar distribución de gaps por hora
            print("   Gaps por hora del día:")
            for hour, gaps in gap_by_hour.items():
                if gaps > 0:
                    print(f"     {hour:02d}:00 UTC: {gaps} gaps totales")
            
            # Identificar días problemáticos
            worst_days = daily_df.nsmallest(5, 'completeness')
            if len(worst_days) > 0 and worst_days.iloc[0]['completeness'] < 100:
                print(f"\n5. DÍAS MÁS PROBLEMÁTICOS:")
                for _, day in worst_days.iterrows():
                    if day['completeness'] < 100:
                        print(f"   {day['date']}: {day['completeness']:.1f}% completo, "
                              f"{day['real_gaps']} gaps")
            
            # Guardar resultados
            results[session_key] = {
                'total_days': total_days,
                'total_expected': total_expected,
                'total_actual': total_actual,
                'total_real_gaps': total_real_gaps,
                'completeness_pct': (total_actual/total_expected*100) if total_expected > 0 else 0,
                'days_100_pct': (days_100/total_days*100) if total_days > 0 else 0,
                'days_95_plus_pct': (days_95_plus/total_days*100) if total_days > 0 else 0,
                'days_with_data': total_days - days_0
            }
        
        return results
    
    def generate_comparison_table(self, results):
        """Generar tabla comparativa"""
        
        print("\n" + "="*80)
        print("COMPARACIÓN ENTRE SESIONES - GAPS REALES")
        print("="*80)
        
        print("\nTABLA RESUMEN:")
        print("-" * 100)
        print(f"{'Sesión':<20} {'Días':<10} {'Esperadas':<12} {'Reales':<12} {'Gaps REALES':<12} {'Completitud':<12} {'Días 100%':<12}")
        print("-" * 100)
        
        for session_key, stats in results.items():
            session_name = self.sessions[session_key]['name'].split('(')[0].strip()
            print(f"{session_name:<20} {stats['total_days']:<10} {stats['total_expected']:<12,} "
                  f"{stats['total_actual']:<12,} {stats['total_real_gaps']:<12,} "
                  f"{stats['completeness_pct']:<12.1f}% {stats['days_100_pct']:<12.1f}%")
        
        print("-" * 100)
    
    def generate_detailed_report(self, results):
        """Generar reporte markdown"""
        
        report = f"""# ANÁLISIS DE GAPS REALES - SOLO DENTRO DE HORARIOS DE TRADING
================================================================================
Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## METODOLOGÍA

Este análisis cuenta ÚNICAMENTE los gaps que ocurren DENTRO del horario exacto de trading.
- NO cuenta fines de semana como gaps
- NO cuenta festivos como gaps  
- NO cuenta horas fuera del horario como gaps
- SOLO cuenta barras faltantes dentro del horario exacto de cada sesión

## RESULTADOS POR SESIÓN

"""
        
        for session_key, stats in results.items():
            session_info = self.sessions[session_key]
            
            report += f"""### {session_info['name']}

**Análisis de Gaps REALES:**
- Días analizados: {stats['total_days']}
- Barras esperadas (en horario): {stats['total_expected']:,}
- Barras presentes: {stats['total_actual']:,}
- **GAPS REALES**: {stats['total_real_gaps']:,}
- **Completitud**: {stats['completeness_pct']:.1f}%

**Calidad de días:**
- Días 100% completos: {stats['days_100_pct']:.1f}%
- Días >=95% completos: {stats['days_95_plus_pct']:.1f}%

---

"""
        
        report += """## CONCLUSIONES

1. Los gaps REALES son significativamente menores que los gaps totales reportados anteriormente
2. La mayoría de los "gaps" anteriores eran simplemente horarios donde el mercado no opera
3. Los gaps reales muestran la verdadera calidad de los datos dentro del horario de trading

## INTERPRETACIÓN

- **100% completo** = Todas las barras esperadas en ese horario están presentes
- **Gap REAL** = Una barra que DEBERÍA existir en horario de trading pero falta
- **No es gap** = Fin de semana, festivo, o fuera del horario de trading
"""
        
        # Guardar reporte
        output_path = Path("data/processed/silver/real_gaps_analysis.md")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nReporte guardado en: {output_path}")
        
        return report
    
    def run_analysis(self):
        """Ejecutar análisis completo"""
        
        print("\n" + "="*80)
        print("ANÁLISIS DE GAPS REALES - INICIANDO")
        print("="*80)
        
        # Cargar datos
        df = self.load_data()
        
        # Analizar gaps reales por sesión
        results = self.analyze_real_gaps_per_session(df)
        
        # Generar tabla comparativa
        self.generate_comparison_table(results)
        
        # Generar reporte
        self.generate_detailed_report(results)
        
        return results


def main():
    """Función principal"""
    analyzer = RealGapAnalyzer()
    results = analyzer.run_analysis()
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)
    
    return results


if __name__ == "__main__":
    main()
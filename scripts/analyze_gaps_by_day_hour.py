#!/usr/bin/env python3
"""
ANÁLISIS DETALLADO DE GAPS POR DÍA DE SEMANA Y HORA
====================================================
Discrimina gaps por:
- Día de la semana (Lun, Mar, Mie, Jue, Vie)
- Hora específica dentro de cada sesión
- Calidad por día/hora
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import holidays

class DetailedGapAnalyzer:
    """Analizador detallado de gaps por día y hora"""
    
    def __init__(self):
        self.data_root = Path("data")
        self.bronze_dir = self.data_root / "processed" / "bronze"
        
        # Festivos
        self.us_holidays = holidays.US(years=range(2020, 2026))
        self.co_holidays = holidays.CO(years=range(2020, 2026))
        
        # Mapeo de días
        self.day_names = {
            0: 'Lunes',
            1: 'Martes', 
            2: 'Miércoles',
            3: 'Jueves',
            4: 'Viernes'
        }
        
        # Sesiones detalladas
        self.sessions = {
            'Premium': {
                'name': 'Premium',
                'hours_cot': (8, 14),    # 08:00-14:00 COT
                'hours_utc': (13, 19),   # 13:00-19:00 UTC
                'days': [0, 1, 2, 3],    # Lun-Jue
                'days_names': 'Lun-Jue',
                'bars_per_hour': 12
            },
            'London': {
                'name': 'London',
                'hours_cot': (3, 8),     # 03:00-08:00 COT
                'hours_utc': (8, 13),    # 08:00-13:00 UTC
                'days': [0, 1, 2, 3],
                'days_names': 'Lun-Jue',
                'bars_per_hour': 12
            },
            'Afternoon': {
                'name': 'Afternoon',
                'hours_cot': (14, 17),   # 14:00-17:00 COT
                'hours_utc': (19, 22),   # 19:00-22:00 UTC
                'days': [0, 1, 2, 3],
                'days_names': 'Lun-Jue',
                'bars_per_hour': 12
            },
            'Friday': {
                'name': 'Friday',
                'hours_cot': (8, 15),    # 08:00-15:00 COT
                'hours_utc': (13, 20),   # 13:00-20:00 UTC
                'days': [4],             # Solo Vie
                'days_names': 'Viernes',
                'bars_per_hour': 12
            }
        }
    
    def load_data(self):
        """Cargar y preparar datos"""
        bronze_path = self.bronze_dir / "utc" / "TWELVEDATA_M5_UTC_FILTERED.csv"
        df = pd.read_csv(bronze_path)
        df['time'] = pd.to_datetime(df['time'])
        
        # Columnas temporales
        df['date'] = df['time'].dt.date
        df['hour_utc'] = df['time'].dt.hour
        df['hour_cot'] = ((df['hour_utc'] - 5) % 24)  # Convertir a COT
        df['minute'] = df['time'].dt.minute
        df['dow'] = df['time'].dt.dayofweek
        df['dow_name'] = df['dow'].map(self.day_names)
        df['year'] = df['time'].dt.year
        
        return df
    
    def analyze_by_day_and_hour(self, df):
        """Analizar gaps por día de semana y hora"""
        
        print("\n" + "="*120)
        print("ANÁLISIS DETALLADO POR DÍA DE SEMANA Y HORA")
        print("="*120)
        
        all_results = []
        
        for session_key, session_info in self.sessions.items():
            
            # Filtrar datos de la sesión
            session_mask = (
                df['dow'].isin(session_info['days']) &
                (df['hour_utc'] >= session_info['hours_utc'][0]) &
                (df['hour_utc'] < session_info['hours_utc'][1])
            )
            
            session_data = df[session_mask].copy()
            
            if len(session_data) == 0:
                continue
            
            # Analizar por día de la semana
            for day in session_info['days']:
                day_name = self.day_names[day]
                day_data = session_data[session_data['dow'] == day]
                
                if len(day_data) == 0:
                    continue
                
                # Calcular estadísticas para este día
                unique_dates = day_data['date'].nunique()
                
                # Barras esperadas
                hours_in_session = session_info['hours_utc'][1] - session_info['hours_utc'][0]
                bars_per_day = hours_in_session * 12
                expected_total = unique_dates * bars_per_day
                actual_total = len(day_data)
                gaps_total = expected_total - actual_total
                completeness = (actual_total / expected_total * 100) if expected_total > 0 else 0
                
                # Analizar por hora dentro del día
                hour_details = []
                for hour_utc in range(session_info['hours_utc'][0], session_info['hours_utc'][1]):
                    hour_cot = (hour_utc - 5) % 24
                    hour_data = day_data[day_data['hour_utc'] == hour_utc]
                    
                    expected_hour = unique_dates * 12  # 12 barras por hora
                    actual_hour = len(hour_data)
                    gaps_hour = expected_hour - actual_hour
                    completeness_hour = (actual_hour / expected_hour * 100) if expected_hour > 0 else 0
                    
                    hour_details.append({
                        'hour_cot': hour_cot,
                        'hour_utc': hour_utc,
                        'completeness': completeness_hour,
                        'gaps': gaps_hour
                    })
                
                # Encontrar mejor y peor hora
                best_hour = max(hour_details, key=lambda x: x['completeness'])
                worst_hour = min(hour_details, key=lambda x: x['completeness'])
                
                # Agregar resultado
                result = {
                    'sesion': session_info['name'],
                    'horario_cot': f"{session_info['hours_cot'][0]:02d}:00-{session_info['hours_cot'][1]:02d}:00",
                    'horario_utc': f"{session_info['hours_utc'][0]:02d}:00-{session_info['hours_utc'][1]:02d}:00",
                    'dia_semana': day_name,
                    'dias_analizados': unique_dates,
                    'barras_esperadas': expected_total,
                    'barras_reales': actual_total,
                    'gaps_reales': gaps_total,
                    'completitud': completeness,
                    'mejor_hora': f"{best_hour['hour_cot']:02d}:00 COT ({best_hour['completeness']:.1f}%)",
                    'peor_hora': f"{worst_hour['hour_cot']:02d}:00 COT ({worst_hour['completeness']:.1f}%)",
                    'hour_details': hour_details
                }
                
                all_results.append(result)
        
        return all_results
    
    def print_detailed_table(self, results):
        """Imprimir tabla detallada"""
        
        print("\n" + "="*140)
        print("TABLA DETALLADA: COMPLETITUD POR SESIÓN, DÍA Y HORA")
        print("="*140)
        
        # Encabezados
        print(f"\n{'Sesión':<12} {'Horario COT':<15} {'Horario UTC':<15} {'Día':<12} {'Días':<6} "
              f"{'Esperadas':<10} {'Reales':<10} {'Gaps':<10} {'Complet':<8} {'Mejor Hora':<20} {'Peor Hora':<20}")
        print("-"*140)
        
        current_session = None
        
        for r in results:
            # Separador por sesión
            if current_session != r['sesion']:
                if current_session is not None:
                    print("-"*140)
                current_session = r['sesion']
            
            print(f"{r['sesion']:<12} {r['horario_cot']:<15} {r['horario_utc']:<15} {r['dia_semana']:<12} "
                  f"{r['dias_analizados']:<6} {r['barras_esperadas']:<10,} {r['barras_reales']:<10,} "
                  f"{r['gaps_reales']:<10,} {r['completitud']:<8.1f}% {r['mejor_hora']:<20} {r['peor_hora']:<20}")
        
        print("-"*140)
    
    def analyze_patterns(self, results):
        """Analizar patrones en los resultados"""
        
        print("\n" + "="*120)
        print("PATRONES IDENTIFICADOS")
        print("="*120)
        
        # Convertir a DataFrame para análisis
        df_results = pd.DataFrame(results)
        
        # 1. Mejor día por sesión
        print("\n1. MEJOR DÍA POR SESIÓN:")
        for session in df_results['sesion'].unique():
            session_data = df_results[df_results['sesion'] == session]
            best_day = session_data.loc[session_data['completitud'].idxmax()]
            print(f"   {session}: {best_day['dia_semana']} ({best_day['completitud']:.1f}% completo)")
        
        # 2. Peor día por sesión
        print("\n2. PEOR DÍA POR SESIÓN:")
        for session in df_results['sesion'].unique():
            session_data = df_results[df_results['sesion'] == session]
            worst_day = session_data.loc[session_data['completitud'].idxmin()]
            print(f"   {session}: {worst_day['dia_semana']} ({worst_day['completitud']:.1f}% completo)")
        
        # 3. Promedio por día de la semana
        print("\n3. PROMEDIO POR DÍA DE LA SEMANA (todas las sesiones):")
        for day in ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']:
            day_data = df_results[df_results['dia_semana'] == day]
            if len(day_data) > 0:
                avg_completeness = day_data['completitud'].mean()
                print(f"   {day}: {avg_completeness:.1f}% promedio")
        
        # 4. Resumen por sesión
        print("\n4. RESUMEN CONSOLIDADO POR SESIÓN:")
        for session in df_results['sesion'].unique():
            session_data = df_results[df_results['sesion'] == session]
            total_expected = session_data['barras_esperadas'].sum()
            total_actual = session_data['barras_reales'].sum()
            total_gaps = session_data['gaps_reales'].sum()
            total_completeness = (total_actual / total_expected * 100) if total_expected > 0 else 0
            
            print(f"\n   {session}:")
            print(f"      Total esperadas: {total_expected:,}")
            print(f"      Total reales: {total_actual:,}")
            print(f"      Total gaps: {total_gaps:,}")
            print(f"      Completitud global: {total_completeness:.1f}%")
    
    def generate_markdown_report(self, results):
        """Generar reporte markdown"""
        
        report = f"""# ANÁLISIS DETALLADO DE GAPS POR DÍA Y HORA
================================================================================
Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## TABLA COMPLETA: COMPLETITUD POR SESIÓN, DÍA Y HORA

| Sesión | Horario COT | Horario UTC | Día Semana | Días Analizados | Barras Esperadas | Barras Reales | Gaps REALES | Completitud | Mejor Hora | Peor Hora |
|--------|-------------|-------------|------------|-----------------|------------------|---------------|-------------|-------------|------------|-----------|
"""
        
        for r in results:
            report += f"| {r['sesion']} | {r['horario_cot']} | {r['horario_utc']} | {r['dia_semana']} | "
            report += f"{r['dias_analizados']} | {r['barras_esperadas']:,} | {r['barras_reales']:,} | "
            report += f"{r['gaps_reales']:,} | **{r['completitud']:.1f}%** | {r['mejor_hora']} | {r['peor_hora']} |\n"
        
        report += """

## HALLAZGOS CLAVE

1. **Premium (Lun-Jue 08:00-14:00 COT)**: La sesión más confiable
2. **London (Lun-Jue 03:00-08:00 COT)**: Problemática, especialmente en madrugada
3. **Afternoon (Lun-Jue 14:00-17:00 COT)**: Calidad variable
4. **Friday (Vie 08:00-15:00 COT)**: Solo viernes, calidad aceptable

## RECOMENDACIONES

- Usar principalmente datos de sesión Premium
- Evitar London y Afternoon para análisis críticos
- Considerar el día de la semana al evaluar calidad
"""
        
        # Guardar reporte
        output_path = Path("data/processed/silver/gaps_by_day_hour_analysis.md")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nReporte guardado: {output_path}")
        
        return report
    
    def run_analysis(self):
        """Ejecutar análisis completo"""
        
        print("\n" + "="*120)
        print("INICIANDO ANÁLISIS DETALLADO POR DÍA Y HORA")
        print("="*120)
        
        # Cargar datos
        df = self.load_data()
        print(f"\nDatos cargados: {len(df):,} registros")
        
        # Analizar por día y hora
        results = self.analyze_by_day_and_hour(df)
        
        # Imprimir tabla detallada
        self.print_detailed_table(results)
        
        # Analizar patrones
        self.analyze_patterns(results)
        
        # Generar reporte
        self.generate_markdown_report(results)
        
        return results


def main():
    """Función principal"""
    analyzer = DetailedGapAnalyzer()
    results = analyzer.run_analysis()
    
    print("\n" + "="*120)
    print("ANÁLISIS COMPLETADO")
    print("="*120)
    
    return results


if __name__ == "__main__":
    main()
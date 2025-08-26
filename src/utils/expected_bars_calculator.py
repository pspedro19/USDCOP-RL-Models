#!/usr/bin/env python3
"""
CALCULADOR DE BARRAS ESPERADAS SEGÚN REGLAS DE HORARIO
=======================================================
Calcula con precisión el número de barras M5 esperadas según:
- Horarios de trading definidos
- Días laborables
- Feriados
- Gaps de fin de semana
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import holidays

class ExpectedBarsCalculator:
    """Calculador preciso de barras esperadas"""
    
    def __init__(self):
        """Inicializar con reglas de horario estándar"""
        
        # Horarios de trading en COT (UTC-5)
        self.trading_sessions = {
            'premium': {
                'days': [0, 1, 2, 3],  # Lun-Jue (0=Monday en Python)
                'start_hour': 8,
                'end_hour': 14,  # Realmente 13:30, pero simplificamos a 14
                'bars_per_hour': 12,  # M5 = 12 barras por hora
                'weight': 1.0
            },
            'good_london': {
                'days': [0, 1, 2, 3],  # Lun-Jue
                'start_hour': 3,
                'end_hour': 8,
                'bars_per_hour': 12,
                'weight': 0.7
            },
            'good_afternoon': {
                'days': [0, 1, 2, 3],  # Lun-Jue
                'start_hour': 14,
                'end_hour': 17,
                'bars_per_hour': 12,
                'weight': 0.7
            },
            'friday': {
                'days': [4],  # Viernes
                'start_hour': 8,
                'end_hour': 15,
                'bars_per_hour': 12,
                'weight': 0.5
            }
        }
        
        # Feriados USA y Colombia
        self.us_holidays = holidays.US(years=range(2020, 2026))
        self.co_holidays = holidays.CO(years=range(2020, 2026))
        
    def calculate_expected_bars(self, start_date: datetime, end_date: datetime, 
                               include_holidays: bool = False) -> Dict:
        """
        Calcular barras esperadas en un período
        
        Args:
            start_date: Fecha inicio
            end_date: Fecha fin
            include_holidays: Si incluir días feriados
            
        Returns:
            Dict con estadísticas detalladas
        """
        
        # Convertir a pandas para facilitar cálculos
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        results = {
            'period': {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'total_days': len(date_range),
                'business_days': 0,
                'weekend_days': 0,
                'holiday_days': 0
            },
            'expected_bars': {
                'premium': 0,
                'good_london': 0,
                'good_afternoon': 0,
                'friday': 0,
                'total': 0
            },
            'expected_by_time': {
                'daily': {},
                'weekly': {},
                'monthly': {},
                'yearly': {}
            },
            'excluded_time': {
                'asian_session': 0,
                'weekends': 0,
                'holidays': 0,
                'total_excluded': 0
            }
        }
        
        # Calcular por cada día
        for date in date_range:
            day_of_week = date.dayofweek
            
            # Verificar si es fin de semana
            if day_of_week in [5, 6]:  # Sábado, Domingo
                results['period']['weekend_days'] += 1
                results['excluded_time']['weekends'] += 288  # 24h * 12 barras/h
                continue
            
            # Verificar si es feriado
            if not include_holidays:
                if date.date() in self.us_holidays or date.date() in self.co_holidays:
                    results['period']['holiday_days'] += 1
                    results['excluded_time']['holidays'] += 288
                    continue
            
            results['period']['business_days'] += 1
            
            # Calcular barras para cada sesión
            for session_name, session_config in self.trading_sessions.items():
                if day_of_week in session_config['days']:
                    hours = session_config['end_hour'] - session_config['start_hour']
                    
                    # Ajuste especial para premium (termina a las 13:30, no 14:00)
                    if session_name == 'premium':
                        hours = 5.5
                    
                    bars = int(hours * session_config['bars_per_hour'])
                    results['expected_bars'][session_name] += bars
            
            # Calcular tiempo excluido (sesión asiática)
            # Asiática: 17:00 - 03:00 (+1 día) = 10 horas
            results['excluded_time']['asian_session'] += 10 * 12  # 120 barras
        
        # Calcular totales
        results['expected_bars']['total'] = sum([
            results['expected_bars'][k] for k in ['premium', 'good_london', 'good_afternoon', 'friday']
        ])
        
        results['excluded_time']['total_excluded'] = sum([
            results['excluded_time'][k] for k in ['weekends', 'holidays', 'asian_session']
        ])
        
        # Calcular estadísticas por período
        total_days = (end_date - start_date).days + 1
        
        # Por día (promedio)
        results['expected_by_time']['daily'] = {
            'avg_bars': results['expected_bars']['total'] / total_days if total_days > 0 else 0,
            'business_day_bars': results['expected_bars']['total'] / results['period']['business_days'] 
                                if results['period']['business_days'] > 0 else 0
        }
        
        # Por semana
        weeks = total_days / 7
        results['expected_by_time']['weekly'] = {
            'weeks': round(weeks, 2),
            'avg_bars_per_week': results['expected_bars']['total'] / weeks if weeks > 0 else 0,
            'theoretical_max': 756  # Máximo teórico por semana con todas las sesiones
        }
        
        # Por mes
        months = total_days / 30.44  # Promedio días por mes
        results['expected_by_time']['monthly'] = {
            'months': round(months, 2),
            'avg_bars_per_month': results['expected_bars']['total'] / months if months > 0 else 0,
            'theoretical_max': 3276  # Aproximado mensual
        }
        
        # Por año
        years = total_days / 365.25
        results['expected_by_time']['yearly'] = {
            'years': round(years, 2),
            'avg_bars_per_year': results['expected_bars']['total'] / years if years > 0 else 0,
            'theoretical_max': 39312  # Aproximado anual
        }
        
        # Calcular eficiencia
        total_possible_bars = total_days * 24 * 12  # Total si fuera 24/7
        results['efficiency'] = {
            'total_possible_bars_24_7': total_possible_bars,
            'expected_trading_bars': results['expected_bars']['total'],
            'efficiency_percentage': (results['expected_bars']['total'] / total_possible_bars * 100) 
                                    if total_possible_bars > 0 else 0,
            'excluded_percentage': (results['excluded_time']['total_excluded'] / total_possible_bars * 100)
                                  if total_possible_bars > 0 else 0
        }
        
        return results
    
    def calculate_nulls_percentage(self, actual_bars: int, start_date: datetime, 
                                  end_date: datetime) -> Dict:
        """
        Calcular porcentaje de nulos basado en barras esperadas
        
        Args:
            actual_bars: Número de barras reales en el dataset
            start_date: Fecha inicio
            end_date: Fecha fin
            
        Returns:
            Dict con análisis de completitud
        """
        
        expected = self.calculate_expected_bars(start_date, end_date)
        
        analysis = {
            'actual_bars': actual_bars,
            'expected_bars': expected['expected_bars']['total'],
            'missing_bars': max(0, expected['expected_bars']['total'] - actual_bars),
            'completeness_percentage': min(100, (actual_bars / expected['expected_bars']['total'] * 100))
                                      if expected['expected_bars']['total'] > 0 else 0,
            'nulls_percentage': max(0, ((expected['expected_bars']['total'] - actual_bars) / 
                                       expected['expected_bars']['total'] * 100))
                               if expected['expected_bars']['total'] > 0 else 0,
            'quality_assessment': ''
        }
        
        # Evaluación de calidad
        if analysis['completeness_percentage'] >= 95:
            analysis['quality_assessment'] = 'EXCELENTE - Dataset casi completo'
        elif analysis['completeness_percentage'] >= 85:
            analysis['quality_assessment'] = 'BUENO - Gaps menores aceptables'
        elif analysis['completeness_percentage'] >= 70:
            analysis['quality_assessment'] = 'REGULAR - Gaps significativos'
        elif analysis['completeness_percentage'] >= 50:
            analysis['quality_assessment'] = 'POBRE - Muchos datos faltantes'
        else:
            analysis['quality_assessment'] = 'CRÍTICO - Mayoría de datos faltantes'
        
        # Desglose por sesión (estimado)
        analysis['missing_by_session'] = {
            'premium': int(analysis['missing_bars'] * 0.40),  # 40% del tiempo es premium
            'good_london': int(analysis['missing_bars'] * 0.25),  # 25% london
            'good_afternoon': int(analysis['missing_bars'] * 0.20),  # 20% afternoon
            'friday': int(analysis['missing_bars'] * 0.15)  # 15% friday
        }
        
        return analysis
    
    def generate_detailed_report(self, start_date: datetime, end_date: datetime,
                                actual_bars: int = None) -> str:
        """
        Generar reporte detallado en formato markdown
        
        Args:
            start_date: Fecha inicio
            end_date: Fecha fin
            actual_bars: Barras reales (opcional)
            
        Returns:
            String con reporte en markdown
        """
        
        expected = self.calculate_expected_bars(start_date, end_date)
        
        report = "# REPORTE DE BARRAS ESPERADAS\n\n"
        report += f"**Período analizado**: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}\n\n"
        
        # Resumen del período
        report += "## 📅 Resumen del Período\n\n"
        report += f"- **Total días**: {expected['period']['total_days']}\n"
        report += f"- **Días laborables**: {expected['period']['business_days']}\n"
        report += f"- **Fines de semana**: {expected['period']['weekend_days']}\n"
        report += f"- **Feriados**: {expected['period']['holiday_days']}\n\n"
        
        # Barras esperadas
        report += "## 📊 Barras M5 Esperadas\n\n"
        report += "### Por Sesión de Trading\n"
        report += f"- **Premium (08:00-13:30 COT)**: {expected['expected_bars']['premium']:,} barras\n"
        report += f"- **London (03:00-08:00 COT)**: {expected['expected_bars']['good_london']:,} barras\n"
        report += f"- **Afternoon (14:00-17:00 COT)**: {expected['expected_bars']['good_afternoon']:,} barras\n"
        report += f"- **Friday (08:00-15:00 COT)**: {expected['expected_bars']['friday']:,} barras\n"
        report += f"- **TOTAL ESPERADO**: {expected['expected_bars']['total']:,} barras\n\n"
        
        # Estadísticas por tiempo
        report += "### Promedios por Período\n"
        report += f"- **Por día calendario**: {expected['expected_by_time']['daily']['avg_bars']:.1f} barras\n"
        report += f"- **Por día laborable**: {expected['expected_by_time']['daily']['business_day_bars']:.1f} barras\n"
        report += f"- **Por semana**: {expected['expected_by_time']['weekly']['avg_bars_per_week']:.1f} barras\n"
        report += f"- **Por mes**: {expected['expected_by_time']['monthly']['avg_bars_per_month']:.1f} barras\n"
        report += f"- **Por año**: {expected['expected_by_time']['yearly']['avg_bars_per_year']:.1f} barras\n\n"
        
        # Tiempo excluido
        report += "## ⏰ Tiempo Excluido\n\n"
        report += f"- **Sesión asiática**: {expected['excluded_time']['asian_session']:,} barras\n"
        report += f"- **Fines de semana**: {expected['excluded_time']['weekends']:,} barras\n"
        report += f"- **Feriados**: {expected['excluded_time']['holidays']:,} barras\n"
        report += f"- **Total excluido**: {expected['excluded_time']['total_excluded']:,} barras\n\n"
        
        # Eficiencia
        report += "## 📈 Eficiencia\n\n"
        report += f"- **Barras posibles 24/7**: {expected['efficiency']['total_possible_bars_24_7']:,}\n"
        report += f"- **Barras en horario trading**: {expected['efficiency']['expected_trading_bars']:,}\n"
        report += f"- **Eficiencia**: {expected['efficiency']['efficiency_percentage']:.2f}%\n"
        report += f"- **Tiempo excluido**: {expected['efficiency']['excluded_percentage']:.2f}%\n\n"
        
        # Si tenemos barras reales, calcular completitud
        if actual_bars is not None:
            nulls_analysis = self.calculate_nulls_percentage(actual_bars, start_date, end_date)
            
            report += "## 🎯 Análisis de Completitud\n\n"
            report += f"- **Barras reales**: {nulls_analysis['actual_bars']:,}\n"
            report += f"- **Barras esperadas**: {nulls_analysis['expected_bars']:,}\n"
            report += f"- **Barras faltantes**: {nulls_analysis['missing_bars']:,}\n"
            report += f"- **Completitud**: {nulls_analysis['completeness_percentage']:.2f}%\n"
            report += f"- **Nulos**: {nulls_analysis['nulls_percentage']:.2f}%\n"
            report += f"- **Evaluación**: {nulls_analysis['quality_assessment']}\n\n"
            
            if nulls_analysis['missing_bars'] > 0:
                report += "### Distribución Estimada de Faltantes\n"
                for session, missing in nulls_analysis['missing_by_session'].items():
                    report += f"- **{session}**: ~{missing:,} barras\n"
        
        return report


def main():
    """Función de prueba"""
    
    calculator = ExpectedBarsCalculator()
    
    # Ejemplo 1: Calcular para el último año
    print("\n" + "="*80)
    print("CÁLCULO DE BARRAS ESPERADAS - ÚLTIMO AÑO")
    print("="*80)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    results = calculator.calculate_expected_bars(start_date, end_date)
    
    print(f"\nPeríodo: {start_date.date()} a {end_date.date()}")
    print(f"Días laborables: {results['period']['business_days']}")
    print(f"\nBarras esperadas por sesión:")
    print(f"  Premium: {results['expected_bars']['premium']:,}")
    print(f"  London: {results['expected_bars']['good_london']:,}")
    print(f"  Afternoon: {results['expected_bars']['good_afternoon']:,}")
    print(f"  Friday: {results['expected_bars']['friday']:,}")
    print(f"  TOTAL: {results['expected_bars']['total']:,}")
    
    print(f"\nEficiencia: {results['efficiency']['efficiency_percentage']:.2f}%")
    
    # Ejemplo 2: Análisis de completitud con datos reales
    print("\n" + "="*80)
    print("ANÁLISIS DE COMPLETITUD - DATOS REALES")
    print("="*80)
    
    # Supongamos que tenemos 35,000 barras reales
    actual_bars = 35000
    
    nulls_analysis = calculator.calculate_nulls_percentage(actual_bars, start_date, end_date)
    
    print(f"\nBarras reales: {nulls_analysis['actual_bars']:,}")
    print(f"Barras esperadas: {nulls_analysis['expected_bars']:,}")
    print(f"Completitud: {nulls_analysis['completeness_percentage']:.2f}%")
    print(f"Nulos: {nulls_analysis['nulls_percentage']:.2f}%")
    print(f"Evaluación: {nulls_analysis['quality_assessment']}")
    
    # Ejemplo 3: Generar reporte completo
    print("\n" + "="*80)
    print("GENERANDO REPORTE DETALLADO")
    print("="*80)
    
    report = calculator.generate_detailed_report(start_date, end_date, actual_bars)
    
    # Guardar reporte
    with open('expected_bars_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n✅ Reporte guardado en: expected_bars_report.md")


if __name__ == "__main__":
    main()
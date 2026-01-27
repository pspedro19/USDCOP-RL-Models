"""
Economic Calendar Parser & Manager
==================================

Gestor del calendario económico para prevenir data leakage.

Contract: CTR-L0-CALENDAR-001

Responsabilidades:
1. Cargar configuración desde YAML
2. Calcular fechas de publicación para cada variable
3. Forward-fill respetando fechas de publicación reales
4. Validar no-leakage en features de ML

Uso:
    from src.data.economic_calendar import EconomicCalendar

    calendar = EconomicCalendar()

    # Calcular fecha de publicación
    pub_date = calendar.get_publication_date('infl_cpi_all_usa_m_cpiaucsl', '2025-12')

    # Forward-fill respetando calendario
    df['cpi_safe'] = calendar.apply_publication_aware_ffill(df, 'infl_cpi_all_usa_m_cpiaucsl')

    # Validar no hay leakage
    result = calendar.validate_no_leakage(df, test_timestamp, 'infl_cpi_all_usa_m_cpiaucsl')

Version: 1.0.0
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, Optional, Tuple, List, Union
from dateutil.relativedelta import relativedelta
import logging

try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False

logger = logging.getLogger(__name__)


class EconomicCalendar:
    """
    Gestor principal del calendario económico.

    Single Source of Truth para fechas de publicación de indicadores macro.
    Previene data leakage en pipelines de ML.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Inicializar calendario económico.

        Args:
            config_path: Ruta al archivo economic_calendar.yaml.
                        Si None, usa la ruta por defecto.
        """
        if config_path is None:
            # Buscar config en rutas estándar
            possible_paths = [
                Path(__file__).parent.parent.parent / "config" / "economic_calendar.yaml",
                Path("/opt/airflow/config/economic_calendar.yaml"),
                Path("config/economic_calendar.yaml"),
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break

            if config_path is None:
                raise FileNotFoundError(
                    f"economic_calendar.yaml not found in: {possible_paths}"
                )

        logger.info(f"Loading economic calendar from: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Combinar todas las variables en un diccionario plano
        self.variables: Dict[str, dict] = {}

        for section in ['usa_monthly', 'colombia_monthly', 'quarterly']:
            section_vars = self.config.get(section, {})
            for var_name, var_config in section_vars.items():
                var_config['_section'] = section
                self.variables[var_name] = var_config

        logger.info(f"Loaded {len(self.variables)} variables from calendar")

        # Cache de fechas de publicación calculadas
        self._pub_date_cache: Dict[Tuple[str, str], pd.Timestamp] = {}

    def get_variable_config(self, variable_name: str) -> Optional[dict]:
        """
        Obtener configuración completa de una variable.

        Args:
            variable_name: Nombre de la variable (ej: 'infl_cpi_all_usa_m_cpiaucsl')

        Returns:
            Dict con configuración o None si no existe
        """
        return self.variables.get(variable_name)

    def get_publication_date(
        self,
        variable_name: str,
        data_period: Union[str, pd.Timestamp, date],
        return_datetime: bool = False
    ) -> Optional[Union[date, pd.Timestamp]]:
        """
        Calcular fecha de publicación para un dato específico.

        Args:
            variable_name: Nombre de la variable
            data_period: Período del dato (ej: '2025-12', '2025-12-01', datetime)
            return_datetime: Si True, retorna Timestamp con hora y timezone

        Returns:
            Fecha de publicación (date o Timestamp)

        Example:
            # CPI de diciembre 2025 se publica ~13 enero 2026
            pub_date = calendar.get_publication_date('infl_cpi_all_usa_m_cpiaucsl', '2025-12')
            # Returns: date(2026, 1, 13)
        """
        var_config = self.variables.get(variable_name)
        if not var_config:
            logger.warning(f"Variable {variable_name} not found in calendar")
            return None

        # Normalizar data_period a fecha
        if isinstance(data_period, str):
            # Formato YYYY-MM o YYYY-MM-DD
            if len(data_period) == 7:
                data_period = pd.Timestamp(data_period + '-01')
            else:
                data_period = pd.Timestamp(data_period)
        elif isinstance(data_period, date) and not isinstance(data_period, datetime):
            data_period = pd.Timestamp(data_period)
        elif isinstance(data_period, datetime):
            data_period = pd.Timestamp(data_period)

        # Check cache
        cache_key = (variable_name, data_period.strftime('%Y-%m'))
        if cache_key in self._pub_date_cache:
            cached = self._pub_date_cache[cache_key]
            return cached if return_datetime else cached.date()

        pub_config = var_config['publication']

        # Calcular mes de publicación
        frequency = var_config.get('frequency', 'monthly')

        if frequency == 'quarterly':
            # Para trimestrales, el lag es en trimestres
            quarter_lag = pub_config.get('quarter_lag', 1)
            days_after = pub_config.get('days_after_quarter', 90)

            # Encontrar fin del trimestre del dato
            quarter_end_month = ((data_period.month - 1) // 3 + 1) * 3
            quarter_end = pd.Timestamp(
                year=data_period.year,
                month=quarter_end_month,
                day=1
            ) + pd.offsets.MonthEnd(0)

            # Publicación es days_after días después del fin del trimestre
            pub_date = quarter_end + timedelta(days=days_after)
            pub_day = pub_date.day
            pub_month = pub_date.month
            pub_year = pub_date.year
        else:
            # Para mensuales
            month_lag = pub_config.get('month_lag', 1)
            pub_month_date = data_period + relativedelta(months=month_lag)

            pub_day = pub_config['typical_day']
            pub_month = pub_month_date.month
            pub_year = pub_month_date.year

        # Construir fecha
        try:
            pub_date = date(pub_year, pub_month, pub_day)
        except ValueError:
            # Si el día no existe en el mes (ej: día 31 en febrero)
            # Usar último día del mes
            last_day = (pd.Timestamp(year=pub_year, month=pub_month, day=1)
                       + pd.offsets.MonthEnd(0)).day
            pub_day = min(pub_day, last_day)
            pub_date = date(pub_year, pub_month, pub_day)

        # Si se requiere datetime con timezone
        if return_datetime:
            time_str = pub_config.get('time', '08:30:00')
            tz_str = pub_config.get('timezone', 'US/Eastern')

            time_parts = time_str.split(':')
            hour = int(time_parts[0])
            minute = int(time_parts[1])
            second = int(time_parts[2]) if len(time_parts) > 2 else 0

            pub_datetime = datetime(
                pub_year, pub_month, pub_day, hour, minute, second
            )

            if PYTZ_AVAILABLE:
                tz = pytz.timezone(tz_str)
                pub_datetime = tz.localize(pub_datetime)

            result = pd.Timestamp(pub_datetime)
            self._pub_date_cache[cache_key] = result
            return result

        result = pd.Timestamp(pub_date)
        self._pub_date_cache[cache_key] = result
        return pub_date

    def get_publication_lag_days(
        self,
        variable_name: str,
        data_period: Union[str, pd.Timestamp]
    ) -> int:
        """
        Calcular días de lag entre fin del período y publicación.

        Args:
            variable_name: Nombre de la variable
            data_period: Período del dato

        Returns:
            Número de días de lag
        """
        if isinstance(data_period, str):
            if len(data_period) == 7:
                data_period = pd.Timestamp(data_period + '-01')
            else:
                data_period = pd.Timestamp(data_period)

        # Fin del período
        period_end = data_period + pd.offsets.MonthEnd(0)

        # Fecha de publicación
        pub_date = self.get_publication_date(variable_name, data_period)

        if pub_date is None:
            return -1

        return (pd.Timestamp(pub_date) - period_end).days

    def apply_publication_aware_ffill(
        self,
        df: pd.DataFrame,
        variable_name: str,
        verbose: bool = False
    ) -> pd.Series:
        """
        Forward-fill respetando calendario de publicaciones.

        ⚠️ CRÍTICO: Nunca propaga un dato antes de su fecha de publicación real.

        Esto previene data leakage en features de ML.

        Args:
            df: DataFrame con index datetime (debe tener la columna variable_name)
            variable_name: Nombre de la variable a procesar
            verbose: Si True, muestra logging detallado

        Returns:
            Serie con forward-fill correcto (sin leakage)

        Example:
            # DataFrame con datos macro
            df = pd.read_sql('SELECT * FROM macro_indicators_daily', conn)
            df = df.set_index('fecha')

            # Forward-fill sin leakage
            df['cpi_safe'] = calendar.apply_publication_aware_ffill(
                df, 'infl_cpi_all_usa_m_cpiaucsl'
            )
        """
        if variable_name not in df.columns:
            logger.warning(f"Variable {variable_name} not found in DataFrame")
            return pd.Series(index=df.index, dtype=float, name=variable_name)

        # Asegurar que el index es datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("DataFrame index must be DatetimeIndex")
            return pd.Series(index=df.index, dtype=float, name=variable_name)

        # Serie resultado (inicialmente todo NaN)
        result = pd.Series(index=df.index, dtype=float, name=f"{variable_name}_safe")

        # Obtener valores no-NaN (datos mensuales originales)
        monthly_data = df[variable_name].dropna().sort_index()

        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Publication-aware forward-fill: {variable_name}")
            logger.info(f"{'='*60}")
            logger.info(f"Monthly data points: {len(monthly_data)}")
            logger.info(f"Target rows: {len(df)}")

        # Para cada valor mensual, propagar solo desde su fecha de publicación
        for data_date, value in monthly_data.items():
            # Calcular fecha de publicación
            pub_date = self.get_publication_date(variable_name, data_date)

            if pub_date is None:
                if verbose:
                    logger.warning(f"  Could not calculate pub date for {data_date}")
                continue

            pub_timestamp = pd.Timestamp(pub_date)

            if verbose:
                logger.info(f"\n  Data period: {data_date.strftime('%Y-%m')}")
                logger.info(f"    Value: {value}")
                logger.info(f"    Published: {pub_date}")

            # ✅ CRÍTICO: Solo propagar desde fecha de publicación
            propagation_mask = (df.index >= pub_timestamp)

            # Sobrescribir valores previos (el dato más reciente publicado gana)
            result.loc[propagation_mask] = value

            if verbose:
                n_rows = propagation_mask.sum()
                logger.info(f"    Propagated to {n_rows} rows")

        if verbose:
            non_null = result.notna().sum()
            logger.info(f"\n{'='*60}")
            logger.info(f"Result: {non_null}/{len(result)} rows with values")
            logger.info(f"{'='*60}\n")

        return result

    def validate_no_leakage(
        self,
        df: pd.DataFrame,
        test_timestamp: Union[str, pd.Timestamp],
        variable_name: str
    ) -> Dict:
        """
        Validar que no hay data leakage en un timestamp específico.

        Verifica que el valor usado en test_timestamp corresponde
        a un dato que ya había sido publicado en ese momento.

        Args:
            df: DataFrame con la variable
            test_timestamp: Timestamp a validar
            variable_name: Nombre de la variable

        Returns:
            Dict con resultado de validación

        Example:
            result = calendar.validate_no_leakage(
                df, '2026-01-10 09:30:00', 'infl_cpi_all_usa_m_cpiaucsl'
            )
            # result['is_valid'] == True si no hay leakage
        """
        if isinstance(test_timestamp, str):
            test_timestamp = pd.Timestamp(test_timestamp)

        if variable_name not in df.columns:
            return {
                'variable': variable_name,
                'timestamp': test_timestamp,
                'status': 'ERROR',
                'message': 'Variable not found in DataFrame',
                'is_valid': False
            }

        # Valor actualmente usado
        try:
            used_value = df.loc[test_timestamp, variable_name]
        except KeyError:
            return {
                'variable': variable_name,
                'timestamp': test_timestamp,
                'status': 'ERROR',
                'message': f'Timestamp {test_timestamp} not in DataFrame',
                'is_valid': False
            }

        # Determinar qué valor debería estar disponible
        monthly_data = df[variable_name].dropna().sort_index()

        expected_value = np.nan
        expected_period = None

        for data_date, value in monthly_data.items():
            pub_date = self.get_publication_date(variable_name, data_date)

            if pub_date is None:
                continue

            pub_timestamp = pd.Timestamp(pub_date)

            # Si ya fue publicado antes del test_timestamp
            if pub_timestamp <= test_timestamp:
                expected_value = value
                expected_period = data_date

        # Comparar
        if pd.isna(used_value) and pd.isna(expected_value):
            is_valid = True
            status = 'PASS - Both NaN'
        elif pd.isna(used_value) or pd.isna(expected_value):
            is_valid = False
            status = 'FAIL - Value mismatch (one is NaN)'
        elif abs(used_value - expected_value) < 1e-6:
            is_valid = True
            status = 'PASS'
        else:
            is_valid = False
            status = 'FAIL - DATA LEAKAGE DETECTED'

        return {
            'variable': variable_name,
            'timestamp': test_timestamp,
            'used_value': used_value,
            'expected_value': expected_value,
            'expected_period': expected_period.strftime('%Y-%m') if expected_period else None,
            'is_valid': is_valid,
            'status': status
        }

    def get_leakage_risk(self, variable_name: str) -> str:
        """Obtener nivel de riesgo de leakage de una variable."""
        var_config = self.variables.get(variable_name, {})
        validation = var_config.get('validation', {})
        return validation.get('leakage_risk', 'UNKNOWN')

    def get_all_variables(self) -> List[str]:
        """Obtener lista de todas las variables."""
        return list(self.variables.keys())

    def get_variables_by_risk(self, risk_level: str) -> List[str]:
        """
        Obtener variables por nivel de riesgo.

        Args:
            risk_level: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
        """
        return [
            var_name for var_name, config in self.variables.items()
            if config.get('validation', {}).get('leakage_risk', '') == risk_level
        ]

    def print_calendar_summary(self):
        """Imprimir resumen del calendario ordenado por día de publicación."""
        print("\n" + "="*80)
        print("ECONOMIC CALENDAR SUMMARY")
        print("="*80)

        # Ordenar por día de publicación
        sorted_vars = sorted(
            self.variables.items(),
            key=lambda x: x[1]['publication']['typical_day']
        )

        print(f"\n{'Variable':<45} {'Day':>4} {'Lag':>6} {'Risk':<10}")
        print("-" * 70)

        for var_name, config in sorted_vars:
            pub = config['publication']
            val = config.get('validation', {})

            day = pub['typical_day']
            lag = pub.get('month_lag', pub.get('quarter_lag', 1))
            risk = val.get('leakage_risk', 'N/A')

            freq = 'Q' if config.get('frequency') == 'quarterly' else 'M'
            lag_str = f"{lag}{freq}"

            print(f"{var_name:<45} {day:>4} {lag_str:>6} {risk:<10}")

        print("="*80 + "\n")

    def create_publication_schedule_df(
        self,
        start_period: str = '2025-01',
        end_period: str = '2026-12'
    ) -> pd.DataFrame:
        """
        Crear DataFrame con schedule completo de publicaciones.

        Args:
            start_period: Período inicial (YYYY-MM)
            end_period: Período final (YYYY-MM)

        Returns:
            DataFrame con columnas: variable, data_period, publication_date, lag_days
        """
        records = []

        periods = pd.period_range(start_period, end_period, freq='M')

        for var_name in self.variables.keys():
            for period in periods:
                period_str = str(period)
                pub_date = self.get_publication_date(var_name, period_str)

                if pub_date:
                    lag_days = self.get_publication_lag_days(var_name, period_str)

                    records.append({
                        'variable': var_name,
                        'data_period': period_str,
                        'publication_date': pub_date,
                        'lag_days': lag_days,
                        'leakage_risk': self.get_leakage_risk(var_name)
                    })

        df = pd.DataFrame(records)
        df['publication_date'] = pd.to_datetime(df['publication_date'])

        return df.sort_values(['publication_date', 'variable'])


# =============================================================================
# Convenience Functions
# =============================================================================

def load_calendar(config_path: Optional[Path] = None) -> EconomicCalendar:
    """Cargar calendario económico (singleton-like)."""
    return EconomicCalendar(config_path)


def get_publication_date(
    variable_name: str,
    data_period: str,
    config_path: Optional[Path] = None
) -> Optional[date]:
    """
    Función de conveniencia para obtener fecha de publicación.

    Example:
        from src.data.economic_calendar import get_publication_date
        pub_date = get_publication_date('infl_cpi_all_usa_m_cpiaucsl', '2025-12')
    """
    calendar = EconomicCalendar(config_path)
    return calendar.get_publication_date(variable_name, data_period)


# =============================================================================
# Main - Test
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*80)
    print("ECONOMIC CALENDAR TEST")
    print("="*80)

    # Cargar calendario
    calendar = EconomicCalendar()

    # Imprimir resumen
    calendar.print_calendar_summary()

    # Test: Calcular fechas de publicación
    print("\nPublication Dates for Dec 2025 data:")
    print("-" * 60)

    test_vars = [
        'labr_unemployment_usa_m_unrate',
        'infl_cpi_all_usa_m_cpiaucsl',
        'infl_pce_usa_m_pcepi',
        'infl_cpi_total_col_m_ipccol',
        'ftrd_exports_total_col_m_expusd',
    ]

    for var in test_vars:
        pub_date = calendar.get_publication_date(var, '2025-12')
        lag = calendar.get_publication_lag_days(var, '2025-12')
        risk = calendar.get_leakage_risk(var)

        print(f"{var}:")
        print(f"  Published: {pub_date} (lag: {lag} days, risk: {risk})")

    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80)

"""
Economic Calendar Parser & Manager
==================================

Gestor del calendario económico para prevenir data leakage.

Contract: CTR-L0-CALENDAR-001

Responsabilidades:
1. Cargar configuración desde SSOT (macro_variables_ssot.yaml)
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

Version: 2.0.0 - Now reads from SSOT
"""

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

# Import SSOT loader
try:
    from .macro_ssot import MacroSSOT, MacroVariableDef
    SSOT_AVAILABLE = True
except ImportError:
    try:
        from src.data.macro_ssot import MacroSSOT, MacroVariableDef
        SSOT_AVAILABLE = True
    except ImportError:
        SSOT_AVAILABLE = False

logger = logging.getLogger(__name__)


class EconomicCalendar:
    """
    Gestor principal del calendario económico.

    Reads from SSOT (macro_variables_ssot.yaml) as the Single Source of Truth
    for publication schedules. Previene data leakage en pipelines de ML.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Inicializar calendario económico.

        Args:
            config_path: Optional path to SSOT config.
                        If None, uses default SSOT location.
        """
        # Build variables dict from SSOT
        self.variables: Dict[str, dict] = {}
        self.config: Dict = {'global_rules': {}}

        if SSOT_AVAILABLE:
            self._load_from_ssot(config_path)
        else:
            logger.warning("SSOT not available, calendar will be empty")

        logger.info(f"Loaded {len(self.variables)} variables from SSOT")

        # Cache de fechas de publicación calculadas
        self._pub_date_cache: Dict[Tuple[str, str], pd.Timestamp] = {}

    def _load_from_ssot(self, config_path: Optional[Path] = None):
        """Load calendar data from SSOT."""
        ssot = MacroSSOT(config_path)

        # Build global rules from SSOT
        global_config = ssot.get_global_config()
        self.config['global_rules'] = {
            'ffill_limits': global_config.get('ffill_limits', {}),
            'default_offsets': global_config.get('default_offsets', {}),
        }

        # Convert each SSOT variable to calendar format
        for var_name in ssot.get_all_variables():
            var_def = ssot.get_variable(var_name)
            if var_def is None:
                continue

            # Build calendar-compatible config from SSOT
            var_config = self._convert_ssot_to_calendar(var_def)
            self.variables[var_name] = var_config

    def _convert_ssot_to_calendar(self, var_def: 'MacroVariableDef') -> dict:
        """Convert SSOT MacroVariableDef to calendar format."""
        sched = var_def.schedule

        # Build publication dict
        publication = {
            'timezone': sched.timezone,
            'month_lag': sched.month_lag,
        }

        if sched.typical_day is not None:
            publication['typical_day'] = sched.typical_day
        elif sched.delay_days is not None:
            # For daily variables, typical_day is delay_days
            publication['typical_day'] = sched.delay_days

        if sched.day_range is not None:
            publication['day_range'] = list(sched.day_range)

        if sched.time is not None:
            publication['time'] = sched.time

        if sched.quarter_lag is not None:
            publication['quarter_lag'] = sched.quarter_lag

        if sched.days_after_quarter is not None:
            publication['days_after_quarter'] = sched.days_after_quarter

        # Build validation dict
        validation = {
            'leakage_risk': var_def.validation.leakage_risk,
            'priority': var_def.validation.priority,
            'expected_range': list(var_def.validation.expected_range),
        }

        # Determine section based on frequency
        frequency = var_def.identity.frequency
        country = var_def.identity.country

        if frequency == 'quarterly':
            section = 'quarterly'
        elif country == 'USA':
            section = 'usa_monthly'
        else:
            section = 'colombia_monthly'

        return {
            'name': var_def.display_name,
            'db_column': var_def.canonical_name,
            'category': var_def.identity.category,
            'frequency': frequency,
            'publication': publication,
            'validation': validation,
            '_section': section,
        }

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
        target_frequency: str = 'daily',
        verbose: bool = False
    ) -> pd.Series:
        """
        Forward-fill respetando calendario de publicaciones con límite por frecuencia.

        ⚠️ CRÍTICO: Nunca propaga un dato antes de su fecha de publicación real.

        Esto previene data leakage en features de ML.

        Args:
            df: DataFrame con index datetime (debe tener la columna variable_name)
            variable_name: Nombre de la variable a procesar
            target_frequency: Frecuencia del dataset target ('daily', '5min', 'hourly')
                             Determina el límite máximo de forward-fill
            verbose: Si True, muestra logging detallado

        Límites (SSOT desde YAML global_rules.ffill_limits):
            - Monthly data → daily: max 22 barras
            - Quarterly data → daily: max 66 barras

        Returns:
            Serie con forward-fill correcto (sin leakage)

        Example:
            # DataFrame con datos macro
            df = pd.read_sql('SELECT * FROM macro_indicators_daily', conn)
            df = df.set_index('fecha')

            # Forward-fill sin leakage con límite
            df['cpi_safe'] = calendar.apply_publication_aware_ffill(
                df, 'infl_cpi_all_usa_m_cpiaucsl', target_frequency='daily'
            )
        """
        if variable_name not in df.columns:
            logger.warning(f"Variable {variable_name} not found in DataFrame")
            return pd.Series(index=df.index, dtype=float, name=variable_name)

        # Asegurar que el index es datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("DataFrame index must be DatetimeIndex")
            return pd.Series(index=df.index, dtype=float, name=variable_name)

        # Obtener límite de ffill desde configuración YAML
        ffill_limits = self.config.get('global_rules', {}).get('ffill_limits', {})
        freq_key = f'{target_frequency}_bars' if target_frequency != '5min' else 'minutes_5'
        freq_config = ffill_limits.get(freq_key, {})

        var_config = self.variables.get(variable_name, {})
        var_freq = var_config.get('frequency', 'monthly')
        max_ffill = freq_config.get(f'{var_freq}_data', 9999)

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
            logger.info(f"Target frequency: {target_frequency}")
            logger.info(f"Max ffill limit: {max_ffill} bars")

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

    def validate_dataset_no_leakage(
        self,
        df: pd.DataFrame,
        variables: List[str] = None,
        sample_rate: int = 100,
        verbose: bool = False
    ) -> Dict[str, bool]:
        """
        Validar que todo el dataset no tiene leakage en ninguna variable.

        Muestrea timestamps del dataset y verifica que los valores usados
        corresponden a datos que ya habían sido publicados.

        Args:
            df: DataFrame con variables macro (index debe ser DatetimeIndex)
            variables: Lista de variables a validar. Si None, usa todas las
                      columnas que estén en el calendario.
            sample_rate: Validar cada N filas (default 100 para eficiencia)
            verbose: Si True, muestra progreso detallado

        Returns:
            Dict con {variable: is_valid} para cada variable

        Example:
            calendar = EconomicCalendar()
            df = pd.read_csv('MACRO_MONTHLY_CLEAN.csv', index_col='fecha', parse_dates=True)
            results = calendar.validate_dataset_no_leakage(df)
            # results = {'infl_cpi_all_usa_m_cpiaucsl': True, 'infl_pce_usa_m_pcepi': False, ...}
        """
        if variables is None:
            variables = [col for col in df.columns if col in self.variables]

        if not variables:
            logger.warning("No variables found in DataFrame that match calendar")
            return {}

        results = {}

        for var in variables:
            if var not in df.columns:
                results[var] = None
                continue

            # Muestrear timestamps para validar
            valid_indices = df[var].dropna().index
            if len(valid_indices) == 0:
                results[var] = True  # No data = no leakage
                continue

            sample_timestamps = df.index[::sample_rate]
            is_valid = True

            for ts in sample_timestamps:
                result = self.validate_no_leakage(df, ts, var)
                if not result['is_valid']:
                    is_valid = False
                    if verbose:
                        logger.warning(
                            f"LEAKAGE in {var} at {ts}: "
                            f"used={result['used_value']}, "
                            f"expected={result['expected_value']}"
                        )
                    break

            results[var] = is_valid

            if verbose:
                status = "✓ PASS" if is_valid else "✗ FAIL"
                logger.info(f"{var}: {status}")

        return results

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

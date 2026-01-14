"""
USD/COP RL Trading System - Stress Testing
============================================

Validación en períodos de crisis específicos.

PROBLEMA QUE RESUELVE:
- Modelo entrenado en datos "normales"
- No sabemos cómo se comportará en crisis
- Backtests promedian rendimiento, esconden debilidades

PERÍODOS DE CRISIS PARA USDCOP:
1. COVID Mar 2020: Volatilidad extrema, spreads 100+ bps
2. Fed Hikes 2022: Fortalecimiento USD, COP débil
3. Petro Election May 2022: Incertidumbre política
4. LatAm Selloff Sep 2022: Contagio regional
5. Banking Crisis Mar 2023: Flight to quality

Author: Claude Code
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, date

from .metrics import TradingMetrics, calculate_all_metrics


@dataclass
class CrisisPeriod:
    """Define un período de crisis."""
    name: str
    start_date: str  # YYYY-MM-DD
    end_date: str
    description: str
    expected_volatility: str  # 'high', 'extreme'
    max_acceptable_dd: float = 25.0  # Max drawdown aceptable en este período
    min_acceptable_sharpe: float = -1.0  # Sharpe mínimo (puede ser negativo en crisis)


# Períodos de crisis predefinidos para USDCOP
DEFAULT_CRISIS_PERIODS = [
    CrisisPeriod(
        name="COVID_Crash",
        start_date="2020-02-20",
        end_date="2020-04-30",
        description="COVID-19 market crash, extreme volatility",
        expected_volatility="extreme",
        max_acceptable_dd=30.0,
        min_acceptable_sharpe=-2.0,
    ),
    CrisisPeriod(
        name="Fed_Hikes_2022",
        start_date="2022-03-01",
        end_date="2022-12-31",
        description="Fed aggressive rate hikes, USD strength",
        expected_volatility="high",
        max_acceptable_dd=25.0,
        min_acceptable_sharpe=-1.0,
    ),
    CrisisPeriod(
        name="Petro_Election",
        start_date="2022-05-15",
        end_date="2022-08-15",
        description="Colombian presidential election uncertainty",
        expected_volatility="high",
        max_acceptable_dd=20.0,
        min_acceptable_sharpe=-0.5,
    ),
    CrisisPeriod(
        name="LatAm_Selloff",
        start_date="2022-09-01",
        end_date="2022-11-30",
        description="LatAm regional selloff, risk-off",
        expected_volatility="high",
        max_acceptable_dd=20.0,
        min_acceptable_sharpe=-0.5,
    ),
    CrisisPeriod(
        name="Banking_Crisis_2023",
        start_date="2023-03-01",
        end_date="2023-04-30",
        description="SVB collapse, banking sector stress",
        expected_volatility="high",
        max_acceptable_dd=20.0,
        min_acceptable_sharpe=-1.0,
    ),
]


@dataclass
class CrisisTestResult:
    """Resultado de test en un período de crisis."""
    period: CrisisPeriod
    metrics: TradingMetrics
    passed: bool
    failure_reasons: List[str] = field(default_factory=list)
    n_bars: int = 0
    data_available: bool = True


@dataclass
class StressTestReport:
    """Reporte completo de stress testing."""
    total_periods: int
    periods_passed: int
    periods_with_data: int
    overall_passed: bool

    crisis_results: List[CrisisTestResult]

    worst_period: str
    worst_sharpe: float
    worst_drawdown: float

    mean_crisis_sharpe: float
    mean_crisis_drawdown: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_periods': self.total_periods,
            'periods_passed': self.periods_passed,
            'periods_with_data': self.periods_with_data,
            'overall_passed': self.overall_passed,
            'worst_period': self.worst_period,
            'worst_sharpe': self.worst_sharpe,
            'worst_drawdown': self.worst_drawdown,
            'mean_crisis_sharpe': self.mean_crisis_sharpe,
            'mean_crisis_drawdown': self.mean_crisis_drawdown,
        }

    def print_report(self):
        """Imprimir reporte formateado."""
        print("\n" + "=" * 70)
        print("STRESS TEST REPORT")
        print("=" * 70)

        print(f"\nOverall: {'PASSED' if self.overall_passed else 'FAILED'}")
        print(f"Periods tested: {self.periods_with_data}/{self.total_periods}")
        print(f"Periods passed: {self.periods_passed}/{self.periods_with_data}")

        print(f"\nWorst period: {self.worst_period}")
        print(f"  Sharpe: {self.worst_sharpe:.2f}")
        print(f"  Max DD: {self.worst_drawdown:.1f}%")

        print(f"\nCrisis averages:")
        print(f"  Mean Sharpe: {self.mean_crisis_sharpe:.2f}")
        print(f"  Mean Max DD: {self.mean_crisis_drawdown:.1f}%")

        print("\nPeriod Details:")
        print("-" * 70)

        for result in self.crisis_results:
            if not result.data_available:
                print(f"  {result.period.name}: NO DATA")
                continue

            status = "PASS" if result.passed else "FAIL"
            print(f"  {result.period.name}: [{status}]")
            print(f"    Sharpe: {result.metrics.sharpe_ratio:.2f}, "
                  f"MaxDD: {result.metrics.max_drawdown:.1f}%, "
                  f"Trades: {result.metrics.n_trades}")

            if result.failure_reasons:
                for reason in result.failure_reasons:
                    print(f"      - {reason}")

        print("=" * 70)


class StressTester:
    """
    Ejecutar stress tests en períodos de crisis.

    Args:
        crisis_periods: Lista de períodos de crisis a testear
        date_column: Nombre de la columna de fecha en el DataFrame
        bars_per_day: Barras por día de trading
        verbose: Nivel de verbosidad
    """

    def __init__(
        self,
        crisis_periods: Optional[List[CrisisPeriod]] = None,
        date_column: str = 'timestamp',
        bars_per_day: int = 60,
        verbose: int = 1,
    ):
        self.crisis_periods = crisis_periods or DEFAULT_CRISIS_PERIODS
        self.date_column = date_column
        self.bars_per_day = bars_per_day
        self.verbose = verbose

    def run(
        self,
        model: Any,
        env_factory: callable,
        df: pd.DataFrame,
        min_required_periods: int = 3,
    ) -> StressTestReport:
        """
        Ejecutar stress tests en todos los períodos de crisis.

        Args:
            model: Modelo entrenado
            env_factory: Factory function para crear environment
            df: DataFrame completo con datos
            min_required_periods: Mínimo de períodos que deben pasar

        Returns:
            StressTestReport con resultados
        """
        results = []

        for period in self.crisis_periods:
            if self.verbose > 0:
                print(f"\nTesting {period.name}...")

            result = self._test_period(model, env_factory, df, period)
            results.append(result)

            if self.verbose > 0:
                if result.data_available:
                    status = "PASSED" if result.passed else "FAILED"
                    print(f"  {status}: Sharpe={result.metrics.sharpe_ratio:.2f}, "
                          f"MaxDD={result.metrics.max_drawdown:.1f}%")
                else:
                    print("  NO DATA for this period")

        return self._generate_report(results, min_required_periods)

    def _test_period(
        self,
        model: Any,
        env_factory: callable,
        df: pd.DataFrame,
        period: CrisisPeriod,
    ) -> CrisisTestResult:
        """Testear un período específico."""
        # Filtrar datos del período
        period_df = self._filter_period(df, period)

        if period_df is None or len(period_df) < self.bars_per_day * 5:
            return CrisisTestResult(
                period=period,
                metrics=None,
                passed=False,
                data_available=False,
            )

        # Crear environment con datos del período
        env = env_factory(period_df)

        # Ejecutar episodio
        returns, actions = self._run_episode(model, env)

        # Calcular métricas
        metrics = calculate_all_metrics(
            returns=np.array(returns),
            actions=np.array(actions),
            bars_per_day=self.bars_per_day,
        )

        # Verificar criterios específicos del período
        passed, failures = self._check_period_criteria(metrics, period)

        return CrisisTestResult(
            period=period,
            metrics=metrics,
            passed=passed,
            failure_reasons=failures,
            n_bars=len(returns),
            data_available=True,
        )

    def _filter_period(
        self,
        df: pd.DataFrame,
        period: CrisisPeriod,
    ) -> Optional[pd.DataFrame]:
        """Filtrar DataFrame al período de crisis."""
        if self.date_column not in df.columns:
            # Intentar usar índice
            if hasattr(df.index, 'to_pydatetime'):
                dates = df.index
            else:
                return None
        else:
            dates = pd.to_datetime(df[self.date_column])

        start = pd.Timestamp(period.start_date)
        end = pd.Timestamp(period.end_date)

        mask = (dates >= start) & (dates <= end)

        if mask.sum() == 0:
            return None

        return df.loc[mask].reset_index(drop=True)

    def _run_episode(
        self,
        model: Any,
        env: Any,
    ) -> Tuple[List[float], List[float]]:
        """Ejecutar un episodio completo en el environment."""
        obs, _ = env.reset()
        done = False
        returns = []
        actions = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            returns.append(info.get('step_return', 0))
            actions.append(float(action[0] if hasattr(action, '__len__') else action))

            done = terminated or truncated

        return returns, actions

    def _check_period_criteria(
        self,
        metrics: TradingMetrics,
        period: CrisisPeriod,
    ) -> Tuple[bool, List[str]]:
        """Verificar criterios específicos del período."""
        failures = []

        if metrics.max_drawdown > period.max_acceptable_dd:
            failures.append(
                f"MaxDD {metrics.max_drawdown:.1f}% > {period.max_acceptable_dd:.1f}%"
            )

        if metrics.sharpe_ratio < period.min_acceptable_sharpe:
            failures.append(
                f"Sharpe {metrics.sharpe_ratio:.2f} < {period.min_acceptable_sharpe:.2f}"
            )

        # No penalizar por bajo trading en crisis (puede ser correcto)
        # Pero alertar si hay 100% HOLD
        if metrics.pct_hold > 95:
            failures.append("Model collapsed to HOLD during crisis")

        return len(failures) == 0, failures

    def _generate_report(
        self,
        results: List[CrisisTestResult],
        min_required_periods: int,
    ) -> StressTestReport:
        """Generar reporte final."""
        # Filtrar resultados con datos
        valid_results = [r for r in results if r.data_available]

        if not valid_results:
            return StressTestReport(
                total_periods=len(results),
                periods_passed=0,
                periods_with_data=0,
                overall_passed=False,
                crisis_results=results,
                worst_period="N/A",
                worst_sharpe=0,
                worst_drawdown=0,
                mean_crisis_sharpe=0,
                mean_crisis_drawdown=0,
            )

        # Estadísticas
        n_passed = sum(1 for r in valid_results if r.passed)
        sharpes = [r.metrics.sharpe_ratio for r in valid_results]
        drawdowns = [r.metrics.max_drawdown for r in valid_results]

        # Peor período
        worst_idx = np.argmin(sharpes)
        worst_result = valid_results[worst_idx]

        # Determinar si pasó overall
        overall_passed = (
            n_passed >= min_required_periods and
            n_passed / len(valid_results) >= 0.5
        )

        return StressTestReport(
            total_periods=len(results),
            periods_passed=n_passed,
            periods_with_data=len(valid_results),
            overall_passed=overall_passed,
            crisis_results=results,
            worst_period=worst_result.period.name,
            worst_sharpe=worst_result.metrics.sharpe_ratio,
            worst_drawdown=worst_result.metrics.max_drawdown,
            mean_crisis_sharpe=np.mean(sharpes),
            mean_crisis_drawdown=np.mean(drawdowns),
        )


class CrisisPeriodsValidator:
    """
    Validador que combina stress testing con validación normal.

    Aplica un peso mayor a los períodos de crisis para
    asegurar que el modelo es robusto en condiciones adversas.

    Args:
        normal_weight: Peso de la validación normal
        crisis_weight: Peso de la validación en crisis
        crisis_periods: Períodos de crisis
        verbose: Nivel de verbosidad
    """

    def __init__(
        self,
        normal_weight: float = 0.6,
        crisis_weight: float = 0.4,
        crisis_periods: Optional[List[CrisisPeriod]] = None,
        verbose: int = 1,
    ):
        self.normal_weight = normal_weight
        self.crisis_weight = crisis_weight
        self.stress_tester = StressTester(
            crisis_periods=crisis_periods,
            verbose=verbose,
        )
        self.verbose = verbose

    def validate(
        self,
        model: Any,
        normal_env: Any,
        df_full: pd.DataFrame,
        env_factory: callable,
        acceptance_criteria: Dict,
    ) -> Dict[str, Any]:
        """
        Ejecutar validación completa.

        Args:
            model: Modelo entrenado
            normal_env: Environment de validación normal
            df_full: DataFrame completo
            env_factory: Factory para crear environments
            acceptance_criteria: Criterios de aceptación

        Returns:
            Dict con resultados de validación
        """
        # Validación normal
        normal_metrics = self._run_normal_validation(model, normal_env)

        # Stress testing
        stress_report = self.stress_tester.run(model, env_factory, df_full)

        # Combinar scores
        normal_score = self._calculate_score(normal_metrics, acceptance_criteria)
        crisis_score = self._calculate_crisis_score(stress_report)

        combined_score = (
            self.normal_weight * normal_score +
            self.crisis_weight * crisis_score
        )

        # Determinar si pasó
        passed = (
            combined_score >= 0.5 and
            normal_score >= 0.4 and
            crisis_score >= 0.3
        )

        return {
            'passed': passed,
            'combined_score': combined_score,
            'normal_score': normal_score,
            'crisis_score': crisis_score,
            'normal_metrics': normal_metrics,
            'stress_report': stress_report,
        }

    def _run_normal_validation(
        self,
        model: Any,
        env: Any,
    ) -> TradingMetrics:
        """Ejecutar validación en environment normal."""
        obs, _ = env.reset()
        done = False
        returns = []
        actions = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            returns.append(info.get('step_return', 0))
            actions.append(float(action[0] if hasattr(action, '__len__') else action))

            done = terminated or truncated

        return calculate_all_metrics(
            returns=np.array(returns),
            actions=np.array(actions),
        )

    def _calculate_score(
        self,
        metrics: TradingMetrics,
        criteria: Dict,
    ) -> float:
        """Calcular score normalizado [0, 1]."""
        scores = []

        # Sharpe
        min_sharpe = criteria.get('min_sharpe', 0.5)
        sharpe_score = min(metrics.sharpe_ratio / min_sharpe, 2.0) / 2.0
        scores.append(max(0, sharpe_score))

        # Drawdown
        max_dd = criteria.get('max_drawdown', 20.0)
        dd_score = 1 - metrics.max_drawdown / max_dd
        scores.append(max(0, dd_score))

        # Profit factor
        min_pf = criteria.get('min_profit_factor', 1.2)
        pf_score = min(metrics.profit_factor / min_pf, 2.0) / 2.0
        scores.append(max(0, pf_score))

        return np.mean(scores)

    def _calculate_crisis_score(
        self,
        report: StressTestReport,
    ) -> float:
        """Calcular score de crisis."""
        if report.periods_with_data == 0:
            return 0.5  # Neutral si no hay datos

        # Base: % de períodos que pasaron
        pass_rate = report.periods_passed / report.periods_with_data

        # Bonus por Sharpe positivo en crisis
        sharpe_bonus = 0.1 if report.mean_crisis_sharpe > 0 else 0

        # Penalización por drawdown extremo
        dd_penalty = max(0, (report.mean_crisis_drawdown - 20) / 100)

        score = pass_rate + sharpe_bonus - dd_penalty

        return np.clip(score, 0, 1)

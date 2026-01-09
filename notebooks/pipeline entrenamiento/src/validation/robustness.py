"""
USD/COP RL Trading System - Robustness Validation
===================================================

Multi-seed training y bootstrap confidence intervals.

PROBLEMA QUE RESUELVE:
- Un solo seed puede dar resultados por suerte
- Sin intervalos de confianza no sabemos la incertidumbre
- Hiperparámetros pueden estar overfitted a un seed

SOLUCIÓN:
- Multi-seed training (5 seeds mínimo)
- Bootstrap CI para métricas
- Consensus requirements: 3/5 folds deben ser positivos

Author: Claude Code
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from .metrics import TradingMetrics, calculate_all_metrics, MetricsAggregator


@dataclass
class SeedResult:
    """Resultado de training con un seed específico."""
    seed: int
    metrics: TradingMetrics
    model_path: Optional[str] = None
    training_history: Optional[Dict] = None
    passed_acceptance: bool = False
    failure_reasons: Optional[List[str]] = None


@dataclass
class RobustnessReport:
    """Reporte de robustez multi-seed."""
    seeds_tested: int
    seeds_passed: int
    consensus_achieved: bool
    consensus_requirement: float

    mean_sharpe: float
    std_sharpe: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float

    mean_max_dd: float
    std_max_dd: float

    mean_calmar: float
    std_calmar: float

    all_results: List[SeedResult]
    best_seed: int
    best_sharpe: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'seeds_tested': self.seeds_tested,
            'seeds_passed': self.seeds_passed,
            'consensus_achieved': self.consensus_achieved,
            'mean_sharpe': self.mean_sharpe,
            'std_sharpe': self.std_sharpe,
            'sharpe_ci': (self.sharpe_ci_lower, self.sharpe_ci_upper),
            'mean_max_dd': self.mean_max_dd,
            'mean_calmar': self.mean_calmar,
            'best_seed': self.best_seed,
            'best_sharpe': self.best_sharpe,
        }


class MultiSeedTrainer:
    """
    Entrenador multi-seed para validación de robustez.

    CONFIGURACIÓN RECOMENDADA:
    - n_seeds: 5 (mínimo para estadísticas)
    - consensus_ratio: 0.6 (3/5 deben pasar)
    - parallel: True si tienes múltiples CPUs

    Args:
        train_fn: Función de training que acepta (config, seed) -> (model, metrics)
        evaluate_fn: Función de evaluación (model, eval_env) -> metrics
        n_seeds: Número de seeds a probar
        base_seed: Seed base (se usa base_seed + i)
        consensus_ratio: Fracción de seeds que deben pasar
        parallel: Si ejecutar en paralelo
        max_workers: Máximo de workers paralelos
        verbose: Nivel de verbosidad
    """

    def __init__(
        self,
        train_fn: Callable,
        evaluate_fn: Callable,
        n_seeds: int = 5,
        base_seed: int = 42,
        consensus_ratio: float = 0.6,
        parallel: bool = False,
        max_workers: int = 4,
        verbose: int = 1,
    ):
        self.train_fn = train_fn
        self.evaluate_fn = evaluate_fn
        self.n_seeds = n_seeds
        self.base_seed = base_seed
        self.consensus_ratio = consensus_ratio
        self.parallel = parallel
        self.max_workers = max_workers
        self.verbose = verbose

    def run(
        self,
        config: Any,
        eval_env: Any,
        acceptance_criteria: Optional[Dict] = None,
    ) -> RobustnessReport:
        """
        Ejecutar training multi-seed.

        Args:
            config: Configuración de training
            eval_env: Environment de evaluación
            acceptance_criteria: Criterios de aceptación

        Returns:
            RobustnessReport con resultados
        """
        if acceptance_criteria is None:
            acceptance_criteria = {
                'min_sharpe': 0.5,
                'max_drawdown': 20.0,
                'min_calmar': 0.3,
                'min_profit_factor': 1.1,
            }

        seeds = [self.base_seed + i for i in range(self.n_seeds)]
        results = []

        if self.parallel and self.n_seeds > 1:
            results = self._run_parallel(seeds, config, eval_env, acceptance_criteria)
        else:
            results = self._run_sequential(seeds, config, eval_env, acceptance_criteria)

        # Generar reporte
        return self._generate_report(results, acceptance_criteria)

    def _run_sequential(
        self,
        seeds: List[int],
        config: Any,
        eval_env: Any,
        acceptance_criteria: Dict,
    ) -> List[SeedResult]:
        """Ejecutar training secuencialmente."""
        results = []

        for seed in seeds:
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"Training with seed {seed}")
                print(f"{'='*60}")

            try:
                result = self._train_single_seed(
                    seed, config, eval_env, acceptance_criteria
                )
                results.append(result)

                if self.verbose > 0:
                    status = "PASSED" if result.passed_acceptance else "FAILED"
                    print(f"Seed {seed}: {status}, Sharpe={result.metrics.sharpe_ratio:.3f}")

            except Exception as e:
                warnings.warn(f"Seed {seed} failed: {e}")
                continue

        return results

    def _run_parallel(
        self,
        seeds: List[int],
        config: Any,
        eval_env: Any,
        acceptance_criteria: Dict,
    ) -> List[SeedResult]:
        """Ejecutar training en paralelo."""
        results = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._train_single_seed, seed, config, eval_env, acceptance_criteria
                ): seed
                for seed in seeds
            }

            for future in as_completed(futures):
                seed = futures[future]
                try:
                    result = future.result()
                    results.append(result)

                    if self.verbose > 0:
                        status = "PASSED" if result.passed_acceptance else "FAILED"
                        print(f"Seed {seed}: {status}, Sharpe={result.metrics.sharpe_ratio:.3f}")

                except Exception as e:
                    warnings.warn(f"Seed {seed} failed: {e}")

        return results

    def _train_single_seed(
        self,
        seed: int,
        config: Any,
        eval_env: Any,
        acceptance_criteria: Dict,
    ) -> SeedResult:
        """Entrenar con un seed específico."""
        # Training
        model, training_history = self.train_fn(config, seed)

        # Evaluación
        metrics = self.evaluate_fn(model, eval_env)

        # Verificar criterios
        passed, failures = metrics.passes_acceptance(**acceptance_criteria)

        return SeedResult(
            seed=seed,
            metrics=metrics,
            training_history=training_history,
            passed_acceptance=passed,
            failure_reasons=failures if not passed else None,
        )

    def _generate_report(
        self,
        results: List[SeedResult],
        acceptance_criteria: Dict,
    ) -> RobustnessReport:
        """Generar reporte de robustez."""
        if not results:
            raise ValueError("No successful training runs")

        # Estadísticas de métricas
        sharpes = [r.metrics.sharpe_ratio for r in results]
        max_dds = [r.metrics.max_drawdown for r in results]
        calmars = [r.metrics.calmar_ratio for r in results]

        # Consensus
        n_passed = sum(1 for r in results if r.passed_acceptance)
        consensus_achieved = n_passed / len(results) >= self.consensus_ratio

        # Mejor resultado
        best_result = max(results, key=lambda r: r.metrics.sharpe_ratio)

        # Bootstrap CI para Sharpe
        ci_lower, ci_upper = self._bootstrap_ci(sharpes)

        return RobustnessReport(
            seeds_tested=len(results),
            seeds_passed=n_passed,
            consensus_achieved=consensus_achieved,
            consensus_requirement=self.consensus_ratio,
            mean_sharpe=np.mean(sharpes),
            std_sharpe=np.std(sharpes),
            sharpe_ci_lower=ci_lower,
            sharpe_ci_upper=ci_upper,
            mean_max_dd=np.mean(max_dds),
            std_max_dd=np.std(max_dds),
            mean_calmar=np.mean(calmars),
            std_calmar=np.std(calmars),
            all_results=results,
            best_seed=best_result.seed,
            best_sharpe=best_result.metrics.sharpe_ratio,
        )

    def _bootstrap_ci(
        self,
        values: List[float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Calcular intervalo de confianza por bootstrap."""
        if len(values) < 2:
            return (values[0], values[0]) if values else (0, 0)

        values = np.array(values)
        bootstrap_means = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(sample.mean())

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

        return (lower, upper)


class BootstrapConfidenceInterval:
    """
    Calcular intervalos de confianza por bootstrap para métricas de trading.

    METODOLOGÍA:
    1. Resamplear retornos con reemplazo
    2. Calcular métrica en cada muestra
    3. Obtener percentiles para CI

    Args:
        n_bootstrap: Número de muestras bootstrap
        confidence: Nivel de confianza (0.95 = 95%)
        block_size: Tamaño de bloque para block bootstrap (preserva autocorrelación)
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        block_size: Optional[int] = None,
    ):
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self.block_size = block_size

    def calculate_sharpe_ci(
        self,
        returns: np.ndarray,
        bars_per_day: int = 60,
    ) -> Tuple[float, float, float]:
        """
        Calcular CI para Sharpe ratio.

        Args:
            returns: Array de retornos
            bars_per_day: Barras por día

        Returns:
            Tuple de (point_estimate, lower, upper)
        """
        # Point estimate
        n_days = len(returns) // bars_per_day
        if n_days < 2:
            return (0, 0, 0)

        daily_returns = returns[:n_days * bars_per_day].reshape(
            n_days, bars_per_day
        ).sum(axis=1)

        point_estimate = daily_returns.mean() / daily_returns.std() * np.sqrt(252)

        # Bootstrap
        bootstrap_sharpes = []

        for _ in range(self.n_bootstrap):
            if self.block_size:
                sample_daily = self._block_bootstrap(daily_returns, self.block_size)
            else:
                sample_daily = np.random.choice(daily_returns, size=len(daily_returns), replace=True)

            if sample_daily.std() > 1e-10:
                sharpe = sample_daily.mean() / sample_daily.std() * np.sqrt(252)
                bootstrap_sharpes.append(sharpe)

        if not bootstrap_sharpes:
            return (point_estimate, point_estimate, point_estimate)

        alpha = 1 - self.confidence
        lower = np.percentile(bootstrap_sharpes, alpha / 2 * 100)
        upper = np.percentile(bootstrap_sharpes, (1 - alpha / 2) * 100)

        return (point_estimate, lower, upper)

    def calculate_all_ci(
        self,
        returns: np.ndarray,
        actions: Optional[np.ndarray] = None,
        bars_per_day: int = 60,
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Calcular CI para múltiples métricas.

        Returns:
            Dict con (point_estimate, lower, upper) para cada métrica
        """
        results = {}

        # Sharpe
        results['sharpe'] = self.calculate_sharpe_ci(returns, bars_per_day)

        # Sortino
        results['sortino'] = self._bootstrap_metric(
            returns, lambda r: self._calc_sortino(r, bars_per_day)
        )

        # Max Drawdown
        results['max_drawdown'] = self._bootstrap_metric(
            returns, lambda r: self._calc_max_dd(r)
        )

        # Total Return
        results['total_return'] = self._bootstrap_metric(
            returns, lambda r: (1 + r).prod() - 1
        )

        return results

    def _bootstrap_metric(
        self,
        data: np.ndarray,
        metric_fn: Callable,
    ) -> Tuple[float, float, float]:
        """Bootstrap genérico para una métrica."""
        point_estimate = metric_fn(data)

        bootstrap_values = []
        for _ in range(self.n_bootstrap):
            if self.block_size:
                sample = self._block_bootstrap(data, self.block_size)
            else:
                sample = np.random.choice(data, size=len(data), replace=True)

            try:
                value = metric_fn(sample)
                if np.isfinite(value):
                    bootstrap_values.append(value)
            except:
                continue

        if not bootstrap_values:
            return (point_estimate, point_estimate, point_estimate)

        alpha = 1 - self.confidence
        lower = np.percentile(bootstrap_values, alpha / 2 * 100)
        upper = np.percentile(bootstrap_values, (1 - alpha / 2) * 100)

        return (point_estimate, lower, upper)

    def _block_bootstrap(self, data: np.ndarray, block_size: int) -> np.ndarray:
        """Block bootstrap para series temporales."""
        n = len(data)
        n_blocks = int(np.ceil(n / block_size))

        blocks = []
        for _ in range(n_blocks):
            start = np.random.randint(0, n - block_size + 1)
            blocks.append(data[start:start + block_size])

        result = np.concatenate(blocks)[:n]
        return result

    def _calc_sortino(self, returns: np.ndarray, bars_per_day: int) -> float:
        """Calcular Sortino ratio."""
        n_days = len(returns) // bars_per_day
        if n_days < 2:
            return 0

        daily_returns = returns[:n_days * bars_per_day].reshape(
            n_days, bars_per_day
        ).sum(axis=1)

        downside = daily_returns[daily_returns < 0]
        if len(downside) < 2 or downside.std() < 1e-10:
            return 0

        return daily_returns.mean() / downside.std() * np.sqrt(252)

    def _calc_max_dd(self, returns: np.ndarray) -> float:
        """Calcular max drawdown."""
        pv = (1 + returns).cumprod()
        peak = np.maximum.accumulate(pv)
        dd = (peak - pv) / (peak + 1e-10)
        return dd.max() * 100


def ensemble_predictions(
    models: List[Any],
    obs: np.ndarray,
    method: str = 'mean',
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Combinar predicciones de múltiples modelos.

    Args:
        models: Lista de modelos entrenados
        obs: Observación de entrada
        method: Método de combinación ('mean', 'median', 'weighted', 'vote')
        weights: Pesos para weighted average

    Returns:
        Acción combinada
    """
    predictions = []

    for model in models:
        action, _ = model.predict(obs, deterministic=True)
        predictions.append(action)

    predictions = np.array(predictions)

    if method == 'mean':
        return predictions.mean(axis=0)
    elif method == 'median':
        return np.median(predictions, axis=0)
    elif method == 'weighted' and weights is not None:
        weights = np.array(weights) / np.sum(weights)
        return np.average(predictions, axis=0, weights=weights)
    elif method == 'vote':
        # Discretizar y votar
        discrete = np.sign(predictions)
        return np.sign(discrete.sum(axis=0))
    else:
        return predictions.mean(axis=0)

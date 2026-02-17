"""
L4: Backtest + Promotion Proposal DAG (Primer Voto)
===================================================
Contract: CTR-L4-PROMOTION-001
Version: 1.0.0

Este DAG implementa el PRIMER VOTO del sistema de promoción de doble voto:

1. Recibe modelo entrenado de L3 via XCom
2. Ejecuta backtest out-of-sample en test.parquet (2025-07-01 → HOY)
   - Datos NUNCA vistos durante training (anti-leakage)
3. Evalúa success_criteria del experiment.yaml
4. Compara vs baseline (modelo en producción actual)
5. Genera promotion_proposal con recomendación
6. SIEMPRE requiere aprobación humana (segundo voto en Dashboard)

Data Flow:
    L3 (model.zip) → L4 (backtest + evaluate) → promotion_proposals table
                                               → Dashboard (segundo voto)
                                               → model_registry.stage='production'

Schedule: Triggered after L3 completes successfully

Author: Trading Team
Created: 2026-01-31
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
import hashlib
import logging
import numpy as np

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# DAG Registry
from contracts.dag_registry import (
    RL_L4_BACKTEST_PROMOTION,
    RL_L3_MODEL_TRAINING,
)

# XCom contracts
try:
    from contracts.xcom_contracts import L3Output
    XCOM_CONTRACTS_AVAILABLE = True
except ImportError:
    XCOM_CONTRACTS_AVAILABLE = False
    L3Output = None

# Utils
from utils.dag_common import get_db_connection

logger = logging.getLogger(__name__)
DAG_ID = RL_L4_BACKTEST_PROMOTION


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """
    Motor de backtest para evaluación out-of-sample.

    Simula trading con costos de transacción realistas para USDCOP.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        transaction_cost_bps: float = 75.0,  # USDCOP spread típico
        position_size: float = 1.0,
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost_bps / 10_000
        self.position_size = position_size

    def run(
        self,
        model_path: str,
        test_data_path: str,
        norm_stats_path: str,
    ) -> Tuple[Dict[str, Any], List[Dict]]:
        """
        Ejecutar backtest en datos out-of-sample.

        Args:
            model_path: Path to model.zip
            test_data_path: Path to test.parquet (OOS data)
            norm_stats_path: Path to norm_stats.json

        Returns:
            Tuple of (metrics_dict, detailed_trades_list)
        """
        import pandas as pd

        try:
            from stable_baselines3 import PPO
        except ImportError:
            logger.error("stable_baselines3 not available")
            return self._empty_metrics(), []

        # Load model
        try:
            model = PPO.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return self._empty_metrics(), []

        # Load test data
        test_path = Path(test_data_path)
        if not test_path.exists():
            # Try alternative paths
            alt_paths = [
                Path(f"data/pipeline/07_output/5min/{test_path.name}"),
                PROJECT_ROOT / "data" / "pipeline" / "07_output" / "5min" / test_path.name,
            ]
            for alt in alt_paths:
                if alt.exists():
                    test_path = alt
                    break

        if not test_path.exists():
            logger.error(f"Test data not found: {test_data_path}")
            return self._empty_metrics(), []

        try:
            df = pd.read_parquet(test_path)
            logger.info(f"Loaded {len(df)} test samples from {test_path}")
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return self._empty_metrics(), []

        # Load norm stats
        norm_stats = {}
        norm_path = Path(norm_stats_path)
        if norm_path.exists():
            with open(norm_path) as f:
                norm_stats = json.load(f)

        # Run simulation
        return self._simulate(model, df, norm_stats)

    def _simulate(
        self,
        model,
        df,
        norm_stats: Dict,
    ) -> Tuple[Dict[str, Any], List[Dict]]:
        """
        Run backtest simulation.

        Returns:
            Tuple of (metrics_dict, detailed_trades_list)
        """
        from src.core.contracts.feature_contract import FEATURE_ORDER

        capital = self.initial_capital
        peak_capital = capital
        position = 0
        entry_price = 0
        entry_time = None
        entry_confidence = None
        entry_raw_action = None
        entry_equity = capital
        trades = []
        detailed_trades = []  # For persistence
        equity_curve = [capital]
        trade_id = 0

        for idx in range(len(df)):
            row = df.iloc[idx]

            # Get timestamp
            if hasattr(df.index, '__getitem__'):
                timestamp = df.index[idx]
            else:
                timestamp = row.get('timestamp', row.get('Timestamp', None))

            # Build observation (15 features)
            obs = self._build_observation(row, position, idx, len(df), FEATURE_ORDER)

            # Normalize
            obs = self._normalize(obs, norm_stats, FEATURE_ORDER)

            # Get action from model
            try:
                action, _ = model.predict(obs, deterministic=True)
                raw_action = float(action[0]) if hasattr(action, '__len__') else float(action)
                signal = self._discretize(raw_action)
                confidence = min(abs(raw_action), 1.0)  # Confidence based on action magnitude
            except Exception as e:
                logger.warning(f"Prediction error at idx {idx}: {e}")
                signal = 0
                raw_action = 0.0
                confidence = 0.0

            # Get close price
            close_price = float(row.get('close', row.get('Close', 4200.0)))

            # Execute trade if signal changed
            if signal != position:
                # Close existing position
                if position != 0:
                    pnl = (close_price - entry_price) * position * self.position_size
                    pnl -= abs(pnl) * self.transaction_cost
                    capital += pnl
                    pnl_pct = pnl / self.initial_capital

                    # Simple trade record
                    trades.append({
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'direction': 'LONG' if position > 0 else 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': close_price,
                    })

                    # Detailed trade for persistence
                    trade_id += 1
                    detailed_trades.append({
                        'trade_id': trade_id,
                        'timestamp': str(timestamp) if timestamp is not None else None,
                        'entry_time': str(entry_time) if entry_time is not None else None,
                        'exit_time': str(timestamp) if timestamp is not None else None,
                        'side': 'LONG' if position > 0 else 'SHORT',
                        'entry_price': float(entry_price),
                        'exit_price': float(close_price),
                        'pnl': float(pnl),
                        'pnl_usd': float(pnl),  # For USDCOP, pnl is already in USD terms
                        'pnl_percent': float(pnl_pct * 100),
                        'status': 'closed',
                        'duration_minutes': self._calc_duration_minutes(entry_time, timestamp),
                        'exit_reason': 'signal',
                        'equity_at_entry': float(entry_equity),
                        'equity_at_exit': float(capital),
                        'entry_confidence': float(entry_confidence) if entry_confidence else 0.0,
                        'exit_confidence': float(confidence),
                        'raw_action': float(raw_action),
                    })

                # Open new position
                if signal != 0:
                    entry_price = close_price
                    entry_time = timestamp
                    entry_confidence = confidence
                    entry_raw_action = raw_action
                    entry_equity = capital
                    # Apply transaction cost on entry
                    capital -= abs(close_price * self.transaction_cost * self.position_size)
                    position = signal
                else:
                    position = 0
                    entry_time = None

                peak_capital = max(peak_capital, capital)

            equity_curve.append(capital)

        # Close any remaining position
        if position != 0 and len(df) > 0:
            last_row = df.iloc[-1]
            close_price = float(last_row.get('close', last_row.get('Close', 4200.0)))

            if hasattr(df.index, '__getitem__'):
                last_timestamp = df.index[-1]
            else:
                last_timestamp = last_row.get('timestamp', last_row.get('Timestamp', None))

            pnl = (close_price - entry_price) * position * self.position_size
            pnl -= abs(pnl) * self.transaction_cost
            capital += pnl
            pnl_pct = pnl / self.initial_capital

            trades.append({
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'direction': 'LONG' if position > 0 else 'SHORT',
            })

            # Detailed trade for remaining position
            trade_id += 1
            detailed_trades.append({
                'trade_id': trade_id,
                'timestamp': str(last_timestamp) if last_timestamp is not None else None,
                'entry_time': str(entry_time) if entry_time is not None else None,
                'exit_time': str(last_timestamp) if last_timestamp is not None else None,
                'side': 'LONG' if position > 0 else 'SHORT',
                'entry_price': float(entry_price),
                'exit_price': float(close_price),
                'pnl': float(pnl),
                'pnl_usd': float(pnl),
                'pnl_percent': float(pnl_pct * 100),
                'status': 'closed',
                'duration_minutes': self._calc_duration_minutes(entry_time, last_timestamp),
                'exit_reason': 'end_of_period',
                'equity_at_entry': float(entry_equity),
                'equity_at_exit': float(capital),
                'entry_confidence': float(entry_confidence) if entry_confidence else 0.0,
                'exit_confidence': 0.0,
                'raw_action': 0.0,
            })

        # Calculate metrics
        metrics = self._calculate_metrics(trades, equity_curve, df)
        return metrics, detailed_trades

    def _calc_duration_minutes(self, entry_time, exit_time) -> Optional[int]:
        """Calculate trade duration in minutes."""
        if entry_time is None or exit_time is None:
            return None
        try:
            import pandas as pd
            entry = pd.Timestamp(entry_time)
            exit_t = pd.Timestamp(exit_time)
            return int((exit_t - entry).total_seconds() / 60)
        except Exception:
            return None

    def _build_observation(
        self,
        row,
        position: float,
        idx: int,
        total: int,
        feature_order: tuple,
    ) -> np.ndarray:
        """Build 15-dim observation from row."""
        obs = []
        for feature in feature_order[:-2]:  # Exclude state features
            val = row.get(feature, 0.0)
            obs.append(float(val) if val is not None and not np.isnan(val) else 0.0)

        # Add state features
        obs.append(float(position))
        obs.append(idx / max(total, 1))  # time_normalized

        return np.array(obs, dtype=np.float32)

    def _normalize(
        self,
        obs: np.ndarray,
        norm_stats: Dict,
        feature_order: tuple,
    ) -> np.ndarray:
        """Apply normalization using stored stats."""
        normalized = []
        for i, feature in enumerate(feature_order):
            if feature in norm_stats:
                stats = norm_stats[feature]
                mean = stats.get('mean', 0)
                std = stats.get('std', 1)
                normalized.append((obs[i] - mean) / std if std > 0 else 0.0)
            else:
                normalized.append(obs[i])

        return np.array(normalized, dtype=np.float32)

    def _discretize(self, action: float) -> int:
        """Convert continuous action to discrete position."""
        if action > 0.33:
            return 1  # LONG
        elif action < -0.33:
            return -1  # SHORT
        return 0  # HOLD

    def _calculate_metrics(
        self,
        trades: List[Dict],
        equity_curve: List[float],
        df,
    ) -> Dict[str, Any]:
        """Calculate backtest metrics."""
        if not trades:
            return self._empty_metrics()

        returns = [t['pnl_pct'] for t in trades]
        winning = [r for r in returns if r > 0]
        losing = [r for r in returns if r < 0]

        # Sharpe ratio (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Win rate
        win_rate = len(winning) / len(trades) if trades else 0

        # Profit factor
        gross_profit = sum(winning) if winning else 0
        gross_loss = abs(sum(losing)) if losing else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0

        # Test period
        test_start = ""
        test_end = ""
        if len(df) > 0:
            if hasattr(df.index, 'min'):
                test_start = str(df.index.min())
                test_end = str(df.index.max())
            elif 'timestamp' in df.columns:
                test_start = str(df['timestamp'].min())
                test_end = str(df['timestamp'].max())

        return {
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'win_rate': float(win_rate),
            'profit_factor': float(min(profit_factor, 999.0)),
            'total_trades': len(trades),
            'avg_trade_pnl': float(np.mean(returns)) if returns else 0.0,
            'test_period_start': test_start,
            'test_period_end': test_end,
            'final_equity': float(equity_curve[-1]) if equity_curve else self.initial_capital,
            'total_return': float((equity_curve[-1] - self.initial_capital) / self.initial_capital) if equity_curve else 0.0,
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'avg_trade_pnl': 0.0,
            'test_period_start': '',
            'test_period_end': '',
            'final_equity': self.initial_capital,
            'total_return': 0.0,
        }


# =============================================================================
# DAG TASKS
# =============================================================================

def load_l3_output(**context) -> Dict[str, Any]:
    """
    Task 1: Cargar output de L3 vía XCom.

    Obtiene la información del modelo entrenado incluyendo:
    - model_path, model_hash
    - dataset_hash, config_hash
    - experiment_name
    """
    ti = context['ti']
    dag_run_conf = context.get('dag_run').conf or {}

    # Try to get experiment name from dag_run config
    experiment_name = dag_run_conf.get('experiment_name', 'default')

    # Try to pull L3 output from XCom
    l3_data = None

    if XCOM_CONTRACTS_AVAILABLE and L3Output is not None:
        try:
            l3_output = L3Output.pull_from_xcom(ti, dag_id=RL_L3_MODEL_TRAINING)
            if l3_output:
                l3_data = l3_output.to_dict()
        except Exception as e:
            logger.warning(f"Could not pull L3Output: {e}")

    # Fallback: try direct XCom pull
    if l3_data is None:
        l3_data = ti.xcom_pull(dag_id=RL_L3_MODEL_TRAINING, key='l3_output')

    # Fallback: construct from dag_run config
    if l3_data is None:
        l3_data = {
            'model_path': dag_run_conf.get('model_path', f'models/{experiment_name}/model.zip'),
            'model_hash': dag_run_conf.get('model_hash', ''),
            'dataset_hash': dag_run_conf.get('dataset_hash', ''),
            'config_hash': dag_run_conf.get('config_hash', ''),
            'experiment_name': experiment_name,
            'norm_stats_path': dag_run_conf.get('norm_stats_path', f'data/pipeline/07_output/5min/DS_{experiment_name}_norm_stats.json'),
            'test_data_path': dag_run_conf.get('test_data_path', f'data/pipeline/07_output/5min/DS_{experiment_name}_test.parquet'),
        }

    logger.info("=" * 60)
    logger.info("L4 BACKTEST + PROMOTION (PRIMER VOTO)")
    logger.info("=" * 60)
    logger.info(f"Experiment: {l3_data.get('experiment_name')}")
    logger.info(f"Model path: {l3_data.get('model_path')}")
    logger.info(f"Model hash: {l3_data.get('model_hash', 'N/A')}")
    logger.info(f"Dataset hash: {l3_data.get('dataset_hash', 'N/A')}")

    ti.xcom_push(key='l3_data', value=l3_data)
    return l3_data


def run_oos_backtest(**context) -> Dict[str, Any]:
    """
    Task 2: Ejecutar backtest out-of-sample.

    Usa test.parquet que contiene datos de 2025-07-01 → HOY
    (datos NUNCA vistos durante training).
    """
    ti = context['ti']
    l3_data = ti.xcom_pull(task_ids='load_l3_output', key='l3_data')

    experiment_name = l3_data.get('experiment_name', 'default')

    # Paths
    model_path = l3_data.get('model_path', f'models/{experiment_name}/model.zip')
    test_data_path = l3_data.get('test_data_path', f'data/pipeline/07_output/5min/DS_{experiment_name}_test.parquet')
    norm_stats_path = l3_data.get('norm_stats_path', f'data/pipeline/07_output/5min/DS_{experiment_name}_norm_stats.json')

    # Resolve paths
    model_path = str(PROJECT_ROOT / model_path) if not Path(model_path).is_absolute() else model_path
    test_data_path = str(PROJECT_ROOT / test_data_path) if not Path(test_data_path).is_absolute() else test_data_path
    norm_stats_path = str(PROJECT_ROOT / norm_stats_path) if not Path(norm_stats_path).is_absolute() else norm_stats_path

    logger.info(f"Running OOS backtest:")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Test data: {test_data_path}")
    logger.info(f"  Norm stats: {norm_stats_path}")

    # Run backtest
    engine = BacktestEngine()
    metrics, detailed_trades = engine.run(model_path, test_data_path, norm_stats_path)

    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS (Out-of-Sample)")
    logger.info("=" * 60)
    logger.info(f"  Test Period: {metrics.get('test_period_start')} to {metrics.get('test_period_end')}")
    logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
    logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
    logger.info(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    logger.info(f"  Total Trades: {metrics.get('total_trades', 0)}")
    logger.info(f"  Total Return: {metrics.get('total_return', 0):.2%}")
    logger.info(f"  Detailed Trades: {len(detailed_trades)} captured for replay")
    logger.info("=" * 60)

    ti.xcom_push(key='backtest_metrics', value=metrics)
    ti.xcom_push(key='detailed_trades', value=detailed_trades)
    return {'status': 'success', 'sharpe': metrics.get('sharpe_ratio', 0), 'trades_captured': len(detailed_trades)}


def compare_vs_baseline(**context) -> Dict[str, Any]:
    """
    Task 3: Comparar vs modelo baseline (producción actual).
    """
    ti = context['ti']
    l3_data = ti.xcom_pull(task_ids='load_l3_output', key='l3_data')
    new_metrics = ti.xcom_pull(task_ids='run_oos_backtest', key='backtest_metrics')

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Get baseline model (current production)
        cur.execute("""
            SELECT model_id, model_path, metrics
            FROM model_registry
            WHERE stage = 'production' AND is_active = TRUE
            ORDER BY promoted_at DESC LIMIT 1
        """)
        baseline_row = cur.fetchone()

        if not baseline_row:
            logger.info("No baseline model found - this is the first model")
            ti.xcom_push(key='vs_baseline', value=None)
            ti.xcom_push(key='baseline_model_id', value=None)
            return {'has_baseline': False}

        baseline_id = baseline_row[0]
        baseline_metrics = baseline_row[2]
        if isinstance(baseline_metrics, str):
            baseline_metrics = json.loads(baseline_metrics) if baseline_metrics else {}

        # Calculate improvements
        baseline_sharpe = baseline_metrics.get('sharpe_ratio', 0)
        baseline_dd = baseline_metrics.get('max_drawdown', 1)
        new_sharpe = new_metrics.get('sharpe_ratio', 0)
        new_dd = new_metrics.get('max_drawdown', 0)

        sharpe_improvement = (
            (new_sharpe - baseline_sharpe) / baseline_sharpe
            if baseline_sharpe > 0 else 1.0 if new_sharpe > 0 else 0.0
        )
        dd_improvement = (
            (baseline_dd - new_dd) / baseline_dd
            if baseline_dd > 0 else 0.0
        )

        comparison = {
            'baseline_model_id': baseline_id,
            'sharpe_improvement': sharpe_improvement,
            'drawdown_improvement': dd_improvement,
            'baseline_sharpe': baseline_sharpe,
            'baseline_max_dd': baseline_dd,
            'new_sharpe': new_sharpe,
            'new_max_dd': new_dd,
        }

        logger.info(f"Comparison vs baseline ({baseline_id}):")
        logger.info(f"  Sharpe: {new_sharpe:.3f} vs {baseline_sharpe:.3f} ({sharpe_improvement:+.1%})")
        logger.info(f"  MaxDD: {new_dd:.2%} vs {baseline_dd:.2%} ({dd_improvement:+.1%})")

        ti.xcom_push(key='vs_baseline', value=comparison)
        ti.xcom_push(key='baseline_model_id', value=baseline_id)

        return {'has_baseline': True, 'baseline_id': baseline_id}

    finally:
        cur.close()
        conn.close()


def evaluate_criteria(**context) -> Dict[str, Any]:
    """
    Task 4: Evaluar success_criteria del experiment.yaml.
    """
    ti = context['ti']
    l3_data = ti.xcom_pull(task_ids='load_l3_output', key='l3_data')
    metrics = ti.xcom_pull(task_ids='run_oos_backtest', key='backtest_metrics')
    comparison = ti.xcom_pull(task_ids='compare_vs_baseline', key='vs_baseline')

    experiment_name = l3_data.get('experiment_name', 'default')

    # Try to load experiment contract for success criteria
    success_criteria = {
        'min_sharpe': 0.5,
        'max_drawdown': 0.15,
        'min_win_rate': 0.45,
        'min_trades': 50,
        'improvement_threshold': 0.05,
    }

    try:
        from src.core.contracts.experiment_contract import ExperimentContract
        yaml_path = PROJECT_ROOT / "config" / "experiments" / f"{experiment_name}.yaml"
        if yaml_path.exists():
            contract = ExperimentContract.from_yaml(yaml_path)
            success_criteria = contract.get_success_criteria()
            logger.info(f"Loaded success_criteria from {yaml_path}")
    except Exception as e:
        logger.warning(f"Could not load experiment contract: {e}. Using defaults.")

    # Evaluate each criterion
    criteria_results = []

    # Min Sharpe
    passed = metrics.get('sharpe_ratio', 0) >= success_criteria.get('min_sharpe', 0.5)
    criteria_results.append({
        'name': 'min_sharpe',
        'threshold': success_criteria.get('min_sharpe', 0.5),
        'actual': metrics.get('sharpe_ratio', 0),
        'passed': passed,
    })

    # Max Drawdown
    passed = metrics.get('max_drawdown', 1) <= success_criteria.get('max_drawdown', 0.15)
    criteria_results.append({
        'name': 'max_drawdown',
        'threshold': success_criteria.get('max_drawdown', 0.15),
        'actual': metrics.get('max_drawdown', 0),
        'passed': passed,
    })

    # Min Win Rate
    passed = metrics.get('win_rate', 0) >= success_criteria.get('min_win_rate', 0.45)
    criteria_results.append({
        'name': 'min_win_rate',
        'threshold': success_criteria.get('min_win_rate', 0.45),
        'actual': metrics.get('win_rate', 0),
        'passed': passed,
    })

    # Min Trades
    passed = metrics.get('total_trades', 0) >= success_criteria.get('min_trades', 50)
    criteria_results.append({
        'name': 'min_trades',
        'threshold': float(success_criteria.get('min_trades', 50)),
        'actual': float(metrics.get('total_trades', 0)),
        'passed': passed,
    })

    # Improvement threshold (if comparison available)
    if comparison:
        improvement = comparison.get('sharpe_improvement', 0)
        threshold = success_criteria.get('improvement_threshold', 0.05)
        passed = improvement >= threshold
        criteria_results.append({
            'name': 'improvement_threshold',
            'threshold': threshold,
            'actual': improvement,
            'passed': passed,
        })

    all_passed = all(cr['passed'] for cr in criteria_results)

    logger.info("=" * 60)
    logger.info("CRITERIA EVALUATION")
    logger.info("=" * 60)
    for cr in criteria_results:
        status = "PASS" if cr['passed'] else "FAIL"
        logger.info(f"  {cr['name']}: {status} ({cr['actual']:.4f} vs {cr['threshold']:.4f})")
    logger.info(f"  OVERALL: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    logger.info("=" * 60)

    ti.xcom_push(key='criteria_results', value=criteria_results)
    ti.xcom_push(key='all_criteria_passed', value=all_passed)

    return {'all_passed': all_passed, 'criteria_count': len(criteria_results)}


def persist_backtest_trades(
    conn,
    proposal_id: str,
    model_id: str,
    detailed_trades: List[Dict],
) -> int:
    """
    Persist individual trades to backtest_trades table for frontend replay.

    Args:
        conn: Database connection
        proposal_id: The proposal ID to link trades to
        model_id: The model ID
        detailed_trades: List of detailed trade dictionaries

    Returns:
        Number of trades inserted
    """
    if not detailed_trades:
        return 0

    cur = conn.cursor()
    inserted = 0

    try:
        for trade in detailed_trades:
            cur.execute("""
                INSERT INTO backtest_trades (
                    proposal_id, trade_id, model_id,
                    timestamp, entry_time, exit_time,
                    side, entry_price, exit_price,
                    pnl, pnl_usd, pnl_percent,
                    status, duration_minutes, exit_reason,
                    equity_at_entry, equity_at_exit,
                    entry_confidence, exit_confidence, raw_action
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT DO NOTHING
            """, (
                proposal_id,
                trade.get('trade_id'),
                model_id,
                trade.get('timestamp'),
                trade.get('entry_time'),
                trade.get('exit_time'),
                trade.get('side'),
                trade.get('entry_price'),
                trade.get('exit_price'),
                trade.get('pnl'),
                trade.get('pnl_usd'),
                trade.get('pnl_percent'),
                trade.get('status', 'closed'),
                trade.get('duration_minutes'),
                trade.get('exit_reason'),
                trade.get('equity_at_entry'),
                trade.get('equity_at_exit'),
                trade.get('entry_confidence'),
                trade.get('exit_confidence'),
                trade.get('raw_action'),
            ))
            inserted += 1

        conn.commit()
        logger.info(f"Persisted {inserted} trades for proposal {proposal_id}")
        return inserted

    except Exception as e:
        logger.error(f"Failed to persist trades: {e}")
        conn.rollback()
        return 0
    finally:
        cur.close()


def generate_promotion_proposal(**context) -> Dict[str, Any]:
    """
    Task 5: Generar promotion proposal (PRIMER VOTO).

    Esta propuesta se guarda en la base de datos y requiere
    aprobación humana en el Dashboard (segundo voto).

    Also persists individual trades to backtest_trades for exact replay.
    """
    ti = context['ti']
    l3_data = ti.xcom_pull(task_ids='load_l3_output', key='l3_data')
    metrics = ti.xcom_pull(task_ids='run_oos_backtest', key='backtest_metrics')
    detailed_trades = ti.xcom_pull(task_ids='run_oos_backtest', key='detailed_trades') or []
    comparison = ti.xcom_pull(task_ids='compare_vs_baseline', key='vs_baseline')
    criteria_results = ti.xcom_pull(task_ids='evaluate_criteria', key='criteria_results')
    all_passed = ti.xcom_pull(task_ids='evaluate_criteria', key='all_criteria_passed')
    baseline_model_id = ti.xcom_pull(task_ids='compare_vs_baseline', key='baseline_model_id')

    experiment_name = l3_data.get('experiment_name', 'default')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_id = f"{experiment_name}_{timestamp}"
    proposal_id = f"PROP-{model_id}"

    # Determine recommendation
    improvement_met = True
    if comparison:
        improvement_met = comparison.get('sharpe_improvement', 0) >= 0.05

    if all_passed and improvement_met:
        recommendation = "PROMOTE"
        confidence = 0.85
        reason = f"All criteria passed. Sharpe {metrics.get('sharpe_ratio', 0):.2f}"
        if comparison:
            reason += f" (+{comparison.get('sharpe_improvement', 0):.0%} vs baseline)"
    elif all_passed and not improvement_met:
        recommendation = "REVIEW"
        confidence = 0.60
        reason = "All criteria passed but improvement below threshold"
    else:
        recommendation = "REJECT"
        confidence = 0.90
        failed = [cr['name'] for cr in criteria_results if not cr['passed']]
        reason = f"Criteria failed: {', '.join(failed)}"

    # Build lineage
    lineage = {
        'model_hash': l3_data.get('model_hash', ''),
        'dataset_hash': l3_data.get('dataset_hash', ''),
        'config_hash': l3_data.get('config_hash', ''),
        'test_period': f"{metrics.get('test_period_start', '')} to {metrics.get('test_period_end', '')}",
        'baseline_model_id': baseline_model_id,
        'l3_experiment_name': experiment_name,
    }

    # Save to database
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Ensure table exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS promotion_proposals (
                id SERIAL PRIMARY KEY,
                proposal_id VARCHAR(255) UNIQUE NOT NULL,
                model_id VARCHAR(255) NOT NULL,
                experiment_name VARCHAR(255) NOT NULL,
                recommendation VARCHAR(20) NOT NULL,
                confidence DECIMAL(5,4),
                reason TEXT,
                metrics JSONB NOT NULL,
                vs_baseline JSONB,
                criteria_results JSONB NOT NULL,
                lineage JSONB NOT NULL,
                status VARCHAR(30) DEFAULT 'PENDING_APPROVAL',
                reviewer VARCHAR(255),
                reviewer_notes TEXT,
                reviewed_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                expires_at TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '7 days')
            )
        """)

        # Insert proposal
        cur.execute("""
            INSERT INTO promotion_proposals (
                proposal_id, model_id, experiment_name,
                recommendation, confidence, reason,
                metrics, vs_baseline, criteria_results,
                lineage, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'PENDING_APPROVAL')
            ON CONFLICT (proposal_id) DO UPDATE SET
                recommendation = EXCLUDED.recommendation,
                confidence = EXCLUDED.confidence,
                reason = EXCLUDED.reason,
                metrics = EXCLUDED.metrics
            RETURNING id
        """, (
            proposal_id,
            model_id,
            experiment_name,
            recommendation,
            confidence,
            reason,
            json.dumps(metrics),
            json.dumps(comparison) if comparison else None,
            json.dumps(criteria_results),
            json.dumps(lineage),
        ))

        result = cur.fetchone()
        conn.commit()

        # Persist individual trades for exact replay in frontend
        trades_persisted = 0
        if detailed_trades:
            trades_persisted = persist_backtest_trades(
                conn, proposal_id, model_id, detailed_trades
            )

        logger.info("=" * 60)
        logger.info("PROMOTION PROPOSAL GENERATED (PRIMER VOTO)")
        logger.info("=" * 60)
        logger.info(f"  Proposal ID: {proposal_id}")
        logger.info(f"  Model ID: {model_id}")
        logger.info(f"  Recommendation: {recommendation}")
        logger.info(f"  Confidence: {confidence:.0%}")
        logger.info(f"  Reason: {reason}")
        logger.info(f"  Trades Persisted: {trades_persisted} (for exact replay)")
        logger.info(f"  Status: PENDING_APPROVAL (requiere segundo voto en Dashboard)")
        logger.info("=" * 60)

        proposal = {
            'proposal_id': proposal_id,
            'model_id': model_id,
            'experiment_name': experiment_name,
            'recommendation': recommendation,
            'confidence': confidence,
            'reason': reason,
            'metrics': metrics,
            'vs_baseline': comparison,
            'criteria_results': criteria_results,
            'lineage': lineage,
            'status': 'PENDING_APPROVAL',
        }

        ti.xcom_push(key='proposal', value=proposal)

        return {
            'proposal_id': proposal_id,
            'recommendation': recommendation,
            'status': 'PENDING_APPROVAL',
            'db_id': result[0] if result else None,
            'trades_persisted': trades_persisted,
        }

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to save promotion proposal: {e}")
        raise
    finally:
        cur.close()
        conn.close()


def send_notification(**context) -> Dict[str, Any]:
    """
    Task 6: Enviar notificación de propuesta pendiente.
    """
    ti = context['ti']
    proposal = ti.xcom_pull(task_ids='generate_promotion_proposal', key='proposal')

    if not proposal:
        return {'status': 'no_proposal'}

    # TODO: Implement Slack/email notification
    logger.info(f"[NOTIFICATION] New model ready for review: {proposal.get('proposal_id')}")
    logger.info(f"[NOTIFICATION] Recommendation: {proposal.get('recommendation')}")
    logger.info(f"[NOTIFICATION] Review at: /experiments/pending")

    return {'status': 'notification_logged', 'proposal_id': proposal.get('proposal_id')}


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'trading-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email': ['trading@company.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='L4: Backtest out-of-sample + Promotion proposal (Primer Voto)',
    schedule_interval=None,  # Triggered by L3 or manual
    catchup=False,
    max_active_runs=1,
    tags=['v3', 'l4', 'backtest', 'promotion', 'two-vote'],
)

with dag:

    # Wait for L3 to complete (optional - can also be triggered manually)
    wait_l3 = ExternalTaskSensor(
        task_id='wait_for_l3',
        external_dag_id=RL_L3_MODEL_TRAINING,
        external_task_id='training_summary',  # Fixed: L3 ends with training_summary, not save_model_artifacts
        mode='reschedule',
        timeout=3600,
        poke_interval=60,
        allowed_states=['success'],
        failed_states=['failed', 'skipped'],
        soft_fail=True,  # Don't fail if L3 hasn't run
    )

    # Load L3 output
    task_load = PythonOperator(
        task_id='load_l3_output',
        python_callable=load_l3_output,
        provide_context=True,
    )

    # Run out-of-sample backtest
    task_backtest = PythonOperator(
        task_id='run_oos_backtest',
        python_callable=run_oos_backtest,
        provide_context=True,
    )

    # Compare vs baseline
    task_compare = PythonOperator(
        task_id='compare_vs_baseline',
        python_callable=compare_vs_baseline,
        provide_context=True,
    )

    # Evaluate criteria
    task_evaluate = PythonOperator(
        task_id='evaluate_criteria',
        python_callable=evaluate_criteria,
        provide_context=True,
    )

    # Generate promotion proposal
    task_proposal = PythonOperator(
        task_id='generate_promotion_proposal',
        python_callable=generate_promotion_proposal,
        provide_context=True,
    )

    # Send notification
    task_notify = PythonOperator(
        task_id='send_notification',
        python_callable=send_notification,
        provide_context=True,
    )

    # Task chain
    wait_l3 >> task_load >> task_backtest >> task_compare >> task_evaluate >> task_proposal >> task_notify

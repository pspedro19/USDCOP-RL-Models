#!/usr/bin/env python3
"""
Script to populate backtest_trades table for existing pending proposals.

This is a one-time migration script to backfill L4 trades for proposals
that were created before the trade persistence feature was added.

Usage:
    python scripts/populate_l4_trades.py
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import psycopg2
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="usdcop_trading",
        user="admin",
        password="admin123"
    )


class BacktestEngine:
    """Simplified backtest engine for trade generation."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        transaction_cost_bps: float = 75.0,
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
        """Run backtest and return metrics + detailed trades."""
        try:
            from stable_baselines3 import PPO
        except ImportError:
            logger.error("stable_baselines3 not available")
            return {}, []

        # Load model
        model_path = Path(model_path)
        if not model_path.exists():
            model_path = PROJECT_ROOT / model_path

        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return {}, []

        try:
            model = PPO.load(str(model_path))
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return {}, []

        # Load test data
        test_path = Path(test_data_path)
        if not test_path.exists():
            test_path = PROJECT_ROOT / test_data_path

        if not test_path.exists():
            # Try alternative paths
            alt_paths = [
                PROJECT_ROOT / "data" / "pipeline" / "07_output" / "5min" / "DS_production_test.parquet",
                PROJECT_ROOT / "data" / "pipeline" / "07_output" / "5min" / "DS_v3_close_only_test.parquet",
                PROJECT_ROOT / "data" / "pipeline" / "07_output" / "5min" / "DS_default_test.parquet",
                PROJECT_ROOT / "data" / "pipeline" / "07_output" / "5min" / "test.parquet",
            ]
            for alt in alt_paths:
                if alt.exists():
                    test_path = alt
                    logger.info(f"Using alternative test data: {alt}")
                    break

        if not test_path.exists():
            logger.error(f"Test data not found: {test_data_path}")
            return {}, []

        try:
            df = pd.read_parquet(test_path)
            logger.info(f"Loaded {len(df)} test samples from {test_path}")
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return {}, []

        # Load norm stats
        norm_stats = {}
        norm_path = Path(norm_stats_path)
        if not norm_path.exists():
            norm_path = PROJECT_ROOT / norm_stats_path

        if norm_path.exists():
            with open(norm_path) as f:
                norm_stats = json.load(f)

        # Run simulation
        return self._simulate(model, df, norm_stats)

    def _simulate(self, model, df, norm_stats: Dict) -> Tuple[Dict[str, Any], List[Dict]]:
        """Run backtest simulation."""
        from src.core.contracts.feature_contract import FEATURE_ORDER

        capital = self.initial_capital
        peak_capital = capital
        position = 0
        entry_price = 0
        entry_time = None
        entry_confidence = None
        entry_equity = capital
        trades = []
        detailed_trades = []
        equity_curve = [capital]
        trade_id = 0

        for idx in range(len(df)):
            row = df.iloc[idx]

            # Get timestamp
            if hasattr(df.index, '__getitem__'):
                timestamp = df.index[idx]
            else:
                timestamp = row.get('timestamp', row.get('Timestamp', None))

            # Build observation
            obs = self._build_observation(row, position, idx, len(df), FEATURE_ORDER)
            obs = self._normalize(obs, norm_stats, FEATURE_ORDER)

            # Get action from model
            try:
                action, _ = model.predict(obs, deterministic=True)
                raw_action = float(action[0]) if hasattr(action, '__len__') else float(action)
                signal = self._discretize(raw_action)
                confidence = min(abs(raw_action), 1.0)
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

                    trades.append({
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'direction': 'LONG' if position > 0 else 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': close_price,
                    })

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
                        'pnl_usd': float(pnl),
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
                    entry_equity = capital
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

    def _build_observation(self, row, position: float, idx: int, total: int, feature_order: tuple) -> np.ndarray:
        """Build observation from row."""
        obs = []
        for feature in feature_order[:-2]:
            val = row.get(feature, 0.0)
            obs.append(float(val) if val is not None and not np.isnan(val) else 0.0)
        obs.append(float(position))
        obs.append(idx / max(total, 1))
        return np.array(obs, dtype=np.float32)

    def _normalize(self, obs: np.ndarray, norm_stats: Dict, feature_order: tuple) -> np.ndarray:
        """Apply normalization."""
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
            return 1
        elif action < -0.33:
            return -1
        return 0

    def _calc_duration_minutes(self, entry_time, exit_time) -> Optional[int]:
        """Calculate trade duration in minutes."""
        if entry_time is None or exit_time is None:
            return None
        try:
            entry = pd.Timestamp(entry_time)
            exit_t = pd.Timestamp(exit_time)
            return int((exit_t - entry).total_seconds() / 60)
        except Exception:
            return None

    def _calculate_metrics(self, trades: List[Dict], equity_curve: List[float], df) -> Dict[str, Any]:
        """Calculate backtest metrics."""
        if not trades:
            return {}

        returns = [t['pnl_pct'] for t in trades]
        winning = [r for r in returns if r > 0]
        losing = [r for r in returns if r < 0]

        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        peak = equity_curve[0]
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        win_rate = len(winning) / len(trades) if trades else 0
        gross_profit = sum(winning) if winning else 0
        gross_loss = abs(sum(losing)) if losing else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0

        return {
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'win_rate': float(win_rate),
            'profit_factor': float(min(profit_factor, 999.0)),
            'total_trades': len(trades),
            'avg_trade_pnl': float(np.mean(returns)) if returns else 0.0,
            'final_equity': float(equity_curve[-1]) if equity_curve else self.initial_capital,
            'total_return': float((equity_curve[-1] - self.initial_capital) / self.initial_capital) if equity_curve else 0.0,
        }


def persist_trades(conn, proposal_id: str, model_id: str, trades: List[Dict]) -> int:
    """Persist trades to database."""
    if not trades:
        return 0

    cur = conn.cursor()
    inserted = 0

    try:
        for trade in trades:
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


def main():
    """Main function."""
    conn = get_db_connection()
    cur = conn.cursor()

    # Get pending proposals
    cur.execute("""
        SELECT proposal_id, model_id, lineage
        FROM promotion_proposals
        WHERE status = 'PENDING_APPROVAL'
        AND recommendation = 'PROMOTE'
        ORDER BY created_at DESC
        LIMIT 5
    """)
    proposals = cur.fetchall()
    cur.close()

    logger.info(f"Found {len(proposals)} pending proposals with PROMOTE recommendation")

    engine = BacktestEngine()

    for proposal_id, model_id, lineage in proposals:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing proposal: {proposal_id}")
        logger.info(f"Model ID: {model_id}")

        # Check if trades already exist
        check_cur = conn.cursor()
        check_cur.execute(
            "SELECT COUNT(*) FROM backtest_trades WHERE proposal_id = %s",
            (proposal_id,)
        )
        existing_count = check_cur.fetchone()[0]
        check_cur.close()

        if existing_count > 0:
            logger.info(f"Skipping - already has {existing_count} trades")
            continue

        # Parse lineage
        if isinstance(lineage, str):
            lineage = json.loads(lineage)

        model_path = lineage.get('modelPath', lineage.get('model_path', ''))
        norm_stats_path = lineage.get('normStatsPath', lineage.get('norm_stats_path', ''))
        test_data_path = lineage.get('testDataPath', 'data/pipeline/07_output/5min/test.parquet')

        logger.info(f"Model path: {model_path}")
        logger.info(f"Norm stats: {norm_stats_path}")
        logger.info(f"Test data: {test_data_path}")

        # Run backtest
        metrics, detailed_trades = engine.run(model_path, test_data_path, norm_stats_path)

        if detailed_trades:
            logger.info(f"Generated {len(detailed_trades)} trades")
            persist_trades(conn, proposal_id, model_id, detailed_trades)
        else:
            logger.warning("No trades generated")

    conn.close()
    logger.info("\nDone!")


if __name__ == "__main__":
    main()

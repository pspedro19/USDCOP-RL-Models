#!/usr/bin/env python3
"""
Strategy Data Backfill Script - TEMPLATE
=========================================

This script backfills multi-strategy trading data by:
1. Reading historical OHLCV data from PostgreSQL
2. Generating signals using 5 strategies
3. Calculating equity curves based on position changes
4. Inserting results into dw.fact_* tables

Usage:
    python backfill_strategy_data.py --days 30 --initial-capital 10000

Requirements:
    pip install psycopg2-binary pandas numpy anthropic
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import argparse
import logging
from typing import Dict, List
import sys
import os

# Add strategies to path
sys.path.append('/home/azureuser/USDCOP-RL-Models/services')
from strategies.rl_strategy import RLStrategy
from strategies.ml_strategy import MLStrategy
from strategies.llm_strategy import LLMStrategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
POSTGRES_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', '5432')),
    'database': os.getenv('POSTGRES_DB', 'usdcop_trading'),
    'user': os.getenv('POSTGRES_USER', 'admin'),
    'password': os.getenv('POSTGRES_PASSWORD', 'admin123')
}

# ============================================================
# STRATEGY WRAPPERS
# ============================================================

class XGBStrategy(MLStrategy):
    """XGBoost strategy - same as LGBM but more aggressive"""

    def __init__(self):
        super().__init__(strategy_code='ML_XGB')
        # More aggressive thresholds
        self.thresholds = {
            'long': 0.12,   # Lower than LGBM (0.15)
            'short': -0.12
        }

class EnsembleStrategy:
    """Ensemble strategy - weighted average of all models"""

    def __init__(self):
        self.strategy_code = 'ENSEMBLE'
        self.weights = {
            'RL_PPO': 0.30,
            'ML_LGBM': 0.25,
            'ML_XGB': 0.25,
            'LLM_CLAUDE': 0.20
        }
        logger.info(f"✅ Ensemble Strategy initialized with weights: {self.weights}")

    async def generate_signal(self, market_data: Dict, signals: Dict[str, Dict]) -> Dict:
        """Combine signals from all strategies"""
        # Map signals to numeric values
        signal_values = {'long': 1, 'short': -1, 'flat': 0}

        # Calculate weighted score
        score = 0
        total_confidence = 0

        for strategy_code, weight in self.weights.items():
            if strategy_code in signals:
                signal = signals[strategy_code]
                value = signal_values.get(signal.get('signal', 'flat'), 0)
                confidence = signal.get('confidence', 0)

                score += value * weight * confidence
                total_confidence += weight * confidence

        # Determine ensemble signal
        if score > 0.3:
            signal = 'long'
            confidence = min(0.95, total_confidence)
        elif score < -0.3:
            signal = 'short'
            confidence = min(0.95, total_confidence)
        else:
            signal = 'flat'
            confidence = 0.5

        # Size based on confidence
        size = confidence * 0.8 if signal != 'flat' else 0.0

        close = market_data.get('close', 0)
        atr = market_data.get('atr_norm', 0.001) * close

        if signal == 'long':
            stop_loss = close - (2 * atr)
            take_profit = close + (3 * atr)
        elif signal == 'short':
            stop_loss = close + (2 * atr)
            take_profit = close - (3 * atr)
        else:
            stop_loss = None
            take_profit = None

        risk_usd = abs(close - stop_loss) * size * 10000 / close if stop_loss else 0

        return {
            'signal': signal,
            'side': 'buy' if signal == 'long' else ('sell' if signal == 'short' else 'hold'),
            'size': size,
            'confidence': confidence,
            'entry_price': close,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_usd': risk_usd,
            'notional_usd': size * 10000,
            'reasoning': f"Ensemble: weighted score {score:.3f}",
            'features': {'score': float(score), 'consensus': len(signals)}
        }

# ============================================================
# ACCOUNT SIMULATOR
# ============================================================

class TradingAccount:
    """Simulates a trading account with position tracking"""

    def __init__(self, strategy_code: str, initial_capital: float = 10000.0):
        self.strategy_code = strategy_code
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = None  # {'side': 'long', 'entry_price': 4320, 'size': 0.5}
        self.equity_history = []
        self.trades = []

        logger.info(f"  {strategy_code}: ${initial_capital:.2f} initial capital")

    def process_signal(self, signal: Dict, current_price: float, timestamp: datetime) -> Dict:
        """Process a signal and update account state"""
        new_signal = signal.get('signal', 'flat')
        size = signal.get('size', 0)

        # Check if position change is needed
        current_position = self.position['side'] if self.position else 'flat'

        if new_signal == current_position:
            # No change
            return None

        # Close existing position (if any)
        if self.position:
            pnl = self._close_position(current_price, timestamp, "signal_change")

        # Open new position (if not flat)
        if new_signal != 'flat':
            self._open_position(new_signal, current_price, size, timestamp)

        return None

    def _open_position(self, side: str, entry_price: float, size: float, timestamp: datetime):
        """Open a new position"""
        notional = self.cash * size
        quantity = notional / entry_price

        self.position = {
            'side': side,
            'entry_price': entry_price,
            'size': size,
            'quantity': quantity,
            'entry_time': timestamp,
            'notional': notional
        }

        logger.debug(f"    {self.strategy_code}: Opened {side} at ${entry_price:.2f} (size: {size:.2f})")

    def _close_position(self, exit_price: float, timestamp: datetime, reason: str) -> float:
        """Close current position and calculate P&L"""
        if not self.position:
            return 0.0

        entry_price = self.position['entry_price']
        quantity = self.position['quantity']
        side = self.position['side']

        # Calculate P&L
        if side == 'long':
            pnl = (exit_price - entry_price) * quantity
        else:  # short
            pnl = (entry_price - exit_price) * quantity

        # Update cash
        self.cash += pnl

        # Record trade
        self.trades.append({
            'strategy_code': self.strategy_code,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': pnl,
            'entry_time': self.position['entry_time'],
            'exit_time': timestamp,
            'reason': reason
        })

        logger.debug(f"    {self.strategy_code}: Closed {side} at ${exit_price:.2f} (P&L: ${pnl:.2f})")

        self.position = None
        return pnl

    def mark_to_market(self, current_price: float) -> float:
        """Calculate current equity (mark-to-market)"""
        if not self.position:
            return self.cash

        entry_price = self.position['entry_price']
        quantity = self.position['quantity']
        side = self.position['side']

        if side == 'long':
            unrealized_pnl = (current_price - entry_price) * quantity
        else:
            unrealized_pnl = (entry_price - current_price) * quantity

        return self.cash + unrealized_pnl

    def get_equity(self, current_price: float) -> float:
        """Get current equity"""
        return self.mark_to_market(current_price)

    def get_return_pct(self, current_price: float) -> float:
        """Get return percentage since start"""
        equity = self.get_equity(current_price)
        return ((equity / self.initial_capital) - 1) * 100

    def get_drawdown_pct(self, current_price: float) -> float:
        """Get current drawdown percentage"""
        equity = self.get_equity(current_price)
        if not self.equity_history:
            return 0.0

        peak = max(self.equity_history)
        if peak == 0:
            return 0.0

        return ((equity - peak) / peak) * 100

# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(**POSTGRES_CONFIG)

def fetch_historical_ohlcv(days: int = 30) -> pd.DataFrame:
    """Fetch historical OHLCV data"""
    logger.info(f"Fetching last {days} days of OHLCV data...")

    query = """
        SELECT
            time,
            symbol,
            open,
            high,
            low,
            close,
            volume
        FROM usdcop_m5_ohlcv
        WHERE symbol = 'USD/COP'
          AND time >= NOW() - INTERVAL %s
        ORDER BY time ASC
    """

    conn = get_db_connection()
    df = pd.read_sql_query(query, conn, params=(f"{days} days",))
    conn.close()

    logger.info(f"  Fetched {len(df)} rows from {df['time'].min()} to {df['time'].max()}")
    return df

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical features for strategies"""
    logger.info("Calculating technical features...")

    # Returns
    df['r_close_5'] = df['close'].pct_change(5)
    df['r_close_10'] = df['close'].pct_change(10)

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26

    # Bollinger Bands
    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = bb_middle + (2 * bb_std)
    df['bb_lower'] = bb_middle - (2 * bb_std)
    df['bb_middle'] = bb_middle
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # ATR (normalized)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    atr = ranges.max(axis=1).rolling(14).mean()
    df['atr_norm'] = atr / df['close']

    # Realized volatility
    df['rv_20'] = df['close'].pct_change().rolling(20).std()

    # Volume ratio
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    # Fill NaN
    df = df.fillna(method='bfill').fillna(0)

    logger.info(f"  Calculated {len(df.columns)} features")
    return df

def insert_signals_batch(signals: List[Dict]):
    """Insert signals in batch"""
    if not signals:
        return

    conn = get_db_connection()
    cur = conn.cursor()

    query = """
        INSERT INTO dw.fact_strategy_signals (
            timestamp_utc, timestamp_cot, strategy_id, symbol_id,
            signal, side, size, confidence,
            entry_price, stop_loss, take_profit,
            risk_usd, notional_usd,
            reasoning, features_snapshot
        )
        SELECT
            %s, %s AT TIME ZONE 'America/Bogota',
            ds.strategy_id, dsym.symbol_id,
            %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s,
            %s, %s::jsonb
        FROM dw.dim_strategy ds, dw.dim_symbol dsym
        WHERE ds.strategy_code = %s
        AND dsym.symbol_code = 'USDCOP'
    """

    for signal in signals:
        cur.execute(query, (
            signal['timestamp'],
            signal['timestamp'],
            signal['signal'],
            signal['side'],
            signal['size'],
            signal['confidence'],
            signal['entry_price'],
            signal['stop_loss'],
            signal['take_profit'],
            signal['risk_usd'],
            signal['notional_usd'],
            signal['reasoning'],
            signal['features'],
            signal['strategy_code']
        ))

    conn.commit()
    cur.close()
    conn.close()

def insert_equity_batch(equity_points: List[Dict]):
    """Insert equity curve points in batch"""
    if not equity_points:
        return

    conn = get_db_connection()
    cur = conn.cursor()

    query = """
        INSERT INTO dw.fact_equity_curve (
            timestamp_utc, timestamp_cot, strategy_id,
            equity_value, cash_balance, positions_value,
            return_since_start_pct, current_drawdown_pct
        )
        SELECT
            %s, %s AT TIME ZONE 'America/Bogota',
            ds.strategy_id,
            %s, %s, %s,
            %s, %s
        FROM dw.dim_strategy ds
        WHERE ds.strategy_code = %s
    """

    for point in equity_points:
        cur.execute(query, (
            point['timestamp'],
            point['timestamp'],
            point['equity_value'],
            point['cash_balance'],
            point['positions_value'],
            point['return_pct'],
            point['drawdown_pct'],
            point['strategy_code']
        ))

    conn.commit()
    cur.close()
    conn.close()

# ============================================================
# MAIN BACKFILL LOGIC
# ============================================================

async def backfill_strategy_data(days: int = 30, initial_capital: float = 10000.0):
    """Main backfill function"""
    logger.info("="*60)
    logger.info("MULTI-STRATEGY DATA BACKFILL")
    logger.info("="*60)

    # 1. Fetch historical data
    df = fetch_historical_ohlcv(days)
    if df.empty:
        logger.error("No historical data found!")
        return

    # 2. Calculate features
    df = calculate_features(df)

    # 3. Initialize strategies
    logger.info("\nInitializing strategies...")
    strategies = {
        'RL_PPO': RLStrategy('RL_PPO'),
        'ML_LGBM': MLStrategy('ML_LGBM'),
        'ML_XGB': XGBStrategy(),
        'LLM_CLAUDE': LLMStrategy('LLM_CLAUDE'),
        'ENSEMBLE': EnsembleStrategy()
    }

    # 4. Initialize accounts
    logger.info(f"\nInitializing accounts with ${initial_capital:.2f}...")
    accounts = {
        code: TradingAccount(code, initial_capital)
        for code in strategies.keys()
    }

    # 5. Process each bar
    logger.info(f"\nProcessing {len(df)} bars...")
    signals_batch = []
    equity_batch = []
    batch_size = 100

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            logger.info(f"  Progress: {idx}/{len(df)} ({idx/len(df)*100:.1f}%)")

        # Convert row to dict for strategies
        bar_data = {
            'timestamp': row['time'],
            'close': row['close'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'volume': row['volume'],
            'rsi_14': row['rsi_14'],
            'macd': row['macd'],
            'bb_position': row['bb_position'],
            'atr_norm': row['atr_norm'],
            'rv_20': row['rv_20'],
            'volume_ratio': row['volume_ratio'],
            'r_close_5': row['r_close_5']
        }

        # Generate signals
        strategy_signals = {}
        for code, strategy in strategies.items():
            if code == 'ENSEMBLE':
                continue  # Process ENSEMBLE after other strategies

            signal = await strategy.generate_signal(bar_data)
            strategy_signals[code] = signal

            # Store signal
            signals_batch.append({
                'strategy_code': code,
                'timestamp': row['time'],
                'signal': signal['signal'],
                'side': signal['side'],
                'size': signal['size'],
                'confidence': signal['confidence'],
                'entry_price': signal.get('entry_price'),
                'stop_loss': signal.get('stop_loss'),
                'take_profit': signal.get('take_profit'),
                'risk_usd': signal.get('risk_usd', 0),
                'notional_usd': signal.get('notional_usd', 0),
                'reasoning': signal.get('reasoning', ''),
                'features': str(signal.get('features', {}))
            })

        # Generate ENSEMBLE signal
        ensemble_signal = await strategies['ENSEMBLE'].generate_signal(bar_data, strategy_signals)
        strategy_signals['ENSEMBLE'] = ensemble_signal

        signals_batch.append({
            'strategy_code': 'ENSEMBLE',
            'timestamp': row['time'],
            'signal': ensemble_signal['signal'],
            'side': ensemble_signal['side'],
            'size': ensemble_signal['size'],
            'confidence': ensemble_signal['confidence'],
            'entry_price': ensemble_signal.get('entry_price'),
            'stop_loss': ensemble_signal.get('stop_loss'),
            'take_profit': ensemble_signal.get('take_profit'),
            'risk_usd': ensemble_signal.get('risk_usd', 0),
            'notional_usd': ensemble_signal.get('notional_usd', 0),
            'reasoning': ensemble_signal.get('reasoning', ''),
            'features': str(ensemble_signal.get('features', {}))
        })

        # Update accounts
        for code, account in accounts.items():
            signal = strategy_signals[code]
            account.process_signal(signal, row['close'], row['time'])

            # Record equity
            equity = account.get_equity(row['close'])
            account.equity_history.append(equity)

            equity_batch.append({
                'strategy_code': code,
                'timestamp': row['time'],
                'equity_value': equity,
                'cash_balance': account.cash,
                'positions_value': equity - account.cash,
                'return_pct': account.get_return_pct(row['close']),
                'drawdown_pct': account.get_drawdown_pct(row['close'])
            })

        # Insert batches
        if len(signals_batch) >= batch_size:
            insert_signals_batch(signals_batch)
            signals_batch = []

        if len(equity_batch) >= batch_size:
            insert_equity_batch(equity_batch)
            equity_batch = []

    # Insert remaining batches
    if signals_batch:
        insert_signals_batch(signals_batch)
    if equity_batch:
        insert_equity_batch(equity_batch)

    # 6. Summary
    logger.info("\n" + "="*60)
    logger.info("BACKFILL COMPLETE")
    logger.info("="*60)

    for code, account in accounts.items():
        final_equity = account.equity_history[-1] if account.equity_history else initial_capital
        total_return = ((final_equity / initial_capital) - 1) * 100
        n_trades = len(account.trades)

        logger.info(f"\n{code}:")
        logger.info(f"  Final Equity: ${final_equity:.2f}")
        logger.info(f"  Total Return: {total_return:+.2f}%")
        logger.info(f"  Trades: {n_trades}")

    logger.info("\n✅ Data backfill successful!")
    logger.info(f"  Signals inserted: {len(df) * 5}")
    logger.info(f"  Equity points inserted: {len(df) * 5}")

# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill multi-strategy trading data")
    parser.add_argument('--days', type=int, default=30, help="Number of days to backfill")
    parser.add_argument('--initial-capital', type=float, default=10000.0, help="Initial capital per strategy")

    args = parser.parse_args()

    import asyncio
    asyncio.run(backfill_strategy_data(args.days, args.initial_capital))

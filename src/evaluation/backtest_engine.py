"""
BacktestEngine - Unified backtesting implementation for L4 evaluation.

This module provides a single, consistent backtesting engine that ensures
exact parity between training (L3) and evaluation (L4) environments.

Used by:
- scripts/run_full_pipeline.py (L4 evaluation)
- scripts/full_2025_backtest.py (standalone backtest)

Contract: CTR-BACKTEST-001
Version: 1.0.0
Date: 2026-02-02

SSOT Compliance:
- Reads configuration from config/experiment_ssot.yaml
- State features: position (obs[13]), unrealized_pnl (obs[14])
- Uses raw_log_ret_5m for PnL calculation
"""

import json
import logging
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BacktestConfig:
    """
    Backtest configuration loaded from SSOT.

    All values should match what was used during L3 training.

    IMPORTANT: Default values now match pipeline_ssot.yaml for training/backtest parity:
    - Transaction costs: 2.5 bps (MEXC pricing)
    - Thresholds: 0.50/-0.50
    - Risk management: stop_loss=-2.5%, take_profit=+3%
    """
    # Capital
    initial_capital: float = 10000.0

    # Transaction costs (MATCH TRAINING!)
    spread_bps: float = 2.5       # Changed from 75.0 to match training
    slippage_bps: float = 2.5     # Changed from 15.0 to match training

    # Position limits
    max_position_holding: int = 576

    # Action thresholds (MATCH TRAINING!)
    threshold_long: float = 0.50   # Changed from 0.40 to match training
    threshold_short: float = -0.50  # Changed from -0.40 to match training

    # Risk management (MATCH TRAINING!)
    stop_loss_pct: float = -0.025   # -2.5% stop loss
    take_profit_pct: float = 0.03   # +3% take profit
    trailing_stop_enabled: bool = True
    trailing_stop_activation_pct: float = 0.015
    trailing_stop_trail_factor: float = 0.5

    # Validation gates
    min_return_pct: float = -10.0   # Changed from -20.0
    max_action_imbalance: float = 0.85
    min_trades: int = 20            # Changed from 10
    min_win_rate: float = 0.30      # Added
    min_sharpe_ratio: float = 0.3   # Added
    max_drawdown_pct: float = 25.0  # Added

    @classmethod
    def from_pipeline_ssot(cls, config_path: Optional[Path] = None) -> "BacktestConfig":
        """Load configuration from pipeline_ssot.yaml (PREFERRED)."""
        try:
            from src.config.pipeline_config import load_pipeline_config
            config = load_pipeline_config(str(config_path) if config_path else None)

            return cls(
                initial_capital=config.backtest.initial_capital,
                spread_bps=config.backtest.transaction_cost_bps,
                slippage_bps=config.backtest.slippage_bps,
                max_position_holding=config.backtest.max_position_duration,
                threshold_long=config.backtest.threshold_long,
                threshold_short=config.backtest.threshold_short,
                stop_loss_pct=config.backtest.stop_loss_pct,
                take_profit_pct=config.backtest.take_profit_pct,
                trailing_stop_enabled=config.backtest.trailing_stop_enabled,
                trailing_stop_activation_pct=config.backtest.trailing_stop_activation_pct,
                trailing_stop_trail_factor=config.backtest.trailing_stop_trail_factor,
                min_return_pct=config.backtest.min_return_pct,
                max_action_imbalance=config.backtest.max_action_imbalance,
                min_trades=config.backtest.min_trades,
                min_win_rate=config.backtest.min_win_rate,
                min_sharpe_ratio=config.backtest.min_sharpe_ratio,
                max_drawdown_pct=config.backtest.max_drawdown_pct,
            )
        except ImportError:
            logger.warning("pipeline_config not available, using defaults")
            return cls()

    @classmethod
    def from_ssot(cls, ssot_path: Optional[Path] = None) -> "BacktestConfig":
        """Load configuration from SSOT file (legacy - prefer from_pipeline_ssot)."""
        # First try pipeline_ssot.yaml
        try:
            return cls.from_pipeline_ssot()
        except Exception as e:
            logger.warning(f"Failed to load from pipeline_ssot: {e}")

        # Fallback to experiment_ssot.yaml
        if ssot_path is None:
            ssot_path = Path(__file__).parent.parent.parent / "config" / "experiment_ssot.yaml"

        if not ssot_path.exists():
            logger.warning(f"SSOT not found at {ssot_path}, using defaults")
            return cls()

        with open(ssot_path) as f:
            ssot = yaml.safe_load(f)

        env = ssot.get('environment', {})
        backtest = ssot.get('backtest', {})
        tx_costs = backtest.get('transaction_costs', {})
        gates = backtest.get('gates', {})
        thresholds = env.get('thresholds', {})

        return cls(
            initial_capital=backtest.get('initial_capital', 10000.0),
            spread_bps=tx_costs.get('spread_bps', 2.5),
            slippage_bps=tx_costs.get('slippage_bps', 2.5),
            max_position_holding=env.get('max_position_holding', 576),
            threshold_long=thresholds.get('long', 0.50),
            threshold_short=thresholds.get('short', -0.50),
            stop_loss_pct=env.get('stop_loss_pct', -0.025),
            take_profit_pct=env.get('take_profit_pct', 0.03),
            min_return_pct=gates.get('min_return_pct', -10.0),
            max_action_imbalance=gates.get('max_action_imbalance', 0.85),
            min_trades=gates.get('min_trades', 20),
        )

    @property
    def total_cost_rate(self) -> float:
        """Total transaction cost rate per position change."""
        return (self.spread_bps + self.slippage_bps) / 10000


@dataclass
class TradeRecord:
    """Immutable record of a completed trade."""
    entry_bar: int
    exit_bar: int
    entry_time: str
    exit_time: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    pnl: float
    bars_held: int

    @property
    def pnl_pct(self) -> float:
        """PnL as percentage of entry price."""
        if self.entry_price > 0:
            return (self.exit_price - self.entry_price) / self.entry_price * 100
        return 0.0


@dataclass
class BacktestResult:
    """Complete backtest results."""
    # Period info
    start_date: str
    end_date: str
    total_bars: int
    trading_days: float

    # Capital
    initial_capital: float
    final_capital: float
    total_pnl: float
    total_return_pct: float

    # Annualized returns
    apr_simple: float
    apr_compound: float

    # Risk metrics
    max_drawdown_pct: float
    drawdown_duration_bars: int
    sharpe_annual: float
    sortino_annual: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    avg_pnl: float
    avg_duration_hours: float

    # Action distribution
    action_distribution: Dict[str, float]

    # Detailed records
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    # Validation
    passed: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "period": {
                "start": self.start_date,
                "end": self.end_date,
                "bars": self.total_bars,
                "trading_days": self.trading_days,
            },
            "capital": {
                "initial": self.initial_capital,
                "final": self.final_capital,
                "total_pnl": self.total_pnl,
                "total_return_pct": self.total_return_pct,
            },
            "annualized": {
                "apr_simple": self.apr_simple,
                "apr_compound": self.apr_compound,
            },
            "risk": {
                "max_drawdown_pct": self.max_drawdown_pct,
                "drawdown_duration_bars": self.drawdown_duration_bars,
                "sharpe_annual": self.sharpe_annual,
                "sortino_annual": self.sortino_annual,
            },
            "trades": {
                "total": self.total_trades,
                "winning": self.winning_trades,
                "losing": self.losing_trades,
                "win_rate_pct": self.win_rate_pct,
                "profit_factor": self.profit_factor,
                "avg_pnl": self.avg_pnl,
                "avg_duration_hours": self.avg_duration_hours,
            },
            "action_distribution": self.action_distribution,
            "trade_details": [
                {
                    "entry_bar": t.entry_bar,
                    "exit_bar": t.exit_bar,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "direction": t.direction,
                    "pnl": t.pnl,
                    "bars_held": t.bars_held,
                }
                for t in self.trades[:20]  # First 20 trades
            ],
            "passed": self.passed,
            "warnings": self.warnings,
        }


# =============================================================================
# Backtest Engine
# =============================================================================

class BacktestEngine:
    """
    Unified backtesting engine with exact L3 TradingEnv parity.

    This engine ensures that:
    1. Observations are built identically to TradingEnv
    2. State features (position, unrealized_pnl) match SSOT
    3. PnL is calculated using raw returns
    4. Transaction costs match training configuration

    Usage:
        config = BacktestConfig.from_ssot()
        engine = BacktestEngine(config, norm_stats, feature_cols)

        for i in range(1, len(df)):
            obs = engine.build_observation(df.iloc[i])
            action, _ = model.predict(obs, deterministic=True)
            engine.step(float(action[0]), df.iloc[i], df.iloc[i-1])

        result = engine.get_result(df)
    """

    def __init__(
        self,
        config: BacktestConfig,
        norm_stats: Dict[str, Dict[str, float]],
        feature_cols: List[str],
    ):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration from SSOT
            norm_stats: Normalization statistics from L2
            feature_cols: List of feature column names (must match training order)
        """
        self.config = config
        self.norm_stats = norm_stats
        self.feature_cols = feature_cols

        # Dynamic observation dimension based on features
        self.n_market_features = len(feature_cols)
        self.n_state_features = 2  # position, unrealized_pnl
        self.observation_dim = self.n_market_features + self.n_state_features

        logger.info(f"BacktestEngine: {self.n_market_features} market features + {self.n_state_features} state = {self.observation_dim} obs_dim")

        # Track synthetic close price when actual close is not available
        self.use_synthetic_close = False
        self.synthetic_close = 4200.0  # Starting price approximation for USDCOP

        self.reset()

    def reset(self) -> None:
        """Reset all state for a new backtest run."""
        self.capital = self.config.initial_capital
        self.position = 0  # -1=SHORT, 0=FLAT, 1=LONG
        self.entry_bar = 0
        self.entry_price = 0.0

        # Tracking
        self.trades: List[TradeRecord] = []
        self.trade_pnl_accumulator = 0.0
        self.equity_curve: List[float] = [self.capital]
        self.returns: List[float] = []
        self.actions_taken = {'long': 0, 'hold': 0, 'short': 0}
        self._bar_idx = 0

        # Reset synthetic close
        self.synthetic_close = 4200.0  # Starting approximation for USDCOP

    def _get_close_price(self, row: pd.Series) -> float:
        """Get close price from row, or use synthetic price based on returns."""
        if 'close' in row.index:
            return float(row['close'])

        # Use raw_log_ret_* to update synthetic close
        self.use_synthetic_close = True
        raw_ret_col = next((c for c in row.index if c.startswith("raw_log_ret_")), None)
        if raw_ret_col:
            log_ret = row[raw_ret_col]
            self.synthetic_close = self.synthetic_close * np.exp(log_ret)

        return self.synthetic_close

    def build_observation(self, row: pd.Series) -> np.ndarray:
        """
        Build observation vector exactly like TradingEnv._get_observation().

        Args:
            row: DataFrame row with feature columns and 'close' price

        Returns:
            observation array (n_market_features market + 2 state)
        """
        obs = np.zeros(self.observation_dim, dtype=np.float32)

        # Market features (0 to n_market_features-1)
        for j, col in enumerate(self.feature_cols):
            if col not in row.index:
                continue
            value = row[col]

            # Skip normalization for already z-scored features
            if col.endswith('_z'):
                obs[j] = np.clip(value, -5, 5)
            elif col in self.norm_stats and '_meta' not in col:
                stats = self.norm_stats[col]
                mean = stats.get('mean', 0)
                std = stats.get('std', 1)
                if std < 1e-8:
                    std = 1.0
                obs[j] = np.clip((value - mean) / std, -5, 5)
            else:
                obs[j] = np.clip(value, -5, 5)

        # State features (V22: 9 state features after market features)
        pos_idx = self.n_market_features
        obs[pos_idx] = float(self.position)

        # unrealized_pnl
        pnl_idx = self.n_market_features + 1
        unrealized_pnl_pct = 0.0
        if self.position != 0 and self.entry_price > 0:
            current_price = self._get_close_price(row)
            unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price * self.position
            obs[pnl_idx] = np.clip(unrealized_pnl_pct, -1.0, 1.0)
        else:
            obs[pnl_idx] = 0.0

        # V21: sl_proximity, tp_proximity, bars_held (if observation_dim allows)
        if self.observation_dim > self.n_market_features + 2:
            sl_pct = getattr(self.config, 'stop_loss_pct', -0.04)
            tp_pct = getattr(self.config, 'take_profit_pct', 0.04)
            max_dur = getattr(self.config, 'max_position_holding', 864)

            if self.position != 0:
                sl_distance = unrealized_pnl_pct - sl_pct
                sl_proximity = float(np.clip(sl_distance / abs(sl_pct), 0.0, 2.0) / 2.0)
                tp_proximity = float(np.clip(unrealized_pnl_pct / tp_pct, 0.0, 1.0))
                bars_held = float(np.clip((self._bar_idx - self.entry_bar) / max_dur, 0.0, 1.0))
            else:
                sl_proximity = 1.0
                tp_proximity = 0.0
                bars_held = 0.0

            sl_idx = self.n_market_features + 2
            tp_idx = self.n_market_features + 3
            bh_idx = self.n_market_features + 4

            if sl_idx < self.observation_dim:
                obs[sl_idx] = sl_proximity
            if tp_idx < self.observation_dim:
                obs[tp_idx] = tp_proximity
            if bh_idx < self.observation_dim:
                obs[bh_idx] = bars_held

        # V22 P1: Temporal features (hour_sin, hour_cos, dow_sin, dow_cos)
        temporal_start = self.n_market_features + 5
        if self.observation_dim > temporal_start:
            try:
                ts = row.name if hasattr(row, 'name') and hasattr(row.name, 'hour') else None
                if ts is None and 'datetime' in row.index:
                    ts = pd.to_datetime(row['datetime'])
                if ts is not None:
                    hour_frac = (ts.hour + ts.minute / 60.0 - 13.0) / 5.0
                    hour_frac = max(0.0, min(1.0, hour_frac))
                    dow_frac = ts.dayofweek / 5.0  # 5 trading days, NOT 7
                    if temporal_start < self.observation_dim:
                        obs[temporal_start] = float(np.sin(2 * np.pi * hour_frac))
                    if temporal_start + 1 < self.observation_dim:
                        obs[temporal_start + 1] = float(np.cos(2 * np.pi * hour_frac))
                    if temporal_start + 2 < self.observation_dim:
                        obs[temporal_start + 2] = float(np.sin(2 * np.pi * dow_frac))
                    if temporal_start + 3 < self.observation_dim:
                        obs[temporal_start + 3] = float(np.cos(2 * np.pi * dow_frac))
            except (AttributeError, TypeError):
                pass

        # Handle NaN
        obs = np.nan_to_num(obs, nan=0.0)

        return obs

    def step(
        self,
        action,
        row: pd.Series,
        prev_row: pd.Series,
    ) -> float:
        """
        Execute one backtest step.

        Args:
            action: int for Discrete(4) or float for continuous [-1, 1]
            row: Current bar data
            prev_row: Previous bar data (for return calculation)

        Returns:
            Net PnL for this step
        """
        self._bar_idx += 1

        # V22: Handle both discrete and continuous actions
        action_type = getattr(self.config, 'action_type', 'continuous')

        if action_type == "discrete" or isinstance(action, (int, np.integer)):
            # V22 P2: Discrete(4) — HOLD=0, BUY=1, SELL=2, CLOSE=3
            action_int = int(action)
            if action_int == 0:  # HOLD
                target = self.position  # Maintain current position
                self.actions_taken['hold'] += 1
            elif action_int == 1:  # BUY
                target = 1
                self.actions_taken['long'] += 1
            elif action_int == 2:  # SELL
                target = -1
                self.actions_taken['short'] += 1
            elif action_int == 3:  # CLOSE
                target = 0  # Flatten
                if self.position != 0:
                    self.actions_taken.setdefault('close', 0)
                    self.actions_taken['close'] += 1
                else:
                    self.actions_taken['hold'] += 1
            else:
                target = 0
                self.actions_taken['hold'] += 1
        else:
            # V21 legacy: Continuous action
            action_val = float(action) if not isinstance(action, np.ndarray) else float(action[0])
            if action_val > self.config.threshold_long:
                target = 1  # LONG
                self.actions_taken['long'] += 1
            elif action_val < self.config.threshold_short:
                target = -1  # SHORT
                self.actions_taken['short'] += 1
            else:
                target = 0  # HOLD
                self.actions_taken['hold'] += 1

        # Force close if max position holding exceeded
        bars_in_position = self._bar_idx - self.entry_bar
        if self.position != 0 and bars_in_position >= self.config.max_position_holding:
            target = 0

        # Calculate market return (use raw if available)
        raw_ret_col = next((c for c in row.index if c.startswith("raw_log_ret_")), None)
        if raw_ret_col:
            market_return = row[raw_ret_col]
        else:
            current_close = self._get_close_price(row)
            prev_close = self._get_close_price(prev_row)
            market_return = np.log(current_close / prev_close) if prev_close > 0 else 0.0

        # PnL from current position
        position_pnl = self.position * market_return * self.capital

        # Transaction cost on position change
        cost = 0.0
        if target != self.position:
            cost = abs(target - self.position) * self.config.total_cost_rate * self.capital

            # Record completed trade
            if self.position != 0:
                current_close = self._get_close_price(row)
                self.trades.append(TradeRecord(
                    entry_bar=self.entry_bar,
                    exit_bar=self._bar_idx,
                    entry_time=str(prev_row.get('timestamp', '') if hasattr(prev_row, 'get') else prev_row.name),
                    exit_time=str(row.get('timestamp', '') if hasattr(row, 'get') else row.name),
                    direction='LONG' if self.position == 1 else 'SHORT',
                    entry_price=self.entry_price,
                    exit_price=current_close,
                    pnl=self.trade_pnl_accumulator,
                    bars_held=self._bar_idx - self.entry_bar,
                ))

            # Update position state
            current_close = self._get_close_price(row)
            self.position = target
            self.entry_bar = self._bar_idx
            self.entry_price = current_close if target != 0 else 0.0
            self.trade_pnl_accumulator = 0.0
        else:
            self.trade_pnl_accumulator += position_pnl

        # Update capital
        net_pnl = position_pnl - cost
        self.capital += net_pnl

        self.equity_curve.append(self.capital)
        if len(self.equity_curve) > 1:
            bar_return = net_pnl / self.equity_curve[-2] if self.equity_curve[-2] > 0 else 0
            self.returns.append(bar_return)

        return net_pnl

    def close_position(self, row: pd.Series) -> None:
        """Close any remaining position at end of backtest."""
        if self.position != 0:
            current_close = self._get_close_price(row)
            self.trades.append(TradeRecord(
                entry_bar=self.entry_bar,
                exit_bar=self._bar_idx,
                entry_time="",  # Unknown
                exit_time=str(row.get('timestamp', '') if hasattr(row, 'get') else row.name),
                direction='LONG' if self.position == 1 else 'SHORT',
                entry_price=self.entry_price,
                exit_price=current_close,
                pnl=self.trade_pnl_accumulator,
                bars_held=self._bar_idx - self.entry_bar,
            ))

    def get_result(self, df: pd.DataFrame) -> BacktestResult:
        """
        Calculate and return comprehensive backtest metrics.

        Args:
            df: Full test dataframe (for date info)

        Returns:
            BacktestResult with all metrics
        """
        equity = np.array(self.equity_curve)
        returns = np.array(self.returns)

        # Capital metrics
        total_pnl = self.capital - self.config.initial_capital
        total_return_pct = (self.capital / self.config.initial_capital - 1) * 100

        # Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = float(np.max(drawdown)) * 100
        max_dd_idx = int(np.argmax(drawdown))
        peak_idx = int(np.argmax(equity[:max_dd_idx+1])) if max_dd_idx > 0 else 0
        dd_duration = max_dd_idx - peak_idx

        # Trade statistics
        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(winning) / len(self.trades) * 100 if self.trades else 0

        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_pnl = np.mean([t.pnl for t in self.trades]) if self.trades else 0
        avg_bars = np.mean([t.bars_held for t in self.trades]) if self.trades else 0
        avg_hours = avg_bars * 5 / 60  # 5 min bars

        # Daily returns for ratio calculation
        bars_per_day = 144  # 12h trading day
        daily_returns = []
        for d in range(0, len(returns), bars_per_day):
            chunk = returns[d:d+bars_per_day]
            if len(chunk) > 0:
                daily_returns.append(np.sum(chunk))
        daily_returns = np.array(daily_returns) if daily_returns else np.array([0])

        # Sharpe ratio
        if len(daily_returns) > 0 and np.std(daily_returns) > 1e-10:
            sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
        else:
            sharpe = 0.0

        # Sortino ratio
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0 and np.std(downside) > 1e-10:
            sortino = float(np.mean(daily_returns) / np.std(downside) * np.sqrt(252))
        else:
            sortino = 0.0

        # APR calculation
        trading_days = len(df) / bars_per_day
        if trading_days > 0:
            daily_return_avg = total_return_pct / trading_days
            apr_simple = daily_return_avg * 252
            apr_compound = ((1 + total_return_pct/100) ** (252/trading_days) - 1) * 100
        else:
            apr_simple = 0
            apr_compound = 0

        # Action distribution
        total_actions = sum(self.actions_taken.values())
        action_pct = {k: v/total_actions*100 if total_actions > 0 else 0
                      for k, v in self.actions_taken.items()}

        # Validation
        passed = True
        warnings = []

        if total_return_pct < self.config.min_return_pct:
            warnings.append(f"Total return {total_return_pct:.1f}% < {self.config.min_return_pct}%")
            passed = False

        max_action_pct = max(action_pct.values()) / 100
        if max_action_pct > self.config.max_action_imbalance:
            warnings.append(f"Action imbalance: {max_action_pct*100:.1f}% > {self.config.max_action_imbalance*100}%")
            passed = False

        if len(self.trades) < self.config.min_trades:
            warnings.append(f"Only {len(self.trades)} trades < {self.config.min_trades}")
            passed = False

        # Date info
        timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'datetime'
        start_date = str(df[timestamp_col].min())
        end_date = str(df[timestamp_col].max())

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            total_bars=len(df),
            trading_days=trading_days,
            initial_capital=self.config.initial_capital,
            final_capital=self.capital,
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            apr_simple=apr_simple,
            apr_compound=apr_compound,
            max_drawdown_pct=max_dd,
            drawdown_duration_bars=dd_duration,
            sharpe_annual=sharpe,
            sortino_annual=sortino,
            total_trades=len(self.trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate_pct=win_rate,
            profit_factor=profit_factor if profit_factor != float('inf') else 999.99,
            avg_pnl=avg_pnl,
            avg_duration_hours=avg_hours,
            action_distribution=action_pct,
            trades=self.trades,
            equity_curve=self.equity_curve,
            passed=passed,
            warnings=warnings,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def run_backtest(
    model,
    df: pd.DataFrame,
    norm_stats: Dict[str, Dict[str, float]],
    config: Optional[BacktestConfig] = None,
    test_start: Optional[str] = None,
) -> BacktestResult:
    """
    Run a complete backtest with a trained model.

    Args:
        model: Trained PPO model with predict() method
        df: Full dataset with features
        norm_stats: Normalization statistics
        config: Backtest configuration (defaults to SSOT)
        test_start: Start date for test period (defaults to 2025-01-01)

    Returns:
        BacktestResult with all metrics
    """
    if config is None:
        config = BacktestConfig.from_ssot()

    # Filter to test period
    timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'datetime'
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    if test_start:
        test_df = df[df[timestamp_col] >= test_start].reset_index(drop=True)
    else:
        test_df = df.reset_index(drop=True)

    # Feature columns from norm_stats (dynamic)
    if "features" in norm_stats:
        # New format with nested "features" key
        feature_stats = norm_stats["features"]
    else:
        # Legacy format - filter out metadata keys
        feature_stats = {k: v for k, v in norm_stats.items() if not k.startswith("_")}

    feature_cols = list(feature_stats.keys())
    logger.info(f"run_backtest: Using {len(feature_cols)} features from norm_stats")

    # Initialize engine
    engine = BacktestEngine(config, feature_stats, feature_cols)

    # Run backtest
    for i in range(1, len(test_df)):
        obs = engine.build_observation(test_df.iloc[i])
        action, _ = model.predict(obs, deterministic=True)
        action_value = float(action[0]) if hasattr(action, '__len__') else float(action)
        engine.step(action_value, test_df.iloc[i], test_df.iloc[i-1])

    # Close final position
    engine.close_position(test_df.iloc[-1])

    return engine.get_result(test_df)

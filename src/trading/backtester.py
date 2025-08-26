"""
Walk-Forward Backtesting System
===============================
Production-ready backtesting with walk-forward optimization for USDCOP trading strategies.
Supports both RL agents and rule-based strategies with realistic cost modeling.
"""

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from src.markets.usdcop.agent import USDCOPAgent
from src.markets.usdcop.environment import USDCOPEnvironment
from src.markets.usdcop.metrics import USDCOPMetrics
from .config import USDCOP_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    train_ratio: float = 0.8
    walk_forward_periods: int = 5
    step_ratio: float = 0.5  # Step size as ratio of test period
    
    # Cost model
    spread_pips: float = 10.0
    slippage_pips: float = 2.0
    commission_pct: float = 0.0
    
    # Monte Carlo
    monte_carlo_runs: int = 100
    confidence_level: float = 0.95
    
    # Optimization
    optimization_metric: str = "sharpe_ratio"
    min_trades: int = 30
    
    # Output
    save_results: bool = True
    output_dir: str = "./backtest_results"


@dataclass
class FoldResult:
    """Result for a single walk-forward fold"""
    fold_num: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    
    # Metrics
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    
    # Performance
    test_return: float
    test_sharpe: float
    test_max_dd: float
    test_trades: int
    
    # Degradation
    performance_ratio: float  # test/train metric ratio


@dataclass
class BacktestResult:
    """Complete backtest result"""
    strategy_name: str
    config: BacktestConfig
    parameters: Dict[str, Any]
    
    # Aggregated metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    
    # Walk-forward results
    fold_results: List[FoldResult]
    
    # Monte Carlo results (optional)
    monte_carlo: Optional[Dict[str, Any]] = None
    
    # Validation
    is_robust: bool = True
    warnings: List[str] = None


class Backtester:
    """Main walk-forward backtesting engine"""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.metrics_calculator = USDCOPMetrics()
        
        # Ensure output directory exists
        if self.config.save_results:
            os.makedirs(self.config.output_dir, exist_ok=True)
    
    def run_walk_forward(self, data: pd.DataFrame, strategy: Any,
                        optimize_params: bool = True) -> BacktestResult:
        """
        Run walk-forward analysis
        
        Args:
            data: Historical OHLCV data
            strategy: Strategy instance or class
            optimize_params: Whether to optimize parameters on each fold
            
        Returns:
            BacktestResult with complete analysis
        """
        logger.info(f"Starting walk-forward backtest with {self.config.walk_forward_periods} periods")
        
        # Validate data
        if data.empty or len(data) < 1000:
            raise ValueError("Insufficient data for walk-forward analysis")
        
        data = data.sort_index()
        
        # Calculate fold boundaries
        folds = self._calculate_folds(len(data))
        
        # Run each fold
        fold_results = []
        all_test_returns = []
        
        for i, (train_slice, test_slice) in enumerate(folds):
            logger.info(f"Processing fold {i+1}/{len(folds)}")
            
            train_data = data.iloc[train_slice]
            test_data = data.iloc[test_slice]
            
            # Optimize parameters if requested
            if optimize_params and hasattr(strategy, 'optimize'):
                optimal_params = self._optimize_strategy(strategy, train_data)
                strategy.set_parameters(optimal_params)
            
            # Run backtest on both periods
            train_result = self._run_single_backtest(strategy, train_data)
            test_result = self._run_single_backtest(strategy, test_data)
            
            # Calculate performance ratio
            train_metric = train_result['metrics'].get(self.config.optimization_metric, 0)
            test_metric = test_result['metrics'].get(self.config.optimization_metric, 0)
            perf_ratio = test_metric / train_metric if train_metric != 0 else 0
            
            # Store fold result
            fold = FoldResult(
                fold_num=i+1,
                train_start=str(train_data.index[0]),
                train_end=str(train_data.index[-1]),
                test_start=str(test_data.index[0]),
                test_end=str(test_data.index[-1]),
                train_metrics=train_result['metrics'],
                test_metrics=test_result['metrics'],
                test_return=test_result['metrics']['total_return'],
                test_sharpe=test_result['metrics']['sharpe_ratio'],
                test_max_dd=test_result['metrics']['max_drawdown'],
                test_trades=test_result['metrics']['total_trades'],
                performance_ratio=perf_ratio
            )
            
            fold_results.append(fold)
            all_test_returns.extend(test_result['returns'])
        
        # Aggregate results
        aggregated = self._aggregate_results(fold_results, all_test_returns)
        
        # Run Monte Carlo if configured
        monte_carlo = None
        if self.config.monte_carlo_runs > 0 and all_test_returns:
            monte_carlo = self._run_monte_carlo(all_test_returns)
        
        # Validate robustness
        is_robust, warnings = self._validate_robustness(fold_results)
        
        # Create final result
        result = BacktestResult(
            strategy_name=strategy.__class__.__name__,
            config=self.config,
            parameters=getattr(strategy, 'parameters', {}),
            total_return=aggregated['total_return'],
            sharpe_ratio=aggregated['sharpe_ratio'],
            max_drawdown=aggregated['max_drawdown'],
            win_rate=aggregated['win_rate'],
            total_trades=aggregated['total_trades'],
            fold_results=fold_results,
            monte_carlo=monte_carlo,
            is_robust=is_robust,
            warnings=warnings
        )
        
        # Save results if configured
        if self.config.save_results:
            self._save_results(result)
        
        return result
    
    def _calculate_folds(self, data_length: int) -> List[Tuple[slice, slice]]:
        """Calculate train/test splits for walk-forward"""
        folds = []
        
        # Calculate sizes
        total_usable = int(data_length * 0.95)  # Leave some buffer
        test_size = int(total_usable * (1 - self.config.train_ratio) / self.config.walk_forward_periods)
        train_size = int(test_size * self.config.train_ratio / (1 - self.config.train_ratio))
        step_size = int(test_size * self.config.step_ratio)
        
        # Generate folds
        start = 0
        for i in range(self.config.walk_forward_periods):
            train_end = start + train_size
            test_end = train_end + test_size
            
            if test_end > data_length:
                break
                
            train_slice = slice(start, train_end)
            test_slice = slice(train_end, test_end)
            
            folds.append((train_slice, test_slice))
            
            start += step_size
        
        return folds
    
    def _run_single_backtest(self, strategy: Any, data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest on a single period"""
        if hasattr(strategy, 'train_model'):
            # RL Agent strategy
            return self._run_rl_backtest(strategy, data)
        else:
            # Rule-based strategy
            return self._run_rule_backtest(strategy, data)
    
    def _run_rl_backtest(self, agent: USDCOPAgent, data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest for RL agent"""
        # Train agent
        agent.train_model(data)
        
        # Create environment for testing
        env = USDCOPEnvironment(data, self.config.__dict__)
        
        # Run episode
        obs, _ = env.reset()
        done = False
        
        equity_curve = [env.initial_capital]
        trades = []
        returns = []
        
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, done, _, info = env.step(action)
            
            equity_curve.append(info.get('equity', equity_curve[-1]))
            
            if info.get('trade_closed', False):
                trades.append(info['trade_info'])
                returns.append(info['trade_info']['return_pct'])
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_forex_metrics(data, trades)
        
        return {
            'metrics': metrics,
            'equity_curve': equity_curve,
            'trades': trades,
            'returns': returns
        }
    
    def _run_rule_backtest(self, strategy: Any, data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest for rule-based strategy"""
        equity = [10000.0]  # Initial capital
        position = 0
        trades = []
        returns = []
        
        for i in range(len(data)):
            # Update strategy
            strategy.update(data.iloc[:i+1])
            
            # Get signal
            signal = strategy.get_signal()
            
            # Simple position management
            if signal > 0 and position <= 0:
                # Buy
                position = 1
                entry_price = data.iloc[i]['close']
                entry_time = data.index[i]
                
            elif signal < 0 and position >= 0:
                # Sell/Close
                if position > 0:
                    # Close long
                    exit_price = data.iloc[i]['close']
                    pnl = (exit_price - entry_price) / entry_price
                    pnl -= (self.config.spread_pips + self.config.slippage_pips * 2) * 0.0001
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': data.index[i],
                        'return_pct': pnl * 100
                    })
                    returns.append(pnl)
                    
                position = -1 if signal < 0 else 0
            
            # Update equity
            if position != 0 and i > 0:
                price_change = (data.iloc[i]['close'] - data.iloc[i-1]['close']) / data.iloc[i-1]['close']
                equity.append(equity[-1] * (1 + position * price_change))
            else:
                equity.append(equity[-1])
        
        # Calculate metrics
        equity_series = pd.Series(equity, index=data.index[:len(equity)])
        
        metrics = {
            'total_return': (equity[-1] / equity[0] - 1) * 100,
            'sharpe_ratio': self._calculate_sharpe(equity_series),
            'max_drawdown': self._calculate_max_drawdown(equity_series),
            'total_trades': len(trades),
            'win_rate': sum(1 for r in returns if r > 0) / len(returns) if returns else 0
        }
        
        return {
            'metrics': metrics,
            'equity_curve': equity,
            'trades': trades,
            'returns': returns
        }
    
    def _optimize_strategy(self, strategy: Any, train_data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize strategy parameters"""
        if not hasattr(strategy, 'get_param_grid'):
            return {}
        
        param_grid = strategy.get_param_grid()
        best_score = -np.inf
        best_params = {}
        
        # Grid search (simple for now)
        for params in self._generate_param_combinations(param_grid):
            strategy.set_parameters(params)
            result = self._run_single_backtest(strategy, train_data)
            
            score = result['metrics'].get(self.config.optimization_metric, 0)
            if score > best_score:
                best_score = score
                best_params = params.copy()
        
        logger.info(f"Best parameters: {best_params} (score: {best_score:.3f})")
        return best_params
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """Generate all parameter combinations"""
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _aggregate_results(self, fold_results: List[FoldResult], 
                          all_returns: List[float]) -> Dict[str, float]:
        """Aggregate metrics across all folds"""
        # Out-of-sample metrics only
        test_returns = [f.test_return for f in fold_results]
        test_sharpes = [f.test_sharpe for f in fold_results]
        test_dds = [f.test_max_dd for f in fold_results]
        
        # Calculate aggregate equity curve
        equity = 10000.0
        for ret in all_returns:
            equity *= (1 + ret / 100)
        
        total_return = (equity / 10000.0 - 1) * 100
        
        return {
            'total_return': total_return,
            'sharpe_ratio': np.mean(test_sharpes),
            'max_drawdown': np.max(test_dds),
            'win_rate': sum(1 for r in all_returns if r > 0) / len(all_returns) if all_returns else 0,
            'total_trades': sum(f.test_trades for f in fold_results)
        }
    
    def _run_monte_carlo(self, returns: List[float]) -> Dict[str, Any]:
        """Run Monte Carlo simulation"""
        n_sims = self.config.monte_carlo_runs
        n_trades = len(returns)
        
        sim_results = []
        
        for _ in range(n_sims):
            # Bootstrap sample returns
            sampled_returns = np.random.choice(returns, size=n_trades, replace=True)
            
            # Calculate cumulative return
            equity = 10000.0
            for ret in sampled_returns:
                equity *= (1 + ret / 100)
            
            total_return = (equity / 10000.0 - 1) * 100
            sim_results.append(total_return)
        
        # Calculate statistics
        sim_results = np.array(sim_results)
        
        return {
            'expected_return': np.mean(sim_results),
            'return_std': np.std(sim_results),
            'var_95': np.percentile(sim_results, 5),
            'confidence_interval': (
                np.percentile(sim_results, (1 - self.config.confidence_level) / 2 * 100),
                np.percentile(sim_results, (1 + self.config.confidence_level) / 2 * 100)
            ),
            'prob_profit': np.mean(sim_results > 0)
        }
    
    def _validate_robustness(self, fold_results: List[FoldResult]) -> Tuple[bool, List[str]]:
        """Validate strategy robustness"""
        warnings = []
        
        # Check performance degradation
        perf_ratios = [f.performance_ratio for f in fold_results]
        avg_degradation = np.mean(perf_ratios)
        
        if avg_degradation < 0.5:
            warnings.append(f"High performance degradation: {avg_degradation:.2f}")
        
        # Check consistency
        test_sharpes = [f.test_sharpe for f in fold_results]
        if np.std(test_sharpes) > np.mean(test_sharpes):
            warnings.append("High variability in out-of-sample performance")
        
        # Check minimum trades
        for fold in fold_results:
            if fold.test_trades < self.config.min_trades:
                warnings.append(f"Fold {fold.fold_num}: Insufficient trades ({fold.test_trades})")
        
        # Determine if robust
        is_robust = len(warnings) == 0 and avg_degradation > 0.5
        
        return is_robust, warnings
    
    def _calculate_sharpe(self, equity_series: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        returns = equity_series.pct_change().dropna()
        if len(returns) < 2:
            return 0.0
        
        return returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = equity_series.expanding().max()
        dd = (equity_series - peak) / peak
        return abs(dd.min())
    
    def _save_results(self, result: BacktestResult):
        """Save backtest results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.strategy_name}_{timestamp}.json"
        filepath = os.path.join(self.config.output_dir, filename)
        
        # Convert to serializable format
        result_dict = asdict(result)
        
        # Save JSON
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
        
        # Also save summary
        summary = {
            'strategy': result.strategy_name,
            'total_return': f"{result.total_return:.2f}%",
            'sharpe_ratio': f"{result.sharpe_ratio:.2f}",
            'max_drawdown': f"{result.max_drawdown:.2f}%",
            'total_trades': result.total_trades,
            'is_robust': result.is_robust,
            'warnings': result.warnings
        }
        
        summary_file = os.path.join(self.config.output_dir, f"summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)


# Example usage functions
def backtest_strategy(strategy, data: pd.DataFrame, config: Optional[BacktestConfig] = None) -> BacktestResult:
    """Convenience function to run backtest"""
    backtester = Backtester(config)
    return backtester.run_walk_forward(data, strategy)


def compare_strategies(strategies: List[Any], data: pd.DataFrame, 
                      config: Optional[BacktestConfig] = None) -> pd.DataFrame:
    """Compare multiple strategies"""
    results = []
    backtester = Backtester(config)
    
    for strategy in strategies:
        logger.info(f"Backtesting {strategy.__class__.__name__}")
        result = backtester.run_walk_forward(data, strategy)
        
        results.append({
            'Strategy': result.strategy_name,
            'Total Return': f"{result.total_return:.2f}%",
            'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
            'Max Drawdown': f"{result.max_drawdown:.2f}%",
            'Win Rate': f"{result.win_rate:.1%}",
            'Total Trades': result.total_trades,
            'Is Robust': result.is_robust
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example test
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='5min')
    data = pd.DataFrame({
        'open': 4000 + np.random.randn(len(dates)).cumsum() * 10,
        'high': 4000 + np.random.randn(len(dates)).cumsum() * 10 + 5,
        'low': 4000 + np.random.randn(len(dates)).cumsum() * 10 - 5,
        'close': 4000 + np.random.randn(len(dates)).cumsum() * 10,
        'volume': np.random.randint(100, 1000, len(dates))
    }, index=dates)
    
    # Create simple test strategy
    class SimpleMAStrategy:
        def __init__(self, fast_period=10, slow_period=30):
            self.fast_period = fast_period
            self.slow_period = slow_period
            self.parameters = {'fast_period': fast_period, 'slow_period': slow_period}
        
        def update(self, data):
            self.data = data
        
        def get_signal(self):
            if len(self.data) < self.slow_period:
                return 0
            
            fast_ma = self.data['close'].iloc[-self.fast_period:].mean()
            slow_ma = self.data['close'].iloc[-self.slow_period:].mean()
            
            if fast_ma > slow_ma:
                return 1
            elif fast_ma < slow_ma:
                return -1
            return 0
        
        def set_parameters(self, params):
            self.fast_period = params.get('fast_period', self.fast_period)
            self.slow_period = params.get('slow_period', self.slow_period)
            self.parameters = params
        
        def get_param_grid(self):
            return {
                'fast_period': [5, 10, 15],
                'slow_period': [20, 30, 50]
            }
    
    # Run backtest
    strategy = SimpleMAStrategy()
    config = BacktestConfig(
        walk_forward_periods=3,
        monte_carlo_runs=50,
        output_dir="./test_backtest"
    )
    
    result = backtest_strategy(strategy, data, config)
    
    print(f"\nBacktest Results for {result.strategy_name}:")
    print(f"Total Return: {result.total_return:.2f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2f}%")
    print(f"Total Trades: {result.total_trades}")
    print(f"Is Robust: {result.is_robust}")
    
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    if result.monte_carlo:
        print(f"\nMonte Carlo Analysis:")
        print(f"  Expected Return: {result.monte_carlo['expected_return']:.2f}%")
        print(f"  95% VaR: {result.monte_carlo['var_95']:.2f}%")
        print(f"  Probability of Profit: {result.monte_carlo['prob_profit']:.1%}")
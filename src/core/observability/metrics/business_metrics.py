"""
Business Metrics
===============
Trading-specific metrics for business KPIs.
"""
from typing import Dict, Any, Optional
from .prometheus_registry import MetricsRegistry

class BusinessMetrics:
    """Business metrics for trading operations."""
    
    def __init__(self, metrics_registry: MetricsRegistry):
        self.registry = metrics_registry
        self._init_business_metrics()
    
    def _init_business_metrics(self):
        """Initialize business-specific metrics."""
        # Trading metrics
        self.trades_total = self.registry.counter(
            'trades_total',
            'Total number of trades',
            ['side', 'outcome', 'symbol']
        )
        
        self.trade_volume = self.registry.counter(
            'trade_volume_total',
            'Total trading volume',
            ['side', 'symbol']
        )
        
        self.trade_pnl = self.registry.counter(
            'trade_pnl_total',
            'Total P&L from trades',
            ['side', 'symbol']
        )
        
        # P&L metrics
        self.pnl_gauge = self.registry.gauge(
            'pnl_total',
            'Current total P&L',
            ['symbol']
        )
        
        self.pnl_distribution = self.registry.summary(
            'pnl_distribution',
            'P&L distribution statistics',
            ['symbol']
        )
        
        # Performance metrics
        self.win_rate = self.registry.gauge(
            'win_rate',
            'Current win rate percentage',
            ['symbol']
        )
        
        self.sharpe_ratio = self.registry.gauge(
            'sharpe_ratio',
            'Current Sharpe ratio',
            ['symbol']
        )
        
        self.max_drawdown = self.registry.gauge(
            'max_drawdown',
            'Maximum drawdown percentage',
            ['symbol']
        )
        
        # Position metrics
        self.active_positions = self.registry.gauge(
            'active_positions',
            'Number of active positions',
            ['symbol']
        )
        
        self.position_value = self.registry.gauge(
            'position_value_total',
            'Total value of active positions',
            ['symbol']
        )
        
        # Model metrics
        self.model_predictions = self.registry.counter(
            'model_predictions_total',
            'Total model predictions',
            ['model', 'symbol', 'confidence_level']
        )
        
        self.prediction_accuracy = self.registry.gauge(
            'prediction_accuracy',
            'Model prediction accuracy',
            ['model', 'symbol']
        )
        
        self.prediction_latency = self.registry.histogram(
            'prediction_latency_seconds',
            'Model prediction latency',
            ['model', 'symbol']
        )
    
    def record_trade(self, side: str, outcome: str, symbol: str, volume: float, pnl: float):
        """Record a completed trade."""
        # Increment trade counters
        self.trades_total.labels(side=side, outcome=outcome, symbol=symbol).inc()
        self.trade_volume.labels(side=side, symbol=symbol).inc(volume)
        self.trade_pnl.labels(side=side, symbol=symbol).inc(pnl)
        
        # Update P&L distribution
        self.pnl_distribution.labels(symbol=symbol).observe(pnl)
    
    def update_pnl(self, symbol: str, total_pnl: float):
        """Update total P&L for a symbol."""
        self.pnl_gauge.labels(symbol=symbol).set(total_pnl)
    
    def update_performance(self, symbol: str, win_rate: float, sharpe: float, drawdown: float):
        """Update performance metrics."""
        self.win_rate.labels(symbol=symbol).set(win_rate)
        self.sharpe_ratio.labels(symbol=symbol).set(sharpe)
        self.max_drawdown.labels(symbol=symbol).set(drawdown)
    
    def update_positions(self, symbol: str, count: int, total_value: float):
        """Update position metrics."""
        self.active_positions.labels(symbol=symbol).set(count)
        self.position_value.labels(symbol=symbol).set(total_value)
    
    def record_prediction(self, model: str, symbol: str, confidence: str, accuracy: float, latency: float):
        """Record model prediction metrics."""
        self.model_predictions.labels(
            model=model, symbol=symbol, confidence_level=confidence
        ).inc()
        
        self.prediction_accuracy.labels(model=model, symbol=symbol).set(accuracy)
        self.prediction_latency.labels(model=model, symbol=symbol).observe(latency)
    
    def get_trading_summary(self, symbol: str) -> Dict[str, Any]:
        """Get trading summary for a symbol."""
        return {
            'symbol': symbol,
            'total_trades': self.trades_total.labels(symbol=symbol)._value.get(),
            'total_pnl': self.pnl_gauge.labels(symbol=symbol)._value.get(),
            'win_rate': self.win_rate.labels(symbol=symbol)._value.get(),
            'sharpe_ratio': self.sharpe_ratio.labels(symbol=symbol)._value.get(),
            'max_drawdown': self.max_drawdown.labels(symbol=symbol)._value.get(),
            'active_positions': self.active_positions.labels(symbol=symbol)._value.get()
        }

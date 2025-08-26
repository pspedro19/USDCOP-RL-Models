"""
Sample Data Fixtures
====================
Provides sample data for testing purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_ohlcv_data(symbol='USDCOP', size=1000, freq='5min', start_date='2024-01-01'):
    """Generate sample OHLCV data"""
    np.random.seed(42)
    
    dates = pd.date_range(start_date, periods=size, freq=freq)
    
    # Generate realistic price movement
    returns = np.random.randn(size) * 0.001  # 0.1% volatility
    log_prices = np.cumsum(returns)
    base_price = 4000
    prices = base_price * np.exp(log_prices)
    
    # Generate OHLCV
    data = pd.DataFrame({
        'time': dates,
        'open': prices * (1 + np.random.randn(size) * 0.0002),
        'high': prices * (1 + np.abs(np.random.randn(size)) * 0.0005),
        'low': prices * (1 - np.abs(np.random.randn(size)) * 0.0005),
        'close': prices,
        'volume': np.random.lognormal(10, 1, size),
        'tick_volume': np.random.randint(10, 1000, size),
        'spread': np.random.uniform(1, 5, size)
    })
    
    # Ensure OHLC consistency
    data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
    data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
    
    return data


def generate_featured_data(size=500):
    """Generate data with features for model training"""
    base_data = generate_ohlcv_data(size=size)
    
    # Add technical indicators
    base_data['sma_10'] = base_data['close'].rolling(10).mean()
    base_data['sma_20'] = base_data['close'].rolling(20).mean()
    base_data['sma_50'] = base_data['close'].rolling(50).mean()
    
    # RSI
    delta = base_data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    base_data['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = base_data['close'].ewm(span=12, adjust=False).mean()
    exp2 = base_data['close'].ewm(span=26, adjust=False).mean()
    base_data['macd'] = exp1 - exp2
    base_data['macd_signal'] = base_data['macd'].ewm(span=9, adjust=False).mean()
    base_data['macd_diff'] = base_data['macd'] - base_data['macd_signal']
    
    # Bollinger Bands
    sma = base_data['close'].rolling(20).mean()
    std = base_data['close'].rolling(20).std()
    base_data['bb_upper'] = sma + (std * 2)
    base_data['bb_lower'] = sma - (std * 2)
    base_data['bb_width'] = base_data['bb_upper'] - base_data['bb_lower']
    
    # Volume indicators
    base_data['volume_sma'] = base_data['volume'].rolling(20).mean()
    base_data['volume_ratio'] = base_data['volume'] / base_data['volume_sma']
    
    # Price patterns
    base_data['higher_high'] = (base_data['high'] > base_data['high'].shift(1)).astype(int)
    base_data['lower_low'] = (base_data['low'] < base_data['low'].shift(1)).astype(int)
    
    # Market microstructure
    base_data['log_returns'] = np.log(base_data['close'] / base_data['close'].shift(1))
    base_data['realized_volatility'] = base_data['log_returns'].rolling(20).std() * np.sqrt(252 * 288)
    
    return base_data.dropna()


def generate_trades_data(n_trades=100):
    """Generate sample trades data"""
    np.random.seed(42)
    
    trades = []
    current_time = datetime.now()
    
    for i in range(n_trades):
        trade = {
            'trade_id': f'T{i:05d}',
            'symbol': 'USDCOP',
            'timestamp': current_time - timedelta(hours=n_trades-i),
            'action': np.random.choice(['BUY', 'SELL']),
            'volume': np.random.choice([0.01, 0.1, 0.5, 1.0]),
            'price': 4000 + np.random.randn() * 20,
            'sl': None,
            'tp': None,
            'commission': np.random.uniform(0.1, 1.0),
            'profit': np.random.randn() * 50,
            'status': np.random.choice(['CLOSED', 'OPEN'], p=[0.8, 0.2])
        }
        
        # Add SL/TP for some trades
        if np.random.random() > 0.5:
            if trade['action'] == 'BUY':
                trade['sl'] = trade['price'] - np.random.uniform(10, 30)
                trade['tp'] = trade['price'] + np.random.uniform(20, 50)
            else:
                trade['sl'] = trade['price'] + np.random.uniform(10, 30)
                trade['tp'] = trade['price'] - np.random.uniform(20, 50)
        
        trades.append(trade)
    
    return pd.DataFrame(trades)


def generate_model_metrics():
    """Generate sample model training metrics"""
    metrics = {
        'training': {
            'loss': np.random.exponential(0.1, 100).tolist(),
            'accuracy': (0.5 + np.random.random(100) * 0.4).tolist(),
            'reward': np.cumsum(np.random.randn(100) * 10).tolist()
        },
        'validation': {
            'loss': np.random.exponential(0.15, 100).tolist(),
            'accuracy': (0.45 + np.random.random(100) * 0.35).tolist(),
            'sharpe_ratio': np.random.randn(100).tolist()
        },
        'test': {
            'total_return': 0.15,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.08,
            'win_rate': 0.58,
            'profit_factor': 1.8
        }
    }
    
    return metrics


# Export sample data generators
__all__ = [
    'generate_ohlcv_data',
    'generate_featured_data',
    'generate_trades_data',
    'generate_model_metrics'
]
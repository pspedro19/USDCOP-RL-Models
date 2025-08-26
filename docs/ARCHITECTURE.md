# USDCOP Trading RL System - Architecture Guide

## 🏗️ System Overview

The USDCOP Trading RL System is built with a modular, scalable architecture that separates concerns and enables easy testing and maintenance. The system follows the **Bronze→Silver→Gold** data quality pipeline and implements **Reinforcement Learning** for automated trading decisions.

## 🏛️ Architecture Principles

1. **Separation of Concerns**: Each component has a single responsibility
2. **Dependency Injection**: Components receive dependencies through configuration
3. **Interface Segregation**: Clean interfaces between components
4. **Single Responsibility**: Each class/module has one reason to change
5. **Open/Closed**: Open for extension, closed for modification

## 📊 Data Flow Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   MT5/API  │───▶│   Bronze    │───▶│   Silver    │
│  (Raw Data)│    │  (Raw Data) │    │(Processed)  │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Fallback   │    │   Quality   │    │    Gold     │
│  Manager    │    │   Tracker   │    │ (Validated) │
└─────────────┘    └─────────────┘    └─────────────┘
                           │                   │
                           ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐
                   │   Reports   │    │    Agent    │
                   │  & Alerts   │    │  Training   │
                   └─────────────┘    └─────────────┘
```

## 🧩 Component Architecture

### 1. Core Layer (`src/core/`)

#### Connectors (`src/core/connectors/`)
- **MT5Connector**: Primary data source connection
- **FallbackManager**: Automatic fallback when primary source fails
- **DataSimulator**: Brownian Bridge simulation for testing

#### Data Management (`src/core/data/`)
- **DataCollector**: Unified data collection interface
- **QualityTracker**: Bronze→Silver→Gold pipeline management
- **DataValidator**: OHLC and data integrity validation
- **StorageManager**: Parquet/CSV storage with compression

#### Base Classes (`src/core/base/`)
- **BaseAgent**: Abstract RL agent interface
- **BaseEnvironment**: Gym environment base class
- **BaseMetrics**: Performance metrics interface
- **BasePipeline**: Pipeline execution framework

### 2. Market-Specific Layer (`src/markets/usdcop/`)

#### USDCOP Implementation
- **Environment**: RL environment with realistic costs
- **Agent**: PPO/DQN implementation for forex
- **Metrics**: Forex-specific metrics (pips, Sharpe ratio)
- **FeatureEngine**: Technical indicators and features
- **Pipeline**: Complete USDCOP processing pipeline

### 3. Trading Layer (`src/trading/`)

#### Trading Components
- **Backtester**: Walk-forward backtesting engine
- **RiskManager**: Position and risk controls
- **PositionManager**: Position sizing and management
- **SignalGenerator**: Trading signal generation

### 4. Visualization Layer (`src/visualization/`)

#### Dashboard Components
- **Dashboard**: Main Dash application
- **HealthMonitor**: System health and status
- **Charts**: Interactive candlestick charts
- **MetricsCards**: Performance metrics display
- **EquityCurve**: Equity curve and drawdown

### 5. Utilities (`src/utils/`)

#### Utility Components
- **Logger**: Centralized structured logging
- **ConfigLoader**: YAML/JSON configuration management
- **Helpers**: Timezone and encoding utilities
- **Performance**: Ray/Numba optimizations

## 🔄 Data Pipeline Architecture

### Bronze Layer (Raw Data)
```python
# Raw data from MT5 or fallback sources
bronze_data = {
    'time': timestamp,
    'open': float,
    'high': float,
    'low': float,
    'close': float,
    'volume': int,
    'source': 'mt5|fallback|simulation'
}
```

### Silver Layer (Processed Data)
```python
# Data with technical indicators and features
silver_data = {
    # OHLC data
    'time': timestamp,
    'open': float,
    'high': float,
    'low': float,
    'close': float,
    'volume': int,
    
    # Technical indicators
    'rsi': float,
    'macd': float,
    'bb_position': float,
    'atr': float,
    
    # Features
    'returns': float,
    'volatility': float,
    'trend_strength': float
}
```

### Gold Layer (Validated Data)
```python
# Final validated dataset for training/trading
gold_data = {
    # All silver data
    ...silver_data,
    
    # Quality metrics
    'quality_score': float,
    'validation_status': 'valid|warning|error',
    'last_updated': timestamp,
    
    # Metadata
    'data_version': str,
    'processing_pipeline': str
}
```

## 🤖 Reinforcement Learning Architecture

### Agent Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   State Space   │───▶│   Neural Net    │───▶│   Action Space  │
│   (30 features) │    │   (Policy/Value)│    │  (BUY/SELL/HOLD)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Environment   │◀───│     Reward      │◀───│   Execution     │
│   (Market Sim)  │    │   Function      │    │   (Trades)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Environment Design
```python
class USDCOPEnvironment:
    def __init__(self):
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(30,))
        self.action_space = Discrete(3)  # BUY, SELL, HOLD
        
    def step(self, action):
        # Execute action
        # Calculate reward
        # Update state
        # Return observation, reward, done, info
        
    def reset(self):
        # Reset to initial state
        # Return initial observation
```

## 🚀 Performance Architecture

### Optimization Strategies
1. **Ray**: Distributed computing for large datasets
2. **Numba**: JIT compilation for numerical operations
3. **Parquet**: Columnar storage with compression
4. **Async I/O**: Non-blocking data collection
5. **Caching**: Intelligent data caching strategies

### Scalability Features
- **Horizontal Scaling**: Multiple worker processes
- **Load Balancing**: Distributed task distribution
- **Resource Management**: Memory and CPU optimization
- **Fault Tolerance**: Automatic retry and recovery

## 🔒 Security Architecture

### Security Layers
1. **Authentication**: API key management
2. **Authorization**: Role-based access control
3. **Encryption**: Data encryption at rest and in transit
4. **Audit Logging**: Comprehensive activity logging
5. **Rate Limiting**: API usage throttling

## 🧪 Testing Architecture

### Testing Strategy
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **End-to-End Tests**: Complete system testing
- **Performance Tests**: Load and stress testing
- **Mock Testing**: External dependency simulation

### Test Coverage
```
src/
├── core/           # 90%+ coverage
├── markets/        # 85%+ coverage
├── trading/        # 80%+ coverage
├── visualization/  # 75%+ coverage
└── utils/          # 95%+ coverage
```

## 📊 Monitoring Architecture

### Monitoring Components
1. **Health Checks**: System status monitoring
2. **Performance Metrics**: Real-time performance tracking
3. **Alerting**: Automated alert generation
4. **Logging**: Structured logging with correlation IDs
5. **Tracing**: Request flow tracing

### Metrics Collection
```python
# Key Performance Indicators
metrics = {
    'data_quality': {
        'bronze_score': float,
        'silver_score': float,
        'gold_score': float
    },
    'trading_performance': {
        'total_return': float,
        'sharpe_ratio': float,
        'max_drawdown': float
    },
    'system_health': {
        'mt5_connection': bool,
        'data_freshness': timedelta,
        'error_rate': float
    }
}
```

## 🔧 Deployment Architecture

### Container Strategy
```dockerfile
# Multi-stage Docker build
FROM python:3.9-slim as base
# Install system dependencies
# Install Python packages
# Configure runtime environment
```

### Orchestration
```yaml
# Docker Compose services
services:
  usdcop_trading:
    build: .
    environment:
      - MT5_LOGIN=${MT5_LOGIN}
      - MT5_PASSWORD=${MT5_PASSWORD}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
```

## 📈 Future Architecture Considerations

### Planned Enhancements
1. **Microservices**: Break down into smaller services
2. **Event Streaming**: Kafka/RabbitMQ for real-time data
3. **Machine Learning Pipeline**: MLflow for model management
4. **Cloud Native**: Kubernetes deployment
5. **Multi-Market**: Support for additional forex pairs

### Scalability Roadmap
- **Phase 1**: Single market optimization
- **Phase 2**: Multi-market support
- **Phase 3**: Cloud deployment
- **Phase 4**: Enterprise features

---

This architecture provides a solid foundation for building a robust, scalable, and maintainable trading system while maintaining flexibility for future enhancements.

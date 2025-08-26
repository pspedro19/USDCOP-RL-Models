# 🚀 USDCOP Advanced Trading Dashboard

## 📊 Overview

A comprehensive real-time trading dashboard for USD/COP forex trading with trained RL models. Features model selection, backtesting visualization, and live trading capabilities.

## ✨ Features

### 1. **Model Selection Panel**
- Choose from 8+ trained models (TD3, DQN, Enhanced DQN, Rainbow DQN, PPO, PPO-LSTM, A2C, SAC, Ensemble)
- View model performance metrics (best return, average trades, win rate)
- One-click model loading

### 2. **Backtesting Visualization**
- Test models on historical data at 1-10x speed
- Real-time chart updates showing price and portfolio value
- Performance metrics:
  - Total Return
  - Sharpe Ratio
  - Win Rate
  - Max Drawdown
  - Total Trades
  - Profit Factor

### 3. **Live Trading Section**
- Real-time price charts
- Trading signals (BUY/HOLD/SELL) with confidence levels
- Market depth / Order book visualization
- Connection to MT5 and Kafka data streams

### 4. **Trade History**
- Complete log of all trades
- Shows time, model, action, price, size, P&L, confidence
- Color-coded for easy analysis

## 🚀 Quick Start

### Installation

1. Install required packages:
```bash
pip install -r dashboard/requirements.txt
```

2. Launch the dashboard:
```bash
python launch_trading_dashboard.py
```

3. Open browser at: http://localhost:8082

### Manual Start

1. Start the API server:
```bash
cd dashboard
python trading_api_server.py
```

2. Open `advanced_trading_dashboard.html` in your browser

## 📁 File Structure

```
dashboard/
├── advanced_trading_dashboard.html  # Main dashboard UI
├── trading_api_server.py           # Flask API backend
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## 🔌 API Endpoints

### Model Management
- `GET /api/models` - List available models
- `POST /api/load-model` - Load a specific model
- `POST /api/predict` - Get model prediction

### Backtesting
- `GET /api/get-test-data` - Get test dataset
- `POST /api/backtest` - Start backtest
- `POST /api/stop-backtest` - Stop running backtest

### Live Trading
- `GET /mt5/status` - MT5 connection status
- `POST /mt5/place-order` - Place MT5 order
- WebSocket: `ws://localhost:8082` - Real-time data stream

## 🎮 Usage Guide

### 1. Select a Model
- Click on any model card in the selection panel
- Green checkmark indicates selected model
- Click "Load Model" to load into memory

### 2. Run Backtest
- After loading a model, click "Start Backtest"
- Adjust speed slider (1-10x)
- Watch real-time visualization
- Monitor performance metrics

### 3. Live Trading
- Click "Live Trading" to connect to real-time data
- Monitor BUY/HOLD/SELL signals
- View order book and market depth
- Execute trades through MT5 integration

## 📊 Model Performance

| Model | Best Return | Avg Trades/Episode | Status |
|-------|------------|-------------------|---------|
| TD3 | +1.87% | 5.7 | ✅ Trained |
| DQN | -0.02% | 115.8 | ✅ Trained |
| Enhanced DQN | TBD | TBD | 🔄 Training |
| Rainbow DQN | TBD | TBD | 🔄 Training |
| PPO | -9.12% | 664.8 | ✅ Trained |
| PPO-LSTM | TBD | TBD | 🔄 Training |
| A2C | -6.01% | 613.4 | ✅ Trained |
| SAC | -4.43% | 226.3 | ✅ Trained |
| Ensemble | +2.5%* | 25.3 | ✅ Trained |

*Estimated based on weighted combination of TD3, DQN, and SAC

## 🔧 Configuration

Edit `CONFIG` object in dashboard HTML:

```javascript
const CONFIG = {
    wsUrl: 'ws://localhost:8765',      // WebSocket URL
    kafkaUrl: 'http://localhost:8080', // Kafka endpoint
    mt5Url: 'http://localhost:8081',   // MT5 bridge
    apiUrl: 'http://localhost:8082',   // API server
    updateInterval: 1000,               // Update frequency (ms)
    backtestSpeed: 5                    // Default backtest speed
};
```

## 🔗 Integrations

### MT5 Integration
- Requires MT5 Python API bridge (not included)
- Configure MT5 connection in `trading_api_server.py`

### Kafka Integration
- Requires Kafka server and producer (not included)
- Configure Kafka connection in `trading_api_server.py`

## 🎨 Customization

### Add New Models
1. Train model and save as `.pkl` file
2. Place in `models/trained/complete_5_models/`
3. Add to `MODELS` array in dashboard HTML

### Modify Charts
- Edit Chart.js configuration in `initializeCharts()` function
- Add new chart types or indicators

### Custom Metrics
- Add new metrics in `updateBacktestMetrics()` function
- Display in metrics grid

## ⚠️ Important Notes

1. **Demo Mode**: Current implementation simulates live data for demonstration
2. **Model Compatibility**: Ensure models have `predict()` method
3. **Performance**: Dashboard optimized for Chrome/Edge browsers
4. **Data Format**: Expects models trained on 26-feature state space

## 🐛 Troubleshooting

### Server won't start
```bash
# Check if port 8082 is in use
netstat -an | findstr :8082

# Kill process using port
taskkill /PID <process_id> /F
```

### Models not loading
- Verify `.pkl` files exist in `models/trained/complete_5_models/`
- Check console for error messages
- Ensure pickle protocol compatibility

### No live data
- Check WebSocket connection status
- Verify MT5/Kafka services are running
- Review browser console for errors

## 📈 Future Enhancements

- [ ] Real MT5 integration
- [ ] Kafka streaming integration  
- [ ] Multi-timeframe analysis
- [ ] Risk management controls
- [ ] Portfolio optimization
- [ ] Automated trading execution
- [ ] Performance reports export
- [ ] Mobile responsive design

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please submit pull requests or open issues.

---

**Built with**: Flask, Socket.IO, Chart.js, HTML5, CSS3, JavaScript

**Author**: USDCOP Trading Team

**Version**: 1.0.0
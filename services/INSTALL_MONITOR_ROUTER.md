# Installing the Model Monitor Router

Due to OneDrive sync issues, please add the following code manually to `multi_model_trading_api.py`.

## Step 1: Add Router Import

After the CORS middleware configuration (around line 257), add:

```python
# Include model monitoring router
try:
    from monitor_router import router as monitor_router
    app.include_router(monitor_router, prefix="/api/monitor", tags=["monitoring"])
    logger.info("Model monitoring router included successfully")
except ImportError as e:
    logger.warning(f"Could not import monitor_router: {e}")
```

## Step 2: Verify Installation

After adding the code, restart the API service and verify the endpoints are available:

- GET `/api/monitor/health` - Get health status of all models
- GET `/api/monitor/{model_id}/health` - Get health for specific model
- POST `/api/monitor/{model_id}/record-action` - Record model action
- POST `/api/monitor/{model_id}/record-pnl` - Record trade PnL
- POST `/api/monitor/{model_id}/set-baseline` - Set baseline from backtest
- POST `/api/monitor/{model_id}/reset` - Reset monitor history

## Files Created

1. `src/monitoring/__init__.py` - Module initialization
2. `src/monitoring/model_monitor.py` - ModelMonitor class implementation
3. `services/monitor_router.py` - FastAPI router for monitoring endpoints

## Testing

```bash
# Test the health endpoint
curl http://localhost:8006/api/monitor/health

# Record an action
curl -X POST "http://localhost:8006/api/monitor/ppo_primary/record-action?action=0.5"

# Get model-specific health
curl http://localhost:8006/api/monitor/ppo_primary/health
```

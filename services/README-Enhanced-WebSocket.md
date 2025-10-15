# Enhanced WebSocket Service with L0 Integration

## Overview

The Enhanced WebSocket Service is a sophisticated real-time market data service that intelligently integrates with the L0 backup system to provide seamless data flow from historical to real-time modes. It operates exclusively during Colombian market hours (8:00-12:55 COT, Monday-Friday) and includes comprehensive health monitoring, API key management, and fallback mechanisms.

## Key Features

### 🔍 L0 Integration & Smart Startup
- **Ready Signal Detection**: Waits for `/data/ready-signals/l0_ready.flag` before starting
- **Data Freshness Checks**: Validates L0 data is recent (< 24 hours) and complete (> 95%)
- **Smart Startup Sequence**: Only starts real-time collection when L0 foundation is ready
- **Seamless Handover**: Smooth transition from L0 historical to WebSocket real-time data

### ⏰ Market Hours Intelligence
- **Colombian Business Hours**: Operates only L-V 8:00-12:55 COT
- **Holiday Awareness**: Integrates with Colombian holiday calendar
- **Session Management**: Automatic start/stop based on market schedule
- **Next Session Info**: Provides information about upcoming market sessions

### 🔑 API Key Management
- **Multi-Key Support**: Manages up to 16 TwelveData API keys across 2 groups
- **Intelligent Rotation**: Automatic key rotation with 8-second rate limiting
- **Usage Tracking**: Monitors API call distribution and key performance
- **Fallback Support**: Multiple environment variable formats supported

### 💾 Data Processing & Storage
- **Real-time Data**: Stores tick-by-tick data in `realtime_market_data` table
- **OHLC Aggregation**: Creates 5-minute bars in `market_data` table
- **Redis Caching**: Ultra-fast price caching with pub/sub broadcasting
- **Buffer Management**: Intelligent batching for optimal database performance

### 📊 Health Monitoring
- **Comprehensive Metrics**: Database, Redis, WebSocket, and system resource monitoring
- **Status Tracking**: Real-time service status with detailed error reporting
- **Connection Statistics**: WebSocket client connection tracking
- **Performance Metrics**: Response times, processing rates, and error counts

### 🔄 Error Handling & Recovery
- **Fallback Mechanisms**: Falls back to L0 incremental if real-time fails
- **Connection Recovery**: Automatic WebSocket reconnection during market hours
- **Graceful Degradation**: Service continues with available components
- **Detailed Logging**: Comprehensive error tracking and debugging information

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   L0 Backup     │    │  Enhanced        │    │  WebSocket      │
│   System        │───▶│  WebSocket       │───▶│  Clients        │
│                 │    │  Service         │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Ready Signals   │    │  PostgreSQL      │    │  Redis Pub/Sub  │
│ /data/ready-    │    │  market_data     │    │  Price Cache    │
│ signals/        │    │  realtime_data   │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Service States

The service operates through several states:

1. **INITIALIZING**: Setting up connections and components
2. **WAITING_L0**: Waiting for L0 ready signal
3. **CHECKING_DATA**: Validating L0 data quality
4. **READY**: Ready to start real-time collection
5. **ACTIVE**: Actively collecting real-time data
6. **MARKET_CLOSED**: Market hours inactive
7. **ERROR**: Error state with automatic recovery
8. **SHUTDOWN**: Graceful shutdown in progress

## Installation & Setup

### 1. Environment Configuration

Copy and configure the environment file:

```bash
cp .env.enhanced-websocket .env
```

Edit the `.env` file with your specific configuration:

```bash
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/usdcop_trading

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# TwelveData API Keys
TWELVEDATA_API_KEY_G1_1=your_real_api_key_here
TWELVEDATA_API_KEY_G1_2=your_real_api_key_here
# ... configure all 16 keys

# Directories
READY_SIGNALS_DIR=/data/ready-signals
BACKUP_DIR=/home/GlobalForex/USDCOP-RL-Models/data/backups
```

### 2. Directory Setup

Create necessary directories:

```bash
mkdir -p /data/ready-signals
mkdir -p /data/backups
mkdir -p ./logs
```

### 3. Database Setup

Ensure your PostgreSQL database has the required tables:

```sql
-- Market data table
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    datetime TIMESTAMP WITH TIME ZONE NOT NULL,
    open DECIMAL(12,6),
    high DECIMAL(12,6),
    low DECIMAL(12,6),
    close DECIMAL(12,6),
    volume INTEGER DEFAULT 0,
    source VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, datetime, source)
);

-- Real-time market data table
CREATE TABLE IF NOT EXISTS realtime_market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    bid DECIMAL(12,6),
    ask DECIMAL(12,6),
    last DECIMAL(12,6),
    volume INTEGER DEFAULT 0,
    spread DECIMAL(12,6),
    session_date DATE,
    trading_session BOOLEAN DEFAULT true,
    source VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System health table
CREATE TABLE IF NOT EXISTS system_health (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(100),
    status VARCHAR(50),
    details JSONB,
    response_time_ms INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### 4. Docker Deployment

#### Option A: Docker Compose (Recommended)

```bash
# Start all services
docker-compose -f docker-compose.enhanced-websocket.yml up -d

# View logs
docker-compose -f docker-compose.enhanced-websocket.yml logs -f enhanced-websocket

# Stop services
docker-compose -f docker-compose.enhanced-websocket.yml down
```

#### Option B: Standalone Docker

```bash
# Build the image
docker build -f Dockerfile.enhanced-websocket -t enhanced-websocket-service .

# Run the container
docker run -d \
  --name enhanced-websocket \
  -p 8080:8080 \
  --env-file .env \
  -v /data/ready-signals:/data/ready-signals \
  -v ./data/backups:/data/backups \
  enhanced-websocket-service
```

### 5. Local Development

```bash
# Install dependencies
pip install -r requirements-enhanced-websocket.txt

# Set environment variables
export DATABASE_URL="postgresql://username:password@localhost:5432/usdcop_trading"
export REDIS_URL="redis://localhost:6379/0"
export TWELVEDATA_API_KEY_G1_1="your_api_key_here"

# Run the service
python enhanced_websocket_service.py
```

## Usage Examples

### Health Check

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "service": "enhanced-websocket-service",
  "version": "4.0.0",
  "service_info": {
    "name": "enhanced_websocket_service",
    "status": "active",
    "uptime_seconds": 1234
  },
  "connectivity": {
    "database": {"connected": true, "response_time_ms": 5},
    "redis": {"connected": true, "response_time_ms": 2}
  },
  "system_resources": {
    "cpu_percent": 15.2,
    "memory_percent": 45.8,
    "disk_percent": 67.3
  }
}
```

### Detailed Status

```bash
curl http://localhost:8080/status
```

Response:
```json
{
  "service_status": "active",
  "is_running": true,
  "l0_integration": {
    "ready_signal_exists": true,
    "data_freshness_hours": 2.5,
    "completeness_percentage": 98.7,
    "is_fresh": true,
    "is_complete": true
  },
  "market_session": {
    "status": "open",
    "current_session_start": "2025-01-20T08:00:00-05:00",
    "current_session_end": "2025-01-20T12:55:00-05:00",
    "minutes_until_close": 145
  },
  "api_key_usage": {
    "total_keys": 16,
    "current_key_index": 3,
    "total_calls": 127
  },
  "realtime_collection": {
    "active": true,
    "websocket_connected": true
  }
}
```

### Latest Price

```bash
curl http://localhost:8080/market/latest
```

Response:
```json
{
  "symbol": "USDCOP",
  "price": 4275.45,
  "bid": 4275.20,
  "ask": 4275.70,
  "spread": 0.50,
  "volume": 125000,
  "timestamp": "2025-01-20T10:30:15-05:00",
  "source": "twelvedata_websocket"
}
```

### WebSocket Connection

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = function(event) {
    console.log('Connected to Enhanced WebSocket Service');

    // Send heartbeat
    ws.send(JSON.stringify({
        type: 'heartbeat'
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);

    if (data.type === 'market_data') {
        console.log('Price update:', data.data);
    } else if (data.type === 'heartbeat_response') {
        console.log('Server status:', data.server_status);
    }
};

// Symbol-specific WebSocket
const symbolWs = new WebSocket('ws://localhost:8080/ws/USDCOP');
```

## Integration with L0 System

### Ready Signal Creation

The L0 system should create a ready signal file when historical data is complete:

```bash
# L0 system creates this file when ready
echo '{"ready": true, "timestamp": "2025-01-20T07:00:00-05:00"}' > /data/ready-signals/l0_ready.flag
```

### Data Handover Process

1. **L0 Completion**: L0 system creates ready signal
2. **Signal Detection**: WebSocket service detects signal
3. **Data Validation**: Service validates L0 data quality
4. **Market Check**: Service checks if market is open
5. **Real-time Start**: Service begins WebSocket collection
6. **Seamless Transition**: Continuous data flow established

### Backup Integration

The service works with L0 backup files:

```
/data/backups/
├── usdcop_m5_complete_backup.parquet    # Historical data
└── usdcop_m5_backup_metadata.json       # Metadata
```

## Monitoring & Troubleshooting

### Log Analysis

```bash
# View service logs
docker logs enhanced-websocket-service

# Follow logs in real-time
docker logs -f enhanced-websocket-service

# Search for specific events
docker logs enhanced-websocket-service | grep "L0 ready signal"
```

### Common Issues

#### 1. Service Stuck in WAITING_L0

**Symptoms**: Service status shows "waiting_l0"

**Solutions**:
- Check if L0 ready signal exists: `ls -la /data/ready-signals/`
- Verify L0 system completed successfully
- Check L0 data freshness in database
- Manually create ready signal for testing

#### 2. Database Connection Issues

**Symptoms**: Database connectivity false in health check

**Solutions**:
- Verify DATABASE_URL environment variable
- Check PostgreSQL service status
- Validate database credentials
- Ensure required tables exist

#### 3. API Key Exhaustion

**Symptoms**: WebSocket connection errors, API rate limits

**Solutions**:
- Check API key usage in status endpoint
- Verify all 16 API keys are configured
- Ensure keys are valid and active
- Check TwelveData account limits

#### 4. Redis Connection Issues

**Symptoms**: Caching failures, pub/sub not working

**Solutions**:
- Verify REDIS_URL environment variable
- Check Redis service status
- Test Redis connectivity: `redis-cli ping`
- Check Redis memory usage

### Performance Tuning

#### Database Performance

```bash
# Monitor database connections
SELECT * FROM pg_stat_activity WHERE datname = 'usdcop_trading';

# Check table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(tablename::text))
FROM pg_tables WHERE schemaname = 'public';
```

#### Redis Performance

```bash
# Monitor Redis performance
redis-cli info memory
redis-cli info stats
```

#### System Resources

```bash
# Monitor service resource usage
docker stats enhanced-websocket-service

# Check disk usage for data directories
du -sh /data/ready-signals /data/backups
```

## Security Considerations

### Environment Variables

- Store sensitive API keys in secure environment files
- Use different API keys for different environments
- Rotate API keys regularly
- Monitor API key usage for anomalies

### Network Security

- Use HTTPS in production (configure SSL_CERT_PATH and SSL_KEY_PATH)
- Implement proper CORS configuration
- Consider VPN or private networks for database access
- Use Redis AUTH in production environments

### Database Security

- Use strong database passwords
- Implement database connection encryption
- Regular security updates for PostgreSQL
- Monitor database access logs

## API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Comprehensive health check |
| `/status` | GET | Detailed service status |
| `/market/latest` | GET | Latest cached price |
| `/market/session` | GET | Market session information |

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `/ws` | General real-time data updates |
| `/ws/{symbol}` | Symbol-specific updates |

### Response Formats

All API responses include:
- Timestamp in ISO 8601 format with timezone
- Service status information
- Error details when applicable
- Comprehensive metadata

## Development & Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/

# Run with coverage
pytest --cov=enhanced_websocket_service tests/
```

### Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements-enhanced-websocket.txt`
3. Configure environment variables
4. Set up local PostgreSQL and Redis
5. Run the service: `python enhanced_websocket_service.py`

### Contributing

1. Follow Python PEP 8 style guidelines
2. Add comprehensive docstrings
3. Write tests for new features
4. Update documentation
5. Test with Colombian market hours simulation

## License

This software is part of the USDCOP Trading System and follows the project's licensing terms.

## Support

For technical support and questions:
- Review logs for detailed error information
- Check health and status endpoints
- Verify L0 system integration
- Monitor database and Redis connectivity
- Validate API key configuration and usage

The Enhanced WebSocket Service is designed to provide reliable, intelligent real-time market data with seamless L0 integration and comprehensive monitoring capabilities.
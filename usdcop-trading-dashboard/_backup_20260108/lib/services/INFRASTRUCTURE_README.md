# Infrastructure Support Components

Complete infrastructure stack for the USDCOP Trading System including caching, logging, health monitoring, and audit trails.

## Components

### 1. Redis Integration (`lib/services/cache/`)

In-memory caching system with Redis-compatible API (uses Map-based implementation for development).

#### Files
- `RedisClient.ts` - Core cache client with TTL support
- `SignalCache.ts` - Trading signal cache manager
- `MetricsCache.ts` - Financial metrics and position cache
- `types.ts` - Type definitions
- `index.ts` - Barrel export

#### Usage

```typescript
import { getSignalCache, getMetricsCache } from '@/lib/services/cache';

// Store and retrieve signals
const signalCache = getSignalCache();

await signalCache.setLatest({
  signal_id: 'sig_123',
  timestamp: new Date().toISOString(),
  symbol: 'USDCOP',
  action: 'BUY',
  confidence: 0.85,
  price: 4250.50,
  features: { rsi: 45, macd: 0.2 },
  model_version: 'v1.0',
  execution_latency_ms: 45
});

const latest = await signalCache.getLatest();
const history = await signalCache.getHistory('USDCOP', 50);
const stats = await signalCache.getSignalStats('USDCOP');

// Store metrics
const metricsCache = getMetricsCache();

await metricsCache.setFinancialMetrics({
  timestamp: new Date().toISOString(),
  total_pnl: 1250.75,
  win_rate: 0.62,
  sharpe_ratio: 1.8,
  max_drawdown: -0.15,
  total_trades: 45,
  active_positions: 3,
  portfolio_value: 50000
});

const metrics = await metricsCache.getFinancialMetrics();
```

### 2. Structured Logging (`lib/services/logging/`)

Multi-level logging system with context support, audit trails, and performance tracking.

#### Files
- `StructuredLogger.ts` - Main logger with context
- `AuditLogger.ts` - Audit trail for compliance
- `PerformanceLogger.ts` - Latency and performance tracking
- `types.ts` - Type definitions
- `index.ts` - Barrel export

#### Usage

```typescript
import { getLogger, getAuditLogger, getPerformanceLogger } from '@/lib/services/logging';

// Structured logging
const logger = getLogger({ service: 'TradingEngine' });

logger.info('Processing signal', { signal_id: 'sig_123' }, { confidence: 0.85 });
logger.warn('High latency detected', { service: 'api' }, { latency_ms: 1200 });
logger.error('Failed to execute trade', new Error('Connection timeout'),
  { symbol: 'USDCOP' }
);

// Create child logger with additional context
const childLogger = logger.child({ user_id: 'user_456' });

// Time async operations
const result = await logger.time('fetch_data', async () => {
  return await fetchData();
});

// Audit logging
const auditLogger = getAuditLogger();

await auditLogger.logSignalGenerated({
  signal_id: 'sig_123',
  symbol: 'USDCOP',
  action: 'BUY',
  confidence: 0.85,
  price: 4250.50,
  model_version: 'v1.0'
});

await auditLogger.logPositionOpened({
  position_id: 'pos_789',
  symbol: 'USDCOP',
  entry_price: 4250.50,
  quantity: 1000,
  entry_time: new Date().toISOString()
});

// Get audit trail
const trail = await auditLogger.getAuditTrail({
  eventType: 'SIGNAL_GENERATED',
  symbol: 'USDCOP',
  startTime: new Date('2025-01-01'),
  limit: 100
});

// Performance logging
const perfLogger = getPerformanceLogger();

const opId = perfLogger.startOperation('model_inference');
// ... do work ...
perfLogger.endOperation(opId, true, { model: 'xgboost' });

// Or measure directly
await perfLogger.measure('api_call', async () => {
  return await fetch('/api/data');
});

// Get metrics
const metrics = perfLogger.getMetrics('model_inference');
const summary = perfLogger.getSummary();
const slowOps = perfLogger.getSlowOperations(1000); // > 1000ms
```

### 3. Health Monitoring (`lib/services/health/`)

Service health checking, latency monitoring, and service registry.

#### Files
- `HealthChecker.ts` - Service health monitoring
- `LatencyMonitor.ts` - Latency tracking and analysis
- `ServiceRegistry.ts` - Service registration and status
- `types.ts` - Type definitions
- `index.ts` - Barrel export

#### Usage

```typescript
import { getHealthChecker, getLatencyMonitor, getServiceRegistry } from '@/lib/services/health';

// Register services
const registry = getServiceRegistry();

registry.register({
  name: 'trading-api',
  url: 'http://localhost:8001/health',
  checkInterval: 30000,
  timeout: 5000
});

registry.register({
  name: 'custom-service',
  checkInterval: 60000,
  healthCheck: async () => {
    // Custom health check logic
    const isHealthy = await checkCustomService();
    return {
      success: isHealthy,
      latency_ms: 50,
      metadata: { custom: 'data' }
    };
  }
});

// Health checking
const healthChecker = getHealthChecker();

// Check single service
const serviceHealth = await healthChecker.checkService('trading-api');

// Check all services
const systemHealth = await healthChecker.checkAllServices();

// Start automated monitoring
healthChecker.startMonitoring();

// Latency monitoring
const latencyMonitor = getLatencyMonitor();

latencyMonitor.recordLatency('trading-api', 'execute_trade', 125, true, {
  symbol: 'USDCOP'
});

// Get statistics
const stats = latencyMonitor.getLatencyStats('trading-api', 'execute_trade');
const avgLatency = latencyMonitor.getAverageLatency('trading-api', 5);
const trend = latencyMonitor.getLatencyTrend('trading-api', 'execute_trade', 5);
const highLatency = latencyMonitor.getHighLatencyServices(10);
```

### 4. API Routes (`app/api/`)

RESTful endpoints for accessing infrastructure services.

#### Routes

**Health Endpoints:**
- `GET /api/health/services` - All services health status
- `POST /api/health/services` - Check specific service
- `GET /api/health/latency` - Latency metrics
- `POST /api/health/latency` - Record latency
- `DELETE /api/health/latency` - Clear measurements

**Cache Endpoints:**
- `GET /api/cache/signals` - Get cached signals
- `POST /api/cache/signals` - Update signal cache
- `DELETE /api/cache/signals` - Clear cache

**Audit Endpoints:**
- `GET /api/audit/signals` - Get audit trail
- `POST /api/audit/signals` - Create audit entry
- `DELETE /api/audit/signals` - Clear audit trail

#### Examples

```typescript
// Check system health
const response = await fetch('/api/health/services');
const health = await response.json();
// {
//   overall_status: 'healthy',
//   services: [...],
//   summary: { total: 5, healthy: 5, degraded: 0, unhealthy: 0 }
// }

// Get latency stats
const latency = await fetch('/api/health/latency?service=trading-api&type=stats');
const stats = await latency.json();

// Get cached signals
const signals = await fetch('/api/cache/signals?type=history&symbol=USDCOP&limit=50');
const data = await signals.json();

// Get audit trail
const audit = await fetch('/api/audit/signals?eventType=SIGNAL_GENERATED&limit=100');
const trail = await audit.json();
```

### 5. Dashboard Components (`components/monitoring/`)

React components for visualizing system health and metrics.

#### Components
- `SystemHealthDashboard.tsx` - Complete system overview
- `ServiceStatusCard.tsx` - Individual service status
- `LatencyChart.tsx` - Latency visualization
- `AuditLogViewer.tsx` - Audit trail viewer

#### Usage

```tsx
import { SystemHealthDashboard } from '@/components/monitoring';

export default function MonitoringPage() {
  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">System Monitoring</h1>

      <SystemHealthDashboard
        refreshInterval={30000}
        showLatency={true}
        showAuditLog={true}
      />
    </div>
  );
}
```

Individual components:

```tsx
import {
  ServiceStatusCard,
  LatencyChart,
  AuditLogViewer
} from '@/components/monitoring';

// Service status card
<ServiceStatusCard service={serviceHealth} />

// Latency chart
<LatencyChart refreshInterval={30000} minutes={5} />

// Audit log viewer
<AuditLogViewer limit={50} autoRefresh={true} />
```

## Cache Structure

The Redis-compatible cache uses the following key structure:

```
usdcop:signal:latest                    -> Latest signal (JSON)
usdcop:signal:history:USDCOP            -> Signal history array
usdcop:position:active:USDCOP           -> Active position data
usdcop:metrics:financial                -> Financial metrics snapshot
usdcop:metrics:custom:key_name          -> Custom metric values
usdcop:health:service_name              -> Service health status
```

## Health Check Format

Services report health in this standard format:

```typescript
{
  service: "trading-api",
  status: "healthy" | "degraded" | "unhealthy",
  latency_ms: 45,
  last_check: "2025-01-15T10:30:00Z",
  details: {
    uptime: 86400,              // seconds
    requests_per_minute: 120,
    error_rate: 0.02,           // 2%
    memory_usage_mb: 512
  },
  metadata: {
    // Service-specific data
  }
}
```

## Performance Considerations

1. **Cache TTL**: Default 5 minutes, configurable per operation
2. **Max Cache Size**: 1000 entries by default, enforced with FIFO eviction
3. **Cleanup Interval**: Expired entries removed every 60 seconds
4. **Latency Buffer**: Last 10,000 measurements kept
5. **Audit Trail**: Last 10,000 events retained in memory

## Environment Variables

```env
# Service URLs
TRADING_API_URL=http://localhost:8001
ML_ANALYTICS_URL=http://localhost:8002
PIPELINE_API_URL=http://localhost:8000
WS_URL=http://localhost:8080

# PostgreSQL
POSTGRES_URL=postgresql://user:pass@localhost:5432/usdcop

# Logging
LOG_LEVEL=info  # debug | info | warn | error | fatal
```

## Integration Example

Complete integration in a trading application:

```typescript
import { getSignalCache, getMetricsCache } from '@/lib/services/cache';
import { getLogger, getAuditLogger, getPerformanceLogger } from '@/lib/services/logging';
import { getHealthChecker, getLatencyMonitor } from '@/lib/services/health';

class TradingEngine {
  private signalCache = getSignalCache();
  private metricsCache = getMetricsCache();
  private logger = getLogger({ service: 'TradingEngine' });
  private auditLogger = getAuditLogger();
  private perfLogger = getPerformanceLogger();
  private latencyMonitor = getLatencyMonitor();

  async processSignal(signal: any) {
    const opId = this.perfLogger.startOperation('process_signal');
    const startTime = Date.now();

    try {
      this.logger.info('Processing signal', { signal_id: signal.signal_id });

      // Cache the signal
      await this.signalCache.setLatest(signal);
      await this.signalCache.addToHistory(signal.symbol, signal);

      // Audit trail
      await this.auditLogger.logSignalGenerated(signal);

      // Execute trade logic
      const result = await this.executeTrade(signal);

      // Update metrics
      await this.updateMetrics(result);

      // Record latency
      const latency = Date.now() - startTime;
      this.latencyMonitor.recordLatency('trading-engine', 'process_signal', latency, true);

      this.perfLogger.endOperation(opId, true, { signal_id: signal.signal_id });
      this.logger.info('Signal processed successfully', { signal_id: signal.signal_id });

      return result;
    } catch (error) {
      const latency = Date.now() - startTime;
      this.latencyMonitor.recordLatency('trading-engine', 'process_signal', latency, false);

      this.perfLogger.endOperation(opId, false, { error: (error as Error).message });
      this.logger.error('Failed to process signal', error as Error, {
        signal_id: signal.signal_id
      });

      throw error;
    }
  }
}
```

## Testing

Test infrastructure components:

```typescript
import { RedisClient } from '@/lib/services/cache';
import { StructuredLogger } from '@/lib/services/logging';
import { HealthChecker } from '@/lib/services/health';

describe('Infrastructure Components', () => {
  test('Cache operations', async () => {
    const cache = new RedisClient({ ttl: 60, namespace: 'test' });

    await cache.set('key', 'value');
    const value = await cache.get('key');
    expect(value).toBe('value');

    await cache.clear();
  });

  test('Logging', () => {
    const logger = new StructuredLogger({ service: 'test' });
    logger.info('Test message', { test: true });
    expect(logger.getContext()).toEqual({ service: 'test' });
  });

  test('Health checking', async () => {
    const checker = new HealthChecker();
    checker.registerService({
      name: 'test',
      healthCheck: async () => ({ success: true, latency_ms: 10 })
    });

    const health = await checker.checkService('test');
    expect(health.status).toBe('healthy');
  });
});
```

## Production Deployment

For production, replace the in-memory cache with actual Redis:

```typescript
// Install ioredis
npm install ioredis

// Update RedisClient.ts to use real Redis
import Redis from 'ioredis';

const redis = new Redis({
  host: process.env.REDIS_HOST,
  port: parseInt(process.env.REDIS_PORT || '6379'),
  password: process.env.REDIS_PASSWORD,
});
```

## Monitoring Best Practices

1. **Set up health check monitoring** every 30 seconds
2. **Track latency percentiles** (P50, P95, P99) for all operations
3. **Maintain audit logs** for at least 30 days
4. **Alert on degraded services** with latency > 1000ms
5. **Monitor cache hit rates** and adjust TTLs accordingly
6. **Review audit trails** daily for anomalies
7. **Export metrics** to external monitoring systems (Prometheus, Grafana)

## License

MIT

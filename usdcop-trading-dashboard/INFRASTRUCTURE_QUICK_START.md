# Infrastructure Components - Quick Start Guide

This guide provides a rapid overview of the infrastructure components added to the USDCOP Trading Dashboard.

## Table of Contents

1. [Overview](#overview)
2. [Quick Setup](#quick-setup)
3. [Basic Usage](#basic-usage)
4. [API Endpoints](#api-endpoints)
5. [Dashboard Integration](#dashboard-integration)
6. [Common Patterns](#common-patterns)

## Overview

The infrastructure stack includes:

- **Cache System**: Redis-compatible in-memory caching for signals, positions, and metrics
- **Logging System**: Structured logging with audit trails and performance tracking
- **Health Monitoring**: Service health checks, latency tracking, and service registry
- **API Routes**: RESTful endpoints for accessing all infrastructure services
- **Dashboard Components**: React components for visualizing system health and metrics

## Quick Setup

### 1. Import Components

```typescript
// Cache
import { getSignalCache, getMetricsCache } from '@/lib/services/cache';

// Logging
import { getLogger, getAuditLogger, getPerformanceLogger } from '@/lib/services/logging';

// Health
import { getHealthChecker, getLatencyMonitor } from '@/lib/services/health';

// Dashboard
import { SystemHealthDashboard } from '@/components/monitoring';
```

### 2. Initialize Services

```typescript
// In your app initialization
const signalCache = getSignalCache();
const logger = getLogger({ service: 'TradingEngine' });
const healthChecker = getHealthChecker();

// Register services for monitoring
healthChecker.registerService({
  name: 'trading-api',
  url: 'http://localhost:8001/health',
  checkInterval: 30000,
});

// Start health monitoring
healthChecker.startMonitoring();
```

## Basic Usage

### Caching Signals

```typescript
import { getSignalCache } from '@/lib/services/cache';

const cache = getSignalCache();

// Store signal
await cache.setLatest({
  signal_id: 'sig_123',
  timestamp: new Date().toISOString(),
  symbol: 'USDCOP',
  action: 'BUY',
  confidence: 0.85,
  price: 4250.50,
  features: { rsi: 45 },
  model_version: 'v1.0',
  execution_latency_ms: 45,
});

// Retrieve signal
const latest = await cache.getLatest();
const history = await cache.getHistory('USDCOP', 50);
```

### Logging Events

```typescript
import { getLogger, getAuditLogger } from '@/lib/services/logging';

const logger = getLogger({ service: 'MyService' });
const auditLogger = getAuditLogger();

// Structured logging
logger.info('Processing signal', { signal_id: 'sig_123' });
logger.error('Trade failed', error, { symbol: 'USDCOP' });

// Audit trail
await auditLogger.logSignalGenerated(signal);
await auditLogger.logPositionOpened(position);
```

### Health Monitoring

```typescript
import { getHealthChecker, getLatencyMonitor } from '@/lib/services/health';

const healthChecker = getHealthChecker();
const latencyMonitor = getLatencyMonitor();

// Check service health
const health = await healthChecker.checkService('trading-api');
const systemHealth = await healthChecker.checkAllServices();

// Record latency
latencyMonitor.recordLatency('api', 'execute_trade', 125, true);
const stats = latencyMonitor.getLatencyStats('api');
```

## API Endpoints

### Health Endpoints

```bash
# Get all services health
GET /api/health/services

# Check specific service
POST /api/health/services
Body: { "service": "trading-api" }

# Get latency metrics
GET /api/health/latency?service=trading-api&type=stats

# Record latency
POST /api/health/latency
Body: { "service": "api", "operation": "trade", "latency_ms": 125 }
```

### Cache Endpoints

```bash
# Get latest signal
GET /api/cache/signals?type=latest

# Get signal history
GET /api/cache/signals?type=history&symbol=USDCOP&limit=50

# Store signal
POST /api/cache/signals
Body: { "signal": {...}, "action": "set-and-add" }

# Clear cache
DELETE /api/cache/signals?symbol=USDCOP
```

### Audit Endpoints

```bash
# Get audit trail
GET /api/audit/signals?eventType=SIGNAL_GENERATED&limit=100

# Create audit entry
POST /api/audit/signals
Body: { "event_type": "SIGNAL_GENERATED", "data": {...} }

# Export audit log
GET /api/audit/signals?type=export&format=csv

# Get audit statistics
GET /api/audit/signals?type=stats
```

## Dashboard Integration

### Full System Dashboard

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

### Individual Components

```tsx
import {
  ServiceStatusCard,
  LatencyChart,
  AuditLogViewer,
} from '@/components/monitoring';

// Service status
<ServiceStatusCard service={serviceHealth} />

// Latency chart
<LatencyChart refreshInterval={30000} minutes={5} />

// Audit log
<AuditLogViewer limit={50} autoRefresh={true} />
```

## Common Patterns

### Pattern 1: Complete Signal Processing

```typescript
import { getSignalCache, getAuditLogger, getPerformanceLogger } from '@/lib/services';

async function processSignal(signal: any) {
  const perfLogger = getPerformanceLogger();
  const opId = perfLogger.startOperation('process_signal');

  try {
    // Cache signal
    const cache = getSignalCache();
    await cache.setLatest(signal);
    await cache.addToHistory(signal.symbol, signal);

    // Audit trail
    const auditLogger = getAuditLogger();
    await auditLogger.logSignalGenerated(signal);

    // Execute trade
    const result = await executeTrade(signal);

    // Log success
    perfLogger.endOperation(opId, true);
    return result;
  } catch (error) {
    perfLogger.endOperation(opId, false);
    throw error;
  }
}
```

### Pattern 2: Health Check with Alerts

```typescript
import { getHealthChecker, getLogger } from '@/lib/services';

const healthChecker = getHealthChecker();
const logger = getLogger({ service: 'HealthMonitor' });

healthChecker.registerService({
  name: 'critical-service',
  checkInterval: 10000,
  healthCheck: async () => {
    const health = await checkService();

    if (!health.success) {
      logger.error('Critical service unhealthy',
        new Error('Health check failed')
      );
      // Send alert
      await sendAlert('Service down');
    }

    return health;
  }
});

healthChecker.startMonitoring();
```

### Pattern 3: Performance Tracking

```typescript
import { getPerformanceLogger, getLatencyMonitor } from '@/lib/services';

const perfLogger = getPerformanceLogger();
const latencyMonitor = getLatencyMonitor();

// Measure function
const result = await perfLogger.measure('api_call', async () => {
  return await fetch('/api/data');
});

// Record for monitoring
const stats = perfLogger.getMetrics('api_call');
latencyMonitor.recordLatency('api', 'call', stats.avg_ms, true);
```

### Pattern 4: Audit Trail Query

```typescript
import { getAuditLogger } from '@/lib/services/logging';

const auditLogger = getAuditLogger();

// Get signal audit trail
const signalTrail = await auditLogger.getAuditTrail({
  eventType: 'SIGNAL_GENERATED',
  symbol: 'USDCOP',
  startTime: new Date('2025-01-01'),
  endTime: new Date(),
  limit: 100,
});

// Export for compliance
const csvExport = await auditLogger.exportAuditTrail('csv');
```

## File Structure

```
lib/services/
├── cache/
│   ├── RedisClient.ts          # Core cache client
│   ├── SignalCache.ts          # Signal caching
│   ├── MetricsCache.ts         # Metrics caching
│   ├── types.ts                # Type definitions
│   └── index.ts                # Exports
├── logging/
│   ├── StructuredLogger.ts     # Main logger
│   ├── AuditLogger.ts          # Audit trail
│   ├── PerformanceLogger.ts    # Performance tracking
│   ├── types.ts                # Type definitions
│   └── index.ts                # Exports
├── health/
│   ├── HealthChecker.ts        # Health monitoring
│   ├── LatencyMonitor.ts       # Latency tracking
│   ├── ServiceRegistry.ts      # Service registry
│   ├── types.ts                # Type definitions
│   └── index.ts                # Exports
└── examples/
    └── infrastructure-usage.ts # Usage examples

app/api/
├── health/
│   ├── services/route.ts       # Service health endpoint
│   └── latency/route.ts        # Latency metrics endpoint
├── cache/
│   └── signals/route.ts        # Signal cache endpoint
└── audit/
    └── signals/route.ts        # Audit log endpoint

components/monitoring/
├── SystemHealthDashboard.tsx   # Main dashboard
├── ServiceStatusCard.tsx       # Service status display
├── LatencyChart.tsx            # Latency visualization
├── AuditLogViewer.tsx          # Audit log viewer
└── index.ts                    # Exports
```

## Environment Variables

```env
# Service URLs
TRADING_API_URL=http://localhost:8001
ML_ANALYTICS_URL=http://localhost:8002
PIPELINE_API_URL=http://localhost:8000
WS_URL=http://localhost:8080
POSTGRES_URL=postgresql://user:pass@localhost:5432/usdcop

# Logging
LOG_LEVEL=info
```

## Testing

Quick test of infrastructure:

```typescript
import { getSignalCache, getLogger, getHealthChecker } from '@/lib/services';

async function testInfrastructure() {
  // Test cache
  const cache = getSignalCache();
  await cache.setLatest({ /* signal data */ });
  const latest = await cache.getLatest();
  console.log('✓ Cache working:', latest !== null);

  // Test logging
  const logger = getLogger();
  logger.info('Test log');
  console.log('✓ Logging working');

  // Test health
  const health = getHealthChecker();
  const status = await health.checkAllServices();
  console.log('✓ Health check working:', status.overall_status);
}
```

## Next Steps

1. **Integrate with Trading Engine**: Add caching and logging to your trading logic
2. **Set Up Monitoring Dashboard**: Create a monitoring page in your Next.js app
3. **Configure Health Checks**: Register all your services for monitoring
4. **Set Up Alerts**: Add alerting logic based on health check failures
5. **Export Metrics**: Integrate with external monitoring (Prometheus, Grafana)

## Documentation

- Full documentation: `lib/services/INFRASTRUCTURE_README.md`
- Usage examples: `lib/services/examples/infrastructure-usage.ts`
- API documentation: See individual route files

## Support

For issues or questions about the infrastructure components, refer to:
- Type definitions in `types.ts` files
- Example code in `examples/` directory
- Inline documentation in source files

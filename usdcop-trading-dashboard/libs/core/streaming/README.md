# Enterprise Streaming Infrastructure

A high-performance, scalable streaming data infrastructure designed for professional trading platforms. This system provides enterprise-grade reliability with sub-millisecond latency and 99.99% uptime.

## Features

### 🚀 Core Capabilities
- **Multi-Exchange Aggregation**: Connect to multiple data sources simultaneously
- **Smart Data Normalization**: Standardize data formats across all exchanges
- **Intelligent Throttling**: Max 60 updates/sec with smart buffering
- **Auto-Reconnection**: Exponential backoff with circuit breaker protection
- **Real-Time Processing**: Tick-by-tick aggregation and pattern detection
- **Order Book Reconstruction**: Full depth reconstruction with gap detection
- **Advanced Caching**: IndexedDB with compression and intelligent eviction

### 📊 Analytics & Monitoring
- **Data Quality Monitoring**: Real-time quality scoring and validation
- **Performance Metrics**: Comprehensive system health monitoring
- **Pattern Detection**: Real-time market pattern recognition
- **Microstructure Analysis**: Deep market structure analytics
- **Volume Profile**: Dynamic volume-at-price analysis

### 🔧 Technical Features
- **Web Workers**: Background processing for CPU-intensive tasks
- **TypeScript**: Full type safety with comprehensive interfaces
- **RxJS Reactive Streams**: Powerful data flow management
- **IndexedDB Caching**: Efficient local data persistence
- **Compression**: Smart data compression for optimal performance

## Quick Start

### Installation

```bash
npm install rxjs eventemitter3 idb
```

### Basic Setup

```typescript
import { createStreamingInfrastructure, StreamingUtils } from './libs/core/streaming';

// Create infrastructure with forex-optimized configuration
const config = StreamingUtils.createForexConfig();
const streaming = createStreamingInfrastructure(config);

// Add a data source
const source = StreamingUtils.createStreamSource(
  'TwelveData',
  'forex',
  'wss://ws.twelvedata.com/v1/quotes/price',
  ['USDCOP', 'EURUSD', 'GBPUSD']
);

await streaming.addSource(source);

// Subscribe to USDCOP ticks
const subscriptionId = await streaming.subscribe('USDCOP', 'tick');

// Listen to real-time data
streaming.getTickStream('USDCOP').subscribe(tick => {
  console.log('Real-time USDCOP tick:', tick);
});

// Monitor system health
streaming.getSystemHealthStream().subscribe(health => {
  console.log('System status:', health.status);
  console.log('Active streams:', health.activeStreams);
  console.log('Quality score:', health.qualityScore);
});
```

### Advanced Configuration

```typescript
import { StreamingInfrastructureConfig } from './libs/core/streaming';

const advancedConfig: Partial<StreamingInfrastructureConfig> = {
  // High-frequency throttling
  throttle: {
    enabled: true,
    maxUpdatesPerSecond: 60,
    burstLimit: 100,
    windowSize: 1000,
    strategy: 'merge' // Merge multiple updates into one
  },

  // Smart buffering
  buffer: {
    enabled: true,
    maxSize: 1000,
    maxAge: 300000, // 5 minutes
    compressionEnabled: true,
    persistToDisk: true,
    strategy: 'fifo'
  },

  // Resilient reconnection
  reconnect: {
    globalRetryLimit: 10,
    globalBackoffMultiplier: 2,
    circuitBreakerThreshold: 5,
    circuitBreakerTimeout: 60000,
    enableAdaptiveBackoff: true,
    enableJitter: true
  },

  // Advanced analytics
  tickAggregation: {
    intervals: ['1s', '5s', '15s', '30s', '1m', '5m'],
    enableVWAP: true,
    enableVolumeDelta: true,
    enableMicrostructure: true,
    enablePatternDetection: true,
    enableRealTimeIndicators: true
  },

  // Order book management
  orderBook: {
    maxLevels: 50,
    enableValidation: true,
    enableGapDetection: true,
    enableAnalytics: true,
    autoReconstruct: true
  },

  // High-performance caching
  cache: {
    enabled: true,
    maxSizeBytes: 100 * 1024 * 1024, // 100MB
    maxAge: 3600000, // 1 hour
    compression: true,
    persistToDisk: true,
    evictionStrategy: 'lru'
  }
};

const streaming = createStreamingInfrastructure(advancedConfig);
```

## Data Streams

### Market Ticks

```typescript
// Subscribe to real-time ticks
streaming.getTickStream('USDCOP').subscribe(tick => {
  console.log({
    symbol: tick.symbol,
    price: tick.last,
    bid: tick.bid,
    ask: tick.ask,
    spread: tick.ask - tick.bid,
    volume: tick.volume,
    timestamp: new Date(tick.timestamp),
    quality: tick.quality
  });
});

// Get aggregated data
streaming.getTickProcessor().getAggregatedStream('USDCOP', '1m').subscribe(agg => {
  console.log('1-minute OHLCV:', {
    open: agg.open,
    high: agg.high,
    low: agg.low,
    close: agg.close,
    volume: agg.volume,
    vwap: agg.vwap
  });
});
```

### Order Books

```typescript
// Real-time order book updates
streaming.getOrderBookStream('USDCOP').subscribe(book => {
  console.log('Order Book Update:', {
    symbol: book.symbol,
    bestBid: book.bids[0]?.price,
    bestAsk: book.asks[0]?.price,
    spread: book.spread,
    midPrice: book.midPrice,
    totalBidVolume: book.totalBidVolume,
    totalAskVolume: book.totalAskVolume,
    imbalance: book.imbalance
  });
});

// Order book analytics
streaming.getOrderBookManager().getAnalyticsStream('USDCOP').subscribe(analytics => {
  console.log('Market Microstructure:', {
    depth: analytics.depth,
    liquidity: analytics.liquidity,
    imbalance: analytics.imbalance,
    volatility: analytics.volatility
  });
});
```

### Pattern Detection

```typescript
// Real-time pattern detection
streaming.getTickProcessor().getPatternStream('USDCOP').subscribe(pattern => {
  console.log('Pattern Detected:', {
    type: pattern.type,
    confidence: pattern.confidence,
    description: pattern.description,
    timestamp: new Date(pattern.timestamp)
  });
});

// Volume profile analysis
streaming.getTickProcessor().getVolumeProfileStream('USDCOP').subscribe(profile => {
  console.log('Volume Profile:', {
    pointOfControl: profile.pointOfControl,
    valueAreaHigh: profile.valueAreaHigh,
    valueAreaLow: profile.valueAreaLow,
    totalVolume: profile.totalVolume
  });
});
```

## Configuration Examples

### Forex Trading Setup

```typescript
const forexConfig = StreamingUtils.createForexConfig();
const streaming = createStreamingInfrastructure(forexConfig);

// Add multiple forex sources
const sources = [
  StreamingUtils.createStreamSource('TwelveData', 'forex', 'wss://ws.twelvedata.com/v1/quotes/price', ['USDCOP']),
  StreamingUtils.createStreamSource('AlphaVantage', 'forex', 'wss://ws.alpha-vantage.com', ['USDCOP']),
  StreamingUtils.createStreamSource('Internal', 'internal', 'ws://localhost:8082/ws', ['USDCOP'])
];

for (const source of sources) {
  await streaming.addSource(source);
}

// Subscribe to major pairs
const symbols = ['USDCOP', 'EURUSD', 'GBPUSD', 'USDJPY'];
for (const symbol of symbols) {
  await streaming.subscribe(symbol, 'tick');
}
```

### Crypto Trading Setup

```typescript
const cryptoConfig = StreamingUtils.createCryptoConfig();
const streaming = createStreamingInfrastructure(cryptoConfig);

// Add crypto exchanges
const exchanges = [
  { name: 'Binance', url: 'wss://stream.binance.com:9443/ws', symbols: ['BTCUSDT', 'ETHUSDT'] },
  { name: 'Coinbase', url: 'wss://ws-feed.pro.coinbase.com', symbols: ['BTC-USD', 'ETH-USD'] },
  { name: 'Kraken', url: 'wss://ws.kraken.com', symbols: ['XBT/USD', 'ETH/USD'] }
];

for (const exchange of exchanges) {
  const source = StreamingUtils.createStreamSource(exchange.name, 'crypto', exchange.url, exchange.symbols);
  await streaming.addSource(source);
}

// Subscribe to order books and trades
await streaming.subscribe('BTCUSDT', 'orderbook');
await streaming.subscribe('BTCUSDT', 'trade');
```

### High-Frequency Trading Setup

```typescript
const hftConfig = StreamingUtils.createHFTConfig();
const streaming = createStreamingInfrastructure(hftConfig);

// Maximum performance configuration
streaming.getSystemHealthStream().subscribe(health => {
  if (health.latency > 10) { // 10ms threshold
    console.warn('High latency detected:', health.latency);
  }

  if (health.errorRate > 0.001) { // 0.1% threshold
    console.warn('High error rate:', health.errorRate);
  }
});
```

## Performance Optimization

### Memory Management

```typescript
// Monitor memory usage
streaming.getSystemMetricsStream().subscribe(metrics => {
  if (metrics.performance.memory > 0.8) {
    console.warn('High memory usage detected');

    // Trigger cache cleanup
    streaming.getCacheManager().clear();
  }
});

// Configure cache limits
const cacheConfig = {
  maxSizeBytes: 50 * 1024 * 1024, // 50MB limit
  maxAge: 1800000, // 30 minutes
  evictionStrategy: 'lru' as const
};
```

### Network Optimization

```typescript
// Optimize for low bandwidth
const lowBandwidthConfig = {
  throttle: {
    enabled: true,
    maxUpdatesPerSecond: 30, // Reduced update rate
    strategy: 'merge' as const
  },
  buffer: {
    compressionEnabled: true,
    maxSize: 500 // Smaller buffer
  }
};
```

### CPU Optimization

```typescript
// Reduce CPU usage
const cpuOptimizedConfig = {
  tickAggregation: {
    enablePatternDetection: false, // Disable CPU-intensive features
    enableRealTimeIndicators: false,
    intervals: ['1m', '5m'] // Fewer intervals
  },
  orderBook: {
    enableAnalytics: false,
    maxLevels: 10 // Fewer levels to process
  }
};
```

## Error Handling

### Connection Management

```typescript
// Handle connection issues
streaming.on('connection_failed', (event) => {
  console.error('Connection failed:', event);

  // Implement fallback logic
  activateBackupSource();
});

streaming.on('connection_restored', (event) => {
  console.log('Connection restored:', event);

  // Resume normal operations
  resumeNormalOperations();
});
```

### Data Quality Monitoring

```typescript
// Monitor data quality
streaming.on('data_quality_alert', (alert) => {
  if (alert.qualityScore < 80) {
    console.warn('Poor data quality detected:', alert);

    // Switch to backup source or alert user
    handlePoorDataQuality(alert);
  }
});

// Validation errors
streaming.on('validation_error', (error) => {
  console.error('Data validation failed:', error);

  // Log for analysis
  logValidationError(error);
});
```

### System Health Monitoring

```typescript
// Comprehensive health monitoring
streaming.getSystemHealthStream().subscribe(health => {
  switch (health.status) {
    case 'healthy':
      // All systems operational
      break;

    case 'degraded':
      console.warn('System performance degraded:', health);
      // Implement graceful degradation
      break;

    case 'critical':
      console.error('Critical system issues:', health);
      // Emergency procedures
      handleCriticalIssues(health);
      break;

    case 'offline':
      console.error('System offline');
      // Activate backup systems
      activateBackupSystems();
      break;
  }
});
```

## Integration Examples

### React Integration

```typescript
import React, { useEffect, useState } from 'react';
import { createStreamingInfrastructure } from './libs/core/streaming';

const TradingDashboard: React.FC = () => {
  const [streaming] = useState(() => createStreamingInfrastructure());
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [systemHealth, setSystemHealth] = useState<any>(null);

  useEffect(() => {
    // Setup streaming
    const setupStreaming = async () => {
      // Add source and subscribe
      const source = StreamingUtils.createStreamSource(
        'TwelveData',
        'forex',
        'wss://ws.twelvedata.com/v1/quotes/price',
        ['USDCOP']
      );

      await streaming.addSource(source);
      await streaming.subscribe('USDCOP', 'tick');
    };

    // Subscribe to data streams
    const tickSub = streaming.getTickStream('USDCOP').subscribe(tick => {
      setCurrentPrice(tick.last);
    });

    const healthSub = streaming.getSystemHealthStream().subscribe(health => {
      setSystemHealth(health);
    });

    setupStreaming().catch(console.error);

    return () => {
      tickSub.unsubscribe();
      healthSub.unsubscribe();
      streaming.destroy();
    };
  }, [streaming]);

  return (
    <div>
      <h1>USDCOP: {currentPrice.toFixed(4)}</h1>
      <div>Status: {systemHealth?.status}</div>
      <div>Quality: {systemHealth?.qualityScore}%</div>
    </div>
  );
};
```

### Node.js Backend Integration

```typescript
import { createStreamingInfrastructure } from './libs/core/streaming';
import WebSocket from 'ws';

class TradingDataServer {
  private streaming = createStreamingInfrastructure();
  private wss = new WebSocket.Server({ port: 8080 });

  async start() {
    // Setup streaming infrastructure
    await this.setupStreaming();

    // Setup WebSocket server for clients
    this.wss.on('connection', (ws) => {
      console.log('Client connected');

      // Send real-time data to clients
      const subscription = this.streaming.getTickStream().subscribe(tick => {
        ws.send(JSON.stringify({
          type: 'tick',
          data: tick
        }));
      });

      ws.on('close', () => {
        subscription.unsubscribe();
        console.log('Client disconnected');
      });
    });
  }

  private async setupStreaming() {
    // Add multiple sources for redundancy
    const sources = [
      StreamingUtils.createStreamSource('Primary', 'forex', 'wss://primary.com/ws', ['USDCOP']),
      StreamingUtils.createStreamSource('Backup', 'forex', 'wss://backup.com/ws', ['USDCOP'])
    ];

    for (const source of sources) {
      await this.streaming.addSource(source);
    }

    await this.streaming.subscribe('USDCOP', 'tick');
  }
}

const server = new TradingDataServer();
server.start().catch(console.error);
```

## Performance Benchmarks

### Latency Targets
- **Tick Processing**: < 1ms
- **Order Book Updates**: < 2ms
- **Pattern Detection**: < 5ms
- **Cache Operations**: < 0.5ms
- **Network Reconnection**: < 3s

### Throughput Targets
- **Messages/Second**: 10,000+
- **Concurrent Streams**: 50+
- **Memory Usage**: < 100MB
- **CPU Usage**: < 20%

### Reliability Targets
- **Uptime**: 99.99%
- **Data Completeness**: 99.9%
- **Error Rate**: < 0.1%
- **Recovery Time**: < 5s

## Best Practices

### 1. Resource Management
```typescript
// Always clean up resources
useEffect(() => {
  return () => {
    streaming.destroy();
  };
}, []);
```

### 2. Error Handling
```typescript
// Implement comprehensive error handling
streaming.on('error', (error) => {
  console.error('Streaming error:', error);
  // Log to monitoring service
  monitoring.logError(error);
});
```

### 3. Performance Monitoring
```typescript
// Monitor system performance
streaming.getSystemMetricsStream().subscribe(metrics => {
  if (metrics.performance.memory > 0.8) {
    // Trigger memory cleanup
    performMemoryCleanup();
  }
});
```

### 4. Data Validation
```typescript
// Validate incoming data
streaming.on('validation_error', (error) => {
  // Handle invalid data gracefully
  handleInvalidData(error);
});
```

## Troubleshooting

### Common Issues

#### High Memory Usage
```typescript
// Reduce buffer sizes
const config = {
  buffer: { maxSize: 500 },
  cache: { maxSizeBytes: 50 * 1024 * 1024 }
};
```

#### Connection Drops
```typescript
// Increase reconnection attempts
const config = {
  reconnect: {
    maxAttempts: 10,
    initialDelay: 500,
    maxDelay: 30000
  }
};
```

#### Slow Performance
```typescript
// Disable CPU-intensive features
const config = {
  tickAggregation: {
    enablePatternDetection: false,
    enableRealTimeIndicators: false
  }
};
```

## Support

For issues, questions, or contributions, please refer to the project documentation or contact the development team.

## License

Enterprise License - All rights reserved.
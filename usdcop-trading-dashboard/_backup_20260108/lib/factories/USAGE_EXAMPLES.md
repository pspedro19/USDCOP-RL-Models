# Factory Pattern Usage Examples

This file contains practical examples of how to use the factory pattern in the USD/COP Trading Dashboard.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Data Provider Examples](#data-provider-examples)
3. [Export Handler Examples](#export-handler-examples)
4. [Service Factory Examples](#service-factory-examples)
5. [Testing Examples](#testing-examples)
6. [React Component Examples](#react-component-examples)
7. [Configuration Examples](#configuration-examples)

---

## Basic Usage

### Quick Start

```typescript
import { ServiceFactory } from '@/lib/factories';

// Create a data provider
const dataProvider = ServiceFactory.createDataProvider('api');

// Fetch real-time data
const liveData = await dataProvider.getRealTimeData();
console.log('Current price:', liveData[0].price);

// Export data
await ServiceFactory.export('csv', liveData, 'market-data');
```

---

## Data Provider Examples

### Example 1: Using API Data Provider

```typescript
import { DataProviderFactory } from '@/lib/factories';

async function fetchMarketData() {
  // Create API provider
  const provider = DataProviderFactory.create('api', {
    apiBaseUrl: 'http://localhost:8000/api',
    timeout: 10000
  });

  // Check if API is healthy
  const health = await provider.checkHealth();
  console.log('API Status:', health.status);

  // Get real-time data
  const data = await provider.getRealTimeData();
  console.log('Current Price:', data[0].price);

  // Get historical candlesticks
  const candles = await provider.getCandlestickData(
    'USDCOP',
    '5m',
    undefined,
    undefined,
    100,
    true
  );
  console.log('Fetched', candles.count, 'candles');

  // Get symbol statistics
  const stats = await provider.getSymbolStats('USDCOP');
  console.log('24h Change:', stats.change_percent_24h.toFixed(2) + '%');
}
```

### Example 2: Using Mock Data Provider for Testing

```typescript
import { DataProviderFactory } from '@/lib/factories';

async function testWithMockData() {
  // Create mock provider
  const provider = DataProviderFactory.create('mock');

  // Subscribe to real-time updates
  const unsubscribe = provider.subscribeToRealTimeUpdates((data) => {
    console.log('Mock update:', data.price);
  });

  // Let it run for 10 seconds
  await new Promise(resolve => setTimeout(resolve, 10000));

  // Cleanup
  unsubscribe();
}
```

### Example 3: Using WebSocket Data Provider

```typescript
import { DataProviderFactory } from '@/lib/factories';

function setupRealtimeUpdates() {
  // Create WebSocket provider
  const provider = DataProviderFactory.create('websocket', {
    wsUrl: 'ws://localhost:8082/ws',
    apiBaseUrl: 'http://localhost:8000/api',
    reconnectAttempts: 5,
    reconnectDelay: 2000
  });

  // Subscribe to updates
  const unsubscribe = provider.subscribeToRealTimeUpdates((data) => {
    console.log('Real-time update:', {
      price: data.price,
      timestamp: new Date(data.timestamp).toISOString(),
      source: data.source
    });
  });

  return unsubscribe;
}
```

### Example 4: Automatic Fallback

```typescript
import { DataProviderFactory } from '@/lib/factories';

async function robustDataFetching() {
  // Try to use API, fallback to mock if it fails
  const provider = DataProviderFactory.createWithFallback('api');

  const data = await provider.getRealTimeData();
  console.log('Data source:', data[0].source);
}
```

### Example 5: Environment-Based Provider

```typescript
import { DataProviderFactory } from '@/lib/factories';

// Automatically selects provider based on environment
const provider = DataProviderFactory.createFromEnvironment();

// In test environment: uses mock
// In production: uses API or WebSocket based on config
```

---

## Export Handler Examples

### Example 1: CSV Export

```typescript
import { ExportFactory } from '@/lib/factories';

async function exportTradesToCSV(trades: Trade[]) {
  const csvHandler = ExportFactory.create('csv');

  const exportData = trades.map(trade => ({
    date: trade.timestamp,
    strategy: trade.strategyCode,
    side: trade.side,
    price: trade.price,
    size: trade.size,
    pnl: trade.pnl
  }));

  await csvHandler.export(exportData, 'trades-export');
}
```

### Example 2: Excel Export

```typescript
import { ExportFactory } from '@/lib/factories';

async function exportBacktestToExcel(results: BacktestResults) {
  const excelHandler = ExportFactory.create('excel');

  const exportData = {
    sheets: [
      {
        name: 'Summary',
        data: [
          { metric: 'Total Return', value: results.totalReturn },
          { metric: 'Sharpe Ratio', value: results.sharpeRatio },
          { metric: 'Max Drawdown', value: results.maxDrawdown }
        ]
      },
      {
        name: 'Trades',
        data: results.trades
      }
    ]
  };

  await excelHandler.export(exportData, 'backtest-results');
}
```

### Example 3: PDF Report

```typescript
import { ExportFactory } from '@/lib/factories';
import type { PdfExportData } from '@/lib/services/export';

async function generatePDFReport(backtest: BacktestResults) {
  const pdfHandler = ExportFactory.create('pdf');

  const reportData: PdfExportData = {
    metrics: [
      { label: 'Total Return', value: backtest.totalReturn + '%' },
      { label: 'Sharpe Ratio', value: backtest.sharpeRatio.toFixed(2) },
      { label: 'Max Drawdown', value: backtest.maxDrawdown + '%' },
      { label: 'Win Rate', value: backtest.winRate + '%' }
    ],
    tables: [
      {
        title: 'Top 10 Trades by P&L',
        headers: ['Date', 'Strategy', 'Side', 'Price', 'P&L'],
        rows: backtest.topTrades.map(t => [
          t.date,
          t.strategy,
          t.side,
          t.price.toString(),
          t.pnl.toString()
        ])
      }
    ]
  };

  await pdfHandler.export(reportData, 'backtest-report', {
    title: 'Backtest Analysis Report',
    subtitle: 'USD/COP Trading Strategy Performance',
    author: 'Trading System',
    company: 'Professional Trading Platform',
    watermark: 'CONFIDENTIAL'
  });
}
```

### Example 4: Quick Export

```typescript
import { ExportFactory } from '@/lib/factories';

// One-liner export
await ExportFactory.export('csv', myData, 'data-export');
```

### Example 5: Data Validation Before Export

```typescript
import { ExportFactory } from '@/lib/factories';

async function safeExport(data: any[], filename: string) {
  // Validate data before export
  if (!ExportFactory.validateData('csv', data)) {
    throw new Error('Invalid data format for CSV export');
  }

  await ExportFactory.export('csv', data, filename);
}
```

---

## Service Factory Examples

### Example 1: Global Configuration

```typescript
// app/config/services.ts
import { ServiceFactory } from '@/lib/factories';

export function initializeServices() {
  ServiceFactory.configure({
    dataProvider: {
      type: 'api',
      config: {
        apiBaseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api',
        timeout: 10000
      }
    },
    exportFormat: 'csv'
  });
}

// app/layout.tsx
import { initializeServices } from './config/services';

export default function RootLayout({ children }) {
  useEffect(() => {
    initializeServices();
  }, []);

  return <html>{children}</html>;
}
```

### Example 2: Using Configured Defaults

```typescript
import { ServiceFactory } from '@/lib/factories';

// Uses configured defaults
const provider = ServiceFactory.createDataProvider();
const exporter = ServiceFactory.createExportHandler();

// Override defaults when needed
const mockProvider = ServiceFactory.createDataProvider('mock');
const pdfExporter = ServiceFactory.createExportHandler('pdf');
```

### Example 3: Complete Trading Dashboard Setup

```typescript
import { ServiceFactory } from '@/lib/factories';

class TradingDashboard {
  private dataProvider;
  private exportHandler;

  constructor() {
    // Create services using factory
    this.dataProvider = ServiceFactory.createDataProviderWithFallback('api');
    this.exportHandler = ServiceFactory.createExportHandler('csv');
  }

  async loadData() {
    const data = await this.dataProvider.getRealTimeData();
    return data;
  }

  async exportData(data: any[], filename: string) {
    await this.exportHandler.export(data, filename);
  }

  subscribeToUpdates(callback: (data: MarketDataPoint) => void) {
    return this.dataProvider.subscribeToRealTimeUpdates(callback);
  }
}
```

---

## Testing Examples

### Example 1: Unit Test with Mock Provider

```typescript
import { DataProviderFactory } from '@/lib/factories';
import { describe, it, expect } from 'vitest';

describe('Market Data Component', () => {
  it('should display current price', async () => {
    // Use mock provider for testing
    const provider = DataProviderFactory.create('mock');

    const data = await provider.getRealTimeData();

    expect(data).toHaveLength(1);
    expect(data[0]).toHaveProperty('symbol', 'USDCOP');
    expect(data[0]).toHaveProperty('price');
    expect(typeof data[0].price).toBe('number');
  });

  it('should handle subscriptions', async () => {
    const provider = DataProviderFactory.create('mock');
    const updates: number[] = [];

    const unsubscribe = provider.subscribeToRealTimeUpdates((data) => {
      updates.push(data.price);
    });

    // Wait for updates
    await new Promise(resolve => setTimeout(resolve, 5000));

    expect(updates.length).toBeGreaterThan(0);

    unsubscribe();
  });
});
```

### Example 2: Integration Test

```typescript
import { ServiceFactory } from '@/lib/factories';

describe('Data Export Integration', () => {
  beforeEach(() => {
    ServiceFactory.configure({
      dataProvider: { type: 'mock' },
      exportFormat: 'csv'
    });
  });

  it('should fetch and export data', async () => {
    const provider = ServiceFactory.createDataProvider();
    const data = await provider.getRealTimeData();

    expect(data.length).toBeGreaterThan(0);

    // Test export
    const exporter = ServiceFactory.createExportHandler();
    expect(exporter.validateData(data)).toBe(true);
  });
});
```

---

## React Component Examples

### Example 1: Market Data Display Component

```typescript
'use client';

import { useState, useEffect } from 'react';
import { ServiceFactory } from '@/lib/factories';
import type { MarketDataPoint } from '@/lib/core/interfaces';

export default function MarketDataDisplay() {
  const [price, setPrice] = useState<number | null>(null);
  const [provider] = useState(() =>
    ServiceFactory.createDataProviderWithFallback('api')
  );

  useEffect(() => {
    // Subscribe to real-time updates
    const unsubscribe = provider.subscribeToRealTimeUpdates((data) => {
      setPrice(data.price);
    });

    // Cleanup
    return unsubscribe;
  }, [provider]);

  return (
    <div>
      <h2>USD/COP Price</h2>
      <p>{price ? `$${price.toFixed(4)}` : 'Loading...'}</p>
    </div>
  );
}
```

### Example 2: Export Button Component

```typescript
'use client';

import { ServiceFactory } from '@/lib/factories';
import type { ExportFormat } from '@/lib/factories';

interface ExportButtonProps {
  data: any[];
  filename: string;
  format: ExportFormat;
}

export default function ExportButton({ data, filename, format }: ExportButtonProps) {
  const handleExport = async () => {
    try {
      await ServiceFactory.export(format, data, filename);
      alert('Export successful!');
    } catch (error) {
      console.error('Export failed:', error);
      alert('Export failed');
    }
  };

  return (
    <button onClick={handleExport}>
      Export as {format.toUpperCase()}
    </button>
  );
}
```

### Example 3: Data Provider Selector

```typescript
'use client';

import { useState } from 'react';
import { ServiceFactory, type DataProviderType } from '@/lib/factories';

export default function ProviderSelector() {
  const [providerType, setProviderType] = useState<DataProviderType>('api');
  const [provider, setProvider] = useState(() =>
    ServiceFactory.createDataProvider('api')
  );

  const handleChange = (type: DataProviderType) => {
    setProviderType(type);
    setProvider(ServiceFactory.createDataProvider(type));
  };

  const availableProviders = ServiceFactory.getAvailableDataProviders();

  return (
    <div>
      <select value={providerType} onChange={(e) => handleChange(e.target.value as DataProviderType)}>
        {availableProviders.map(type => (
          <option key={type} value={type}>{type}</option>
        ))}
      </select>
    </div>
  );
}
```

---

## Configuration Examples

### Example 1: Environment-Based Configuration

```typescript
// lib/config/providers.ts
import { ServiceFactory, type ServiceConfig } from '@/lib/factories';

export function getServiceConfig(): ServiceConfig {
  const env = process.env.NODE_ENV;
  const isProduction = env === 'production';
  const isTest = env === 'test';

  return {
    dataProvider: {
      type: isTest ? 'mock' : 'api',
      config: {
        apiBaseUrl: process.env.NEXT_PUBLIC_API_URL,
        timeout: isProduction ? 5000 : 10000
      }
    },
    exportFormat: 'csv'
  };
}

// Initialize
ServiceFactory.configure(getServiceConfig());
```

### Example 2: Feature Flags

```typescript
import { ServiceFactory } from '@/lib/factories';

function configureByFeatureFlags() {
  const useWebSocket = process.env.NEXT_PUBLIC_ENABLE_WEBSOCKET === 'true';

  ServiceFactory.configure({
    dataProvider: {
      type: useWebSocket ? 'websocket' : 'api'
    }
  });
}
```

### Example 3: Dynamic Configuration

```typescript
import { ServiceFactory } from '@/lib/factories';

class DynamicServiceConfig {
  static updateProvider(type: DataProviderType) {
    const currentConfig = ServiceFactory.getConfig();

    ServiceFactory.configure({
      ...currentConfig,
      dataProvider: {
        type,
        config: currentConfig.dataProvider?.config
      }
    });
  }

  static updateExportFormat(format: ExportFormat) {
    const currentConfig = ServiceFactory.getConfig();

    ServiceFactory.configure({
      ...currentConfig,
      exportFormat: format
    });
  }
}
```

---

## Best Practices

1. **Use factories for all service instantiation**
   - Don't create providers directly
   - Always use factories for consistency

2. **Configure once, use everywhere**
   - Initialize ServiceFactory at app startup
   - Override only when necessary

3. **Use mock providers for testing**
   - Never use real APIs in tests
   - Mock providers are fast and reliable

4. **Handle errors gracefully**
   - Use `createWithFallback` for production
   - Always check health before heavy operations

5. **Clean up subscriptions**
   - Always unsubscribe in useEffect cleanup
   - Prevent memory leaks

6. **Type safety**
   - Use TypeScript types from factories
   - Leverage IDataProvider and IExportHandler interfaces

---

## Troubleshooting

### Issue: "Unknown provider type"

```typescript
// Bad
const provider = DataProviderFactory.create('invalid' as any);

// Good - Check if valid first
const type = 'api';
if (DataProviderFactory.isValidType(type)) {
  const provider = DataProviderFactory.create(type);
}
```

### Issue: Export fails with no error

```typescript
// Validate data first
const data = [...];
const format = 'csv';

if (ExportFactory.validateData(format, data)) {
  await ExportFactory.export(format, data, 'filename');
} else {
  console.error('Invalid data format');
}
```

### Issue: Provider not receiving updates

```typescript
// Make sure to store the provider instance
const [provider] = useState(() =>
  ServiceFactory.createDataProvider('api')
);

// Don't create a new instance on every render
// Bad: ServiceFactory.createDataProvider('api')
```

---

For more information, see the [README.md](./README.md) file.

/**
 * Factory Pattern Tests
 * ======================
 *
 * Tests for all factory implementations
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { DataProviderFactory } from '../DataProviderFactory';
import { ExportFactory } from '../ExportFactory';
import { ServiceFactory } from '../ServiceFactory';

describe('DataProviderFactory', () => {
  it('should create an API data provider', () => {
    const provider = DataProviderFactory.create('api');
    expect(provider).toBeDefined();
    expect(provider.getRealTimeData).toBeDefined();
  });

  it('should create a mock data provider', () => {
    const provider = DataProviderFactory.create('mock');
    expect(provider).toBeDefined();
    expect(provider.getRealTimeData).toBeDefined();
  });

  it('should create a WebSocket data provider', () => {
    const provider = DataProviderFactory.create('websocket');
    expect(provider).toBeDefined();
    expect(provider.getRealTimeData).toBeDefined();
  });

  it('should throw error for unknown provider type', () => {
    expect(() => {
      DataProviderFactory.create('invalid' as any);
    }).toThrow('Unknown data provider type: invalid');
  });

  it('should create provider with fallback', () => {
    const provider = DataProviderFactory.createWithFallback('api');
    expect(provider).toBeDefined();
  });

  it('should return available provider types', () => {
    const types = DataProviderFactory.getAvailableTypes();
    expect(types).toEqual(['api', 'mock', 'websocket']);
  });

  it('should validate provider type', () => {
    expect(DataProviderFactory.isValidType('api')).toBe(true);
    expect(DataProviderFactory.isValidType('mock')).toBe(true);
    expect(DataProviderFactory.isValidType('websocket')).toBe(true);
    expect(DataProviderFactory.isValidType('invalid')).toBe(false);
  });
});

describe('ExportFactory', () => {
  it('should create a CSV export handler', () => {
    const handler = ExportFactory.create('csv');
    expect(handler).toBeDefined();
    expect(handler.export).toBeDefined();
    expect(handler.getExtension()).toBe('csv');
    expect(handler.getMimeType()).toBe('text/csv');
  });

  it('should create an Excel export handler', () => {
    const handler = ExportFactory.create('excel');
    expect(handler).toBeDefined();
    expect(handler.getExtension()).toBe('xlsx');
  });

  it('should create a PDF export handler', () => {
    const handler = ExportFactory.create('pdf');
    expect(handler).toBeDefined();
    expect(handler.getExtension()).toBe('pdf');
  });

  it('should throw error for unknown format', () => {
    expect(() => {
      ExportFactory.create('invalid' as any);
    }).toThrow('Unknown export format: invalid');
  });

  it('should return available formats', () => {
    const formats = ExportFactory.getAvailableFormats();
    expect(formats).toEqual(['csv', 'excel', 'pdf']);
  });

  it('should validate format', () => {
    expect(ExportFactory.isValidFormat('csv')).toBe(true);
    expect(ExportFactory.isValidFormat('excel')).toBe(true);
    expect(ExportFactory.isValidFormat('pdf')).toBe(true);
    expect(ExportFactory.isValidFormat('invalid')).toBe(false);
  });

  it('should get handler info', () => {
    const info = ExportFactory.getHandlerInfo('csv');
    expect(info.extension).toBe('csv');
    expect(info.mimeType).toBe('text/csv');
  });

  it('should validate data', () => {
    const validData = [{ name: 'Test', value: 123 }];
    const invalidData = null;

    expect(ExportFactory.validateData('csv', validData)).toBe(true);
    expect(ExportFactory.validateData('csv', invalidData)).toBe(false);
  });
});

describe('ServiceFactory', () => {
  beforeEach(() => {
    ServiceFactory.resetConfig();
  });

  it('should create data provider', () => {
    const provider = ServiceFactory.createDataProvider('api');
    expect(provider).toBeDefined();
  });

  it('should create export handler', () => {
    const handler = ServiceFactory.createExportHandler('csv');
    expect(handler).toBeDefined();
  });

  it('should use configured defaults', () => {
    ServiceFactory.configure({
      dataProvider: { type: 'mock' },
      exportFormat: 'pdf'
    });

    const provider = ServiceFactory.createDataProvider();
    const handler = ServiceFactory.createExportHandler();

    expect(provider).toBeDefined();
    expect(handler).toBeDefined();
    expect(handler.getExtension()).toBe('pdf');
  });

  it('should override configured defaults', () => {
    ServiceFactory.configure({
      dataProvider: { type: 'mock' },
      exportFormat: 'csv'
    });

    const provider = ServiceFactory.createDataProvider('api');
    const handler = ServiceFactory.createExportHandler('pdf');

    expect(provider).toBeDefined();
    expect(handler.getExtension()).toBe('pdf');
  });

  it('should get configuration', () => {
    const config = {
      dataProvider: { type: 'mock' as const },
      exportFormat: 'csv' as const
    };

    ServiceFactory.configure(config);
    const retrievedConfig = ServiceFactory.getConfig();

    expect(retrievedConfig).toEqual(config);
  });

  it('should reset configuration', () => {
    ServiceFactory.configure({
      dataProvider: { type: 'mock' },
      exportFormat: 'pdf'
    });

    ServiceFactory.resetConfig();
    const config = ServiceFactory.getConfig();

    expect(config).toEqual({});
  });

  it('should get available data providers', () => {
    const providers = ServiceFactory.getAvailableDataProviders();
    expect(providers).toEqual(['api', 'mock', 'websocket']);
  });

  it('should get available export formats', () => {
    const formats = ServiceFactory.getAvailableExportFormats();
    expect(formats).toEqual(['csv', 'excel', 'pdf']);
  });

  it('should create provider with fallback', () => {
    const provider = ServiceFactory.createDataProviderWithFallback('api');
    expect(provider).toBeDefined();
  });

  it('should create provider from environment', () => {
    const provider = ServiceFactory.createDataProviderFromEnvironment();
    expect(provider).toBeDefined();
  });
});

describe('Integration Tests', () => {
  it('should work with mock provider end-to-end', async () => {
    const provider = DataProviderFactory.create('mock');

    // Test getRealTimeData
    const data = await provider.getRealTimeData();
    expect(data).toHaveLength(1);
    expect(data[0].symbol).toBe('USDCOP');

    // Test getCandlestickData
    const candles = await provider.getCandlestickData('USDCOP', '5m');
    expect(candles.data.length).toBeGreaterThan(0);

    // Test getSymbolStats
    const stats = await provider.getSymbolStats('USDCOP');
    expect(stats.symbol).toBe('USDCOP');

    // Test checkHealth
    const health = await provider.checkHealth();
    expect(health.status).toBe('healthy');
  });

  it('should handle subscriptions with mock provider', async () => {
    const provider = DataProviderFactory.create('mock');
    const updates: number[] = [];

    const unsubscribe = provider.subscribeToRealTimeUpdates((data) => {
      updates.push(data.price);
    });

    // Wait for some updates
    await new Promise(resolve => setTimeout(resolve, 5000));

    expect(updates.length).toBeGreaterThan(0);

    unsubscribe();
  });

  it('should validate and export CSV data', async () => {
    const handler = ExportFactory.create('csv');

    const data = [
      { symbol: 'USDCOP', price: 4200, volume: 1000 },
      { symbol: 'USDCOP', price: 4201, volume: 1100 }
    ];

    expect(handler.validateData(data)).toBe(true);
    expect(handler.getExtension()).toBe('csv');
  });
});

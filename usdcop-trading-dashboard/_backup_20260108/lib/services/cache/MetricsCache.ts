// Metrics Cache for USDCOP Trading System

import type { IMetricsCache, MetricsCacheData, PositionCacheData } from './types';
import { RedisClient } from './RedisClient';

/**
 * Cache manager for trading metrics and positions
 */
export class MetricsCache implements IMetricsCache {
  private client: RedisClient;
  private readonly FINANCIAL_METRICS_KEY = 'metrics:financial';
  private readonly POSITION_KEY_PREFIX = 'position:active';
  private readonly CUSTOM_METRIC_PREFIX = 'metrics:custom';
  private readonly DEFAULT_TTL = 300; // 5 minutes

  constructor(client?: RedisClient) {
    this.client = client || RedisClient.getInstance();
  }

  /**
   * Store financial metrics
   */
  async setFinancialMetrics(metrics: MetricsCacheData): Promise<void> {
    await this.client.set(this.FINANCIAL_METRICS_KEY, metrics, this.DEFAULT_TTL);
  }

  /**
   * Retrieve financial metrics
   */
  async getFinancialMetrics(): Promise<MetricsCacheData | null> {
    return await this.client.get<MetricsCacheData>(this.FINANCIAL_METRICS_KEY);
  }

  /**
   * Store custom metric
   */
  async setCustomMetric(key: string, value: number | string): Promise<void> {
    const metricKey = `${this.CUSTOM_METRIC_PREFIX}:${key}`;
    await this.client.set(metricKey, value, this.DEFAULT_TTL);
  }

  /**
   * Retrieve custom metric
   */
  async getCustomMetric(key: string): Promise<number | string | null> {
    const metricKey = `${this.CUSTOM_METRIC_PREFIX}:${key}`;
    return await this.client.get<number | string>(metricKey);
  }

  /**
   * Store active position
   */
  async setPosition(symbol: string, position: PositionCacheData): Promise<void> {
    const positionKey = `${this.POSITION_KEY_PREFIX}:${symbol}`;
    await this.client.set(positionKey, position, this.DEFAULT_TTL);
  }

  /**
   * Get active position for symbol
   */
  async getPosition(symbol: string): Promise<PositionCacheData | null> {
    const positionKey = `${this.POSITION_KEY_PREFIX}:${symbol}`;
    return await this.client.get<PositionCacheData>(positionKey);
  }

  /**
   * Remove position from cache
   */
  async deletePosition(symbol: string): Promise<void> {
    const positionKey = `${this.POSITION_KEY_PREFIX}:${symbol}`;
    await this.client.delete(positionKey);
  }

  /**
   * Get all active positions
   */
  async getAllPositions(): Promise<PositionCacheData[]> {
    const pattern = `${this.POSITION_KEY_PREFIX}:*`;
    const keys = await this.client.keys(pattern);

    const positions: PositionCacheData[] = [];
    for (const key of keys) {
      const position = await this.client.get<PositionCacheData>(
        key.replace(`${this.client['config'].namespace}:`, '')
      );
      if (position) {
        positions.push(position);
      }
    }

    return positions;
  }

  /**
   * Get position count
   */
  async getPositionCount(): Promise<number> {
    const pattern = `${this.POSITION_KEY_PREFIX}:*`;
    const keys = await this.client.keys(pattern);
    return keys.length;
  }

  /**
   * Calculate aggregate position metrics
   */
  async getAggregatePositionMetrics(): Promise<{
    total_pnl: number;
    total_positions: number;
    profitable_positions: number;
    losing_positions: number;
    win_rate: number;
  }> {
    const positions = await this.getAllPositions();

    const metrics = {
      total_pnl: 0,
      total_positions: positions.length,
      profitable_positions: 0,
      losing_positions: 0,
      win_rate: 0,
    };

    for (const position of positions) {
      metrics.total_pnl += position.pnl;

      if (position.pnl > 0) {
        metrics.profitable_positions++;
      } else if (position.pnl < 0) {
        metrics.losing_positions++;
      }
    }

    if (metrics.total_positions > 0) {
      metrics.win_rate = metrics.profitable_positions / metrics.total_positions;
    }

    return metrics;
  }

  /**
   * Get all custom metrics
   */
  async getAllCustomMetrics(): Promise<Record<string, number | string>> {
    const pattern = `${this.CUSTOM_METRIC_PREFIX}:*`;
    const keys = await this.client.keys(pattern);

    const metrics: Record<string, number | string> = {};

    for (const key of keys) {
      const metricName = key.replace(`${this.CUSTOM_METRIC_PREFIX}:`, '');
      const value = await this.getCustomMetric(metricName);
      if (value !== null) {
        metrics[metricName] = value;
      }
    }

    return metrics;
  }

  /**
   * Clear all metrics
   */
  async clearMetrics(): Promise<void> {
    // Clear financial metrics
    await this.client.delete(this.FINANCIAL_METRICS_KEY);

    // Clear custom metrics
    const customKeys = await this.client.keys(`${this.CUSTOM_METRIC_PREFIX}:*`);
    for (const key of customKeys) {
      await this.client.delete(key);
    }

    // Clear positions
    const positionKeys = await this.client.keys(`${this.POSITION_KEY_PREFIX}:*`);
    for (const key of positionKeys) {
      await this.client.delete(key);
    }
  }

  /**
   * Update financial metric snapshot
   */
  async updateMetricSnapshot(
    updates: Partial<MetricsCacheData>
  ): Promise<void> {
    const current = await this.getFinancialMetrics();

    const updated: MetricsCacheData = {
      timestamp: new Date().toISOString(),
      total_pnl: updates.total_pnl ?? current?.total_pnl ?? 0,
      win_rate: updates.win_rate ?? current?.win_rate ?? 0,
      sharpe_ratio: updates.sharpe_ratio ?? current?.sharpe_ratio ?? 0,
      max_drawdown: updates.max_drawdown ?? current?.max_drawdown ?? 0,
      total_trades: updates.total_trades ?? current?.total_trades ?? 0,
      active_positions: updates.active_positions ?? current?.active_positions ?? 0,
      portfolio_value: updates.portfolio_value ?? current?.portfolio_value ?? 0,
    };

    await this.setFinancialMetrics(updated);
  }

  /**
   * Get metrics freshness (seconds since last update)
   */
  async getMetricsFreshness(): Promise<number> {
    const metrics = await this.getFinancialMetrics();

    if (!metrics) return -1;

    const lastUpdate = new Date(metrics.timestamp);
    const now = new Date();

    return Math.floor((now.getTime() - lastUpdate.getTime()) / 1000);
  }
}

// Export singleton instance getter
let metricsCacheInstance: MetricsCache | null = null;

export const getMetricsCache = (): MetricsCache => {
  if (!metricsCacheInstance) {
    metricsCacheInstance = new MetricsCache();
  }
  return metricsCacheInstance;
};

// Signal Cache for USDCOP Trading System

import type { ISignalCache, SignalCacheData } from './types';
import { RedisClient } from './RedisClient';

/**
 * Cache manager for trading signals
 * Stores latest signal and maintains history per symbol
 */
export class SignalCache implements ISignalCache {
  private client: RedisClient;
  private readonly LATEST_KEY = 'signal:latest';
  private readonly HISTORY_KEY_PREFIX = 'signal:history';
  private readonly MAX_HISTORY_LENGTH = 100;
  private readonly DEFAULT_TTL = 3600; // 1 hour

  constructor(client?: RedisClient) {
    this.client = client || RedisClient.getInstance();
  }

  /**
   * Store the latest signal
   */
  async setLatest(signal: SignalCacheData): Promise<void> {
    await this.client.set(this.LATEST_KEY, signal, this.DEFAULT_TTL);
  }

  /**
   * Retrieve the latest signal
   */
  async getLatest(): Promise<SignalCacheData | null> {
    return await this.client.get<SignalCacheData>(this.LATEST_KEY);
  }

  /**
   * Add signal to symbol-specific history
   */
  async addToHistory(symbol: string, signal: SignalCacheData): Promise<void> {
    const historyKey = `${this.HISTORY_KEY_PREFIX}:${symbol}`;

    // Get existing history
    const history = (await this.client.get<SignalCacheData[]>(historyKey)) || [];

    // Add new signal at the beginning
    history.unshift(signal);

    // Limit history size
    if (history.length > this.MAX_HISTORY_LENGTH) {
      history.splice(this.MAX_HISTORY_LENGTH);
    }

    // Store updated history
    await this.client.set(historyKey, history, this.DEFAULT_TTL);
  }

  /**
   * Get signal history for a symbol
   */
  async getHistory(symbol: string, limit: number = 50): Promise<SignalCacheData[]> {
    const historyKey = `${this.HISTORY_KEY_PREFIX}:${symbol}`;
    const history = (await this.client.get<SignalCacheData[]>(historyKey)) || [];

    return history.slice(0, limit);
  }

  /**
   * Clear history for a specific symbol
   */
  async clearHistory(symbol: string): Promise<void> {
    const historyKey = `${this.HISTORY_KEY_PREFIX}:${symbol}`;
    await this.client.delete(historyKey);
  }

  /**
   * Get all symbols with cached history
   */
  async getSymbolsWithHistory(): Promise<string[]> {
    const pattern = `${this.HISTORY_KEY_PREFIX}:*`;
    const keys = await this.client.keys(pattern);

    return keys.map((key) =>
      key.replace(`${this.HISTORY_KEY_PREFIX}:`, '')
    );
  }

  /**
   * Get signal statistics
   */
  async getSignalStats(symbol: string): Promise<{
    total: number;
    buy: number;
    sell: number;
    hold: number;
    avg_confidence: number;
  }> {
    const history = await this.getHistory(symbol, this.MAX_HISTORY_LENGTH);

    const stats = {
      total: history.length,
      buy: 0,
      sell: 0,
      hold: 0,
      avg_confidence: 0,
    };

    if (history.length === 0) return stats;

    let totalConfidence = 0;

    for (const signal of history) {
      switch (signal.action) {
        case 'BUY':
          stats.buy++;
          break;
        case 'SELL':
          stats.sell++;
          break;
        case 'HOLD':
          stats.hold++;
          break;
      }
      totalConfidence += signal.confidence;
    }

    stats.avg_confidence = totalConfidence / history.length;

    return stats;
  }

  /**
   * Get signals within time range
   */
  async getSignalsInRange(
    symbol: string,
    startTime: Date,
    endTime: Date
  ): Promise<SignalCacheData[]> {
    const history = await this.getHistory(symbol, this.MAX_HISTORY_LENGTH);

    return history.filter((signal) => {
      const signalTime = new Date(signal.timestamp);
      return signalTime >= startTime && signalTime <= endTime;
    });
  }

  /**
   * Clear all signal data
   */
  async clearAll(): Promise<void> {
    await this.client.delete(this.LATEST_KEY);

    const symbols = await this.getSymbolsWithHistory();
    for (const symbol of symbols) {
      await this.clearHistory(symbol);
    }
  }

  /**
   * Get latest signals for all symbols
   */
  async getLatestBySymbol(): Promise<Record<string, SignalCacheData>> {
    const symbols = await this.getSymbolsWithHistory();
    const result: Record<string, SignalCacheData> = {};

    for (const symbol of symbols) {
      const history = await this.getHistory(symbol, 1);
      if (history.length > 0) {
        result[symbol] = history[0];
      }
    }

    return result;
  }
}

// Export singleton instance getter
let signalCacheInstance: SignalCache | null = null;

export const getSignalCache = (): SignalCache => {
  if (!signalCacheInstance) {
    signalCacheInstance = new SignalCache();
  }
  return signalCacheInstance;
};

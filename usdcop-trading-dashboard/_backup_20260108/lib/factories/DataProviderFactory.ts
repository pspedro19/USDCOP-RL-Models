/**
 * Data Provider Factory
 * ======================
 *
 * Factory for creating data providers based on configuration
 * Implements the Factory Pattern for data provider instantiation
 *
 * Usage:
 * ```typescript
 * const provider = DataProviderFactory.create('api');
 * const data = await provider.getRealTimeData();
 * ```
 */

import type { IDataProvider } from '@/lib/core/interfaces';
import { ApiDataProvider, type ApiDataProviderConfig } from '@/lib/services/providers/ApiDataProvider';
import { WebSocketDataProvider, type WebSocketDataProviderConfig } from '@/lib/services/providers/WebSocketDataProvider';
import { MockDataProvider } from '@/lib/services/mock/MockDataProvider';

export type DataProviderType = 'api' | 'mock' | 'websocket';

export type DataProviderConfig =
  | ApiDataProviderConfig
  | WebSocketDataProviderConfig
  | Record<string, never>;

/**
 * DataProviderFactory
 *
 * Creates instances of IDataProvider implementations
 * based on the specified type
 */
export class DataProviderFactory {
  /**
   * Create a data provider instance
   *
   * @param type - The type of data provider to create
   * @param config - Optional configuration for the provider
   * @returns An instance of IDataProvider
   * @throws Error if the provider type is unknown
   */
  static create(type: DataProviderType, config?: DataProviderConfig): IDataProvider {
    switch (type) {
      case 'api':
        return new ApiDataProvider(config as ApiDataProviderConfig);

      case 'mock':
        return new MockDataProvider();

      case 'websocket':
        return new WebSocketDataProvider(config as WebSocketDataProviderConfig);

      default:
        throw new Error(`Unknown data provider type: ${type}`);
    }
  }

  /**
   * Create a data provider with automatic fallback
   *
   * Attempts to create the primary provider, falls back to mock if it fails
   *
   * @param primaryType - The primary provider type to try
   * @param config - Optional configuration
   * @returns An instance of IDataProvider
   */
  static createWithFallback(
    primaryType: DataProviderType,
    config?: DataProviderConfig
  ): IDataProvider {
    try {
      return this.create(primaryType, config);
    } catch (error) {
      console.warn(
        `[DataProviderFactory] Failed to create ${primaryType} provider, falling back to mock`,
        error
      );
      return this.create('mock');
    }
  }

  /**
   * Create a data provider based on environment
   *
   * @param config - Optional configuration
   * @returns An instance of IDataProvider
   */
  static createFromEnvironment(config?: DataProviderConfig): IDataProvider {
    const env = process.env.NEXT_PUBLIC_DATA_PROVIDER || process.env.NODE_ENV;

    // Use mock provider in test environment
    if (env === 'test') {
      console.log('[DataProviderFactory] Using mock provider for test environment');
      return this.create('mock');
    }

    // Use WebSocket provider if WS URL is configured
    if (process.env.NEXT_PUBLIC_WS_URL) {
      console.log('[DataProviderFactory] Using WebSocket provider');
      return this.create('websocket', config);
    }

    // Default to API provider
    console.log('[DataProviderFactory] Using API provider');
    return this.create('api', config);
  }

  /**
   * Get available provider types
   *
   * @returns Array of available provider types
   */
  static getAvailableTypes(): DataProviderType[] {
    return ['api', 'mock', 'websocket'];
  }

  /**
   * Check if a provider type is valid
   *
   * @param type - The provider type to check
   * @returns True if the type is valid
   */
  static isValidType(type: string): type is DataProviderType {
    return this.getAvailableTypes().includes(type as DataProviderType);
  }
}

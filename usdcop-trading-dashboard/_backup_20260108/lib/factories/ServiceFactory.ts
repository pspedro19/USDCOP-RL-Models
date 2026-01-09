/**
 * Service Factory
 * ================
 *
 * Factory for creating services based on configuration
 * Implements the Factory Pattern for service instantiation
 *
 * This factory serves as a central point for creating all services
 * in the application, enabling dependency injection and testing
 *
 * Usage:
 * ```typescript
 * const dataProvider = ServiceFactory.createDataProvider('api');
 * const exportHandler = ServiceFactory.createExportHandler('csv');
 * ```
 */

import type { IDataProvider, IExportHandler } from '@/lib/core/interfaces';
import { DataProviderFactory, type DataProviderType, type DataProviderConfig } from './DataProviderFactory';
import { ExportFactory, type ExportFormat } from './ExportFactory';

export interface ServiceConfig {
  dataProvider?: {
    type: DataProviderType;
    config?: DataProviderConfig;
  };
  exportFormat?: ExportFormat;
}

/**
 * ServiceFactory
 *
 * Main factory for creating all services in the application
 * Delegates to specialized factories for specific service types
 */
export class ServiceFactory {
  private static config: ServiceConfig = {};

  /**
   * Configure the factory with default settings
   *
   * @param config - Service configuration
   */
  static configure(config: ServiceConfig): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Create a data provider
   *
   * @param type - The type of data provider (optional, uses config default)
   * @param config - Provider configuration (optional)
   * @returns An instance of IDataProvider
   */
  static createDataProvider(
    type?: DataProviderType,
    config?: DataProviderConfig
  ): IDataProvider {
    const providerType = type || this.config.dataProvider?.type || 'api';
    const providerConfig = config || this.config.dataProvider?.config;

    return DataProviderFactory.create(providerType, providerConfig);
  }

  /**
   * Create a data provider with automatic fallback
   *
   * @param type - The primary provider type
   * @param config - Provider configuration (optional)
   * @returns An instance of IDataProvider
   */
  static createDataProviderWithFallback(
    type?: DataProviderType,
    config?: DataProviderConfig
  ): IDataProvider {
    const providerType = type || this.config.dataProvider?.type || 'api';
    const providerConfig = config || this.config.dataProvider?.config;

    return DataProviderFactory.createWithFallback(providerType, providerConfig);
  }

  /**
   * Create a data provider based on environment
   *
   * @param config - Provider configuration (optional)
   * @returns An instance of IDataProvider
   */
  static createDataProviderFromEnvironment(config?: DataProviderConfig): IDataProvider {
    return DataProviderFactory.createFromEnvironment(config);
  }

  /**
   * Create an export handler
   *
   * @param format - The export format (optional, uses config default)
   * @returns An instance of IExportHandler
   */
  static createExportHandler(format?: ExportFormat): IExportHandler {
    const exportFormat = format || this.config.exportFormat || 'csv';
    return ExportFactory.create(exportFormat);
  }

  /**
   * Export data using the specified format
   *
   * @param format - The export format
   * @param data - The data to export
   * @param filename - The filename (without extension)
   * @param config - Optional report configuration
   */
  static async export(
    format: ExportFormat,
    data: any,
    filename: string,
    config?: any
  ): Promise<void> {
    return ExportFactory.export(format, data, filename, config);
  }

  /**
   * Get the current configuration
   *
   * @returns Current service configuration
   */
  static getConfig(): ServiceConfig {
    return { ...this.config };
  }

  /**
   * Reset configuration to defaults
   */
  static resetConfig(): void {
    this.config = {};
  }

  /**
   * Get available data provider types
   *
   * @returns Array of available provider types
   */
  static getAvailableDataProviders(): DataProviderType[] {
    return DataProviderFactory.getAvailableTypes();
  }

  /**
   * Get available export formats
   *
   * @returns Array of available export formats
   */
  static getAvailableExportFormats(): ExportFormat[] {
    return ExportFactory.getAvailableFormats();
  }
}

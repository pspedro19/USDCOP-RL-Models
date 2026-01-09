/**
 * Factories Export
 * ================
 *
 * Central export point for all factory classes
 *
 * Factory Pattern Implementation:
 * - DataProviderFactory: Creates data providers (API, Mock, WebSocket)
 * - ExportFactory: Creates export handlers (CSV, Excel, PDF)
 * - ServiceFactory: Main factory that orchestrates all services
 *
 * Usage:
 * ```typescript
 * import { ServiceFactory } from '@/lib/factories';
 *
 * // Configure factory
 * ServiceFactory.configure({
 *   dataProvider: { type: 'api' },
 *   exportFormat: 'csv'
 * });
 *
 * // Create services
 * const dataProvider = ServiceFactory.createDataProvider();
 * const exportHandler = ServiceFactory.createExportHandler();
 *
 * // Or use specialized factories
 * import { DataProviderFactory, ExportFactory } from '@/lib/factories';
 *
 * const mockProvider = DataProviderFactory.create('mock');
 * const pdfHandler = ExportFactory.create('pdf');
 * ```
 */

// Export all factories
export { DataProviderFactory } from './DataProviderFactory';
export type { DataProviderType, DataProviderConfig } from './DataProviderFactory';

export { ExportFactory } from './ExportFactory';
export type { ExportFormat } from './ExportFactory';

export { ServiceFactory } from './ServiceFactory';
export type { ServiceConfig } from './ServiceFactory';

// Re-export commonly used types for convenience
export type { IDataProvider, IExportHandler } from '@/lib/core/interfaces';

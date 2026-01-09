/**
 * Core Interfaces
 * ===============
 *
 * Central export point for all core interfaces.
 */

// Data Provider interfaces
export type {
  IDataProvider,
  IExtendedDataProvider,
  DataProviderConfig,
} from './IDataProvider';

// WebSocket Provider interfaces
export type {
  IWebSocketProvider,
  IExtendedWebSocketProvider,
  WebSocketProviderConfig,
} from './IWebSocketProvider';

// Risk Calculator interfaces
export type {
  IRiskCalculator,
  IAdvancedRiskCalculator,
  RiskCalculatorConfig,
  RiskMetrics,
  PositionSize,
  RiskLevels,
  RiskLimits,
} from './IRiskCalculator';

// Subscribable interfaces
export type {
  ISubscribable,
  ISubscribableWithErrors,
  IObservable,
  IAsyncObservable,
  IEventEmitter,
  ISubject,
  IBehaviorSubject,
  IReplaySubject,
  SubscriptionHandler,
  ErrorHandler,
  UnsubscribeFn,
  SubscriptionOptions,
} from './ISubscribable';

// Export Handler interfaces
export type {
  IExportHandler,
  ExportOptions,
  ReportConfig,
  ChartData,
  TableData,
  MetricData,
  ExcelSheetData,
} from './IExportHandler';

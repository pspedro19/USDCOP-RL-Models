// Logging Types for USDCOP Trading System

export type LogLevel = 'debug' | 'info' | 'warn' | 'error' | 'fatal';

export interface LogContext {
  service?: string;
  user_id?: string;
  session_id?: string;
  request_id?: string;
  symbol?: string;
  [key: string]: any;
}

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  message: string;
  context: LogContext;
  error?: {
    name: string;
    message: string;
    stack?: string;
  };
  metadata?: Record<string, any>;
}

export interface AuditLogEntry {
  audit_id: string;
  timestamp: string;
  event_type: 'SIGNAL_GENERATED' | 'SIGNAL_EXECUTED' | 'POSITION_OPENED' | 'POSITION_CLOSED' | 'RISK_ALERT' | 'SYSTEM_EVENT';
  user_id?: string;
  symbol: string;
  action: string;
  before_state?: Record<string, any>;
  after_state?: Record<string, any>;
  metadata?: Record<string, any>;
  ip_address?: string;
}

export interface PerformanceLogEntry {
  operation: string;
  duration_ms: number;
  timestamp: string;
  success: boolean;
  metadata?: Record<string, any>;
}

export interface LatencyMetrics {
  operation: string;
  count: number;
  total_ms: number;
  avg_ms: number;
  min_ms: number;
  max_ms: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
}

export interface IStructuredLogger {
  debug(message: string, context?: LogContext, metadata?: Record<string, any>): void;
  info(message: string, context?: LogContext, metadata?: Record<string, any>): void;
  warn(message: string, context?: LogContext, metadata?: Record<string, any>): void;
  error(message: string, error?: Error, context?: LogContext, metadata?: Record<string, any>): void;
  fatal(message: string, error?: Error, context?: LogContext, metadata?: Record<string, any>): void;
  setContext(context: LogContext): void;
  getContext(): LogContext;
}

export interface IAuditLogger {
  logSignalGenerated(signal: any, metadata?: Record<string, any>): Promise<void>;
  logSignalExecuted(signal: any, execution: any, metadata?: Record<string, any>): Promise<void>;
  logPositionOpened(position: any, metadata?: Record<string, any>): Promise<void>;
  logPositionClosed(position: any, pnl: number, metadata?: Record<string, any>): Promise<void>;
  logRiskAlert(alert: any, metadata?: Record<string, any>): Promise<void>;
  logSystemEvent(event: string, metadata?: Record<string, any>): Promise<void>;
  getAuditTrail(filters?: AuditFilters): Promise<AuditLogEntry[]>;
}

export interface AuditFilters {
  startTime?: Date;
  endTime?: Date;
  eventType?: AuditLogEntry['event_type'];
  symbol?: string;
  userId?: string;
  limit?: number;
}

export interface IPerformanceLogger {
  startOperation(operation: string): string;
  endOperation(operationId: string, success?: boolean, metadata?: Record<string, any>): void;
  logLatency(operation: string, durationMs: number, metadata?: Record<string, any>): void;
  getMetrics(operation?: string): LatencyMetrics | LatencyMetrics[];
  clearMetrics(): void;
}

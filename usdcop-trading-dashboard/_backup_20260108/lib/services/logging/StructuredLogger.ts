// Structured Logger for USDCOP Trading System

import type { IStructuredLogger, LogLevel, LogContext, LogEntry } from './types';

/**
 * Structured logger with context support
 * Provides consistent logging format across the application
 */
export class StructuredLogger implements IStructuredLogger {
  private context: LogContext;
  private minLevel: LogLevel;
  private readonly levels: Record<LogLevel, number> = {
    debug: 0,
    info: 1,
    warn: 2,
    error: 3,
    fatal: 4,
  };

  constructor(context: LogContext = {}, minLevel: LogLevel = 'info') {
    this.context = context;
    this.minLevel = minLevel;
  }

  private shouldLog(level: LogLevel): boolean {
    return this.levels[level] >= this.levels[this.minLevel];
  }

  private formatLogEntry(
    level: LogLevel,
    message: string,
    context?: LogContext,
    metadata?: Record<string, any>,
    error?: Error
  ): LogEntry {
    return {
      timestamp: new Date().toISOString(),
      level,
      message,
      context: { ...this.context, ...context },
      ...(error && {
        error: {
          name: error.name,
          message: error.message,
          stack: error.stack,
        },
      }),
      ...(metadata && { metadata }),
    };
  }

  private output(entry: LogEntry): void {
    const color = this.getColorForLevel(entry.level);
    const emoji = this.getEmojiForLevel(entry.level);

    // Console output with color
    const consoleMethod = this.getConsoleMethod(entry.level);
    const contextStr = Object.keys(entry.context).length
      ? ` [${JSON.stringify(entry.context)}]`
      : '';

    consoleMethod(
      `${emoji} ${entry.timestamp} ${color}[${entry.level.toUpperCase()}]${this.resetColor()} ${entry.message}${contextStr}`
    );

    if (entry.metadata) {
      console.log('  Metadata:', entry.metadata);
    }

    if (entry.error) {
      console.error('  Error:', entry.error);
    }
  }

  private getColorForLevel(level: LogLevel): string {
    const colors = {
      debug: '\x1b[36m', // Cyan
      info: '\x1b[32m', // Green
      warn: '\x1b[33m', // Yellow
      error: '\x1b[31m', // Red
      fatal: '\x1b[35m', // Magenta
    };
    return colors[level];
  }

  private resetColor(): string {
    return '\x1b[0m';
  }

  private getEmojiForLevel(level: LogLevel): string {
    const emojis = {
      debug: '🔍',
      info: '📝',
      warn: '⚠️',
      error: '❌',
      fatal: '💀',
    };
    return emojis[level];
  }

  private getConsoleMethod(level: LogLevel): typeof console.log {
    switch (level) {
      case 'error':
      case 'fatal':
        return console.error.bind(console);
      case 'warn':
        return console.warn.bind(console);
      default:
        return console.log.bind(console);
    }
  }

  debug(message: string, context?: LogContext, metadata?: Record<string, any>): void {
    if (!this.shouldLog('debug')) return;

    const entry = this.formatLogEntry('debug', message, context, metadata);
    this.output(entry);
  }

  info(message: string, context?: LogContext, metadata?: Record<string, any>): void {
    if (!this.shouldLog('info')) return;

    const entry = this.formatLogEntry('info', message, context, metadata);
    this.output(entry);
  }

  warn(message: string, context?: LogContext, metadata?: Record<string, any>): void {
    if (!this.shouldLog('warn')) return;

    const entry = this.formatLogEntry('warn', message, context, metadata);
    this.output(entry);
  }

  error(
    message: string,
    error?: Error,
    context?: LogContext,
    metadata?: Record<string, any>
  ): void {
    if (!this.shouldLog('error')) return;

    const entry = this.formatLogEntry('error', message, context, metadata, error);
    this.output(entry);
  }

  fatal(
    message: string,
    error?: Error,
    context?: LogContext,
    metadata?: Record<string, any>
  ): void {
    if (!this.shouldLog('fatal')) return;

    const entry = this.formatLogEntry('fatal', message, context, metadata, error);
    this.output(entry);
  }

  setContext(context: LogContext): void {
    this.context = { ...this.context, ...context };
  }

  getContext(): LogContext {
    return { ...this.context };
  }

  clearContext(): void {
    this.context = {};
  }

  setMinLevel(level: LogLevel): void {
    this.minLevel = level;
  }

  /**
   * Create child logger with additional context
   */
  child(context: LogContext): StructuredLogger {
    return new StructuredLogger({ ...this.context, ...context }, this.minLevel);
  }

  /**
   * Log with custom level
   */
  log(
    level: LogLevel,
    message: string,
    context?: LogContext,
    metadata?: Record<string, any>
  ): void {
    if (!this.shouldLog(level)) return;

    const entry = this.formatLogEntry(level, message, context, metadata);
    this.output(entry);
  }

  /**
   * Measure operation time
   */
  async time<T>(
    operation: string,
    fn: () => Promise<T>,
    context?: LogContext
  ): Promise<T> {
    const start = Date.now();
    this.debug(`Starting operation: ${operation}`, context);

    try {
      const result = await fn();
      const duration = Date.now() - start;
      this.info(`Completed operation: ${operation}`, context, { duration_ms: duration });
      return result;
    } catch (error) {
      const duration = Date.now() - start;
      this.error(
        `Failed operation: ${operation}`,
        error as Error,
        context,
        { duration_ms: duration }
      );
      throw error;
    }
  }
}

// Export singleton instance
let loggerInstance: StructuredLogger | null = null;

export const getLogger = (context?: LogContext): StructuredLogger => {
  if (!loggerInstance) {
    loggerInstance = new StructuredLogger(
      context,
      (process.env.LOG_LEVEL as LogLevel) || 'info'
    );
  }

  if (context) {
    return loggerInstance.child(context);
  }

  return loggerInstance;
};

/**
 * Logger Utility
 * ==============
 *
 * Centralized logging utility with environment-aware behavior.
 * Follows SOLID principles - Single Responsibility for logging.
 *
 * Usage:
 *   import { logger } from '@/lib/utils/logger';
 *   logger.debug('Message');  // Only in development
 *   logger.info('Message');   // Always
 *   logger.warn('Message');   // Always
 *   logger.error('Message');  // Always
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LoggerConfig {
  enableDebug: boolean;
  enableInfo: boolean;
  prefix: string;
}

const defaultConfig: LoggerConfig = {
  enableDebug: process.env.NODE_ENV !== 'production',
  enableInfo: true,
  prefix: '[USDCOP]',
};

class Logger {
  private config: LoggerConfig;

  constructor(config: Partial<LoggerConfig> = {}) {
    this.config = { ...defaultConfig, ...config };
  }

  private formatMessage(level: LogLevel, message: string, ...args: unknown[]): string {
    const timestamp = new Date().toISOString();
    return `${this.config.prefix} [${timestamp}] [${level.toUpperCase()}] ${message}`;
  }

  debug(message: string, ...args: unknown[]): void {
    if (this.config.enableDebug) {
      console.log(this.formatMessage('debug', message), ...args);
    }
  }

  info(message: string, ...args: unknown[]): void {
    if (this.config.enableInfo) {
      console.info(this.formatMessage('info', message), ...args);
    }
  }

  warn(message: string, ...args: unknown[]): void {
    console.warn(this.formatMessage('warn', message), ...args);
  }

  error(message: string, ...args: unknown[]): void {
    console.error(this.formatMessage('error', message), ...args);
  }

  /**
   * Create a scoped logger with a custom prefix
   */
  scope(name: string): Logger {
    return new Logger({
      ...this.config,
      prefix: `${this.config.prefix}[${name}]`,
    });
  }
}

// Singleton instance
export const logger = new Logger();

// Factory for creating scoped loggers
export const createLogger = (scope: string): Logger => logger.scope(scope);

// Type exports
export type { LogLevel, LoggerConfig };

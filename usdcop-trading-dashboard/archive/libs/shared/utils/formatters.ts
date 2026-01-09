/**
 * Shared Utility Formatters for Elite Trading Platform
 * Professional data formatting utilities
 */

import { format, formatDistanceToNow, parseISO } from 'date-fns';

/**
 * Format currency values with proper precision
 */
export function formatCurrency(
  value: number,
  currency: string = 'USD',
  decimals?: number
): string {
  const precision = decimals ?? getCurrencyPrecision(currency);

  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: currency,
    minimumFractionDigits: precision,
    maximumFractionDigits: precision
  }).format(value);
}

/**
 * Format price with appropriate precision for trading
 */
export function formatPrice(value: number, symbol: string = ''): string {
  const precision = getPricePrecision(symbol);

  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: precision,
    maximumFractionDigits: precision
  }).format(value);
}

/**
 * Format percentage values
 */
export function formatPercentage(
  value: number,
  decimals: number = 2,
  showSign: boolean = true
): string {
  const sign = showSign && value > 0 ? '+' : '';

  return `${sign}${new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  }).format(value / 100)}`;
}

/**
 * Format volume values with K, M, B suffixes
 */
export function formatVolume(value: number): string {
  if (value === 0) return '0';

  const units = ['', 'K', 'M', 'B', 'T'];
  const order = Math.floor(Math.log10(Math.abs(value)) / 3);
  const unitname = units[order];
  const num = value / Math.pow(10, order * 3);

  return `${num.toFixed(1)}${unitname}`;
}

/**
 * Format large numbers with appropriate suffixes
 */
export function formatLargeNumber(value: number, decimals: number = 1): string {
  if (Math.abs(value) < 1000) {
    return value.toFixed(decimals);
  }

  return formatVolume(value);
}

/**
 * Format time values
 */
export function formatTime(timestamp: number | string | Date): string {
  const date = typeof timestamp === 'number'
    ? new Date(timestamp)
    : typeof timestamp === 'string'
    ? parseISO(timestamp)
    : timestamp;

  return format(date, 'HH:mm:ss');
}

/**
 * Format date values
 */
export function formatDate(timestamp: number | string | Date): string {
  const date = typeof timestamp === 'number'
    ? new Date(timestamp)
    : typeof timestamp === 'string'
    ? parseISO(timestamp)
    : timestamp;

  return format(date, 'MMM dd, yyyy');
}

/**
 * Format full datetime
 */
export function formatDateTime(timestamp: number | string | Date): string {
  const date = typeof timestamp === 'number'
    ? new Date(timestamp)
    : typeof timestamp === 'string'
    ? parseISO(timestamp)
    : timestamp;

  return format(date, 'MMM dd, yyyy HH:mm:ss');
}

/**
 * Format relative time (e.g., "2 minutes ago")
 */
export function formatRelativeTime(timestamp: number | string | Date): string {
  const date = typeof timestamp === 'number'
    ? new Date(timestamp)
    : typeof timestamp === 'string'
    ? parseISO(timestamp)
    : timestamp;

  return formatDistanceToNow(date, { addSuffix: true });
}

/**
 * Format duration in milliseconds to human readable
 */
export function formatDuration(milliseconds: number): string {
  const seconds = Math.floor(milliseconds / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) {
    return `${days}d ${hours % 24}h`;
  } else if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  } else if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  } else if (seconds > 0) {
    return `${seconds}s`;
  } else {
    return `${milliseconds}ms`;
  }
}

/**
 * Format bytes to human readable sizes
 */
export function formatBytes(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

/**
 * Format latency/response time
 */
export function formatLatency(milliseconds: number): string {
  if (milliseconds < 1) {
    return `${(milliseconds * 1000).toFixed(0)}μs`;
  } else if (milliseconds < 1000) {
    return `${milliseconds.toFixed(1)}ms`;
  } else {
    return `${(milliseconds / 1000).toFixed(2)}s`;
  }
}

/**
 * Format order size based on symbol
 */
export function formatOrderSize(size: number, symbol: string = ''): string {
  const precision = getVolumePrecision(symbol);

  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: precision,
    maximumFractionDigits: precision
  }).format(size);
}

/**
 * Format P&L with color indication
 */
export function formatPnL(value: number, currency: string = 'USD'): {
  formatted: string;
  isPositive: boolean;
  isNegative: boolean;
} {
  const formatted = formatCurrency(value, currency);
  const isPositive = value > 0;
  const isNegative = value < 0;

  return { formatted, isPositive, isNegative };
}

/**
 * Truncate text with ellipsis
 */
export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength - 3) + '...';
}

/**
 * Format symbol for display
 */
export function formatSymbol(symbol: string): string {
  // Handle common forex pairs
  if (symbol.length === 6 && /^[A-Z]{6}$/.test(symbol)) {
    return `${symbol.substring(0, 3)}/${symbol.substring(3, 6)}`;
  }

  return symbol;
}

// Helper functions for precision
function getCurrencyPrecision(currency: string): number {
  const highPrecisionCurrencies = ['BTC', 'ETH', 'XRP'];
  const lowPrecisionCurrencies = ['JPY', 'KRW'];

  if (highPrecisionCurrencies.includes(currency)) return 8;
  if (lowPrecisionCurrencies.includes(currency)) return 0;
  return 2;
}

function getPricePrecision(symbol: string): number {
  // Common forex pairs
  if (symbol.includes('JPY')) return 3;
  if (symbol.includes('USD') || symbol.includes('EUR') || symbol.includes('GBP')) return 5;

  // Crypto pairs
  if (symbol.includes('BTC') || symbol.includes('ETH')) return 8;

  return 4;
}

function getVolumePrecision(symbol: string): number {
  // Crypto typically has higher precision
  if (symbol.includes('BTC') || symbol.includes('ETH')) return 8;

  // Forex pairs
  if (symbol.includes('USD') || symbol.includes('EUR')) return 2;

  return 3;
}
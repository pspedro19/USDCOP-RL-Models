/**
 * Timezone Utilities - Colombia Time (COT)
 * =========================================
 * All market data should be displayed in America/Bogota timezone (COT/UTC-5)
 *
 * Usage:
 *   import { formatCOT, formatCOTTime, formatCOTDate } from '@/lib/utils/timezone-utils';
 *
 *   formatCOT(timestamp)           // Full datetime in COT
 *   formatCOTTime(timestamp)       // Time only (HH:MM:SS)
 *   formatCOTDate(timestamp)       // Date only (MM/DD/YYYY)
 */

const COT_TIMEZONE = 'America/Bogota';

/**
 * Format timestamp to Colombia Time (COT)
 * @param timestamp - ISO string or Date object
 * @param format - 'time' | 'date' | 'datetime'
 * @returns Formatted string in COT timezone
 */
export function formatCOT(
  timestamp: string | Date | null | undefined,
  format: 'time' | 'date' | 'datetime' = 'datetime'
): string {
  if (!timestamp) return '-';

  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;

  if (isNaN(date.getTime())) return 'Invalid Date';

  const baseOptions: Intl.DateTimeFormatOptions = {
    timeZone: COT_TIMEZONE
  };

  switch (format) {
    case 'time':
      return date.toLocaleString('en-US', {
        ...baseOptions,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
      }) + ' COT';

    case 'date':
      return date.toLocaleString('en-US', {
        ...baseOptions,
        month: '2-digit',
        day: '2-digit',
        year: 'numeric'
      });

    case 'datetime':
      return date.toLocaleString('en-US', {
        ...baseOptions,
        month: '2-digit',
        day: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
      }) + ' COT';
  }
}

/**
 * Format time only in COT (HH:MM:SS COT)
 */
export function formatCOTTime(timestamp: string | Date | null | undefined): string {
  return formatCOT(timestamp, 'time');
}

/**
 * Format date only in COT (MM/DD/YYYY)
 */
export function formatCOTDate(timestamp: string | Date | null | undefined): string {
  return formatCOT(timestamp, 'date');
}

/**
 * Format full datetime in COT (MM/DD/YYYY, HH:MM:SS COT)
 */
export function formatCOTDateTime(timestamp: string | Date | null | undefined): string {
  return formatCOT(timestamp, 'datetime');
}

/**
 * Get current time in Colombia timezone
 */
export function getCurrentCOT(): Date {
  // Create a date that represents current moment
  // When formatted with COT timezone, it will show correct local time
  return new Date();
}

/**
 * Check if current time is within trading hours
 * @returns true if within 8:00 AM - 12:55 PM COT, Monday-Friday
 */
export function isMarketOpen(): boolean {
  const now = new Date();

  // Get Colombia time components
  const cotFormatter = new Intl.DateTimeFormat('en-US', {
    timeZone: COT_TIMEZONE,
    hour: 'numeric',
    minute: 'numeric',
    weekday: 'short',
    hour12: false
  });

  const parts = cotFormatter.formatToParts(now);
  const hour = parseInt(parts.find(p => p.type === 'hour')?.value || '0');
  const minute = parseInt(parts.find(p => p.type === 'minute')?.value || '0');
  const weekday = parts.find(p => p.type === 'weekday')?.value || '';

  // Check if weekend
  if (weekday === 'Sat' || weekday === 'Sun') {
    return false;
  }

  // Check trading hours: 8:00 AM - 12:55 PM COT
  if (hour < 8 || hour > 12) {
    return false;
  }

  if (hour === 12 && minute > 55) {
    return false;
  }

  return true;
}

/**
 * Get time until market opens/closes
 * @returns Object with status and minutes until next event
 */
export function getMarketTimeStatus(): {
  isOpen: boolean;
  minutesUntilChange: number;
  nextEvent: 'open' | 'close' | 'none';
} {
  const now = new Date();
  const isOpen = isMarketOpen();

  // Calculate minutes until market open/close
  // This is a simplified version - full implementation would handle weekends properly

  return {
    isOpen,
    minutesUntilChange: 0, // TODO: Calculate actual minutes
    nextEvent: isOpen ? 'close' : 'open'
  };
}

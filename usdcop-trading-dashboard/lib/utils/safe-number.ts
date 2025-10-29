/**
 * Safe Number Utilities
 * Prevents undefined/NaN errors when formatting numbers
 */

/**
 * Safely format a number with toFixed, returning fallback if invalid
 * @param value - The value to format
 * @param decimals - Number of decimal places (default: 2)
 * @param fallback - Fallback string if value is invalid (default: '0.00')
 * @returns Formatted string
 */
export function safeToFixed(
  value: number | string | null | undefined,
  decimals: number = 2,
  fallback: string = '0.00'
): string {
  const num = Number(value);

  if (!Number.isFinite(num)) {
    return fallback;
  }

  return num.toFixed(decimals);
}

/**
 * Safely get a number value, returning 0 if invalid
 * @param value - The value to convert
 * @param fallback - Fallback value if invalid (default: 0)
 * @returns Number
 */
export function safeNumber(
  value: number | string | null | undefined,
  fallback: number = 0
): number {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

/**
 * Format currency safely
 * @param value - The value to format
 * @param currency - Currency symbol (default: '$')
 * @param decimals - Number of decimal places (default: 2)
 * @returns Formatted currency string
 */
export function safeCurrency(
  value: number | string | null | undefined,
  currency: string = '$',
  decimals: number = 2
): string {
  const formatted = safeToFixed(value, decimals, '0.00');
  const num = Number(formatted);

  return `${currency}${num.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  })}`;
}

/**
 * Format percentage safely
 * @param value - The value to format (as decimal, e.g., 0.05 = 5%)
 * @param decimals - Number of decimal places (default: 2)
 * @returns Formatted percentage string
 */
export function safePercent(
  value: number | string | null | undefined,
  decimals: number = 2
): string {
  return `${safeToFixed(safeNumber(value) * 100, decimals, '0.00')}%`;
}

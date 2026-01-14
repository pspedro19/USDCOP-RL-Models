/**
 * Single source of truth para filtrado de equity.
 * Contrato: GTR-008
 *
 * Este modulo centraliza toda la logica de filtrado de equity
 * para evitar duplicacion entre componentes.
 */

export interface EquityDataPoint {
  bar: number;
  timestamp?: string | Date;
  equity: number;
  pnl?: number;
  position?: number;
  action?: string;
}

export interface EquityFilterConfig {
  minEquity: number;
  maxDrawdownPct: number;
  excludeWarmup: boolean;
  warmupBars: number;
}

/**
 * Default filter configuration based on FEATURE_CONTRACT_V20
 */
export const DEFAULT_EQUITY_FILTER: EquityFilterConfig = {
  minEquity: 0,
  maxDrawdownPct: 100,
  excludeWarmup: true,
  warmupBars: 14, // From FEATURE_CONTRACT_V20.warmup_bars
};

/**
 * Filters equity data according to the specified configuration.
 * This is the SINGLE source of truth for equity filtering.
 *
 * @param data - Array of equity data points
 * @param config - Filter configuration
 * @returns Filtered array of equity data points
 *
 * @example
 * const filtered = filterEquityData(data, {
 *   ...DEFAULT_EQUITY_FILTER,
 *   minEquity: 5000,
 * });
 */
export function filterEquityData(
  data: EquityDataPoint[],
  config: EquityFilterConfig = DEFAULT_EQUITY_FILTER
): EquityDataPoint[] {
  if (!data || data.length === 0) {
    return [];
  }

  let filtered = [...data];

  // Step 1: Exclude warmup bars if configured
  if (config.excludeWarmup && config.warmupBars > 0) {
    filtered = filtered.slice(config.warmupBars);
  }

  // Step 2: Filter by minimum equity
  if (config.minEquity > 0) {
    filtered = filtered.filter((d) => d.equity >= config.minEquity);
  }

  // Step 3: Filter by maximum drawdown
  if (config.maxDrawdownPct < 100) {
    let peak = filtered[0]?.equity ?? 0;
    filtered = filtered.filter((d) => {
      peak = Math.max(peak, d.equity);
      if (peak <= 0) return true; // Avoid division by zero
      const drawdown = ((peak - d.equity) / peak) * 100;
      return drawdown <= config.maxDrawdownPct;
    });
  }

  return filtered;
}

/**
 * Calculates drawdown series from equity data.
 *
 * @param data - Array of equity data points
 * @returns Array of drawdown percentages
 */
export function calculateDrawdownSeries(data: EquityDataPoint[]): number[] {
  if (!data || data.length === 0) {
    return [];
  }

  let peak = data[0].equity;
  return data.map((d) => {
    peak = Math.max(peak, d.equity);
    if (peak <= 0) return 0;
    return ((peak - d.equity) / peak) * 100;
  });
}

/**
 * Calculates maximum drawdown from equity data.
 *
 * @param data - Array of equity data points
 * @returns Maximum drawdown percentage
 */
export function calculateMaxDrawdown(data: EquityDataPoint[]): number {
  const drawdowns = calculateDrawdownSeries(data);
  return Math.max(...drawdowns, 0);
}

/**
 * Calculates equity statistics from filtered data.
 *
 * @param data - Array of equity data points
 * @returns Statistics object
 */
export function calculateEquityStats(data: EquityDataPoint[]): {
  startEquity: number;
  endEquity: number;
  minEquity: number;
  maxEquity: number;
  totalReturn: number;
  totalReturnPct: number;
  maxDrawdown: number;
  barCount: number;
} {
  if (!data || data.length === 0) {
    return {
      startEquity: 0,
      endEquity: 0,
      minEquity: 0,
      maxEquity: 0,
      totalReturn: 0,
      totalReturnPct: 0,
      maxDrawdown: 0,
      barCount: 0,
    };
  }

  const equities = data.map((d) => d.equity);
  const startEquity = equities[0];
  const endEquity = equities[equities.length - 1];

  return {
    startEquity,
    endEquity,
    minEquity: Math.min(...equities),
    maxEquity: Math.max(...equities),
    totalReturn: endEquity - startEquity,
    totalReturnPct: startEquity > 0 ? ((endEquity - startEquity) / startEquity) * 100 : 0,
    maxDrawdown: calculateMaxDrawdown(data),
    barCount: data.length,
  };
}

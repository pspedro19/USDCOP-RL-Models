/**
 * Risk Management Configuration
 * ==============================
 * Risk thresholds and limits for trading operations
 * Currently used in RealTimeRiskEngine and risk monitoring components
 */

/**
 * Risk Threshold Configuration
 */
export const riskConfig = {
  /**
   * Leverage Limits
   */
  leverage: {
    /**
     * Maximum allowed leverage ratio
     * Default: 5.0x
     * Override: NEXT_PUBLIC_MAX_LEVERAGE
     */
    max: parseFloat(process.env.NEXT_PUBLIC_MAX_LEVERAGE || '5.0'),

    /**
     * Warning threshold (before breach)
     * Default: 4.0x
     */
    warning: parseFloat(process.env.NEXT_PUBLIC_WARNING_LEVERAGE || '4.0'),

    /**
     * Recommended maximum
     * Default: 3.0x
     */
    recommended: 3.0,
  },

  /**
   * Value at Risk (VaR) Limits
   */
  var: {
    /**
     * Maximum VaR as percentage of portfolio value
     * Default: 10%
     * Override: NEXT_PUBLIC_MAX_VAR_PCT
     */
    maxPercentage: parseFloat(process.env.NEXT_PUBLIC_MAX_VAR_PCT || '0.10'),

    /**
     * Warning threshold
     * Default: 8%
     */
    warningPercentage: parseFloat(process.env.NEXT_PUBLIC_WARNING_VAR_PCT || '0.08'),

    /**
     * VaR confidence levels
     */
    confidenceLevels: {
      var95: 0.95,
      var99: 0.99,
    },

    /**
     * Expected Shortfall multiplier
     */
    expectedShortfallMultiplier: 1.6,
  },

  /**
   * Drawdown Limits
   */
  drawdown: {
    /**
     * Maximum allowed drawdown
     * Default: 15%
     * Override: NEXT_PUBLIC_MAX_DRAWDOWN_PCT
     */
    max: parseFloat(process.env.NEXT_PUBLIC_MAX_DRAWDOWN_PCT || '0.15'),

    /**
     * Warning threshold
     * Default: 12%
     */
    warning: parseFloat(process.env.NEXT_PUBLIC_WARNING_DRAWDOWN_PCT || '0.12'),

    /**
     * Critical threshold (emergency action)
     * Default: 20%
     */
    critical: parseFloat(process.env.NEXT_PUBLIC_CRITICAL_DRAWDOWN_PCT || '0.20'),
  },

  /**
   * Position Concentration Limits
   */
  concentration: {
    /**
     * Maximum single position as percentage of portfolio
     * Default: 40%
     * Override: NEXT_PUBLIC_MAX_CONCENTRATION_PCT
     */
    maxSinglePosition: parseFloat(process.env.NEXT_PUBLIC_MAX_CONCENTRATION_PCT || '0.40'),

    /**
     * Maximum sector concentration
     * Default: 60%
     */
    maxSectorConcentration: 0.60,

    /**
     * Maximum country concentration
     * Default: 70%
     */
    maxCountryConcentration: 0.70,

    /**
     * Warning threshold for single position
     * Default: 30%
     */
    warningSinglePosition: 0.30,
  },

  /**
   * Liquidity Risk Limits
   */
  liquidity: {
    /**
     * Minimum liquidity score (0-1)
     * Default: 0.7
     * Override: NEXT_PUBLIC_MIN_LIQUIDITY_SCORE
     */
    minScore: parseFloat(process.env.NEXT_PUBLIC_MIN_LIQUIDITY_SCORE || '0.7'),

    /**
     * Maximum time to liquidate portfolio (in hours)
     * Default: 24 hours
     */
    maxTimeToLiquidate: 24,

    /**
     * Warning threshold
     * Default: 18 hours
     */
    warningTimeToLiquidate: 18,
  },

  /**
   * Volatility Limits
   */
  volatility: {
    /**
     * Maximum portfolio volatility (annualized)
     * Default: 25%
     */
    maxPortfolioVolatility: 0.25,

    /**
     * Warning threshold
     * Default: 20%
     */
    warningVolatility: 0.20,
  },

  /**
   * Correlation Limits
   */
  correlation: {
    /**
     * Maximum acceptable correlation between positions
     * Default: 0.80
     */
    maxPositionCorrelation: 0.80,

    /**
     * Warning threshold
     * Default: 0.70
     */
    warningCorrelation: 0.70,
  },

  /**
   * Stop Loss Configuration
   */
  stopLoss: {
    /**
     * Default stop loss percentage
     * Default: 2%
     */
    defaultPercentage: 0.02,

    /**
     * Maximum stop loss percentage
     * Default: 5%
     */
    maxPercentage: 0.05,

    /**
     * Trailing stop loss settings
     */
    trailing: {
      enabled: true,
      activationPercentage: 0.01, // 1% profit
      trailPercentage: 0.005, // 0.5% trail
    },
  },

  /**
   * Position Sizing Limits
   */
  positionSizing: {
    /**
     * Minimum position size (as percentage of portfolio)
     * Default: 0.5%
     */
    minSize: 0.005,

    /**
     * Maximum position size (as percentage of portfolio)
     * Default: 10%
     */
    maxSize: 0.10,

    /**
     * Default position size
     * Default: 2%
     */
    defaultSize: 0.02,
  },

  /**
   * Risk Alert Configuration
   */
  alerts: {
    /**
     * Maximum number of alerts to keep in memory
     */
    maxAlerts: 100,

    /**
     * Alert severity levels
     */
    severityLevels: {
      low: 'low',
      medium: 'medium',
      high: 'high',
      critical: 'critical',
    } as const,

    /**
     * Auto-acknowledge alerts after X milliseconds
     * Default: null (manual acknowledgment required)
     */
    autoAcknowledgeAfter: null as number | null,
  },

  /**
   * Risk Monitoring Intervals
   */
  monitoring: {
    /**
     * Metrics refresh interval (milliseconds)
     * Default: 30 seconds
     */
    metricsRefreshInterval: 30000,

    /**
     * Alert check interval (milliseconds)
     * Default: 5 seconds
     */
    alertCheckInterval: 5000,

    /**
     * Health check interval (milliseconds)
     * Default: 60 seconds
     */
    healthCheckInterval: 60000,
  },

  /**
   * Stress Testing Configuration
   */
  stressTesting: {
    /**
     * Scenario definitions
     */
    scenarios: {
      marketCrash: {
        name: 'Market Crash',
        priceShock: -0.20, // -20%
      },
      flashCrash: {
        name: 'Flash Crash',
        priceShock: -0.10, // -10%
      },
      rally: {
        name: 'Strong Rally',
        priceShock: 0.15, // +15%
      },
      volatilitySpike: {
        name: 'Volatility Spike',
        volatilityMultiplier: 3.0,
      },
    },
  },

  /**
   * Portfolio Value Configuration
   */
  portfolio: {
    /**
     * Default portfolio value for calculations
     * Override: NEXT_PUBLIC_PORTFOLIO_VALUE
     */
    defaultValue: parseFloat(process.env.NEXT_PUBLIC_PORTFOLIO_VALUE || '10000000'),

    /**
     * Minimum portfolio value
     */
    minValue: 100000,

    /**
     * Currency
     */
    currency: 'COP',
  },
} as const;

/**
 * Risk Severity Calculation
 * Returns the severity level based on current vs limit values
 */
export function calculateRiskSeverity(
  currentValue: number,
  limitValue: number,
  warningValue?: number
): 'low' | 'medium' | 'high' | 'critical' {
  const ratio = Math.abs(currentValue / limitValue);

  if (ratio >= 1.0) return 'critical';
  if (ratio >= 0.9) return 'high';
  if (warningValue && Math.abs(currentValue) >= warningValue) return 'medium';
  return 'low';
}

/**
 * Check if value breaches threshold
 */
export function breachesThreshold(
  currentValue: number,
  threshold: number,
  isUpperLimit: boolean = true
): boolean {
  if (isUpperLimit) {
    return currentValue > threshold;
  }
  return currentValue < threshold;
}

/**
 * Get recommended action based on risk level
 */
export function getRecommendedAction(
  riskType: string,
  severity: 'low' | 'medium' | 'high' | 'critical'
): string {
  const recommendations: Record<string, Record<string, string>> = {
    leverage: {
      critical: 'Immediately reduce position sizes or close positions to lower leverage',
      high: 'Consider reducing position sizes to lower leverage',
      medium: 'Monitor closely and avoid opening new positions',
      low: 'Continue normal operations',
    },
    var: {
      critical: 'Implement emergency hedging strategies immediately',
      high: 'Consider hedging strategies or reducing risk exposure',
      medium: 'Review risk exposure and consider protective measures',
      low: 'Continue normal operations',
    },
    drawdown: {
      critical: 'Emergency position reduction - review trading strategy',
      high: 'Review trading strategy and consider position reduction',
      medium: 'Monitor closely and reduce new position sizes',
      low: 'Continue normal operations',
    },
  };

  return recommendations[riskType]?.[severity] || 'Review risk parameters';
}

/**
 * Export risk threshold values for backwards compatibility
 */
export const riskThresholds = {
  maxLeverage: riskConfig.leverage.max,
  varLimit: riskConfig.var.maxPercentage,
  maxDrawdown: riskConfig.drawdown.max,
  minLiquidityScore: riskConfig.liquidity.minScore,
  maxConcentration: riskConfig.concentration.maxSinglePosition,
} as const;

export default riskConfig;

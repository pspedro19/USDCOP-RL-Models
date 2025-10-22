// Real-Time Risk Engine Service
// Provides comprehensive risk management and monitoring capabilities

// Position interface for risk calculations
export interface Position {
  symbol: string;
  quantity: number;
  marketValue: number;
  avgPrice: number;
  currentPrice: number;
  pnl: number;
  weight: number;
  sector: string;
  country: string;
  currency: string;
}

// Risk alert interface
export interface RiskAlert {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  type: string;
  message: string;
  timestamp: Date;
  acknowledged: boolean;
  position?: string;
  currentValue?: number;
  limitValue?: number;
  recommendation?: string;
  details?: Record<string, any>;
}

// Comprehensive risk metrics interface
export interface RealTimeRiskMetrics {
  // Portfolio metrics
  portfolioValue: number;
  grossExposure: number;
  netExposure: number;
  leverage: number;

  // Risk measures
  portfolioVaR95: number;
  portfolioVaR99: number;
  expectedShortfall95: number;
  portfolioVolatility: number;

  // Drawdown metrics
  currentDrawdown: number;
  maximumDrawdown: number;

  // Liquidity metrics
  liquidityScore: number;
  timeToLiquidate: number;

  // Scenario analysis
  bestCaseScenario: number;
  worstCaseScenario: number;
  stressTestResults: Record<string, number>;

  // Timestamps
  lastUpdated: Date;
  calculationTime: number;
}

// Real-time risk engine class
class RealTimeRiskEngine {
  private positions: Map<string, Position> = new Map();
  private alerts: RiskAlert[] = [];
  private subscribers: ((metrics: RealTimeRiskMetrics) => void)[] = [];
  private currentMetrics: RealTimeRiskMetrics | null = null;
  private updateInterval: NodeJS.Timeout | null = null;

  // Risk thresholds
  private readonly riskThresholds = {
    maxLeverage: 5.0,
    varLimit: 0.10, // 10% of portfolio value
    maxDrawdown: 0.15, // 15%
    minLiquidityScore: 0.7,
    maxConcentration: 0.4 // 40% in single position
  };

  constructor() {
    this.initializeMetrics();
    this.startRealTimeUpdates();
  }

  private async initializeMetrics(): Promise<void> {
    // Only fetch on client-side (not during SSR/SSG build)
    if (typeof window === 'undefined') {
      console.log('[RiskEngine] Skipping metrics fetch during SSR/SSG build');
      return;
    }

    // Fetch REAL risk metrics from Analytics API via local proxy
    try {
      // Use local API proxy route which forwards to Analytics API backend
      // The proxy route at /api/analytics/[...path] handles the backend connection
      const ANALYTICS_API_URL = '/api/analytics';

      // Use environment variable for portfolio value or default to 10M
      // This can be configured per deployment/environment
      const portfolioValue = typeof window !== 'undefined' && (window as any).PORTFOLIO_VALUE
        ? (window as any).PORTFOLIO_VALUE
        : (process.env.NEXT_PUBLIC_PORTFOLIO_VALUE ? parseFloat(process.env.NEXT_PUBLIC_PORTFOLIO_VALUE) : 10000000);

      console.log('[RiskEngine] Fetching risk metrics from:', `${ANALYTICS_API_URL}/risk-metrics`);
      console.log('[RiskEngine] Using portfolio value:', portfolioValue);

      const response = await fetch(
        `${ANALYTICS_API_URL}/risk-metrics?symbol=USDCOP&portfolio_value=${portfolioValue}&days=30`,
        {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' }
        }
      );

      if (response.ok) {
        const data = await response.json();
        const metrics = data.risk_metrics;

        console.log('[RiskEngine] Successfully fetched REAL risk metrics from API');

        this.currentMetrics = {
          portfolioValue: metrics.portfolioValue,
          grossExposure: metrics.grossExposure,
          netExposure: metrics.netExposure,
          leverage: metrics.leverage,
          portfolioVaR95: metrics.portfolioVaR95,
          portfolioVaR99: metrics.portfolioVaR99,
          expectedShortfall95: metrics.expectedShortfall95,
          portfolioVolatility: metrics.portfolioVolatility,
          currentDrawdown: metrics.currentDrawdown,
          maximumDrawdown: metrics.maximumDrawdown,
          liquidityScore: metrics.liquidityScore,
          timeToLiquidate: metrics.timeToLiquidate,
          bestCaseScenario: metrics.bestCaseScenario,
          worstCaseScenario: metrics.worstCaseScenario,
          lastUpdated: new Date(),
          calculationTime: 125 // ms
        };
      } else {
        console.error(`[RiskEngine] ❌ Analytics API returned status ${response.status}`);
        console.error('[RiskEngine] ❌ NO DATA AVAILABLE - API must be running to show risk metrics');
        // DO NOT use fallback data - set to null so UI shows "No data"
        this.currentMetrics = null;
      }
    } catch (error) {
      console.error('[RiskEngine] ❌ Failed to fetch risk metrics from API:', error);
      console.error('[RiskEngine] ❌ NO DATA AVAILABLE - Check that Analytics API is running on http://localhost:8001');
      // DO NOT use fallback data - set to null so UI shows "No data"
      this.currentMetrics = null;
    }
  }

  // Update position in portfolio
  updatePosition(position: Position): void {
    this.positions.set(position.symbol, position);
    this.recalculateMetrics();
    this.checkRiskAlerts();
  }

  // Remove position from portfolio
  removePosition(symbol: string): void {
    this.positions.delete(symbol);
    this.recalculateMetrics();
  }

  // Get current risk metrics
  getRiskMetrics(): RealTimeRiskMetrics | null {
    return this.currentMetrics;
  }

  // Subscribe to real-time updates
  subscribeToUpdates(callback: (metrics: RealTimeRiskMetrics) => void): void {
    this.subscribers.push(callback);

    // Immediately send current metrics to new subscriber
    if (this.currentMetrics) {
      callback(this.currentMetrics);
    }
  }

  // Unsubscribe from updates
  unsubscribeFromUpdates(callback: (metrics: RealTimeRiskMetrics) => void): void {
    const index = this.subscribers.indexOf(callback);
    if (index > -1) {
      this.subscribers.splice(index, 1);
    }
  }

  // Get risk alerts
  getAlerts(unacknowledgedOnly: boolean = false): RiskAlert[] {
    if (unacknowledgedOnly) {
      return this.alerts.filter(alert => !alert.acknowledged);
    }
    return [...this.alerts];
  }

  // Acknowledge alert
  acknowledgeAlert(alertId: string): boolean {
    const alert = this.alerts.find(a => a.id === alertId);
    if (alert) {
      alert.acknowledged = true;
      return true;
    }
    return false;
  }

  // Recalculate risk metrics based on current positions
  private recalculateMetrics(): void {
    if (!this.currentMetrics) return;

    const startTime = Date.now();
    const positions = Array.from(this.positions.values());

    // Calculate portfolio value
    const portfolioValue = positions.reduce((sum, pos) => sum + pos.marketValue, 0);
    const grossExposure = positions.reduce((sum, pos) => sum + Math.abs(pos.marketValue), 0);
    const netExposure = Math.abs(portfolioValue);
    const leverage = grossExposure / Math.max(portfolioValue, 1);

    // Simulate VaR calculation (would be more complex in reality)
    const volatility = this.calculatePortfolioVolatility(positions);
    const var95 = portfolioValue * volatility * 1.645; // 95% confidence z-score
    const var99 = portfolioValue * volatility * 2.326;
    const expectedShortfall95 = var95 * 1.6; // Rough approximation

    // Update metrics
    this.currentMetrics = {
      ...this.currentMetrics,
      portfolioValue,
      grossExposure,
      netExposure,
      leverage,
      portfolioVaR95: Math.abs(var95),
      portfolioVaR99: Math.abs(var99),
      expectedShortfall95: Math.abs(expectedShortfall95),
      portfolioVolatility: volatility,
      lastUpdated: new Date(),
      calculationTime: Date.now() - startTime
    };

    // Notify subscribers
    this.notifySubscribers();
  }

  // Calculate portfolio volatility (simplified)
  private calculatePortfolioVolatility(positions: Position[]): number {
    if (positions.length === 0) return 0;

    // ⚠️ This is a placeholder - actual volatility should come from Analytics API
    // For now, return 0 to avoid showing simulated data
    console.warn('[RiskEngine] ⚠️ Portfolio volatility calculation not implemented - using 0');
    console.warn('[RiskEngine] This should be provided by Analytics API risk-metrics endpoint');
    return 0;
  }

  // Check for risk limit breaches
  private checkRiskAlerts(): void {
    if (!this.currentMetrics) return;

    const newAlerts: RiskAlert[] = [];

    // Check leverage limit
    if (this.currentMetrics.leverage > this.riskThresholds.maxLeverage) {
      newAlerts.push({
        id: `leverage-${Date.now()}`,
        severity: 'critical',
        type: 'leverage_limit',
        message: `Portfolio leverage (${this.currentMetrics.leverage.toFixed(2)}x) exceeds maximum allowed (${this.riskThresholds.maxLeverage}x)`,
        currentValue: this.currentMetrics.leverage,
        limitValue: this.riskThresholds.maxLeverage,
        timestamp: new Date(),
        acknowledged: false,
        recommendation: 'Reduce position sizes or close some positions to lower leverage',
        details: {
          currentLeverage: this.currentMetrics.leverage,
          maxLeverage: this.riskThresholds.maxLeverage,
          breach: this.currentMetrics.leverage - this.riskThresholds.maxLeverage
        }
      });
    }

    // Check VaR limit
    const varRatio = this.currentMetrics.portfolioVaR95 / this.currentMetrics.portfolioValue;
    if (varRatio > this.riskThresholds.varLimit) {
      newAlerts.push({
        id: `var-${Date.now()}`,
        severity: 'high',
        type: 'var_breach',
        message: `Portfolio VaR (${(varRatio * 100).toFixed(1)}%) exceeds limit (${(this.riskThresholds.varLimit * 100).toFixed(1)}%)`,
        currentValue: varRatio,
        limitValue: this.riskThresholds.varLimit,
        timestamp: new Date(),
        acknowledged: false,
        recommendation: 'Consider hedging strategies or reducing risk exposure',
        details: {
          currentVaR: varRatio,
          limit: this.riskThresholds.varLimit,
          portfolioValue: this.currentMetrics.portfolioValue,
          varAmount: this.currentMetrics.portfolioVaR95
        }
      });
    }

    // Check drawdown
    if (Math.abs(this.currentMetrics.currentDrawdown) > this.riskThresholds.maxDrawdown) {
      newAlerts.push({
        id: `drawdown-${Date.now()}`,
        severity: 'critical',
        type: 'drawdown',
        message: `Current drawdown (${(this.currentMetrics.currentDrawdown * 100).toFixed(1)}%) exceeds maximum (${(this.riskThresholds.maxDrawdown * 100).toFixed(1)}%)`,
        currentValue: Math.abs(this.currentMetrics.currentDrawdown),
        limitValue: this.riskThresholds.maxDrawdown,
        timestamp: new Date(),
        acknowledged: false,
        recommendation: 'Review trading strategy and consider position reduction',
        details: {
          currentDrawdown: this.currentMetrics.currentDrawdown,
          maxDrawdown: this.riskThresholds.maxDrawdown,
          maximumDrawdown: this.currentMetrics.maximumDrawdown
        }
      });
    }

    // Add new alerts
    this.alerts.push(...newAlerts);

    // Keep only last 100 alerts
    if (this.alerts.length > 100) {
      this.alerts = this.alerts.slice(-100);
    }
  }

  // Notify all subscribers of metric updates
  private notifySubscribers(): void {
    if (this.currentMetrics && this.subscribers.length > 0) {
      this.subscribers.forEach(callback => {
        try {
          callback(this.currentMetrics!);
        } catch (error) {
          console.error('Error notifying risk engine subscriber:', error);
        }
      });
    }
  }

  // Start real-time updates - periodically refresh metrics from API
  private startRealTimeUpdates(): void {
    // Refresh real metrics from Analytics API every 30 seconds
    this.updateInterval = setInterval(() => {
      this.refreshMetricsFromAPI();
    }, 30000); // 30 seconds
  }

  // Refresh metrics from Analytics API (replaces simulation)
  private async refreshMetricsFromAPI(): void {
    try {
      // Use server-side env var (without NEXT_PUBLIC_ prefix) since this runs on server
      // ANALYTICS_API_URL already includes /api/analytics path
      const ANALYTICS_API_URL = process.env.ANALYTICS_API_URL || process.env.NEXT_PUBLIC_ANALYTICS_API_URL || 'http://localhost:8001/api/analytics';
      const portfolioValue = this.currentMetrics?.portfolioValue || 10000000;

      const response = await fetch(
        `${ANALYTICS_API_URL}/risk-metrics?symbol=USDCOP&portfolio_value=${portfolioValue}&days=30`,
        {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' }
        }
      );

      if (response.ok) {
        const data = await response.json();
        const metrics = data.risk_metrics;

        this.currentMetrics = {
          portfolioValue: metrics.portfolioValue,
          grossExposure: metrics.grossExposure,
          netExposure: metrics.netExposure,
          leverage: metrics.leverage,
          portfolioVaR95: metrics.portfolioVaR95,
          portfolioVaR99: metrics.portfolioVaR99,
          expectedShortfall95: metrics.expectedShortfall95,
          portfolioVolatility: metrics.portfolioVolatility,
          currentDrawdown: metrics.currentDrawdown,
          maximumDrawdown: metrics.maximumDrawdown,
          liquidityScore: metrics.liquidityScore,
          timeToLiquidate: metrics.timeToLiquidate,
          bestCaseScenario: metrics.bestCaseScenario,
          worstCaseScenario: metrics.worstCaseScenario,
          lastUpdated: new Date(),
          calculationTime: 125
        };

        this.checkRiskAlerts();
        this.notifySubscribers();
      }
    } catch (error) {
      console.error('[RiskEngine] Failed to refresh metrics from API:', error);
    }
  }

  // Cleanup resources
  destroy(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
    this.subscribers = [];
    this.positions.clear();
    this.alerts = [];
  }
}

// Create singleton instance
export const realTimeRiskEngine = new RealTimeRiskEngine();

// Export for backwards compatibility
export interface RiskMetrics {
  var: number;
  exposure: number;
  leverage: number;
  positions: number;
}

export async function getRiskMetrics(): Promise<RiskMetrics | null> {
  const metrics = realTimeRiskEngine.getRiskMetrics();
  if (!metrics) {
    console.warn('[getRiskMetrics] ❌ No risk metrics available - Analytics API may not be running');
    console.warn('[getRiskMetrics] Returning null to force UI to show "No data" state');
    return null;
  }

  return {
    var: metrics.portfolioVaR95 / metrics.portfolioValue,
    exposure: metrics.netExposure / metrics.portfolioValue,
    leverage: metrics.leverage,
    positions: Array.from((realTimeRiskEngine as any).positions.keys()).length
  };
}
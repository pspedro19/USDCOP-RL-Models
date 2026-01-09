/**
 * Equity Curve Builder
 * Builds equity curves and drawdown analysis from trades
 */

import { Trade, Position, EquityPoint, DrawdownInfo } from './types';

export class EquityCurveBuilder {
  /**
   * Build detailed equity curve from trades
   */
  static buildEquityCurve(
    trades: Trade[],
    initialCapital: number,
    includeIntraday: boolean = false
  ): EquityPoint[] {
    const curve: EquityPoint[] = [];

    // Add starting point
    const startTime = this.getStartTime(trades);
    curve.push({
      timestamp: startTime,
      value: initialCapital,
      cumReturn: 0,
      drawdown: 0,
      drawdownPercent: 0,
    });

    let capital = initialCapital;
    let peak = initialCapital;

    // Sort trades by exit time
    const sortedTrades = [...trades]
      .filter(t => t.status === 'closed' && t.exitTime)
      .sort((a, b) => a.exitTime! - b.exitTime!);

    for (const trade of sortedTrades) {
      // Update capital
      capital += trade.pnl;
      peak = Math.max(peak, capital);

      const drawdown = peak - capital;
      const drawdownPercent = peak > 0 ? (drawdown / peak) * 100 : 0;

      curve.push({
        timestamp: trade.exitTime!,
        value: capital,
        cumReturn: ((capital - initialCapital) / initialCapital) * 100,
        drawdown,
        drawdownPercent,
      });

      // Add intraday point if requested
      if (includeIntraday && trade.entryTime !== trade.exitTime) {
        // Interpolate equity at entry point
        const prevEquity = curve[curve.length - 2]?.value || initialCapital;
        const entryEquity = prevEquity + trade.pnl * 0.5; // Rough interpolation

        const entryPeak = Math.max(peak, entryEquity);
        const entryDrawdown = entryPeak - entryEquity;
        const entryDrawdownPercent = entryPeak > 0 ? (entryDrawdown / entryPeak) * 100 : 0;

        curve.push({
          timestamp: trade.entryTime,
          value: entryEquity,
          cumReturn: ((entryEquity - initialCapital) / initialCapital) * 100,
          drawdown: entryDrawdown,
          drawdownPercent: entryDrawdownPercent,
        });
      }
    }

    // Sort by timestamp to ensure chronological order
    return curve.sort((a, b) => a.timestamp - b.timestamp);
  }

  /**
   * Build equity curve with unrealized P&L
   */
  static buildLiveEquityCurve(
    trades: Trade[],
    positions: Position[],
    initialCapital: number
  ): EquityPoint[] {
    // Build closed trades equity curve
    const curve = this.buildEquityCurve(trades, initialCapital);

    // Add current unrealized P&L
    if (positions.length > 0) {
      const lastPoint = curve[curve.length - 1];
      const unrealizedPnL = positions.reduce((sum, p) => sum + p.unrealizedPnL, 0);
      const currentValue = lastPoint.value + unrealizedPnL;
      const peak = Math.max(...curve.map(p => p.value), currentValue);

      const drawdown = peak - currentValue;
      const drawdownPercent = peak > 0 ? (drawdown / peak) * 100 : 0;

      curve.push({
        timestamp: Date.now(),
        value: currentValue,
        cumReturn: ((currentValue - initialCapital) / initialCapital) * 100,
        drawdown,
        drawdownPercent,
      });
    }

    return curve;
  }

  /**
   * Calculate drawdown periods
   */
  static calculateDrawdowns(equityCurve: EquityPoint[]): DrawdownInfo[] {
    if (equityCurve.length === 0) return [];

    const drawdowns: DrawdownInfo[] = [];
    let currentDrawdown: DrawdownInfo | null = null;
    let peak = equityCurve[0].value;
    let peakTime = equityCurve[0].timestamp;

    for (let i = 0; i < equityCurve.length; i++) {
      const point = equityCurve[i];

      if (point.value >= peak) {
        // New peak reached
        if (currentDrawdown) {
          // End current drawdown
          currentDrawdown.end = point.timestamp;
          currentDrawdown.duration = point.timestamp - currentDrawdown.start;
          currentDrawdown.recovered = true;
          drawdowns.push(currentDrawdown);
          currentDrawdown = null;
        }
        peak = point.value;
        peakTime = point.timestamp;
      } else if (point.value < peak) {
        // In drawdown
        if (!currentDrawdown) {
          // Start new drawdown
          currentDrawdown = {
            start: peakTime,
            peak,
            trough: point.value,
            value: peak - point.value,
            percent: ((peak - point.value) / peak) * 100,
            duration: 0,
            recovered: false,
          };
        } else {
          // Update if deeper drawdown
          if (point.value < currentDrawdown.trough) {
            currentDrawdown.trough = point.value;
            currentDrawdown.value = peak - point.value;
            currentDrawdown.percent = ((peak - point.value) / peak) * 100;
          }
        }
      }
    }

    // Handle ongoing drawdown
    if (currentDrawdown) {
      const lastPoint = equityCurve[equityCurve.length - 1];
      currentDrawdown.duration = lastPoint.timestamp - currentDrawdown.start;
      currentDrawdown.recovered = false;
      drawdowns.push(currentDrawdown);
    }

    // Sort by percent (largest first)
    return drawdowns.sort((a, b) => b.percent - a.percent);
  }

  /**
   * Get maximum drawdown info
   */
  static getMaxDrawdown(equityCurve: EquityPoint[]): {
    value: number;
    percent: number;
    start: number;
    end: number | null;
    duration: number;
    recovered: boolean;
  } {
    const drawdowns = this.calculateDrawdowns(equityCurve);

    if (drawdowns.length === 0) {
      return {
        value: 0,
        percent: 0,
        start: Date.now(),
        end: null,
        duration: 0,
        recovered: true,
      };
    }

    const maxDrawdown = drawdowns[0];
    return {
      value: maxDrawdown.value,
      percent: maxDrawdown.percent,
      start: maxDrawdown.start,
      end: maxDrawdown.end || null,
      duration: maxDrawdown.duration,
      recovered: maxDrawdown.recovered,
    };
  }

  /**
   * Get current drawdown info
   */
  static getCurrentDrawdown(equityCurve: EquityPoint[]): {
    value: number;
    percent: number;
    duration: number;
  } {
    if (equityCurve.length === 0) {
      return { value: 0, percent: 0, duration: 0 };
    }

    const lastPoint = equityCurve[equityCurve.length - 1];
    const peak = Math.max(...equityCurve.map(p => p.value));

    if (lastPoint.value >= peak) {
      return { value: 0, percent: 0, duration: 0 };
    }

    // Find when current drawdown started
    let drawdownStart = lastPoint.timestamp;
    for (let i = equityCurve.length - 1; i >= 0; i--) {
      if (equityCurve[i].value >= peak) {
        drawdownStart = equityCurve[i].timestamp;
        break;
      }
    }

    return {
      value: peak - lastPoint.value,
      percent: ((peak - lastPoint.value) / peak) * 100,
      duration: lastPoint.timestamp - drawdownStart,
    };
  }

  /**
   * Smooth equity curve using moving average
   */
  static smoothCurve(curve: EquityPoint[], window: number = 10): EquityPoint[] {
    if (curve.length <= window) return curve;

    const smoothed: EquityPoint[] = [];

    for (let i = 0; i < curve.length; i++) {
      const start = Math.max(0, i - Math.floor(window / 2));
      const end = Math.min(curve.length, i + Math.ceil(window / 2));
      const windowPoints = curve.slice(start, end);

      const avgValue = windowPoints.reduce((sum, p) => sum + p.value, 0) / windowPoints.length;

      smoothed.push({
        ...curve[i],
        value: avgValue,
      });
    }

    return smoothed;
  }

  /**
   * Resample equity curve to fixed time intervals
   */
  static resampleCurve(
    curve: EquityPoint[],
    intervalMs: number
  ): EquityPoint[] {
    if (curve.length === 0) return [];

    const resampled: EquityPoint[] = [];
    const startTime = curve[0].timestamp;
    const endTime = curve[curve.length - 1].timestamp;

    let currentTime = startTime;
    let curveIndex = 0;

    while (currentTime <= endTime) {
      // Find the point just before or at current time
      while (curveIndex < curve.length - 1 && curve[curveIndex + 1].timestamp <= currentTime) {
        curveIndex++;
      }

      // Interpolate if needed
      if (curveIndex < curve.length - 1) {
        const p1 = curve[curveIndex];
        const p2 = curve[curveIndex + 1];
        const ratio = (currentTime - p1.timestamp) / (p2.timestamp - p1.timestamp);

        resampled.push({
          timestamp: currentTime,
          value: p1.value + (p2.value - p1.value) * ratio,
          cumReturn: p1.cumReturn + (p2.cumReturn - p1.cumReturn) * ratio,
          drawdown: p1.drawdown + (p2.drawdown - p1.drawdown) * ratio,
          drawdownPercent: p1.drawdownPercent + (p2.drawdownPercent - p1.drawdownPercent) * ratio,
        });
      } else {
        resampled.push(curve[curveIndex]);
      }

      currentTime += intervalMs;
    }

    return resampled;
  }

  /**
   * Calculate rolling returns
   */
  static calculateRollingReturns(
    curve: EquityPoint[],
    windowMs: number
  ): { timestamp: number; return: number }[] {
    const returns: { timestamp: number; return: number }[] = [];

    for (let i = 0; i < curve.length; i++) {
      const currentPoint = curve[i];
      const windowStart = currentPoint.timestamp - windowMs;

      // Find point at window start
      let startPoint = curve[0];
      for (let j = i - 1; j >= 0; j--) {
        if (curve[j].timestamp <= windowStart) {
          startPoint = curve[j];
          break;
        }
      }

      const returnPct = startPoint.value > 0
        ? ((currentPoint.value - startPoint.value) / startPoint.value) * 100
        : 0;

      returns.push({
        timestamp: currentPoint.timestamp,
        return: returnPct,
      });
    }

    return returns;
  }

  /**
   * Calculate daily returns
   */
  static calculateDailyReturns(curve: EquityPoint[]): number[] {
    const dailyReturns: number[] = [];

    for (let i = 1; i < curve.length; i++) {
      const prevValue = curve[i - 1].value;
      const currValue = curve[i].value;

      if (prevValue > 0) {
        const dailyReturn = (currValue - prevValue) / prevValue;
        dailyReturns.push(dailyReturn);
      }
    }

    return dailyReturns;
  }

  /**
   * Get underwater plot (drawdown over time)
   */
  static getUnderwaterPlot(curve: EquityPoint[]): { timestamp: number; drawdown: number }[] {
    return curve.map(point => ({
      timestamp: point.timestamp,
      drawdown: -point.drawdownPercent, // Negative for plotting below zero
    }));
  }

  /**
   * Helper to get start time
   */
  private static getStartTime(trades: Trade[]): number {
    if (trades.length === 0) {
      return Date.now() - 30 * 24 * 60 * 60 * 1000; // 30 days ago
    }

    const sortedTrades = [...trades].sort((a, b) => a.entryTime - b.entryTime);
    return sortedTrades[0].entryTime;
  }
}

/**
 * Real Market Metrics Calculator
 * Based on actual data structure: timestamp, price, bid, ask
 * Calculates only metrics that are accurate with our data
 */

export interface OHLCData {
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  price: number;
  bid: number;
  ask: number;
  volume?: number;
}

export interface RealMarketMetrics {
  // Spread metrics (REAL - tenemos bid/ask)
  currentSpread: {
    absolute: number;        // ask - bid
    bps: number;            // basis points
    percentage: number;     // % del precio
  };

  // Volatility metrics (REAL - calculadas de OHLC)
  volatility: {
    atr14: number;           // Average True Range 14 periods
    parkinson: number;       // Parkinson estimator (anualizada)
    garmanKlass: number;     // Garman-Klass estimator (anualizada)
    yangZhang: number;       // Yang-Zhang estimator (anualizada)
  };

  // Price action (REAL)
  priceAction: {
    sessionHigh: number;
    sessionLow: number;
    sessionRange: number;
    sessionRangePct: number;
    currentPrice: number;
    pricePosition: number;   // Position en el rango 0-1
  };

  // Returns analysis (REAL)
  returns: {
    current: number;         // Return actual vs anterior
    intraday: number;        // Return intradía
    drawdown: number;        // Drawdown actual desde máximo
    maxDrawdown: number;     // Max drawdown del período
  };

  // Market activity (REAL)
  activity: {
    ticksPerHour: number;    // Ticks recibidos por hora
    avgSpread: number;       // Spread promedio del período
    spreadStability: number; // Estabilidad del spread (1-CV)
    dataQuality: number;     // % de datos válidos
  };

  // Trading session stats (REAL)
  session: {
    isMarketHours: boolean;
    timeInSession: number;   // Minutos desde apertura
    progressPct: number;     // % completado de la sesión
    remainingMinutes: number;
  };
}

export class RealMarketMetricsCalculator {
  private static readonly TRADING_DAYS_YEAR = 252;
  private static readonly MARKET_OPEN_HOUR = 13; // 8:00 AM COT = 13:00 UTC
  private static readonly MARKET_CLOSE_HOUR = 18; // 1:00 PM COT = 18:00 UTC
  private static readonly SESSION_DURATION_MINUTES = 300; // 5 horas = 300 minutos

  /**
   * Calculate comprehensive real metrics from OHLC data
   */
  static calculateMetrics(
    data: OHLCData[],
    currentTick?: { price: number; bid: number; ask: number; timestamp: Date }
  ): RealMarketMetrics {
    if (data.length === 0) {
      throw new Error('No data provided for metrics calculation');
    }

    const latest = currentTick || {
      price: data[data.length - 1].price,
      bid: data[data.length - 1].bid,
      ask: data[data.length - 1].ask,
      timestamp: data[data.length - 1].timestamp
    };

    return {
      currentSpread: this.calculateSpreadMetrics(latest),
      volatility: this.calculateVolatilityMetrics(data),
      priceAction: this.calculatePriceActionMetrics(data, latest.price),
      returns: this.calculateReturnsMetrics(data),
      activity: this.calculateActivityMetrics(data),
      session: this.calculateSessionMetrics(latest.timestamp)
    };
  }

  /**
   * Real spread calculation (we have bid/ask)
   */
  private static calculateSpreadMetrics(tick: { price: number; bid: number; ask: number }) {
    const absolute = tick.ask - tick.bid;
    const mid = (tick.ask + tick.bid) / 2;
    const bps = (absolute / mid) * 10000;
    const percentage = (absolute / tick.price) * 100;

    return {
      absolute,
      bps,
      percentage
    };
  }

  /**
   * Volatility estimators that don't require volume
   */
  private static calculateVolatilityMetrics(data: OHLCData[]) {
    const periods = Math.min(14, data.length);
    const recentData = data.slice(-periods);

    return {
      atr14: this.calculateATR(recentData),
      parkinson: this.calculateParkinsonVolatility(recentData),
      garmanKlass: this.calculateGarmanKlassVolatility(recentData),
      yangZhang: this.calculateYangZhangVolatility(recentData)
    };
  }

  /**
   * Average True Range
   */
  private static calculateATR(data: OHLCData[], periods: number = 14): number {
    if (data.length < 2) return 0;

    const trueRanges: number[] = [];

    for (let i = 1; i < data.length; i++) {
      const current = data[i];
      const previous = data[i - 1];

      const tr = Math.max(
        current.high - current.low,
        Math.abs(current.high - previous.close),
        Math.abs(current.low - previous.close)
      );

      trueRanges.push(tr);
    }

    const periodToUse = Math.min(periods, trueRanges.length);
    const recentTR = trueRanges.slice(-periodToUse);

    return recentTR.reduce((sum, tr) => sum + tr, 0) / recentTR.length;
  }

  /**
   * Parkinson volatility estimator (using High-Low)
   */
  private static calculateParkinsonVolatility(data: OHLCData[]): number {
    if (data.length === 0) return 0;

    const logRanges = data.map(candle => {
      if (candle.high <= 0 || candle.low <= 0) return 0;
      return Math.pow(Math.log(candle.high / candle.low), 2);
    });

    const avgLogRange = logRanges.reduce((sum, lr) => sum + lr, 0) / logRanges.length;
    const variance = avgLogRange / (4 * Math.log(2));

    // Anualizada
    return Math.sqrt(variance * this.TRADING_DAYS_YEAR);
  }

  /**
   * Garman-Klass volatility estimator (using OHLC)
   */
  private static calculateGarmanKlassVolatility(data: OHLCData[]): number {
    if (data.length === 0) return 0;

    const gkValues = data.map(candle => {
      if (candle.high <= 0 || candle.low <= 0 || candle.close <= 0 || candle.open <= 0) {
        return 0;
      }

      const ln_h_l = Math.log(candle.high / candle.low);
      const ln_c_o = Math.log(candle.close / candle.open);

      return 0.5 * Math.pow(ln_h_l, 2) - (2 * Math.log(2) - 1) * Math.pow(ln_c_o, 2);
    });

    const avgGK = gkValues.reduce((sum, gk) => sum + gk, 0) / gkValues.length;

    // Anualizada
    return Math.sqrt(avgGK * this.TRADING_DAYS_YEAR);
  }

  /**
   * Yang-Zhang volatility estimator
   */
  private static calculateYangZhangVolatility(data: OHLCData[]): number {
    if (data.length < 2) return 0;

    let overnightComponent = 0;
    let openToCloseComponent = 0;

    for (let i = 1; i < data.length; i++) {
      const current = data[i];
      const previous = data[i - 1];

      if (current.open > 0 && current.close > 0 && previous.close > 0) {
        // Overnight return component
        const overnightReturn = Math.log(current.open / previous.close);
        overnightComponent += Math.pow(overnightReturn, 2);

        // Open-to-close component (Garman-Klass for intraday)
        const ln_h_o = Math.log(current.high / current.open);
        const ln_l_o = Math.log(current.low / current.open);
        const ln_c_o = Math.log(current.close / current.open);

        const gkIntraday = ln_h_o * ln_l_o - 2 * Math.log(2) * Math.pow(ln_c_o, 2);
        openToCloseComponent += gkIntraday;
      }
    }

    const n = data.length - 1;
    const overnight = overnightComponent / n;
    const openToClose = openToCloseComponent / n;

    const yangZhangVar = overnight + openToClose;

    // Anualizada
    return Math.sqrt(yangZhangVar * this.TRADING_DAYS_YEAR);
  }

  /**
   * Price action metrics for current session
   */
  private static calculatePriceActionMetrics(data: OHLCData[], currentPrice: number) {
    if (data.length === 0) return {
      sessionHigh: currentPrice,
      sessionLow: currentPrice,
      sessionRange: 0,
      sessionRangePct: 0,
      currentPrice,
      pricePosition: 0.5
    };

    // Usar último día de datos como "sesión"
    const today = new Date();
    const sessionStart = new Date(today.getFullYear(), today.getMonth(), today.getDate());

    const sessionData = data.filter(d => d.timestamp >= sessionStart);
    const dataToUse = sessionData.length > 0 ? sessionData : data.slice(-20); // Fallback a últimas 20 barras

    const sessionHigh = Math.max(...dataToUse.map(d => d.high));
    const sessionLow = Math.min(...dataToUse.map(d => d.low));
    const sessionRange = sessionHigh - sessionLow;
    const sessionRangePct = sessionRange / sessionLow * 100;

    // Posición del precio actual en el rango (0 = low, 1 = high)
    const pricePosition = sessionRange > 0 ?
      (currentPrice - sessionLow) / sessionRange : 0.5;

    return {
      sessionHigh,
      sessionLow,
      sessionRange,
      sessionRangePct,
      currentPrice,
      pricePosition
    };
  }

  /**
   * Returns and drawdown metrics
   */
  private static calculateReturnsMetrics(data: OHLCData[]) {
    if (data.length < 2) return {
      current: 0,
      intraday: 0,
      drawdown: 0,
      maxDrawdown: 0
    };

    const latest = data[data.length - 1];
    const previous = data[data.length - 2];

    // Current return vs previous close
    const current = (latest.close - previous.close) / previous.close;

    // Intraday return (open to current)
    const intraday = (latest.close - latest.open) / latest.open;

    // Calcular drawdown
    const prices = data.map(d => d.close);
    let maxPrice = prices[0];
    let maxDrawdown = 0;
    let currentDrawdown = 0;

    for (const price of prices) {
      if (price > maxPrice) {
        maxPrice = price;
      }

      const drawdown = (price - maxPrice) / maxPrice;
      currentDrawdown = drawdown;

      if (drawdown < maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }

    return {
      current,
      intraday,
      drawdown: currentDrawdown,
      maxDrawdown
    };
  }

  /**
   * Market activity metrics
   */
  private static calculateActivityMetrics(data: OHLCData[]) {
    if (data.length === 0) return {
      ticksPerHour: 0,
      avgSpread: 0,
      spreadStability: 0,
      dataQuality: 1
    };

    // Calcular ticks por hora basado en frecuencia de datos
    const timeSpan = data[data.length - 1].timestamp.getTime() - data[0].timestamp.getTime();
    const hoursSpan = timeSpan / (1000 * 60 * 60);
    const ticksPerHour = hoursSpan > 0 ? data.length / hoursSpan : 0;

    // Spread promedio y estabilidad
    const spreads = data.map(d => d.ask - d.bid).filter(s => s > 0);
    const avgSpread = spreads.length > 0 ?
      spreads.reduce((sum, s) => sum + s, 0) / spreads.length : 0;

    // Coeficiente de variación del spread (menor = más estable)
    let spreadStability = 1;
    if (spreads.length > 1) {
      const spreadStd = this.calculateStandardDeviation(spreads);
      const cv = avgSpread > 0 ? spreadStd / avgSpread : 0;
      spreadStability = Math.max(0, 1 - cv); // 1 = perfectamente estable
    }

    // Calidad de datos (% de registros con bid/ask válidos)
    const validRecords = data.filter(d => d.bid > 0 && d.ask > 0 && d.ask > d.bid).length;
    const dataQuality = data.length > 0 ? validRecords / data.length : 0;

    return {
      ticksPerHour,
      avgSpread,
      spreadStability,
      dataQuality
    };
  }

  /**
   * Trading session metrics
   */
  private static calculateSessionMetrics(currentTime: Date) {
    const now = new Date(currentTime);
    const currentHour = now.getUTCHours() + (now.getUTCMinutes() / 60);

    const isMarketHours = currentHour >= this.MARKET_OPEN_HOUR &&
                         currentHour <= this.MARKET_CLOSE_HOUR;

    // Tiempo transcurrido desde apertura
    const minutesSinceOpen = Math.max(0,
      (currentHour - this.MARKET_OPEN_HOUR) * 60
    );

    const timeInSession = Math.min(minutesSinceOpen, this.SESSION_DURATION_MINUTES);
    const progressPct = (timeInSession / this.SESSION_DURATION_MINUTES) * 100;
    const remainingMinutes = Math.max(0, this.SESSION_DURATION_MINUTES - timeInSession);

    return {
      isMarketHours,
      timeInSession,
      progressPct,
      remainingMinutes
    };
  }

  /**
   * Helper: Calculate standard deviation
   */
  private static calculateStandardDeviation(values: number[]): number {
    if (values.length < 2) return 0;

    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    const variance = squaredDiffs.reduce((sum, diff) => sum + diff, 0) / (values.length - 1);

    return Math.sqrt(variance);
  }

  /**
   * Format metrics for display
   */
  static formatMetrics(metrics: RealMarketMetrics) {
    return {
      spread: {
        bps: metrics.currentSpread.bps.toFixed(1),
        percentage: metrics.currentSpread.percentage.toFixed(3),
        absolute: metrics.currentSpread.absolute.toFixed(4)
      },
      volatility: {
        atr: metrics.volatility.atr14.toFixed(2),
        parkinson: (metrics.volatility.parkinson * 100).toFixed(1) + '%',
        garmanKlass: (metrics.volatility.garmanKlass * 100).toFixed(1) + '%'
      },
      returns: {
        current: (metrics.returns.current * 100).toFixed(2) + '%',
        intraday: (metrics.returns.intraday * 100).toFixed(2) + '%',
        drawdown: (metrics.returns.drawdown * 100).toFixed(2) + '%'
      },
      session: {
        progress: metrics.session.progressPct.toFixed(1) + '%',
        remaining: Math.floor(metrics.session.remainingMinutes) + 'min'
      }
    };
  }
}
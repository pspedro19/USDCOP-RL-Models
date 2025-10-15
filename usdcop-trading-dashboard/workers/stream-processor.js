/**
 * Stream Processor Worker - Background Data Processing
 * High-performance worker for tick aggregation, indicator calculation, and pattern detection
 */

// Import ComLink for worker communication
// Note: In production, you would import this properly
// importScripts('https://unpkg.com/comlink/dist/umd/comlink.js');

class StreamProcessor {
  constructor() {
    this.tickBuffer = [];
    this.indicators = new Map();
    this.patterns = [];
    this.processingQueue = [];
    this.isProcessing = false;
  }

  // ==========================================
  // MAIN PROCESSING METHODS
  // ==========================================

  async processTick(tickData) {
    try {
      const result = {
        id: tickData.id || this.generateId(),
        type: 'tick_processed',
        timestamp: Date.now(),
        data: null,
        processingTime: 0,
        memoryUsed: 0
      };

      const startTime = performance.now();

      // Add to buffer
      this.addToBuffer(tickData);

      // Process indicators
      const indicators = await this.calculateIndicators(tickData);

      // Detect patterns
      const patterns = await this.detectPatterns(tickData);

      // Aggregate data
      const aggregated = await this.aggregateData(tickData);

      result.data = {
        tick: tickData,
        indicators,
        patterns,
        aggregated,
        bufferSize: this.tickBuffer.length
      };

      result.processingTime = performance.now() - startTime;
      result.memoryUsed = this.estimateMemoryUsage();

      return result;

    } catch (error) {
      return {
        id: tickData.id || this.generateId(),
        type: 'processing_error',
        timestamp: Date.now(),
        error: error.message,
        data: null
      };
    }
  }

  async aggregateData(tickData) {
    try {
      const symbol = tickData.symbol;
      const intervals = ['1s', '5s', '15s', '30s', '1m', '5m'];
      const aggregations = {};

      for (const interval of intervals) {
        const windowSize = this.getIntervalMs(interval);
        const windowTicks = this.getTicksInWindow(symbol, windowSize);

        if (windowTicks.length > 0) {
          aggregations[interval] = this.createAggregation(windowTicks, interval);
        }
      }

      return aggregations;

    } catch (error) {
      console.error('Aggregation error:', error);
      return {};
    }
  }

  async calculateIndicators(tickData) {
    try {
      const symbol = tickData.symbol;
      const price = tickData.last || tickData.price;

      if (!this.indicators.has(symbol)) {
        this.indicators.set(symbol, {
          prices: [],
          volumes: [],
          ema9: price,
          ema21: price,
          ema50: price,
          ema200: price,
          rsi: 50,
          macd: 0,
          signal: 0,
          bb: { upper: price, middle: price, lower: price },
          atr: 0,
          adx: 0,
          stoch: { k: 50, d: 50 },
          williamsR: -50
        });
      }

      const indicators = this.indicators.get(symbol);

      // Update price and volume history
      indicators.prices.push(price);
      indicators.volumes.push(tickData.volume || 0);

      // Keep only last 200 values for efficiency
      if (indicators.prices.length > 200) {
        indicators.prices = indicators.prices.slice(-200);
        indicators.volumes = indicators.volumes.slice(-200);
      }

      // Calculate indicators
      this.updateEMAs(indicators, price);
      this.updateRSI(indicators);
      this.updateMACD(indicators);
      this.updateBollingerBands(indicators);
      this.updateATR(indicators, tickData);
      this.updateStochastic(indicators, tickData);

      return {
        symbol,
        timestamp: Date.now(),
        price,
        ema9: indicators.ema9,
        ema21: indicators.ema21,
        ema50: indicators.ema50,
        ema200: indicators.ema200,
        rsi: indicators.rsi,
        macd: indicators.macd,
        signal: indicators.signal,
        histogram: indicators.macd - indicators.signal,
        bb_upper: indicators.bb.upper,
        bb_middle: indicators.bb.middle,
        bb_lower: indicators.bb.lower,
        atr: indicators.atr,
        stoch_k: indicators.stoch.k,
        stoch_d: indicators.stoch.d,
        williams_r: indicators.williamsR
      };

    } catch (error) {
      console.error('Indicator calculation error:', error);
      return null;
    }
  }

  async detectPatterns(tickData) {
    try {
      const symbol = tickData.symbol;
      const patterns = [];

      // Get recent price data
      const indicators = this.indicators.get(symbol);
      if (!indicators || indicators.prices.length < 20) {
        return patterns;
      }

      const prices = indicators.prices.slice(-20);
      const volumes = indicators.volumes.slice(-20);

      // Detect various patterns
      patterns.push(...this.detectTrendPatterns(prices));
      patterns.push(...this.detectVolumePatterns(volumes));
      patterns.push(...this.detectCandlestickPatterns(prices));
      patterns.push(...this.detectMomentumPatterns(indicators));

      return patterns.map(pattern => ({
        ...pattern,
        symbol,
        timestamp: Date.now()
      }));

    } catch (error) {
      console.error('Pattern detection error:', error);
      return [];
    }
  }

  async compressBuffer(bufferData) {
    try {
      const startTime = performance.now();

      // Simple compression: remove duplicates and outliers
      const compressed = this.removeDuplicates(bufferData);
      const cleaned = this.removeOutliers(compressed);

      const compressionRatio = bufferData.length / cleaned.length;

      return {
        type: 'compression_complete',
        originalSize: bufferData.length,
        compressedSize: cleaned.length,
        compressionRatio,
        data: cleaned,
        processingTime: performance.now() - startTime
      };

    } catch (error) {
      return {
        type: 'compression_error',
        error: error.message
      };
    }
  }

  async persistData(data) {
    try {
      // In a real implementation, this would persist to IndexedDB
      const startTime = performance.now();

      // Simulate persistence delay
      await new Promise(resolve => setTimeout(resolve, 10));

      return {
        type: 'persistence_complete',
        dataSize: JSON.stringify(data).length,
        processingTime: performance.now() - startTime
      };

    } catch (error) {
      return {
        type: 'persistence_error',
        error: error.message
      };
    }
  }

  async qualityCheck(data) {
    try {
      const checks = {
        completeness: this.checkCompleteness(data),
        timeliness: this.checkTimeliness(data),
        accuracy: this.checkAccuracy(data),
        consistency: this.checkConsistency(data)
      };

      const overallScore = Object.values(checks).reduce((sum, score) => sum + score, 0) / 4;

      return {
        type: 'quality_check_complete',
        checks,
        overallScore,
        timestamp: Date.now()
      };

    } catch (error) {
      return {
        type: 'quality_check_error',
        error: error.message
      };
    }
  }

  // ==========================================
  // INDICATOR CALCULATIONS
  // ==========================================

  updateEMAs(indicators, price) {
    // EMA calculation: EMA = (Price - PrevEMA) * (2 / (Period + 1)) + PrevEMA
    indicators.ema9 = this.calculateEMA(price, indicators.ema9, 9);
    indicators.ema21 = this.calculateEMA(price, indicators.ema21, 21);
    indicators.ema50 = this.calculateEMA(price, indicators.ema50, 50);
    indicators.ema200 = this.calculateEMA(price, indicators.ema200, 200);
  }

  calculateEMA(price, prevEMA, period) {
    const multiplier = 2 / (period + 1);
    return (price - prevEMA) * multiplier + prevEMA;
  }

  updateRSI(indicators) {
    const prices = indicators.prices;
    if (prices.length < 15) return;

    const period = 14;
    const recentPrices = prices.slice(-period - 1);

    let gains = 0;
    let losses = 0;

    for (let i = 1; i < recentPrices.length; i++) {
      const change = recentPrices[i] - recentPrices[i - 1];
      if (change > 0) {
        gains += change;
      } else {
        losses += Math.abs(change);
      }
    }

    const avgGain = gains / period;
    const avgLoss = losses / period;

    if (avgLoss === 0) {
      indicators.rsi = 100;
    } else {
      const rs = avgGain / avgLoss;
      indicators.rsi = 100 - (100 / (1 + rs));
    }
  }

  updateMACD(indicators) {
    // MACD = EMA12 - EMA26
    const ema12 = this.calculateEMA(indicators.prices[indicators.prices.length - 1], indicators.ema9, 12);
    const ema26 = this.calculateEMA(indicators.prices[indicators.prices.length - 1], indicators.ema21, 26);

    indicators.macd = ema12 - ema26;
    indicators.signal = this.calculateEMA(indicators.macd, indicators.signal || indicators.macd, 9);
  }

  updateBollingerBands(indicators) {
    const prices = indicators.prices;
    if (prices.length < 20) return;

    const period = 20;
    const recentPrices = prices.slice(-period);

    // Calculate SMA
    const sma = recentPrices.reduce((sum, price) => sum + price, 0) / period;

    // Calculate standard deviation
    const variance = recentPrices.reduce((sum, price) => sum + Math.pow(price - sma, 2), 0) / period;
    const stdDev = Math.sqrt(variance);

    indicators.bb.middle = sma;
    indicators.bb.upper = sma + (2 * stdDev);
    indicators.bb.lower = sma - (2 * stdDev);
  }

  updateATR(indicators, tickData) {
    const prices = indicators.prices;
    if (prices.length < 2) return;

    // Simplified ATR calculation
    const high = tickData.high || tickData.last;
    const low = tickData.low || tickData.last;
    const prevClose = prices[prices.length - 2];

    const tr1 = high - low;
    const tr2 = Math.abs(high - prevClose);
    const tr3 = Math.abs(low - prevClose);

    const trueRange = Math.max(tr1, tr2, tr3);
    indicators.atr = this.calculateEMA(trueRange, indicators.atr || trueRange, 14);
  }

  updateStochastic(indicators, tickData) {
    const prices = indicators.prices;
    if (prices.length < 14) return;

    const period = 14;
    const recentPrices = prices.slice(-period);
    const currentPrice = tickData.last || tickData.price;

    const highest = Math.max(...recentPrices);
    const lowest = Math.min(...recentPrices);

    const k = ((currentPrice - lowest) / (highest - lowest)) * 100;
    indicators.stoch.k = k;
    indicators.stoch.d = this.calculateEMA(k, indicators.stoch.d, 3);
  }

  // ==========================================
  // PATTERN DETECTION
  // ==========================================

  detectTrendPatterns(prices) {
    const patterns = [];

    if (prices.length < 10) return patterns;

    // Detect uptrend
    const recentPrices = prices.slice(-5);
    let isUptrend = true;
    for (let i = 1; i < recentPrices.length; i++) {
      if (recentPrices[i] <= recentPrices[i - 1]) {
        isUptrend = false;
        break;
      }
    }

    if (isUptrend) {
      patterns.push({
        type: 'uptrend',
        confidence: 0.8,
        description: 'Consecutive higher prices detected'
      });
    }

    // Detect downtrend
    let isDowntrend = true;
    for (let i = 1; i < recentPrices.length; i++) {
      if (recentPrices[i] >= recentPrices[i - 1]) {
        isDowntrend = false;
        break;
      }
    }

    if (isDowntrend) {
      patterns.push({
        type: 'downtrend',
        confidence: 0.8,
        description: 'Consecutive lower prices detected'
      });
    }

    return patterns;
  }

  detectVolumePatterns(volumes) {
    const patterns = [];

    if (volumes.length < 5) return patterns;

    const recent = volumes.slice(-1)[0];
    const historical = volumes.slice(-5, -1);
    const avgVolume = historical.reduce((sum, vol) => sum + vol, 0) / historical.length;

    if (recent > avgVolume * 2) {
      patterns.push({
        type: 'volume_spike',
        confidence: 0.9,
        description: `Volume spike: ${(recent / avgVolume).toFixed(2)}x normal`
      });
    }

    return patterns;
  }

  detectCandlestickPatterns(prices) {
    const patterns = [];

    if (prices.length < 4) return patterns;

    // Simple doji detection (open ≈ close)
    const recent = prices.slice(-4);
    const open = recent[0];
    const close = recent[3];
    const high = Math.max(...recent);
    const low = Math.min(...recent);

    const body = Math.abs(close - open);
    const range = high - low;

    if (body < range * 0.1) {
      patterns.push({
        type: 'doji',
        confidence: 0.7,
        description: 'Doji pattern detected - market indecision'
      });
    }

    return patterns;
  }

  detectMomentumPatterns(indicators) {
    const patterns = [];

    // RSI overbought/oversold
    if (indicators.rsi > 80) {
      patterns.push({
        type: 'overbought',
        confidence: 0.8,
        description: `RSI overbought: ${indicators.rsi.toFixed(2)}`
      });
    } else if (indicators.rsi < 20) {
      patterns.push({
        type: 'oversold',
        confidence: 0.8,
        description: `RSI oversold: ${indicators.rsi.toFixed(2)}`
      });
    }

    // MACD signal crossover
    if (indicators.macd > indicators.signal && indicators.macd > 0) {
      patterns.push({
        type: 'bullish_crossover',
        confidence: 0.75,
        description: 'MACD bullish signal crossover'
      });
    }

    return patterns;
  }

  // ==========================================
  // DATA MANAGEMENT
  // ==========================================

  addToBuffer(tick) {
    this.tickBuffer.push({
      ...tick,
      timestamp: Date.now()
    });

    // Maintain buffer size (keep last 1000 ticks)
    if (this.tickBuffer.length > 1000) {
      this.tickBuffer = this.tickBuffer.slice(-1000);
    }
  }

  getTicksInWindow(symbol, windowMs) {
    const cutoff = Date.now() - windowMs;
    return this.tickBuffer.filter(tick =>
      tick.symbol === symbol && tick.timestamp >= cutoff
    );
  }

  createAggregation(ticks, interval) {
    if (ticks.length === 0) return null;

    const prices = ticks.map(t => t.last || t.price);
    const volumes = ticks.map(t => t.volume || 0);

    return {
      interval,
      timestamp: Date.now(),
      open: prices[0],
      high: Math.max(...prices),
      low: Math.min(...prices),
      close: prices[prices.length - 1],
      volume: volumes.reduce((sum, vol) => sum + vol, 0),
      tickCount: ticks.length,
      vwap: this.calculateVWAP(ticks)
    };
  }

  calculateVWAP(ticks) {
    let totalValue = 0;
    let totalVolume = 0;

    ticks.forEach(tick => {
      const price = tick.last || tick.price;
      const volume = tick.volume || 1;
      totalValue += price * volume;
      totalVolume += volume;
    });

    return totalVolume > 0 ? totalValue / totalVolume : 0;
  }

  removeDuplicates(data) {
    const seen = new Set();
    return data.filter(item => {
      const key = `${item.symbol}_${item.timestamp}_${item.price}`;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }

  removeOutliers(data) {
    if (data.length < 10) return data;

    // Simple outlier removal using IQR method
    const prices = data.map(d => d.last || d.price).sort((a, b) => a - b);
    const q1 = prices[Math.floor(prices.length * 0.25)];
    const q3 = prices[Math.floor(prices.length * 0.75)];
    const iqr = q3 - q1;
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;

    return data.filter(item => {
      const price = item.last || item.price;
      return price >= lowerBound && price <= upperBound;
    });
  }

  // ==========================================
  // QUALITY CHECKS
  // ==========================================

  checkCompleteness(data) {
    if (!Array.isArray(data) || data.length === 0) return 0;

    const requiredFields = ['symbol', 'price', 'timestamp'];
    let completeItems = 0;

    data.forEach(item => {
      const hasAllFields = requiredFields.every(field =>
        item.hasOwnProperty(field) && item[field] != null
      );
      if (hasAllFields) completeItems++;
    });

    return (completeItems / data.length) * 100;
  }

  checkTimeliness(data) {
    if (!Array.isArray(data) || data.length === 0) return 0;

    const now = Date.now();
    const threshold = 60000; // 1 minute
    let timelyItems = 0;

    data.forEach(item => {
      const age = now - (item.timestamp || 0);
      if (age <= threshold) timelyItems++;
    });

    return (timelyItems / data.length) * 100;
  }

  checkAccuracy(data) {
    if (!Array.isArray(data) || data.length === 0) return 0;

    let accurateItems = 0;

    data.forEach(item => {
      const price = item.price || item.last;
      const volume = item.volume;

      // Basic accuracy checks
      let isAccurate = true;

      if (typeof price !== 'number' || price <= 0) isAccurate = false;
      if (volume !== undefined && (typeof volume !== 'number' || volume < 0)) isAccurate = false;
      if (item.bid && item.ask && item.bid >= item.ask) isAccurate = false;

      if (isAccurate) accurateItems++;
    });

    return (accurateItems / data.length) * 100;
  }

  checkConsistency(data) {
    if (!Array.isArray(data) || data.length < 2) return 100;

    let consistentItems = 0;

    for (let i = 1; i < data.length; i++) {
      const current = data[i];
      const previous = data[i - 1];

      // Check timestamp consistency
      let isConsistent = true;

      if (current.timestamp < previous.timestamp) isConsistent = false;

      // Check price consistency (no more than 10% change)
      const priceChange = Math.abs(current.price - previous.price) / previous.price;
      if (priceChange > 0.1) isConsistent = false;

      if (isConsistent) consistentItems++;
    }

    return (consistentItems / (data.length - 1)) * 100;
  }

  // ==========================================
  // UTILITY METHODS
  // ==========================================

  getIntervalMs(interval) {
    const intervals = {
      '1s': 1000,
      '5s': 5000,
      '15s': 15000,
      '30s': 30000,
      '1m': 60000,
      '5m': 300000,
      '15m': 900000,
      '30m': 1800000,
      '1h': 3600000
    };

    return intervals[interval] || 60000;
  }

  estimateMemoryUsage() {
    // Rough estimation of memory usage
    const bufferSize = JSON.stringify(this.tickBuffer).length;
    const indicatorsSize = JSON.stringify([...this.indicators.values()]).length;
    return bufferSize + indicatorsSize;
  }

  generateId() {
    return `proc-${Date.now()}-${Math.random().toString(36).substring(2)}`;
  }

  // Clear old data to prevent memory leaks
  cleanup() {
    // Keep only recent data
    const cutoff = Date.now() - 3600000; // 1 hour
    this.tickBuffer = this.tickBuffer.filter(tick => tick.timestamp > cutoff);

    // Clear old indicators data
    this.indicators.forEach(indicator => {
      indicator.prices = indicator.prices.slice(-100);
      indicator.volumes = indicator.volumes.slice(-100);
    });
  }

  // Periodic cleanup
  startCleanup() {
    setInterval(() => {
      this.cleanup();
    }, 300000); // Every 5 minutes
  }
}

// Initialize the processor
const processor = new StreamProcessor();
processor.startCleanup();

// Handle messages from main thread
self.onmessage = async function(event) {
  const { id, type, data } = event.data;

  try {
    let result;

    switch (type) {
      case 'process_tick':
        result = await processor.processTick(data);
        break;
      case 'aggregate_data':
        result = await processor.aggregateData(data);
        break;
      case 'calculate_indicators':
        result = await processor.calculateIndicators(data);
        break;
      case 'detect_patterns':
        result = await processor.detectPatterns(data);
        break;
      case 'compress_buffer':
        result = await processor.compressBuffer(data);
        break;
      case 'persist_data':
        result = await processor.persistData(data);
        break;
      case 'quality_check':
        result = await processor.qualityCheck(data);
        break;
      default:
        result = {
          type: 'error',
          error: `Unknown message type: ${type}`
        };
    }

    // Send result back to main thread
    self.postMessage({
      id,
      type: result.type || type + '_complete',
      result,
      timestamp: Date.now()
    });

  } catch (error) {
    // Send error back to main thread
    self.postMessage({
      id,
      type: 'error',
      error: error.message,
      timestamp: Date.now()
    });
  }
};

// Export for ComLink if available
if (typeof Comlink !== 'undefined') {
  Comlink.expose(processor);
}
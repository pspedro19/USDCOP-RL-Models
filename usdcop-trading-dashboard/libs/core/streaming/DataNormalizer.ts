/**
 * DataNormalizer - Multi-Exchange Data Normalization Pipeline
 * Standardizes tick formats, timezones, and data structures across all data sources
 */

import type {
  StreamSource,
  StreamMessage,
  ExchangeType,
  DataFormat,
  StreamDataType
} from '../types/streaming-types';

import type { MarketTick, OrderBook, Trade, OHLCV } from '../types/market-data';

export interface NormalizationConfig {
  readonly baseTimezone: string;
  readonly baseCurrency: string;
  readonly precision: {
    readonly price: number;
    readonly volume: number;
    readonly percentage: number;
  };
  readonly validation: {
    readonly enablePriceValidation: boolean;
    readonly enableVolumeValidation: boolean;
    readonly maxPriceDeviation: number;
    readonly minVolume: number;
  };
  readonly conversion: {
    readonly enableCurrencyConversion: boolean;
    readonly exchangeRates: Map<string, number>;
  };
}

export interface NormalizationResult<T = any> {
  readonly success: boolean;
  readonly data?: T;
  readonly errors: readonly string[];
  readonly warnings: readonly string[];
  readonly metadata: {
    readonly originalFormat: string;
    readonly processingTime: number;
    readonly dataQuality: number;
  };
}

export class DataNormalizer {
  private readonly config: NormalizationConfig;
  private readonly exchangeNormalizers = new Map<ExchangeType, ExchangeNormalizer>();
  private readonly symbolMappings = new Map<string, string>();
  private readonly priceHistory = new Map<string, number[]>();

  constructor(config: NormalizationConfig) {
    this.config = config;
    this.initializeNormalizers();
    this.loadSymbolMappings();
  }

  // ==========================================
  // MAIN NORMALIZATION METHODS
  // ==========================================

  public normalize(
    source: StreamSource,
    rawMessage: any,
    dataType: StreamDataType
  ): NormalizationResult {
    const startTime = performance.now();
    const errors: string[] = [];
    const warnings: string[] = [];

    try {
      // Get appropriate normalizer
      const normalizer = this.exchangeNormalizers.get(source.type);
      if (!normalizer) {
        throw new Error(`No normalizer found for exchange type: ${source.type}`);
      }

      // Validate input data
      const validationResult = this.validateInput(rawMessage, dataType);
      if (!validationResult.isValid) {
        errors.push(...validationResult.errors);
        warnings.push(...validationResult.warnings);
      }

      // Normalize based on data type
      let normalizedData: any;
      switch (dataType) {
        case 'tick':
          normalizedData = this.normalizeTick(normalizer, source, rawMessage);
          break;
        case 'orderbook':
          normalizedData = this.normalizeOrderBook(normalizer, source, rawMessage);
          break;
        case 'trade':
          normalizedData = this.normalizeTrade(normalizer, source, rawMessage);
          break;
        case 'ohlcv':
          normalizedData = this.normalizeOHLCV(normalizer, source, rawMessage);
          break;
        default:
          throw new Error(`Unsupported data type: ${dataType}`);
      }

      // Post-processing validation and enhancement
      const enhancedData = this.enhanceData(normalizedData, source, dataType);
      const qualityScore = this.calculateDataQuality(enhancedData, errors, warnings);

      const processingTime = performance.now() - startTime;

      return {
        success: errors.length === 0,
        data: enhancedData,
        errors,
        warnings,
        metadata: {
          originalFormat: source.dataFormat.messageFormat,
          processingTime,
          dataQuality: qualityScore
        }
      };

    } catch (error) {
      errors.push(`Normalization failed: ${error.message}`);

      return {
        success: false,
        errors,
        warnings,
        metadata: {
          originalFormat: 'unknown',
          processingTime: performance.now() - startTime,
          dataQuality: 0
        }
      };
    }
  }

  public normalizeSymbol(symbol: string, fromExchange: string, toStandard: boolean = true): string {
    const key = `${fromExchange}:${symbol}`;

    if (toStandard) {
      return this.symbolMappings.get(key) || symbol;
    } else {
      // Reverse lookup
      for (const [mapped, standard] of this.symbolMappings.entries()) {
        if (standard === symbol && mapped.startsWith(`${fromExchange}:`)) {
          return mapped.split(':')[1];
        }
      }
      return symbol;
    }
  }

  // ==========================================
  // DATA TYPE-SPECIFIC NORMALIZATION
  // ==========================================

  private normalizeTick(
    normalizer: ExchangeNormalizer,
    source: StreamSource,
    rawData: any
  ): MarketTick {
    const tick = normalizer.normalizeTick(rawData);

    return {
      id: this.generateId(),
      symbol: this.normalizeSymbol(tick.symbol, source.name),
      timestamp: this.normalizeTimestamp(tick.timestamp, source),
      bid: this.normalizePrice(tick.bid),
      ask: this.normalizePrice(tick.ask),
      last: this.normalizePrice(tick.last),
      volume: this.normalizeVolume(tick.volume),
      change: this.calculateChange(tick.last, tick.symbol),
      changePercent: this.calculateChangePercent(tick.last, tick.symbol),
      high: this.normalizePrice(tick.high),
      low: this.normalizePrice(tick.low),
      open: this.normalizePrice(tick.open),
      vwap: tick.vwap ? this.normalizePrice(tick.vwap) : undefined,
      source: source.name as any,
      quality: this.assessTickQuality(tick)
    };
  }

  private normalizeOrderBook(
    normalizer: ExchangeNormalizer,
    source: StreamSource,
    rawData: any
  ): OrderBook {
    const book = normalizer.normalizeOrderBook(rawData);

    return {
      symbol: this.normalizeSymbol(book.symbol, source.name),
      timestamp: this.normalizeTimestamp(book.timestamp, source),
      bids: book.bids.map(level => ({
        price: this.normalizePrice(level.price),
        size: this.normalizeVolume(level.size),
        count: level.count
      })).sort((a, b) => b.price - a.price), // Sort bids descending
      asks: book.asks.map(level => ({
        price: this.normalizePrice(level.price),
        size: this.normalizeVolume(level.size),
        count: level.count
      })).sort((a, b) => a.price - b.price), // Sort asks ascending
      sequence: book.sequence,
      checksum: book.checksum
    };
  }

  private normalizeTrade(
    normalizer: ExchangeNormalizer,
    source: StreamSource,
    rawData: any
  ): Trade {
    const trade = normalizer.normalizeTrade(rawData);

    return {
      id: trade.id || this.generateId(),
      symbol: this.normalizeSymbol(trade.symbol, source.name),
      timestamp: this.normalizeTimestamp(trade.timestamp, source),
      price: this.normalizePrice(trade.price),
      size: this.normalizeVolume(trade.size),
      side: trade.side,
      conditions: trade.conditions
    };
  }

  private normalizeOHLCV(
    normalizer: ExchangeNormalizer,
    source: StreamSource,
    rawData: any
  ): OHLCV {
    const ohlcv = normalizer.normalizeOHLCV(rawData);

    return {
      timestamp: this.normalizeTimestamp(ohlcv.timestamp, source),
      open: this.normalizePrice(ohlcv.open),
      high: this.normalizePrice(ohlcv.high),
      low: this.normalizePrice(ohlcv.low),
      close: this.normalizePrice(ohlcv.close),
      volume: this.normalizeVolume(ohlcv.volume),
      interval: ohlcv.interval
    };
  }

  // ==========================================
  // VALUE NORMALIZATION METHODS
  // ==========================================

  private normalizePrice(price: number | string): number {
    if (typeof price === 'string') {
      price = parseFloat(price);
    }

    if (isNaN(price) || price <= 0) {
      throw new Error(`Invalid price: ${price}`);
    }

    // Round to configured precision
    return Math.round(price * Math.pow(10, this.config.precision.price))
      / Math.pow(10, this.config.precision.price);
  }

  private normalizeVolume(volume: number | string): number {
    if (typeof volume === 'string') {
      volume = parseFloat(volume);
    }

    if (isNaN(volume) || volume < 0) {
      return 0; // Allow zero volume
    }

    // Round to configured precision
    return Math.round(volume * Math.pow(10, this.config.precision.volume))
      / Math.pow(10, this.config.precision.volume);
  }

  private normalizeTimestamp(timestamp: number | string, source: StreamSource): number {
    let ts: number;

    if (typeof timestamp === 'string') {
      // Handle different timestamp formats
      if (timestamp.includes('T') || timestamp.includes('-')) {
        // ISO format
        ts = new Date(timestamp).getTime();
      } else {
        // Assume Unix timestamp
        ts = parseInt(timestamp);
      }
    } else {
      ts = timestamp;
    }

    // Convert to milliseconds if needed
    if (ts < 1e12) {
      ts *= 1000;
    }

    // Validate timestamp is reasonable (not in future, not too old)
    const now = Date.now();
    const maxAge = 24 * 60 * 60 * 1000; // 24 hours

    if (ts > now + 60000) { // 1 minute in future
      throw new Error(`Timestamp in future: ${ts}`);
    }

    if (ts < now - maxAge) {
      throw new Error(`Timestamp too old: ${ts}`);
    }

    return ts;
  }

  // ==========================================
  // VALIDATION METHODS
  // ==========================================

  private validateInput(rawData: any, dataType: StreamDataType): {
    isValid: boolean;
    errors: string[];
    warnings: string[];
  } {
    const errors: string[] = [];
    const warnings: string[] = [];

    if (!rawData) {
      errors.push('Raw data is null or undefined');
      return { isValid: false, errors, warnings };
    }

    // Type-specific validation
    switch (dataType) {
      case 'tick':
        this.validateTickData(rawData, errors, warnings);
        break;
      case 'orderbook':
        this.validateOrderBookData(rawData, errors, warnings);
        break;
      case 'trade':
        this.validateTradeData(rawData, errors, warnings);
        break;
      case 'ohlcv':
        this.validateOHLCVData(rawData, errors, warnings);
        break;
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  private validateTickData(data: any, errors: string[], warnings: string[]): void {
    if (!data.symbol && !data.s) {
      errors.push('Missing symbol');
    }

    const price = data.price || data.last || data.p;
    if (!price || isNaN(parseFloat(price)) || parseFloat(price) <= 0) {
      errors.push('Invalid or missing price');
    }

    const volume = data.volume || data.v;
    if (volume !== undefined && (isNaN(parseFloat(volume)) || parseFloat(volume) < 0)) {
      warnings.push('Invalid volume, will default to 0');
    }
  }

  private validateOrderBookData(data: any, errors: string[], warnings: string[]): void {
    if (!data.symbol && !data.s) {
      errors.push('Missing symbol');
    }

    if (!Array.isArray(data.bids) && !Array.isArray(data.b)) {
      errors.push('Missing or invalid bids array');
    }

    if (!Array.isArray(data.asks) && !Array.isArray(data.a)) {
      errors.push('Missing or invalid asks array');
    }

    // Validate bid/ask structure
    const bids = data.bids || data.b || [];
    const asks = data.asks || data.a || [];

    if (bids.length === 0) {
      warnings.push('Empty bids array');
    }

    if (asks.length === 0) {
      warnings.push('Empty asks array');
    }
  }

  private validateTradeData(data: any, errors: string[], warnings: string[]): void {
    if (!data.symbol && !data.s) {
      errors.push('Missing symbol');
    }

    const price = data.price || data.p;
    if (!price || isNaN(parseFloat(price)) || parseFloat(price) <= 0) {
      errors.push('Invalid or missing trade price');
    }

    const size = data.size || data.quantity || data.q;
    if (!size || isNaN(parseFloat(size)) || parseFloat(size) <= 0) {
      errors.push('Invalid or missing trade size');
    }
  }

  private validateOHLCVData(data: any, errors: string[], warnings: string[]): void {
    const prices = [data.open, data.high, data.low, data.close];
    const priceNames = ['open', 'high', 'low', 'close'];

    prices.forEach((price, index) => {
      if (!price || isNaN(parseFloat(price)) || parseFloat(price) <= 0) {
        errors.push(`Invalid or missing ${priceNames[index]} price`);
      }
    });

    if (data.high < data.low) {
      errors.push('High price is less than low price');
    }

    if (data.open < data.low || data.open > data.high) {
      warnings.push('Open price outside high-low range');
    }

    if (data.close < data.low || data.close > data.high) {
      warnings.push('Close price outside high-low range');
    }
  }

  // ==========================================
  // ENHANCEMENT METHODS
  // ==========================================

  private enhanceData(data: any, source: StreamSource, dataType: StreamDataType): any {
    const enhanced = { ...data };

    // Add metadata
    enhanced._metadata = {
      source: source.name,
      exchangeType: source.type,
      normalizedAt: Date.now(),
      dataType
    };

    // Type-specific enhancements
    switch (dataType) {
      case 'tick':
        return this.enhanceTick(enhanced);
      case 'orderbook':
        return this.enhanceOrderBook(enhanced);
      case 'trade':
        return this.enhanceTrade(enhanced);
      default:
        return enhanced;
    }
  }

  private enhanceTick(tick: MarketTick): MarketTick {
    // Calculate spread
    if (tick.bid && tick.ask) {
      (tick as any).spread = this.normalizePrice(tick.ask - tick.bid);
      (tick as any).spreadPercent = this.normalizePrice((tick.spread / tick.ask) * 100);
    }

    // Calculate mid price
    if (tick.bid && tick.ask) {
      (tick as any).mid = this.normalizePrice((tick.bid + tick.ask) / 2);
    }

    // Update price history for change calculations
    this.updatePriceHistory(tick.symbol, tick.last);

    return tick;
  }

  private enhanceOrderBook(book: OrderBook): OrderBook {
    const enhanced = { ...book };

    // Calculate spread
    if (book.bids.length > 0 && book.asks.length > 0) {
      const bestBid = book.bids[0].price;
      const bestAsk = book.asks[0].price;
      (enhanced as any).spread = this.normalizePrice(bestAsk - bestBid);
      (enhanced as any).midPrice = this.normalizePrice((bestBid + bestAsk) / 2);
    }

    // Calculate depth
    const bidDepth = book.bids.reduce((sum, level) => sum + level.size, 0);
    const askDepth = book.asks.reduce((sum, level) => sum + level.size, 0);

    (enhanced as any).bidDepth = this.normalizeVolume(bidDepth);
    (enhanced as any).askDepth = this.normalizeVolume(askDepth);
    (enhanced as any).totalDepth = this.normalizeVolume(bidDepth + askDepth);

    return enhanced;
  }

  private enhanceTrade(trade: Trade): Trade {
    const enhanced = { ...trade };

    // Calculate trade value
    (enhanced as any).value = this.normalizePrice(trade.price * trade.size);

    // Add trade direction indicator (for visualization)
    (enhanced as any).direction = trade.side === 'buy' ? 1 : -1;

    return enhanced;
  }

  // ==========================================
  // CALCULATION METHODS
  // ==========================================

  private calculateChange(currentPrice: number, symbol: string): number {
    const history = this.priceHistory.get(symbol);
    if (!history || history.length < 2) {
      return 0;
    }

    const previousPrice = history[history.length - 2];
    return this.normalizePrice(currentPrice - previousPrice);
  }

  private calculateChangePercent(currentPrice: number, symbol: string): number {
    const change = this.calculateChange(currentPrice, symbol);
    if (change === 0) return 0;

    const history = this.priceHistory.get(symbol);
    if (!history || history.length < 2) {
      return 0;
    }

    const previousPrice = history[history.length - 2];
    const changePercent = (change / previousPrice) * 100;

    return Math.round(changePercent * Math.pow(10, this.config.precision.percentage))
      / Math.pow(10, this.config.precision.percentage);
  }

  private updatePriceHistory(symbol: string, price: number): void {
    if (!this.priceHistory.has(symbol)) {
      this.priceHistory.set(symbol, []);
    }

    const history = this.priceHistory.get(symbol)!;
    history.push(price);

    // Keep only last 100 prices
    if (history.length > 100) {
      history.splice(0, history.length - 100);
    }
  }

  private assessTickQuality(tick: any): any {
    let score = 100;

    // Check for missing essential fields
    if (!tick.bid || !tick.ask) score -= 20;
    if (!tick.volume) score -= 10;
    if (!tick.timestamp) score -= 30;

    // Check spread reasonableness
    if (tick.bid && tick.ask) {
      const spread = tick.ask - tick.bid;
      const spreadPercent = (spread / tick.ask) * 100;

      if (spreadPercent > 5) score -= 20; // Very wide spread
      if (spreadPercent < 0) score -= 50; // Invalid spread
    }

    // Check price consistency
    if (tick.last && tick.bid && tick.ask) {
      if (tick.last < tick.bid || tick.last > tick.ask) {
        score -= 15; // Last price outside bid-ask
      }
    }

    return score >= 80 ? 'excellent' :
           score >= 60 ? 'good' :
           score >= 40 ? 'fair' : 'poor';
  }

  private calculateDataQuality(data: any, errors: string[], warnings: string[]): number {
    let score = 100;

    // Deduct for errors and warnings
    score -= errors.length * 25;
    score -= warnings.length * 5;

    // Bonus for complete data
    if (data && typeof data === 'object') {
      const fieldCount = Object.keys(data).length;
      if (fieldCount > 5) score += 5;
    }

    return Math.max(0, Math.min(100, score));
  }

  // ==========================================
  // INITIALIZATION & SETUP
  // ==========================================

  private initializeNormalizers(): void {
    // Forex normalizer
    this.exchangeNormalizers.set('forex', new ForexNormalizer());

    // Crypto normalizers
    this.exchangeNormalizers.set('crypto', new CryptoNormalizer());

    // Generic normalizer for other types
    this.exchangeNormalizers.set('stocks', new GenericNormalizer());
    this.exchangeNormalizers.set('commodities', new GenericNormalizer());
    this.exchangeNormalizers.set('internal', new GenericNormalizer());
    this.exchangeNormalizers.set('simulation', new GenericNormalizer());
  }

  private loadSymbolMappings(): void {
    // Standard forex symbol mappings
    this.symbolMappings.set('binance:USDCOP', 'USD/COP');
    this.symbolMappings.set('coinbase:USD-COP', 'USD/COP');
    this.symbolMappings.set('kraken:USDCOP', 'USD/COP');
    this.symbolMappings.set('twelvedata:USDCOP', 'USD/COP');

    // Crypto mappings
    this.symbolMappings.set('binance:BTCUSDT', 'BTC/USDT');
    this.symbolMappings.set('coinbase:BTC-USD', 'BTC/USD');
    this.symbolMappings.set('kraken:BTCUSD', 'BTC/USD');

    // Add more mappings as needed
  }

  private generateId(): string {
    return `norm-${Date.now()}-${Math.random().toString(36).substring(2)}`;
  }
}

// ==========================================
// EXCHANGE-SPECIFIC NORMALIZERS
// ==========================================

abstract class ExchangeNormalizer {
  abstract normalizeTick(data: any): any;
  abstract normalizeOrderBook(data: any): any;
  abstract normalizeTrade(data: any): any;
  abstract normalizeOHLCV(data: any): any;
}

class ForexNormalizer extends ExchangeNormalizer {
  normalizeTick(data: any): any {
    return {
      symbol: data.symbol || data.instrument || 'UNKNOWN',
      timestamp: data.timestamp || data.time || data.t || Date.now(),
      bid: parseFloat(data.bid || data.b || '0'),
      ask: parseFloat(data.ask || data.a || '0'),
      last: parseFloat(data.price || data.last || data.close || data.c || data.bid || '0'),
      volume: parseFloat(data.volume || data.v || '0'),
      high: parseFloat(data.high || data.h || data.last || '0'),
      low: parseFloat(data.low || data.l || data.last || '0'),
      open: parseFloat(data.open || data.o || data.last || '0')
    };
  }

  normalizeOrderBook(data: any): any {
    return {
      symbol: data.symbol || data.instrument || 'UNKNOWN',
      timestamp: data.timestamp || data.time || Date.now(),
      bids: this.normalizeLevels(data.bids || data.b || []),
      asks: this.normalizeLevels(data.asks || data.a || []),
      sequence: data.sequence || data.seq
    };
  }

  normalizeTrade(data: any): any {
    return {
      id: data.id || data.trade_id || String(Date.now()),
      symbol: data.symbol || data.instrument || 'UNKNOWN',
      timestamp: data.timestamp || data.time || Date.now(),
      price: parseFloat(data.price || data.p || '0'),
      size: parseFloat(data.size || data.quantity || data.q || '0'),
      side: this.normalizeSide(data.side || data.type || 'unknown')
    };
  }

  normalizeOHLCV(data: any): any {
    return {
      timestamp: data.timestamp || data.time || Date.now(),
      open: parseFloat(data.open || data.o || '0'),
      high: parseFloat(data.high || data.h || '0'),
      low: parseFloat(data.low || data.l || '0'),
      close: parseFloat(data.close || data.c || '0'),
      volume: parseFloat(data.volume || data.v || '0'),
      interval: data.interval || '1m'
    };
  }

  private normalizeLevels(levels: any[]): any[] {
    return levels.map(level => ({
      price: parseFloat(Array.isArray(level) ? level[0] : level.price || level.p || '0'),
      size: parseFloat(Array.isArray(level) ? level[1] : level.size || level.s || '0')
    }));
  }

  private normalizeSide(side: string): 'buy' | 'sell' {
    const normalized = side.toLowerCase();
    return normalized.includes('buy') || normalized.includes('bid') ? 'buy' : 'sell';
  }
}

class CryptoNormalizer extends ExchangeNormalizer {
  normalizeTick(data: any): any {
    // Similar to forex but with crypto-specific field mappings
    return {
      symbol: data.s || data.symbol || 'UNKNOWN',
      timestamp: parseInt(data.E || data.timestamp || data.time || String(Date.now())),
      bid: parseFloat(data.b || data.bid || '0'),
      ask: parseFloat(data.a || data.ask || '0'),
      last: parseFloat(data.c || data.price || data.last || '0'),
      volume: parseFloat(data.v || data.volume || '0'),
      high: parseFloat(data.h || data.high || '0'),
      low: parseFloat(data.l || data.low || '0'),
      open: parseFloat(data.o || data.open || '0')
    };
  }

  normalizeOrderBook(data: any): any {
    return {
      symbol: data.s || data.symbol || 'UNKNOWN',
      timestamp: parseInt(data.E || data.timestamp || String(Date.now())),
      bids: (data.b || data.bids || []).map((level: any) => ({
        price: parseFloat(level[0] || level.price || '0'),
        size: parseFloat(level[1] || level.size || '0')
      })),
      asks: (data.a || data.asks || []).map((level: any) => ({
        price: parseFloat(level[0] || level.price || '0'),
        size: parseFloat(level[1] || level.size || '0')
      })),
      sequence: data.lastUpdateId || data.u
    };
  }

  normalizeTrade(data: any): any {
    return {
      id: String(data.t || data.id || Date.now()),
      symbol: data.s || data.symbol || 'UNKNOWN',
      timestamp: parseInt(data.T || data.timestamp || String(Date.now())),
      price: parseFloat(data.p || data.price || '0'),
      size: parseFloat(data.q || data.quantity || data.size || '0'),
      side: data.m ? 'sell' : 'buy' // Binance format
    };
  }

  normalizeOHLCV(data: any): any {
    return {
      timestamp: parseInt(data.t || data.timestamp || String(Date.now())),
      open: parseFloat(data.o || data.open || '0'),
      high: parseFloat(data.h || data.high || '0'),
      low: parseFloat(data.l || data.low || '0'),
      close: parseFloat(data.c || data.close || '0'),
      volume: parseFloat(data.v || data.volume || '0'),
      interval: data.i || data.interval || '1m'
    };
  }
}

class GenericNormalizer extends ExchangeNormalizer {
  normalizeTick(data: any): any {
    // Generic normalization that tries to handle various formats
    return {
      symbol: data.symbol || data.s || data.instrument || 'UNKNOWN',
      timestamp: this.normalizeTimestamp(data.timestamp || data.time || data.t),
      bid: this.normalizeNumber(data.bid || data.b),
      ask: this.normalizeNumber(data.ask || data.a),
      last: this.normalizeNumber(data.price || data.last || data.close || data.c),
      volume: this.normalizeNumber(data.volume || data.v, 0),
      high: this.normalizeNumber(data.high || data.h),
      low: this.normalizeNumber(data.low || data.l),
      open: this.normalizeNumber(data.open || data.o)
    };
  }

  normalizeOrderBook(data: any): any {
    return {
      symbol: data.symbol || data.s || 'UNKNOWN',
      timestamp: this.normalizeTimestamp(data.timestamp || data.time),
      bids: this.normalizeLevels(data.bids || data.b || []),
      asks: this.normalizeLevels(data.asks || data.a || [])
    };
  }

  normalizeTrade(data: any): any {
    return {
      id: String(data.id || data.trade_id || Date.now()),
      symbol: data.symbol || data.s || 'UNKNOWN',
      timestamp: this.normalizeTimestamp(data.timestamp || data.time),
      price: this.normalizeNumber(data.price || data.p),
      size: this.normalizeNumber(data.size || data.quantity || data.q),
      side: data.side === 'buy' || data.side === 'sell' ? data.side : 'buy'
    };
  }

  normalizeOHLCV(data: any): any {
    return {
      timestamp: this.normalizeTimestamp(data.timestamp || data.time),
      open: this.normalizeNumber(data.open || data.o),
      high: this.normalizeNumber(data.high || data.h),
      low: this.normalizeNumber(data.low || data.l),
      close: this.normalizeNumber(data.close || data.c),
      volume: this.normalizeNumber(data.volume || data.v, 0),
      interval: data.interval || '1m'
    };
  }

  private normalizeTimestamp(value: any): number {
    if (typeof value === 'number') return value;
    if (typeof value === 'string') {
      const parsed = parseInt(value);
      return isNaN(parsed) ? Date.now() : parsed;
    }
    return Date.now();
  }

  private normalizeNumber(value: any, defaultValue: number = 0): number {
    if (typeof value === 'number') return value;
    if (typeof value === 'string') {
      const parsed = parseFloat(value);
      return isNaN(parsed) ? defaultValue : parsed;
    }
    return defaultValue;
  }

  private normalizeLevels(levels: any[]): any[] {
    return levels.map(level => ({
      price: this.normalizeNumber(Array.isArray(level) ? level[0] : level.price),
      size: this.normalizeNumber(Array.isArray(level) ? level[1] : level.size)
    }));
  }
}
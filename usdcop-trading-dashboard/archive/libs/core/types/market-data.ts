/**
 * Core Market Data Types for Elite Trading Platform
 * High-performance, type-safe data structures for market operations
 */

export interface MarketTick {
  readonly id: string;
  readonly symbol: string;
  readonly timestamp: number;
  readonly bid: number;
  readonly ask: number;
  readonly last: number;
  readonly volume: number;
  readonly change: number;
  readonly changePercent: number;
  readonly high: number;
  readonly low: number;
  readonly open: number;
  readonly vwap?: number;
  readonly source: DataSource;
  readonly quality: DataQuality;
}

export interface OHLCV {
  readonly timestamp: number;
  readonly open: number;
  readonly high: number;
  readonly low: number;
  readonly close: number;
  readonly volume: number;
  readonly interval: TimeInterval;
}

export interface OrderBookLevel {
  readonly price: number;
  readonly size: number;
  readonly count?: number;
}

export interface OrderBook {
  readonly symbol: string;
  readonly timestamp: number;
  readonly bids: readonly OrderBookLevel[];
  readonly asks: readonly OrderBookLevel[];
  readonly sequence?: number;
  readonly checksum?: string;
}

export interface Trade {
  readonly id: string;
  readonly symbol: string;
  readonly timestamp: number;
  readonly price: number;
  readonly size: number;
  readonly side: TradeSide;
  readonly conditions?: string[];
}

export interface MarketDepth {
  readonly symbol: string;
  readonly timestamp: number;
  readonly levels: number;
  readonly bids: readonly OrderBookLevel[];
  readonly asks: readonly OrderBookLevel[];
  readonly spread: number;
  readonly midPrice: number;
}

// Enums and Union Types
export type DataSource = 'twelvedata' | 'binance' | 'coinbase' | 'kraken' | 'internal' | 'simulation';
export type DataQuality = 'realtime' | 'delayed' | 'historical' | 'simulated' | 'interpolated';
export type TimeInterval = '1s' | '5s' | '15s' | '30s' | '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w' | '1M';
export type TradeSide = 'buy' | 'sell';
export type MarketStatus = 'open' | 'closed' | 'pre_open' | 'post_close' | 'halted';

// Market Session Info
export interface MarketSession {
  readonly symbol: string;
  readonly status: MarketStatus;
  readonly openTime?: number;
  readonly closeTime?: number;
  readonly timezone: string;
  readonly holidays: readonly string[];
}

// Aggregated Market Statistics
export interface MarketStats {
  readonly symbol: string;
  readonly timestamp: number;
  readonly volume24h: number;
  readonly volumeAvg30d: number;
  readonly priceChange24h: number;
  readonly priceChangePercent24h: number;
  readonly high24h: number;
  readonly low24h: number;
  readonly marketCap?: number;
  readonly circulatingSupply?: number;
}

// Real-time data stream metadata
export interface StreamMetadata {
  readonly streamId: string;
  readonly symbol: string;
  readonly dataType: string;
  readonly lastUpdate: number;
  readonly connectionStatus: 'connected' | 'disconnected' | 'reconnecting' | 'error';
  readonly latency: number;
  readonly messagesPerSecond: number;
  readonly errorCount: number;
}
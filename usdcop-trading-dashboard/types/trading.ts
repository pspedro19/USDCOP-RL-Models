/**
 * Trading Data Types
 * ===================
 *
 * Tipos para datos de trading, OHLCV, señales, posiciones, etc.
 */

import { Nullable } from './common';

// === ENUMS ===

/**
 * Tipo de orden
 */
export enum OrderType {
  BUY = 'BUY',
  SELL = 'SELL',
  HOLD = 'HOLD',
}

/**
 * Lado de la operación
 */
export enum OrderSide {
  LONG = 'long',
  SHORT = 'short',
  FLAT = 'flat',
  CLOSE = 'close',
}

/**
 * Estado de la orden
 */
export enum OrderStatus {
  PENDING = 'pending',
  OPEN = 'open',
  CLOSED = 'closed',
  CANCELLED = 'cancelled',
  FILLED = 'filled',
  PARTIAL = 'partial',
  REJECTED = 'rejected',
}

/**
 * Estado del mercado
 */
export enum MarketStatus {
  OPEN = 'open',
  CLOSED = 'closed',
  PRE_MARKET = 'pre_market',
  POST_MARKET = 'post_market',
  HALTED = 'halted',
}

/**
 * Tipo de timeframe
 */
export enum Timeframe {
  ONE_MIN = '1m',
  FIVE_MIN = '5m',
  FIFTEEN_MIN = '15m',
  THIRTY_MIN = '30m',
  ONE_HOUR = '1h',
  FOUR_HOUR = '4h',
  ONE_DAY = '1d',
  ONE_WEEK = '1w',
  ONE_MONTH = '1M',
}

// === MARKET DATA ===

/**
 * Datos OHLCV base
 */
export interface OHLCVData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/**
 * Datos de candlestick con timestamp
 */
export interface CandlestickData extends OHLCVData {
  timestamp?: number;
  datetime?: string;
}

/**
 * Datos de candlestick extendidos con indicadores
 */
export interface CandlestickExtended extends CandlestickData {
  indicators?: TechnicalIndicators;
}

/**
 * Respuesta de candlesticks de la API
 */
export interface CandlestickResponse {
  symbol: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  count: number;
  data: CandlestickExtended[];
}

/**
 * Punto de datos del mercado en tiempo real
 */
export interface MarketDataPoint {
  symbol: string;
  price: number;
  timestamp: number;
  volume: number;
  bid?: number;
  ask?: number;
  spread?: number;
  source?: string;
}

/**
 * Actualización del mercado (WebSocket)
 */
export interface MarketUpdate {
  timestamp: string;
  symbol: string;
  price: number;
  volume: number;
  bid: number;
  ask: number;
  spread: number;
  change_24h: number;
  change_percent_24h: number;
}

/**
 * Tick de precio
 */
export interface PriceTick {
  timestamp: number;
  price: number;
  volume?: number;
  side?: 'buy' | 'sell';
}

// === TECHNICAL INDICATORS ===

/**
 * Indicadores técnicos
 */
export interface TechnicalIndicators {
  // Moving Averages
  ema_20?: number;
  ema_50?: number;
  ema_200?: number;
  sma_20?: number;
  sma_50?: number;
  sma_200?: number;

  // Bollinger Bands
  bb_upper?: number;
  bb_middle?: number;
  bb_lower?: number;
  bb_width?: number;

  // Oscillators
  rsi?: number;
  rsi_signal?: 'overbought' | 'oversold' | 'neutral';
  macd?: number;
  macd_signal?: number;
  macd_histogram?: number;
  stochastic?: number;
  stochastic_signal?: number;

  // Momentum
  momentum?: number;
  roc?: number; // Rate of Change
  cci?: number; // Commodity Channel Index

  // Volatility
  atr?: number; // Average True Range
  volatility?: number;

  // Volume
  obv?: number; // On Balance Volume
  vwap?: number; // Volume Weighted Average Price

  // Custom indicators
  [key: string]: number | string | undefined;
}

// === TRADING SIGNALS ===

/**
 * Señal de trading
 */
export interface TradingSignal {
  id: string;
  timestamp: string;
  type: OrderType;
  confidence: number;
  price: number;
  stopLoss?: number;
  takeProfit?: number;
  reasoning: string[];
  riskScore: number;
  expectedReturn: number;
  timeHorizon: string;
  modelSource: string;
  latency: number;
}

/**
 * Alerta de señal (WebSocket)
 */
export interface SignalAlert {
  signal_id: number;
  timestamp: string;
  strategy_code: string;
  strategy_name: string;
  signal: OrderSide;
  confidence: number;
  entry_price: number;
  reasoning: string;
}

/**
 * Performance de señales
 */
export interface SignalPerformance {
  winRate: number;
  avgWin: number;
  avgLoss: number;
  profitFactor: number;
  sharpeRatio: number;
  totalSignals: number;
  activeSignals: number;
}

// === POSITIONS & TRADES ===

/**
 * Posición de trading
 */
export interface Position {
  id: string;
  symbol: string;
  side: OrderSide;
  entryPrice: number;
  currentPrice: number;
  quantity: number;
  notionalValue: number;
  unrealizedPnl: number;
  realizedPnl: number;
  openedAt: string;
  closedAt?: string;
  status: OrderStatus;
  stopLoss?: number;
  takeProfit?: number;
  strategy?: string;
}

/**
 * Trade ejecutado
 */
export interface Trade {
  trade_id: number;
  timestamp: string;
  strategy_code: string;
  side: 'buy' | 'sell';
  price: number;
  size: number;
  pnl: number;
  status: OrderStatus;
  commission?: number;
  slippage?: number;
}

/**
 * Actualización de trade (WebSocket)
 */
export interface TradeUpdate extends Trade {}

/**
 * Orden de trading
 */
export interface Order {
  id: string;
  symbol: string;
  type: OrderType;
  side: OrderSide;
  price: number;
  quantity: number;
  status: OrderStatus;
  filledQuantity: number;
  remainingQuantity: number;
  averageFillPrice: number;
  createdAt: string;
  updatedAt: string;
  expiresAt?: string;
  timeInForce?: 'GTC' | 'IOC' | 'FOK' | 'DAY';
  stopPrice?: number;
  limitPrice?: number;
}

// === ORDER BOOK ===

/**
 * Nivel del order book (precio, cantidad)
 */
export type OrderBookLevel = [number, number];

/**
 * Order book completo
 */
export interface OrderBook {
  timestamp: string;
  symbol: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  lastPrice: number;
  spread: number;
  spreadPercent: number;
}

/**
 * Actualización del order book (WebSocket)
 */
export interface OrderBookUpdate extends OrderBook {}

// === STATISTICS ===

/**
 * Estadísticas del símbolo
 */
export interface SymbolStats {
  symbol: string;
  price: number;
  open_24h: number;
  high_24h: number;
  low_24h: number;
  volume_24h: number;
  change_24h: number;
  change_percent_24h: number;
  spread: number;
  timestamp: string;
  source: string;
}

/**
 * Estadísticas de mercado
 */
export interface MarketStats {
  totalVolume: number;
  totalTrades: number;
  avgPrice: number;
  highPrice: number;
  lowPrice: number;
  openPrice: number;
  closePrice: number;
  priceChange: number;
  priceChangePercent: number;
  volatility: number;
  timestamp: string;
}

// === VOLUME PROFILE ===

/**
 * Nivel de perfil de volumen
 */
export interface VolumeProfileLevel {
  price: number;
  volume: number;
  buyVolume: number;
  sellVolume: number;
  delta: number;
}

/**
 * Perfil de volumen completo
 */
export interface VolumeProfile {
  symbol: string;
  timeframe: Timeframe;
  startTime: number;
  endTime: number;
  levels: VolumeProfileLevel[];
  poc: number; // Point of Control
  vah: number; // Value Area High
  val: number; // Value Area Low
}

// === BACKTEST ===

/**
 * Resultado de backtest
 */
export interface BacktestResult {
  strategy: string;
  startDate: string;
  endDate: string;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  profitFactor: number;
  sharpeRatio: number;
  maxDrawdown: number;
  totalReturn: number;
  totalReturnPercent: number;
  avgWin: number;
  avgLoss: number;
  largestWin: number;
  largestLoss: number;
  avgTradeDuration: number;
  equityCurve: number[];
  trades: Trade[];
}

/**
 * Métricas de performance
 */
export interface PerformanceMetrics {
  totalReturn: number;
  annualizedReturn: number;
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  maxDrawdown: number;
  maxDrawdownPercent: number;
  volatility: number;
  winRate: number;
  profitFactor: number;
  expectancy: number;
  avgWin: number;
  avgLoss: number;
  avgTrade: number;
  totalTrades: number;
}

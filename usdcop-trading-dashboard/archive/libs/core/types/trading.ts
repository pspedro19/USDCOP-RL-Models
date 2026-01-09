/**
 * Trading System Types for Elite Trading Platform
 * Professional-grade trading data structures
 */

export interface Order {
  readonly id: string;
  readonly clientOrderId?: string;
  readonly symbol: string;
  readonly side: OrderSide;
  readonly type: OrderType;
  readonly timeInForce: TimeInForce;
  readonly quantity: number;
  readonly price?: number;
  readonly stopPrice?: number;
  readonly limitPrice?: number;
  readonly status: OrderStatus;
  readonly timestamp: number;
  readonly lastUpdateTime: number;
  readonly executedQuantity: number;
  readonly averagePrice?: number;
  readonly commission?: number;
  readonly commissionAsset?: string;
  readonly fills: readonly Fill[];
  readonly workingTime?: number;
  readonly selfTradePreventionMode?: SelfTradePreventionMode;
}

export interface Fill {
  readonly id: string;
  readonly orderId: string;
  readonly price: number;
  readonly quantity: number;
  readonly commission: number;
  readonly commissionAsset: string;
  readonly timestamp: number;
  readonly tradeId?: string;
  readonly isMaker: boolean;
}

export interface Position {
  readonly symbol: string;
  readonly side: PositionSide;
  readonly size: number;
  readonly averagePrice: number;
  readonly unrealizedPnl: number;
  readonly realizedPnl: number;
  readonly marginUsed: number;
  readonly percentage: number;
  readonly timestamp: number;
  readonly leverage?: number;
  readonly maintenanceMargin?: number;
  readonly initialMargin?: number;
}

export interface Portfolio {
  readonly accountId: string;
  readonly timestamp: number;
  readonly totalValue: number;
  readonly availableBalance: number;
  readonly usedMargin: number;
  readonly freeMargin: number;
  readonly equity: number;
  readonly unrealizedPnl: number;
  readonly realizedPnl: number;
  readonly positions: readonly Position[];
  readonly orders: readonly Order[];
  readonly marginLevel: number;
  readonly currency: string;
}

export interface RiskLimits {
  readonly maxPositionSize: number;
  readonly maxDailyLoss: number;
  readonly maxDrawdown: number;
  readonly maxLeverage: number;
  readonly allowedSymbols: readonly string[];
  readonly maxOrderSize: number;
  readonly maxOpenOrders: number;
  readonly marginCallLevel: number;
  readonly stopOutLevel: number;
}

export interface TradingSignal {
  readonly id: string;
  readonly symbol: string;
  readonly timestamp: number;
  readonly type: SignalType;
  readonly side: OrderSide;
  readonly strength: number; // 0-1
  readonly confidence: number; // 0-1
  readonly price: number;
  readonly stopLoss?: number;
  readonly takeProfit?: number;
  readonly timeframe: string;
  readonly source: string;
  readonly metadata?: Record<string, any>;
}

export interface StrategyPerformance {
  readonly strategyId: string;
  readonly name: string;
  readonly timeframe: string;
  readonly totalTrades: number;
  readonly winRate: number;
  readonly profitFactor: number;
  readonly sharpeRatio: number;
  readonly maxDrawdown: number;
  readonly totalReturn: number;
  readonly averageWin: number;
  readonly averageLoss: number;
  readonly largestWin: number;
  readonly largestLoss: number;
  readonly consecutiveWins: number;
  readonly consecutiveLosses: number;
  readonly timestamp: number;
}

// Enums and Union Types
export type OrderSide = 'buy' | 'sell';
export type OrderType = 'market' | 'limit' | 'stop_market' | 'stop_limit' | 'take_profit' | 'take_profit_limit' | 'trailing_stop';
export type TimeInForce = 'GTC' | 'IOC' | 'FOK' | 'GTD' | 'GTT';
export type OrderStatus = 'new' | 'partially_filled' | 'filled' | 'canceled' | 'rejected' | 'expired' | 'pending_cancel';
export type PositionSide = 'long' | 'short' | 'both';
export type SelfTradePreventionMode = 'EXPIRE_TAKER' | 'EXPIRE_MAKER' | 'EXPIRE_BOTH' | 'NONE';
export type SignalType = 'entry' | 'exit' | 'reversal' | 'scalp' | 'swing' | 'position';

// Execution Report
export interface ExecutionReport {
  readonly orderId: string;
  readonly symbol: string;
  readonly side: OrderSide;
  readonly type: OrderType;
  readonly status: OrderStatus;
  readonly executionType: ExecutionType;
  readonly timestamp: number;
  readonly price?: number;
  readonly quantity?: number;
  readonly cumulativeQuantity: number;
  readonly leavesQuantity: number;
  readonly lastPrice?: number;
  readonly lastQuantity?: number;
  readonly commission?: number;
  readonly commissionAsset?: string;
  readonly tradeId?: string;
  readonly rejectReason?: string;
}

export type ExecutionType = 'NEW' | 'CANCELED' | 'REPLACED' | 'REJECTED' | 'TRADE' | 'EXPIRED';

// Account Information
export interface Account {
  readonly accountId: string;
  readonly accountType: AccountType;
  readonly status: AccountStatus;
  readonly permissions: readonly TradingPermission[];
  readonly balances: readonly Balance[];
  readonly commissionRates: CommissionRates;
  readonly timestamp: number;
}

export interface Balance {
  readonly asset: string;
  readonly free: number;
  readonly locked: number;
  readonly total: number;
}

export interface CommissionRates {
  readonly maker: number;
  readonly taker: number;
  readonly buyer: number;
  readonly seller: number;
}

export type AccountType = 'spot' | 'margin' | 'futures' | 'options';
export type AccountStatus = 'normal' | 'margin_call' | 'pre_liquidation' | 'liquidation' | 'bankruptcy' | 'adl_queue';
export type TradingPermission = 'spot' | 'margin' | 'futures' | 'leveraged_tokens' | 'tt_rebalance';

// Risk Management
export interface RiskMetrics {
  readonly timestamp: number;
  readonly var95: number; // Value at Risk 95%
  readonly var99: number; // Value at Risk 99%
  readonly expectedShortfall: number;
  readonly beta: number;
  readonly alpha: number;
  readonly sharpeRatio: number;
  readonly sortinoRatio: number;
  readonly calmarRatio: number;
  readonly maxDrawdown: number;
  readonly volatility: number;
  readonly correlation: number;
  readonly exposureByAsset: Record<string, number>;
  readonly exposureBySector: Record<string, number>;
}
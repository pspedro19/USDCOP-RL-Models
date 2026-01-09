/**
 * Event System Types for Elite Trading Platform
 * Type-safe pub/sub event definitions
 */

import { MarketTick, OrderBook, Trade } from './market-data';
import { Order, Position, ExecutionReport, TradingSignal } from './trading';

// Base Event Interface
export interface BaseEvent<T = any> {
  readonly id: string;
  readonly type: string;
  readonly timestamp: number;
  readonly source: string;
  readonly data: T;
  readonly priority: EventPriority;
  readonly metadata?: Record<string, any>;
}

// Market Data Events
export interface MarketTickEvent extends BaseEvent<MarketTick> {
  readonly type: 'market.tick';
}

export interface OrderBookEvent extends BaseEvent<OrderBook> {
  readonly type: 'market.orderbook';
}

export interface TradeEvent extends BaseEvent<Trade> {
  readonly type: 'market.trade';
}

export interface MarketStatusEvent extends BaseEvent<{ symbol: string; status: string }> {
  readonly type: 'market.status';
}

// Trading Events
export interface OrderEvent extends BaseEvent<Order> {
  readonly type: 'trading.order.created' | 'trading.order.updated' | 'trading.order.filled' | 'trading.order.cancelled';
}

export interface PositionEvent extends BaseEvent<Position> {
  readonly type: 'trading.position.opened' | 'trading.position.updated' | 'trading.position.closed';
}

export interface ExecutionEvent extends BaseEvent<ExecutionReport> {
  readonly type: 'trading.execution';
}

export interface SignalEvent extends BaseEvent<TradingSignal> {
  readonly type: 'trading.signal';
}

// System Events
export interface ConnectionEvent extends BaseEvent<{ status: ConnectionStatus; endpoint: string }> {
  readonly type: 'system.connection';
}

export interface ErrorEvent extends BaseEvent<{ error: Error; context: string }> {
  readonly type: 'system.error';
}

export interface PerformanceEvent extends BaseEvent<PerformanceMetrics> {
  readonly type: 'system.performance';
}

export interface ConfigEvent extends BaseEvent<{ key: string; value: any; previous?: any }> {
  readonly type: 'system.config.changed';
}

// User Interface Events
export interface UIStateEvent extends BaseEvent<{ component: string; state: any }> {
  readonly type: 'ui.state.changed';
}

export interface UserActionEvent extends BaseEvent<{ action: string; params: any }> {
  readonly type: 'ui.user.action';
}

export interface NavigationEvent extends BaseEvent<{ from: string; to: string }> {
  readonly type: 'ui.navigation';
}

// Data Events
export interface DataRequestEvent extends BaseEvent<DataRequest> {
  readonly type: 'data.request';
}

export interface DataResponseEvent extends BaseEvent<DataResponse> {
  readonly type: 'data.response';
}

export interface CacheEvent extends BaseEvent<{ key: string; action: 'hit' | 'miss' | 'set' | 'delete' }> {
  readonly type: 'data.cache';
}

// Union Types for Event Filtering
export type MarketEvent = MarketTickEvent | OrderBookEvent | TradeEvent | MarketStatusEvent;
export type TradingEvent = OrderEvent | PositionEvent | ExecutionEvent | SignalEvent;
export type SystemEvent = ConnectionEvent | ErrorEvent | PerformanceEvent | ConfigEvent;
export type UIEvent = UIStateEvent | UserActionEvent | NavigationEvent;
export type DataEvent = DataRequestEvent | DataResponseEvent | CacheEvent;

export type TradingPlatformEvent = MarketEvent | TradingEvent | SystemEvent | UIEvent | DataEvent;

// Event Priority
export type EventPriority = 'critical' | 'high' | 'normal' | 'low';

// Connection Status
export type ConnectionStatus = 'connected' | 'disconnected' | 'reconnecting' | 'error';

// Performance Metrics
export interface PerformanceMetrics {
  readonly timestamp: number;
  readonly memoryUsage: {
    readonly used: number;
    readonly total: number;
    readonly percentage: number;
  };
  readonly cpuUsage: number;
  readonly networkLatency: number;
  readonly renderTime: number;
  readonly fps: number;
  readonly eventsPerSecond: number;
  readonly websocketLatency: number;
  readonly dataProcessingTime: number;
}

// Data Request/Response
export interface DataRequest {
  readonly id: string;
  readonly type: string;
  readonly params: Record<string, any>;
  readonly timeout?: number;
  readonly priority: EventPriority;
}

export interface DataResponse {
  readonly requestId: string;
  readonly success: boolean;
  readonly data?: any;
  readonly error?: string;
  readonly duration: number;
  readonly cached: boolean;
}

// Event Filter
export interface EventFilter {
  readonly types?: string[];
  readonly sources?: string[];
  readonly priority?: EventPriority[];
  readonly symbols?: string[];
  readonly timeRange?: {
    readonly from: number;
    readonly to: number;
  };
}

// Event Subscription
export interface EventSubscription {
  readonly id: string;
  readonly filter: EventFilter;
  readonly callback: (event: TradingPlatformEvent) => void;
  readonly active: boolean;
  readonly createdAt: number;
  readonly lastTriggered?: number;
  readonly eventCount: number;
}

// Event Statistics
export interface EventStats {
  readonly totalEvents: number;
  readonly eventsByType: Record<string, number>;
  readonly eventsBySource: Record<string, number>;
  readonly eventsByPriority: Record<EventPriority, number>;
  readonly averageLatency: number;
  readonly peakEventsPerSecond: number;
  readonly errorRate: number;
  readonly lastReset: number;
}
/**
 * WebSocket Types
 * ================
 *
 * Tipos para mensajes WebSocket, conexiones en tiempo real, etc.
 */

import { MarketUpdate, OrderBookUpdate, TradeUpdate, SignalAlert } from './trading';

// === CONNECTION ===

/**
 * Estado de conexión WebSocket
 */
export enum WebSocketConnectionState {
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  DISCONNECTING = 'disconnecting',
  DISCONNECTED = 'disconnected',
  RECONNECTING = 'reconnecting',
  ERROR = 'error',
}

/**
 * Estado de conexión
 */
export interface ConnectionState {
  state: WebSocketConnectionState;
  connectedAt?: number;
  disconnectedAt?: number;
  reconnectAttempts?: number;
  lastError?: string;
}

/**
 * Configuración de WebSocket
 */
export interface WebSocketConfig {
  url: string;
  protocols?: string | string[];
  reconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  timeout?: number;
}

// === MESSAGES ===

/**
 * Tipo de mensaje WebSocket
 */
export enum WebSocketMessageType {
  // Connection
  CONNECT = 'connect',
  DISCONNECT = 'disconnect',
  PING = 'ping',
  PONG = 'pong',

  // Subscription
  SUBSCRIBE = 'subscribe',
  UNSUBSCRIBE = 'unsubscribe',
  SUBSCRIBED = 'subscribed',
  UNSUBSCRIBED = 'unsubscribed',

  // Data
  MARKET_DATA = 'market_data',
  PRICE_UPDATE = 'price_update',
  ORDER_BOOK = 'order_book',
  TRADE = 'trade',
  SIGNAL = 'signal',
  ALERT = 'alert',
  STATUS = 'status',

  // Errors
  ERROR = 'error',
  WARNING = 'warning',
}

/**
 * Mensaje base de WebSocket
 */
export interface WebSocketMessage<T = any> {
  type: WebSocketMessageType;
  channel?: string;
  payload?: T;
  timestamp?: string;
  id?: string;
}

/**
 * Mensaje de conexión
 */
export interface ConnectMessage extends WebSocketMessage {
  type: WebSocketMessageType.CONNECT;
  payload: {
    client_id?: string;
    version?: string;
  };
}

/**
 * Mensaje de suscripción
 */
export interface SubscribeMessage extends WebSocketMessage {
  type: WebSocketMessageType.SUBSCRIBE;
  channel: string;
  payload?: {
    symbol?: string;
    filters?: Record<string, any>;
  };
}

/**
 * Mensaje de desuscripción
 */
export interface UnsubscribeMessage extends WebSocketMessage {
  type: WebSocketMessageType.UNSUBSCRIBE;
  channel: string;
}

/**
 * Mensaje de heartbeat (ping)
 */
export interface PingMessage extends WebSocketMessage {
  type: WebSocketMessageType.PING;
  timestamp: string;
}

/**
 * Mensaje de heartbeat (pong)
 */
export interface PongMessage extends WebSocketMessage {
  type: WebSocketMessageType.PONG;
  timestamp: string;
}

/**
 * Mensaje de datos de mercado
 */
export interface MarketDataMessage extends WebSocketMessage<MarketUpdate> {
  type: WebSocketMessageType.MARKET_DATA | WebSocketMessageType.PRICE_UPDATE;
  channel: 'market_data';
}

/**
 * Mensaje de order book
 */
export interface OrderBookMessage extends WebSocketMessage<OrderBookUpdate> {
  type: WebSocketMessageType.ORDER_BOOK;
  channel: 'order_book';
}

/**
 * Mensaje de trade
 */
export interface TradeMessage extends WebSocketMessage<TradeUpdate> {
  type: WebSocketMessageType.TRADE;
  channel: 'trades';
}

/**
 * Mensaje de señal
 */
export interface SignalMessage extends WebSocketMessage<SignalAlert> {
  type: WebSocketMessageType.SIGNAL;
  channel: 'signals';
}

/**
 * Mensaje de alerta
 */
export interface AlertMessage extends WebSocketMessage {
  type: WebSocketMessageType.ALERT;
  payload: {
    level: 'info' | 'warning' | 'error' | 'critical';
    title: string;
    message: string;
    source: string;
  };
}

/**
 * Mensaje de estado
 */
export interface StatusMessage extends WebSocketMessage {
  type: WebSocketMessageType.STATUS;
  payload: {
    status: 'online' | 'offline' | 'maintenance';
    message?: string;
    services?: Record<string, 'up' | 'down' | 'degraded'>;
  };
}

/**
 * Mensaje de error
 */
export interface ErrorMessage extends WebSocketMessage {
  type: WebSocketMessageType.ERROR;
  payload: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
}

/**
 * Unión de todos los tipos de mensajes
 */
export type AnyWebSocketMessage =
  | ConnectMessage
  | SubscribeMessage
  | UnsubscribeMessage
  | PingMessage
  | PongMessage
  | MarketDataMessage
  | OrderBookMessage
  | TradeMessage
  | SignalMessage
  | AlertMessage
  | StatusMessage
  | ErrorMessage;

// === CHANNELS ===

/**
 * Canales disponibles
 */
export enum WebSocketChannel {
  MARKET_DATA = 'market_data',
  ORDER_BOOK = 'order_book',
  TRADES = 'trades',
  SIGNALS = 'signals',
  ALERTS = 'alerts',
  STATUS = 'status',
}

/**
 * Suscripción a canal
 */
export interface ChannelSubscription {
  channel: WebSocketChannel;
  symbol?: string;
  filters?: Record<string, any>;
  active: boolean;
  subscribedAt: number;
}

// === HANDLERS ===

/**
 * Handler de mensaje
 */
export type MessageHandler<T = any> = (data: T) => void;

/**
 * Handler de error
 */
export type ErrorHandler = (error: Error | string) => void;

/**
 * Handler de conexión
 */
export type ConnectionHandler = (state: ConnectionState) => void;

/**
 * Handlers de WebSocket
 */
export interface WebSocketHandlers {
  onConnect?: ConnectionHandler;
  onDisconnect?: ConnectionHandler;
  onError?: ErrorHandler;
  onMessage?: MessageHandler<AnyWebSocketMessage>;
  onMarketData?: MessageHandler<MarketUpdate>;
  onOrderBook?: MessageHandler<OrderBookUpdate>;
  onTrade?: MessageHandler<TradeUpdate>;
  onSignal?: MessageHandler<SignalAlert>;
  onAlert?: MessageHandler<AlertMessage['payload']>;
  onStatus?: MessageHandler<StatusMessage['payload']>;
}

// === MANAGER ===

/**
 * Interface del WebSocket Manager
 */
export interface IWebSocketManager {
  // Connection
  connect(): void;
  disconnect(): void;
  reconnect(): void;
  isConnected(): boolean;
  getConnectionState(): ConnectionState;

  // Subscriptions
  subscribe(channel: WebSocketChannel, options?: { symbol?: string; filters?: Record<string, any> }): void;
  unsubscribe(channel: WebSocketChannel): void;
  getSubscriptions(): ChannelSubscription[];

  // Event handlers
  on(channel: WebSocketChannel | 'connect' | 'disconnect' | 'error', handler: MessageHandler): void;
  off(channel: WebSocketChannel | 'connect' | 'disconnect' | 'error', handler: MessageHandler): void;

  // Messaging
  send(message: AnyWebSocketMessage): void;

  // Heartbeat
  startHeartbeat(): void;
  stopHeartbeat(): void;
}

// === STREAMING DATA ===

/**
 * Stream de datos
 */
export interface DataStream<T> {
  id: string;
  channel: WebSocketChannel;
  active: boolean;
  buffer: T[];
  maxBufferSize: number;
  lastUpdate: number;
  subscribe: (handler: MessageHandler<T>) => () => void;
  unsubscribe: (handler: MessageHandler<T>) => void;
  clear: () => void;
  pause: () => void;
  resume: () => void;
}

/**
 * Configuración de stream
 */
export interface StreamConfig {
  channel: WebSocketChannel;
  bufferSize?: number;
  throttleMs?: number;
  batchUpdates?: boolean;
  autoReconnect?: boolean;
}

// === STATISTICS ===

/**
 * Estadísticas de WebSocket
 */
export interface WebSocketStats {
  messagesReceived: number;
  messagesSent: number;
  bytesReceived: number;
  bytesSent: number;
  errors: number;
  reconnects: number;
  averageLatency: number;
  uptime: number;
  lastMessageAt?: number;
}

/**
 * Métricas de rendimiento
 */
export interface PerformanceMetrics {
  latency: {
    current: number;
    average: number;
    min: number;
    max: number;
  };
  throughput: {
    messagesPerSecond: number;
    bytesPerSecond: number;
  };
  connectionQuality: {
    score: number; // 0-100
    status: 'excellent' | 'good' | 'fair' | 'poor';
    droppedMessages: number;
  };
}

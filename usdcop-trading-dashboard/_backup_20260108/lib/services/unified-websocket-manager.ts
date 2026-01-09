/**
 * Unified WebSocket Manager
 * =========================
 *
 * Consolidated WebSocket implementation with:
 * - Token-based authentication
 * - Message validation with schema
 * - Exponential backoff with jitter for reconnection
 * - Channel and symbol subscriptions
 * - Market hours awareness (USD/COP: 8:00-12:55 COT)
 * - Multiple fallback strategies (WebSocket → Socket.IO → HTTP polling)
 * - Connection quality monitoring
 *
 * Replaces:
 * - lib/services/websocket-manager.ts
 * - lib/services/realtime-websocket-manager.ts
 * - lib/services/market-data/WebSocketConnector.ts
 */

import { io, Socket } from 'socket.io-client';

// ============================================================================
// Types and Interfaces
// ============================================================================

export interface MarketDataPoint {
  timestamp: string;
  symbol: string;
  price: number;
  bid?: number;
  ask?: number;
  volume?: number;
  spread?: number;
  change24h?: number;
  changePercent24h?: number;
  source: string;
  type: 'tick' | 'candlestick' | 'quote' | 'trade';
}

export interface OrderBookUpdate {
  timestamp: string;
  symbol: string;
  bids: Array<[number, number]>;
  asks: Array<[number, number]>;
  lastPrice: number;
  spread: number;
  spreadPercent: number;
}

export interface TradeUpdate {
  tradeId: string;
  timestamp: string;
  strategyCode: string;
  side: 'buy' | 'sell';
  price: number;
  size: number;
  pnl: number;
  status: 'open' | 'closed' | 'pending';
}

export interface SignalAlert {
  signalId: string;
  timestamp: string;
  strategyCode: string;
  strategyName: string;
  signal: 'long' | 'short' | 'flat' | 'close';
  confidence: number;
  entryPrice: number;
  reasoning: string;
}

export interface WebSocketConfig {
  url: string;
  authToken?: string;
  maxReconnectAttempts: number;
  baseReconnectDelay: number;
  maxReconnectDelay: number;
  heartbeatInterval: number;
  heartbeatTimeout: number;
  connectionTimeout: number;
  enableMarketHoursAwareness: boolean;
  enableFallbacks: boolean;
  /**
   * If true, requires server to respond with pong messages.
   * If false, uses any incoming message as heartbeat confirmation.
   * Set to false if backend doesn't support ping/pong protocol.
   */
  requirePongResponse: boolean;
  /**
   * Maximum number of consecutive heartbeat failures before triggering reconnect.
   * Only applies when requirePongResponse is true.
   */
  maxHeartbeatFailures: number;
}

export interface ConnectionStatus {
  connected: boolean;
  authenticated: boolean;
  connectionType: 'websocket' | 'socketio' | 'polling' | 'disconnected';
  lastHeartbeat: Date | null;
  lastPong: Date | null;
  reconnectAttempts: number;
  latency: number;
  quality: 'excellent' | 'good' | 'poor' | 'disconnected';
  marketOpen: boolean;
}

type MessageType =
  | 'market_data'
  | 'order_book'
  | 'trades'
  | 'signals'
  | 'status'
  | 'error'
  | 'auth_response'
  | 'pong';

type DataCallback<T = unknown> = (data: T) => void;
type StatusCallback = (status: ConnectionStatus) => void;
type ErrorCallback = (error: Error) => void;

// ============================================================================
// Message Validation Schema
// ============================================================================

interface MessageSchema {
  type: string;
  required: string[];
  validators: Record<string, (value: unknown) => boolean>;
}

const MESSAGE_SCHEMAS: Record<string, MessageSchema> = {
  market_data: {
    type: 'market_data',
    required: ['timestamp', 'price'],
    validators: {
      timestamp: (v) => typeof v === 'string' || typeof v === 'number',
      price: (v) => typeof v === 'number' && v > 0,
      symbol: (v) => typeof v === 'string',
      volume: (v) => v === undefined || (typeof v === 'number' && v >= 0),
      bid: (v) => v === undefined || typeof v === 'number',
      ask: (v) => v === undefined || typeof v === 'number',
    }
  },
  order_book: {
    type: 'order_book',
    required: ['bids', 'asks'],
    validators: {
      bids: (v) => Array.isArray(v),
      asks: (v) => Array.isArray(v),
      lastPrice: (v) => v === undefined || typeof v === 'number',
    }
  },
  trades: {
    type: 'trades',
    required: ['tradeId', 'side', 'price', 'size'],
    validators: {
      tradeId: (v) => typeof v === 'string' || typeof v === 'number',
      side: (v) => v === 'buy' || v === 'sell',
      price: (v) => typeof v === 'number' && v > 0,
      size: (v) => typeof v === 'number' && v > 0,
    }
  },
  signals: {
    type: 'signals',
    required: ['signalId', 'signal', 'confidence'],
    validators: {
      signalId: (v) => typeof v === 'string' || typeof v === 'number',
      signal: (v) => ['long', 'short', 'flat', 'close'].includes(v as string),
      confidence: (v) => typeof v === 'number' && v >= 0 && v <= 1,
    }
  },
  auth_response: {
    type: 'auth_response',
    required: ['success'],
    validators: {
      success: (v) => typeof v === 'boolean',
    }
  },
  pong: {
    type: 'pong',
    required: ['timestamp'],
    validators: {
      timestamp: (v) => typeof v === 'number',
    }
  }
};

// ============================================================================
// Unified WebSocket Manager
// ============================================================================

class UnifiedWebSocketManager {
  private ws: WebSocket | null = null;
  private socket: Socket | null = null;
  private config: WebSocketConfig;
  private status: ConnectionStatus;

  // Timers
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private heartbeatTimeoutTimer: NodeJS.Timeout | null = null;
  private connectionTimeoutTimer: NodeJS.Timeout | null = null;
  private marketStatusTimer: NodeJS.Timeout | null = null;
  private pollingTimer: NodeJS.Timeout | null = null;
  private pendingPingTimestamp: number | null = null;
  private consecutiveHeartbeatFailures: number = 0;
  private lastMessageTime: Date | null = null;

  // Callbacks
  private messageCallbacks: Map<string, Set<DataCallback>> = new Map();
  private statusCallbacks: Set<StatusCallback> = new Set();
  private errorCallbacks: Set<ErrorCallback> = new Set();

  // Data management
  private dataBuffer: MarketDataPoint[] = [];
  private bufferSize = 1000;
  private lastDataTime: Date | null = null;

  // Market hours (USD/COP: 8:00 AM - 12:55 PM COT, Mon-Fri)
  private tradingHours = {
    start: 8,
    end: 12.917 // 12:55 PM = 12 + 55/60
  };

  constructor(config?: Partial<WebSocketConfig>) {
    this.config = {
      url: config?.url || process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000',
      authToken: config?.authToken,
      maxReconnectAttempts: config?.maxReconnectAttempts ?? 10,
      baseReconnectDelay: config?.baseReconnectDelay ?? 1000,
      maxReconnectDelay: config?.maxReconnectDelay ?? 30000,
      heartbeatInterval: config?.heartbeatInterval ?? 30000,
      heartbeatTimeout: config?.heartbeatTimeout ?? 10000,
      connectionTimeout: config?.connectionTimeout ?? 10000,
      enableMarketHoursAwareness: config?.enableMarketHoursAwareness ?? true,
      enableFallbacks: config?.enableFallbacks ?? true,
      // Default: don't require pong - use any message as heartbeat confirmation
      // This is more compatible with backends that don't implement ping/pong
      requirePongResponse: config?.requirePongResponse ?? false,
      maxHeartbeatFailures: config?.maxHeartbeatFailures ?? 3,
    };

    this.status = {
      connected: false,
      authenticated: false,
      connectionType: 'disconnected',
      lastHeartbeat: null,
      lastPong: null,
      reconnectAttempts: 0,
      latency: 0,
      quality: 'disconnected',
      marketOpen: false,
    };

    this.checkMarketStatus();
    if (this.config.enableMarketHoursAwareness) {
      this.startMarketStatusChecker();
    }
  }

  // ============================================================================
  // Authentication
  // ============================================================================

  /**
   * Set authentication token for WebSocket connection
   */
  setAuthToken(token: string): void {
    this.config.authToken = token;

    // If already connected, send auth message
    if (this.status.connected && !this.status.authenticated) {
      this.authenticate();
    }
  }

  /**
   * Send authentication message to server
   */
  private authenticate(): void {
    if (!this.config.authToken) {
      console.warn('[UnifiedWebSocket] No auth token configured');
      return;
    }

    const authMessage = {
      type: 'authenticate',
      token: this.config.authToken,
      timestamp: Date.now()
    };

    this.sendMessage(authMessage);
    console.log('[UnifiedWebSocket] Authentication request sent');
  }

  // ============================================================================
  // Connection Management
  // ============================================================================

  /**
   * Connect to WebSocket server with fallback chain
   */
  async connect(): Promise<void> {
    // Skip connection if WebSocket is disabled
    if (process.env.NEXT_PUBLIC_DISABLE_WEBSOCKET === 'true') {
      return;
    }

    if (this.status.connected) {
      return;
    }

    // Check if we're in browser
    if (typeof window === 'undefined') {
      return;
    }

    try {
      await this.connectPrimaryWebSocket();
    } catch {
      // Silent fallback
      if (this.config.enableFallbacks) {
        try {
          await this.connectSocketIOFallback();
        } catch {
          // Silent - use HTTP polling
          this.startHttpPolling();
        }
      }
    }
  }

  /**
   * Primary WebSocket connection
   */
  private connectPrimaryWebSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        // Build URL with optional auth token as query param
        let wsUrl = `${this.config.url}/ws`;
        if (this.config.authToken) {
          wsUrl += `?token=${encodeURIComponent(this.config.authToken)}`;
        }

        this.ws = new WebSocket(wsUrl);

        // Connection timeout
        this.connectionTimeoutTimer = setTimeout(() => {
          if (this.ws?.readyState !== WebSocket.OPEN) {
            console.warn('[UnifiedWebSocket] Connection timeout');
            this.ws?.close();
            reject(new Error('Connection timeout'));
          }
        }, this.config.connectionTimeout);

        this.ws.onopen = () => {
          this.clearConnectionTimeout();
          console.log('[UnifiedWebSocket] Primary WebSocket connected');
          this.handleConnectionSuccess('websocket');

          // Authenticate if token provided
          if (this.config.authToken) {
            this.authenticate();
          } else {
            this.status.authenticated = true; // No auth required
          }

          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleIncomingMessage(event.data);
        };

        this.ws.onclose = (event) => {
          console.log(`[UnifiedWebSocket] Connection closed (code: ${event.code})`);
          this.handleConnectionLoss();
        };

        this.ws.onerror = (error) => {
          // Silent fail when backend unavailable - only log in verbose mode
          if (process.env.NODE_ENV === 'development') {
            console.debug('[UnifiedWebSocket] Connection unavailable - backend not running');
          }
          reject(error);
        };

      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Socket.IO fallback connection
   */
  private connectSocketIOFallback(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const socketUrl = this.config.url.replace('ws://', 'http://').replace('wss://', 'https://');

        this.socket = io(socketUrl, {
          transports: ['websocket', 'polling'],
          reconnection: false, // We handle reconnection ourselves
          timeout: this.config.connectionTimeout,
          auth: this.config.authToken ? { token: this.config.authToken } : undefined,
        });

        this.socket.on('connect', () => {
          console.log('[UnifiedWebSocket] Socket.IO connected');
          this.handleConnectionSuccess('socketio');
          this.status.authenticated = true;

          // Subscribe to channels
          this.socket?.emit('subscribe', {
            symbol: 'USDCOP',
            channels: ['market_data', 'order_book', 'trades', 'signals']
          });

          resolve();
        });

        this.socket.on('market_data', (data) => this.handleIncomingMessage(JSON.stringify({ type: 'market_data', payload: data })));
        this.socket.on('order_book', (data) => this.handleIncomingMessage(JSON.stringify({ type: 'order_book', payload: data })));
        this.socket.on('trades', (data) => this.handleIncomingMessage(JSON.stringify({ type: 'trades', payload: data })));
        this.socket.on('signals', (data) => this.handleIncomingMessage(JSON.stringify({ type: 'signals', payload: data })));
        this.socket.on('pong', (timestamp) => this.handlePong(timestamp));

        this.socket.on('disconnect', () => {
          console.log('[UnifiedWebSocket] Socket.IO disconnected');
          this.handleConnectionLoss();
        });

        this.socket.on('connect_error', (err) => {
          // Silent when backend unavailable
          reject(err);
        });

        // Connection timeout
        setTimeout(() => {
          if (!this.socket?.connected) {
            this.socket?.disconnect();
            reject(new Error('Socket.IO connection timeout'));
          }
        }, this.config.connectionTimeout);

      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * HTTP polling as last resort fallback
   */
  private startHttpPolling(): void {
    console.log('[UnifiedWebSocket] Starting HTTP polling fallback');

    this.status.connected = true;
    this.status.connectionType = 'polling';
    this.status.quality = 'poor';
    this.notifyStatusChange();

    const pollInterval = this.status.marketOpen ? 5000 : 30000;

    const poll = async () => {
      if (this.status.connectionType !== 'polling') return;

      try {
        const response = await fetch('/api/proxy/ws', {
          headers: this.config.authToken ? {
            'Authorization': `Bearer ${this.config.authToken}`
          } : {}
        });

        if (response.ok) {
          const data = await response.json();
          if (data && data.price) {
            this.handleIncomingMessage(JSON.stringify({
              type: 'market_data',
              payload: {
                timestamp: data.timestamp || new Date().toISOString(),
                symbol: data.symbol || 'USDCOP',
                price: data.price,
                bid: data.bid,
                ask: data.ask,
                volume: data.volume,
                source: 'http_polling',
                type: 'quote'
              }
            }));
          }
          this.status.lastHeartbeat = new Date();
        }
      } catch (error) {
        console.warn('[UnifiedWebSocket] HTTP polling error:', error);
        this.status.quality = 'disconnected';
        this.notifyStatusChange();
      }

      this.pollingTimer = setTimeout(poll, pollInterval);
    };

    poll();
  }

  /**
   * Handle successful connection
   */
  private handleConnectionSuccess(type: 'websocket' | 'socketio'): void {
    this.status.connected = true;
    this.status.connectionType = type;
    this.status.reconnectAttempts = 0;
    this.status.lastHeartbeat = new Date();
    this.status.quality = 'excellent';

    this.startHeartbeat();
    this.notifyStatusChange();
  }

  /**
   * Handle connection loss
   */
  private handleConnectionLoss(): void {
    this.status.connected = false;
    this.status.authenticated = false;
    this.status.quality = 'disconnected';
    this.status.connectionType = 'disconnected';

    this.stopHeartbeat();
    this.notifyStatusChange();
    this.scheduleReconnect();
  }

  /**
   * Clear connection timeout timer
   */
  private clearConnectionTimeout(): void {
    if (this.connectionTimeoutTimer) {
      clearTimeout(this.connectionTimeoutTimer);
      this.connectionTimeoutTimer = null;
    }
  }

  // ============================================================================
  // Reconnection with Exponential Backoff + Jitter
  // ============================================================================

  /**
   * Schedule reconnection with exponential backoff and jitter
   */
  private scheduleReconnect(): void {
    if (this.status.reconnectAttempts >= this.config.maxReconnectAttempts) {
      console.error('[UnifiedWebSocket] Max reconnection attempts reached');
      this.notifyError(new Error('Max reconnection attempts reached'));
      return;
    }

    // Calculate delay with exponential backoff
    const exponentialDelay = Math.min(
      this.config.baseReconnectDelay * Math.pow(2, this.status.reconnectAttempts),
      this.config.maxReconnectDelay
    );

    // Add jitter (±25% of the delay) to prevent thundering herd
    const jitter = exponentialDelay * 0.25 * (Math.random() * 2 - 1);
    const delay = Math.round(exponentialDelay + jitter);

    console.log(
      `[UnifiedWebSocket] Reconnecting in ${delay}ms ` +
      `(attempt ${this.status.reconnectAttempts + 1}/${this.config.maxReconnectAttempts})`
    );

    this.reconnectTimer = setTimeout(() => {
      this.status.reconnectAttempts++;
      this.connect();
    }, delay);
  }

  // ============================================================================
  // Message Handling and Validation
  // ============================================================================

  /**
   * Handle incoming WebSocket message with validation
   */
  private handleIncomingMessage(rawData: string): void {
    try {
      const message = JSON.parse(rawData);
      const { type, channel, payload } = message;

      // Track last message time for heartbeat monitoring
      this.lastMessageTime = new Date();

      // Any incoming message resets heartbeat failure counter when not requiring pong
      if (!this.config.requirePongResponse) {
        this.consecutiveHeartbeatFailures = 0;
        this.clearHeartbeatTimeout();
      }

      // Handle different message structures
      const messageType = type || channel;
      const data = payload || message;

      // Handle special message types
      if (messageType === 'auth_response') {
        this.handleAuthResponse(data);
        return;
      }

      if (messageType === 'pong') {
        this.handlePong(data.timestamp);
        return;
      }

      if (messageType === 'error') {
        this.notifyError(new Error(data.message || 'Server error'));
        return;
      }

      // Validate message against schema
      if (!this.validateMessage(messageType, data)) {
        console.warn('[UnifiedWebSocket] Message validation failed:', messageType, data);
        return;
      }

      // Process market data
      if (messageType === 'market_data' || messageType === 'price_update') {
        const marketData = this.normalizeMarketData(data);
        this.addToBuffer(marketData);
        this.notifySubscribers('market_data', marketData);
      } else {
        // Pass through other message types
        this.notifySubscribers(messageType, data);
      }

      this.lastDataTime = new Date();
      this.updateConnectionQuality();

    } catch (error) {
      console.error('[UnifiedWebSocket] Error parsing message:', error);
    }
  }

  /**
   * Validate message against schema
   */
  private validateMessage(type: string, data: unknown): boolean {
    const schema = MESSAGE_SCHEMAS[type];
    if (!schema) {
      // Unknown message type - allow through with warning
      console.debug(`[UnifiedWebSocket] No schema for message type: ${type}`);
      return true;
    }

    if (typeof data !== 'object' || data === null) {
      return false;
    }

    const record = data as Record<string, unknown>;

    // Check required fields
    for (const field of schema.required) {
      if (!(field in record)) {
        console.warn(`[UnifiedWebSocket] Missing required field: ${field}`);
        return false;
      }
    }

    // Run validators
    for (const [field, validator] of Object.entries(schema.validators)) {
      if (field in record && !validator(record[field])) {
        console.warn(`[UnifiedWebSocket] Validation failed for field: ${field}`);
        return false;
      }
    }

    return true;
  }

  /**
   * Normalize market data to standard format
   */
  private normalizeMarketData(data: Record<string, unknown>): MarketDataPoint {
    return {
      timestamp: String(data.timestamp || new Date().toISOString()),
      symbol: String(data.symbol || 'USDCOP'),
      price: Number(data.price || data.close || 0),
      bid: data.bid ? Number(data.bid) : undefined,
      ask: data.ask ? Number(data.ask) : undefined,
      volume: data.volume ? Number(data.volume) : undefined,
      spread: data.spread ? Number(data.spread) : undefined,
      change24h: data.change_24h ? Number(data.change_24h) : undefined,
      changePercent24h: data.change_percent_24h ? Number(data.change_percent_24h) : undefined,
      source: String(data.source || 'websocket'),
      type: (data.type as MarketDataPoint['type']) || 'quote',
    };
  }

  /**
   * Handle authentication response
   */
  private handleAuthResponse(data: { success: boolean; error?: string }): void {
    if (data.success) {
      console.log('[UnifiedWebSocket] Authentication successful');
      this.status.authenticated = true;
    } else {
      console.error('[UnifiedWebSocket] Authentication failed:', data.error);
      this.status.authenticated = false;
      this.notifyError(new Error(`Authentication failed: ${data.error}`));
    }
    this.notifyStatusChange();
  }

  /**
   * Handle pong response
   */
  private handlePong(timestamp: number): void {
    this.status.lastPong = new Date();
    this.lastMessageTime = new Date();

    // Clear pending ping and reset failure counter
    if (this.pendingPingTimestamp) {
      this.status.latency = Date.now() - this.pendingPingTimestamp;
      this.pendingPingTimestamp = null;
    }

    this.consecutiveHeartbeatFailures = 0;
    this.clearHeartbeatTimeout();
    this.updateConnectionQuality();
  }

  // ============================================================================
  // Heartbeat Management
  // ============================================================================

  /**
   * Start heartbeat monitoring
   *
   * Heartbeat strategy:
   * - If requirePongResponse is false (default): any incoming message counts as heartbeat
   * - If requirePongResponse is true: requires explicit pong response
   * - Graceful degradation: tracks consecutive failures before triggering reconnect
   */
  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.consecutiveHeartbeatFailures = 0;

    this.heartbeatTimer = setInterval(() => {
      this.pendingPingTimestamp = Date.now();
      this.status.lastHeartbeat = new Date();

      // Only send ping if we're connected
      if (this.ws?.readyState === WebSocket.OPEN) {
        try {
          this.ws.send(JSON.stringify({
            type: 'ping',
            timestamp: this.pendingPingTimestamp
          }));
        } catch (error) {
          console.debug('[UnifiedWebSocket] Failed to send ping:', error);
        }
      } else if (this.socket?.connected) {
        this.socket.emit('ping', this.pendingPingTimestamp);
      }

      // Set up timeout to check for response
      this.clearHeartbeatTimeout();
      this.heartbeatTimeoutTimer = setTimeout(() => {
        this.handleHeartbeatTimeout();
      }, this.config.heartbeatTimeout);

    }, this.config.heartbeatInterval);
  }

  /**
   * Handle heartbeat timeout
   */
  private handleHeartbeatTimeout(): void {
    // If we don't require pong, check if we received any message recently
    if (!this.config.requirePongResponse) {
      const timeSinceLastMessage = this.lastMessageTime
        ? Date.now() - this.lastMessageTime.getTime()
        : Infinity;

      // If we received any message within the heartbeat interval, consider connection healthy
      if (timeSinceLastMessage < this.config.heartbeatInterval * 2) {
        // Connection is healthy based on incoming messages
        this.consecutiveHeartbeatFailures = 0;
        this.pendingPingTimestamp = null;
        return;
      }
    }

    // No pong received and no recent messages
    this.consecutiveHeartbeatFailures++;
    this.pendingPingTimestamp = null;

    if (this.consecutiveHeartbeatFailures >= this.config.maxHeartbeatFailures) {
      console.warn(
        `[UnifiedWebSocket] Connection stale - ${this.consecutiveHeartbeatFailures} consecutive heartbeat failures. Reconnecting...`
      );
      this.status.quality = 'poor';
      this.notifyStatusChange();

      // Trigger graceful reconnection
      this.handleConnectionLoss();
    } else {
      // Log only in debug mode to avoid console spam
      if (process.env.NODE_ENV === 'development') {
        console.debug(
          `[UnifiedWebSocket] Heartbeat timeout (${this.consecutiveHeartbeatFailures}/${this.config.maxHeartbeatFailures})`
        );
      }
      // Degrade quality but don't disconnect yet
      if (this.consecutiveHeartbeatFailures > 1) {
        this.status.quality = 'poor';
        this.notifyStatusChange();
      }
    }
  }

  /**
   * Clear heartbeat timeout timer
   */
  private clearHeartbeatTimeout(): void {
    if (this.heartbeatTimeoutTimer) {
      clearTimeout(this.heartbeatTimeoutTimer);
      this.heartbeatTimeoutTimer = null;
    }
  }

  /**
   * Stop heartbeat monitoring
   */
  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
    this.clearHeartbeatTimeout();
  }

  // ============================================================================
  // Connection Quality
  // ============================================================================

  /**
   * Update connection quality based on latency and data freshness
   */
  private updateConnectionQuality(): void {
    if (!this.status.connected) {
      this.status.quality = 'disconnected';
      return;
    }

    const timeSinceLastData = this.lastDataTime
      ? Date.now() - this.lastDataTime.getTime()
      : Infinity;

    if (this.status.latency < 100 && timeSinceLastData < 10000) {
      this.status.quality = 'excellent';
    } else if (this.status.latency < 300 && timeSinceLastData < 30000) {
      this.status.quality = 'good';
    } else {
      this.status.quality = 'poor';
    }

    this.notifyStatusChange();
  }

  // ============================================================================
  // Market Hours Awareness
  // ============================================================================

  /**
   * Check if market is currently open
   */
  private checkMarketStatus(): void {
    const now = new Date();
    const bogotaTime = new Date(now.toLocaleString("en-US", { timeZone: "America/Bogota" }));
    const currentHour = bogotaTime.getHours() + (bogotaTime.getMinutes() / 60);
    const dayOfWeek = bogotaTime.getDay();

    // Market is open Monday to Friday, 8:00 AM to 12:55 PM COT
    this.status.marketOpen = dayOfWeek >= 1 && dayOfWeek <= 5 &&
      currentHour >= this.tradingHours.start &&
      currentHour <= this.tradingHours.end;
  }

  /**
   * Start market status checker
   */
  private startMarketStatusChecker(): void {
    this.marketStatusTimer = setInterval(() => {
      const wasOpen = this.status.marketOpen;
      this.checkMarketStatus();

      if (wasOpen !== this.status.marketOpen) {
        console.log(`[UnifiedWebSocket] Market status changed: ${this.status.marketOpen ? 'OPEN' : 'CLOSED'}`);

        if (this.status.marketOpen && !this.status.connected) {
          this.connect();
        }

        this.notifyStatusChange();
      }
    }, 60000);
  }

  /**
   * Stop market status checker
   */
  private stopMarketStatusChecker(): void {
    if (this.marketStatusTimer) {
      clearInterval(this.marketStatusTimer);
      this.marketStatusTimer = null;
    }
  }

  // ============================================================================
  // Data Buffer Management
  // ============================================================================

  /**
   * Add data point to buffer
   */
  private addToBuffer(data: MarketDataPoint): void {
    this.dataBuffer.push(data);
    if (this.dataBuffer.length > this.bufferSize) {
      this.dataBuffer.shift();
    }
  }

  /**
   * Get recent data from buffer
   */
  getRecentData(count: number = 100): MarketDataPoint[] {
    return this.dataBuffer.slice(-count);
  }

  // ============================================================================
  // Subscription Management
  // ============================================================================

  /**
   * Subscribe to a message channel
   */
  subscribe(channel: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'subscribe', channel }));
    } else if (this.socket?.connected) {
      this.socket.emit('subscribe', { channel });
    }
    console.log(`[UnifiedWebSocket] Subscribed to channel: ${channel}`);
  }

  /**
   * Unsubscribe from a message channel
   */
  unsubscribe(channel: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'unsubscribe', channel }));
    } else if (this.socket?.connected) {
      this.socket.emit('unsubscribe', { channel });
    }
    console.log(`[UnifiedWebSocket] Unsubscribed from channel: ${channel}`);
  }

  /**
   * Add callback for message type
   */
  on<T = unknown>(type: string, callback: DataCallback<T>): () => void {
    if (!this.messageCallbacks.has(type)) {
      this.messageCallbacks.set(type, new Set());
    }
    this.messageCallbacks.get(type)!.add(callback as DataCallback);

    // Return unsubscribe function
    return () => {
      this.messageCallbacks.get(type)?.delete(callback as DataCallback);
    };
  }

  /**
   * Remove callback for message type
   */
  off(type: string, callback: DataCallback): void {
    this.messageCallbacks.get(type)?.delete(callback);
  }

  /**
   * Subscribe to status changes
   */
  onStatusChange(callback: StatusCallback): () => void {
    this.statusCallbacks.add(callback);
    return () => {
      this.statusCallbacks.delete(callback);
    };
  }

  /**
   * Subscribe to error events
   */
  onError(callback: ErrorCallback): () => void {
    this.errorCallbacks.add(callback);
    return () => {
      this.errorCallbacks.delete(callback);
    };
  }

  // ============================================================================
  // Notification Methods
  // ============================================================================

  /**
   * Notify subscribers of new data
   */
  private notifySubscribers(type: string, data: unknown): void {
    this.messageCallbacks.get(type)?.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error(`[UnifiedWebSocket] Error in ${type} callback:`, error);
      }
    });
  }

  /**
   * Notify status change
   */
  private notifyStatusChange(): void {
    this.statusCallbacks.forEach(callback => {
      try {
        callback({ ...this.status });
      } catch (error) {
        console.error('[UnifiedWebSocket] Error in status callback:', error);
      }
    });
  }

  /**
   * Notify error
   */
  private notifyError(error: Error): void {
    this.errorCallbacks.forEach(callback => {
      try {
        callback(error);
      } catch (callbackError) {
        console.error('[UnifiedWebSocket] Error in error callback:', callbackError);
      }
    });
  }

  // ============================================================================
  // Send Message
  // ============================================================================

  /**
   * Send message through WebSocket
   */
  sendMessage(message: Record<string, unknown>): boolean {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
      return true;
    } else if (this.socket?.connected) {
      this.socket.emit(message.type as string, message);
      return true;
    }

    console.warn('[UnifiedWebSocket] Cannot send message - not connected');
    return false;
  }

  // ============================================================================
  // Public API
  // ============================================================================

  /**
   * Get current connection status
   */
  getStatus(): ConnectionStatus {
    return { ...this.status };
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.status.connected;
  }

  /**
   * Check if authenticated
   */
  isAuthenticated(): boolean {
    return this.status.authenticated;
  }

  /**
   * Check if market is open
   */
  isMarketOpen(): boolean {
    return this.status.marketOpen;
  }

  /**
   * Force reconnection
   */
  forceReconnect(): void {
    this.disconnect();
    this.status.reconnectAttempts = 0;
    setTimeout(() => this.connect(), 1000);
  }

  /**
   * Disconnect from WebSocket
   */
  disconnect(): void {
    // Clear all timers
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.pollingTimer) {
      clearTimeout(this.pollingTimer);
      this.pollingTimer = null;
    }
    this.clearConnectionTimeout();
    this.stopHeartbeat();
    this.stopMarketStatusChecker();

    // Close connections
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }

    // Reset status
    this.status.connected = false;
    this.status.authenticated = false;
    this.status.connectionType = 'disconnected';
    this.status.quality = 'disconnected';

    this.notifyStatusChange();
    console.log('[UnifiedWebSocket] Disconnected');
  }

  /**
   * Clean up all resources (call when unmounting)
   */
  destroy(): void {
    this.disconnect();
    this.messageCallbacks.clear();
    this.statusCallbacks.clear();
    this.errorCallbacks.clear();
    this.dataBuffer = [];
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

let instance: UnifiedWebSocketManager | null = null;

export function getUnifiedWebSocketManager(config?: Partial<WebSocketConfig>): UnifiedWebSocketManager {
  if (typeof window === 'undefined') {
    // Return a mock for SSR
    return {} as UnifiedWebSocketManager;
  }

  if (!instance) {
    instance = new UnifiedWebSocketManager(config);
  }

  return instance;
}

export function resetUnifiedWebSocketManager(): void {
  if (instance) {
    instance.destroy();
    instance = null;
  }
}

export { UnifiedWebSocketManager };
export default UnifiedWebSocketManager;

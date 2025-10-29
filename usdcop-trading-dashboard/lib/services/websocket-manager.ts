/**
 * WebSocket Manager for Real-Time Market Data
 * Connects to trading API WebSocket endpoint for live USDCOP data
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

export interface OrderBookUpdate {
  timestamp: string;
  symbol: string;
  bids: Array<[number, number]>; // [price, size]
  asks: Array<[number, number]>; // [price, size]
  lastPrice: number;
  spread: number;
  spreadPercent: number;
}

export interface TradeUpdate {
  trade_id: number;
  timestamp: string;
  strategy_code: string;
  side: 'buy' | 'sell';
  price: number;
  size: number;
  pnl: number;
  status: 'open' | 'closed' | 'pending';
}

export interface SignalAlert {
  signal_id: number;
  timestamp: string;
  strategy_code: string;
  strategy_name: string;
  signal: 'long' | 'short' | 'flat' | 'close';
  confidence: number;
  entry_price: number;
  reasoning: string;
}

type MessageHandler = (data: any) => void;

class WebSocketManager {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 2000;
  private handlers: Map<string, Set<MessageHandler>> = new Map();
  private isConnecting = false;
  private heartbeatInterval: NodeJS.Timeout | null = null;

  constructor(private url: string) {}

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN || this.isConnecting) {
      console.log('[WebSocketManager] Already connected or connecting');
      return;
    }

    this.isConnecting = true;

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('[WebSocketManager] Connected to', this.url);
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        this.startHeartbeat();

        // Subscribe to all channels
        this.subscribe('market_data');
        this.subscribe('order_book');
        this.subscribe('trades');
        this.subscribe('signals');
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const { channel, payload } = data;

          if (channel && this.handlers.has(channel)) {
            this.handlers.get(channel)?.forEach(handler => handler(payload));
          }
        } catch (error) {
          console.error('[WebSocketManager] Error parsing message:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('[WebSocketManager] WebSocket error:', error);
        this.isConnecting = false;
      };

      this.ws.onclose = () => {
        console.log('[WebSocketManager] Connection closed');
        this.isConnecting = false;
        this.stopHeartbeat();
        this.attemptReconnect();
      };
    } catch (error) {
      console.error('[WebSocketManager] Failed to create WebSocket:', error);
      this.isConnecting = false;
    }
  }

  private startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000); // Every 30 seconds
  }

  private stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(
        `[WebSocketManager] Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`
      );

      setTimeout(() => {
        this.connect();
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      console.error('[WebSocketManager] Max reconnect attempts reached');
    }
  }

  subscribe(channel: string) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'subscribe',
        channel
      }));
      console.log('[WebSocketManager] Subscribed to', channel);
    }
  }

  unsubscribe(channel: string) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'unsubscribe',
        channel
      }));
      console.log('[WebSocketManager] Unsubscribed from', channel);
    }
  }

  on(channel: string, handler: MessageHandler) {
    if (!this.handlers.has(channel)) {
      this.handlers.set(channel, new Set());
    }
    this.handlers.get(channel)?.add(handler);
  }

  off(channel: string, handler: MessageHandler) {
    this.handlers.get(channel)?.delete(handler);
  }

  disconnect() {
    this.stopHeartbeat();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.handlers.clear();
    console.log('[WebSocketManager] Disconnected');
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// Singleton instance
let wsManager: WebSocketManager | null = null;

export const getWebSocketManager = (): WebSocketManager => {
  if (typeof window === 'undefined') {
    // Server-side rendering
    return {} as WebSocketManager;
  }

  if (!wsManager) {
    const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws';
    wsManager = new WebSocketManager(WS_URL);
  }

  return wsManager;
};

export default WebSocketManager;

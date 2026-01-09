/**
 * WebSocket Data Provider
 * ========================
 *
 * Provides real-time market data via WebSocket connection
 * Implements IDataProvider interface with WebSocket streaming
 */

import type {
  IDataProvider,
  MarketDataPoint,
  CandlestickResponse,
  SymbolStats,
} from '@/lib/core/interfaces';
import { ApiDataProvider } from './ApiDataProvider';

export interface WebSocketDataProviderConfig {
  wsUrl?: string;
  apiBaseUrl?: string;
  reconnectAttempts?: number;
  reconnectDelay?: number;
}

export class WebSocketDataProvider implements IDataProvider {
  private ws: WebSocket | null = null;
  private wsUrl: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts: number;
  private reconnectDelay: number;
  private subscribers: Array<(data: MarketDataPoint) => void> = [];
  private isConnecting = false;
  private apiProvider: ApiDataProvider;

  constructor(config?: WebSocketDataProviderConfig) {
    this.wsUrl = config?.wsUrl || 'ws://localhost:8082/ws';
    this.maxReconnectAttempts = config?.reconnectAttempts || 5;
    this.reconnectDelay = config?.reconnectDelay || 2000;

    // Use API provider for non-real-time methods
    this.apiProvider = new ApiDataProvider({
      apiBaseUrl: config?.apiBaseUrl,
    });
  }

  /**
   * Get real-time data (delegates to API provider)
   */
  async getRealTimeData(): Promise<MarketDataPoint[]> {
    return this.apiProvider.getRealTimeData();
  }

  /**
   * Get candlestick data (delegates to API provider)
   */
  async getCandlestickData(
    symbol: string = 'USDCOP',
    timeframe: string = '5m',
    startDate?: string,
    endDate?: string,
    limit: number = 1000,
    includeIndicators: boolean = true
  ): Promise<CandlestickResponse> {
    return this.apiProvider.getCandlestickData(
      symbol,
      timeframe,
      startDate,
      endDate,
      limit,
      includeIndicators
    );
  }

  /**
   * Get symbol statistics (delegates to API provider)
   */
  async getSymbolStats(symbol: string = 'USDCOP'): Promise<SymbolStats> {
    return this.apiProvider.getSymbolStats(symbol);
  }

  /**
   * Check health (checks both WebSocket and API)
   */
  async checkHealth(): Promise<{ status: string; message?: string }> {
    const apiHealth = await this.apiProvider.checkHealth();
    const wsConnected = this.ws?.readyState === WebSocket.OPEN;

    return {
      status: wsConnected && apiHealth.status === 'healthy' ? 'healthy' : 'degraded',
      message: `WebSocket: ${wsConnected ? 'connected' : 'disconnected'}, API: ${apiHealth.status}`,
    };
  }

  /**
   * Subscribe to real-time updates via WebSocket
   */
  subscribeToRealTimeUpdates(callback: (data: MarketDataPoint) => void): () => void {
    this.subscribers.push(callback);

    // Connect WebSocket if this is the first subscriber
    if (this.subscribers.length === 1) {
      this.connectWebSocket();
    }

    // Return unsubscribe function
    return () => {
      const index = this.subscribers.indexOf(callback);
      if (index > -1) {
        this.subscribers.splice(index, 1);
      }

      // Disconnect if no more subscribers
      if (this.subscribers.length === 0) {
        this.disconnectWebSocket();
      }
    };
  }

  /**
   * Connect to WebSocket server
   */
  private connectWebSocket(): void {
    if (this.ws?.readyState === WebSocket.OPEN || this.isConnecting) {
      console.log('[WebSocketDataProvider] Already connected or connecting');
      return;
    }

    if (typeof window === 'undefined') {
      console.warn('[WebSocketDataProvider] WebSocket not available on server-side');
      return;
    }

    this.isConnecting = true;

    try {
      console.log(`[WebSocketDataProvider] Connecting to ${this.wsUrl}`);
      this.ws = new WebSocket(this.wsUrl);

      this.ws.onopen = () => {
        console.log('[WebSocketDataProvider] Connected');
        this.isConnecting = false;
        this.reconnectAttempts = 0;

        // Subscribe to USDCOP
        if (this.ws?.readyState === WebSocket.OPEN) {
          this.ws.send(
            JSON.stringify({
              type: 'subscribe',
              symbol: 'USDCOP',
            })
          );
        }
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // Handle different message types
          if (data.type === 'price_update' || data.type === 'market_data') {
            const marketData: MarketDataPoint = {
              symbol: data.symbol || 'USDCOP',
              price: data.price || data.close,
              timestamp: new Date(data.timestamp).getTime(),
              volume: data.volume || 0,
              bid: data.bid,
              ask: data.ask,
              source: data.source || 'websocket',
            };

            // Notify all subscribers
            this.subscribers.forEach((callback) => callback(marketData));
          }
        } catch (error) {
          console.error('[WebSocketDataProvider] Error parsing message:', error);
        }
      };

      this.ws.onerror = () => {
        // Silent when backend unavailable
        this.isConnecting = false;
      };

      this.ws.onclose = () => {
        console.log('[WebSocketDataProvider] Connection closed');
        this.isConnecting = false;
        this.attemptReconnect();
      };
    } catch (error) {
      console.error('[WebSocketDataProvider] Failed to create WebSocket:', error);
      this.isConnecting = false;
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  private disconnectWebSocket(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /**
   * Attempt to reconnect to WebSocket server
   */
  private attemptReconnect(): void {
    if (this.subscribers.length === 0) {
      // No subscribers, don't reconnect
      return;
    }

    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(
        `[WebSocketDataProvider] Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`
      );

      setTimeout(() => {
        this.connectWebSocket();
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      console.error('[WebSocketDataProvider] Max reconnect attempts reached');
    }
  }
}

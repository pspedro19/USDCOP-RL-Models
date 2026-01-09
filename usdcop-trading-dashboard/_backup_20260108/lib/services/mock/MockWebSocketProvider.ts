/**
 * Mock WebSocket Provider
 * ========================
 *
 * Provides mock WebSocket functionality for testing
 * Simulates real-time data streaming without actual WebSocket connection
 *
 * WARNING: This provider should NEVER be used in production environments
 */

// Production environment guard
if (process.env.NODE_ENV === 'production') {
  throw new Error('MockWebSocketProvider cannot be used in production');
}

import type {
  IWebSocketProvider,
  WebSocketConfig,
  MessageHandler,
} from '@/lib/core/interfaces';

export class MockWebSocketProvider implements IWebSocketProvider {
  private handlers: Map<string, Set<MessageHandler>> = new Map();
  private connected = false;
  private updateInterval: NodeJS.Timeout | null = null;
  private basePrice = 4200;

  constructor(private config?: WebSocketConfig) {}

  /**
   * Simulate WebSocket connection
   */
  connect(): void {
    if (this.connected) {
      console.log('[MockWebSocketProvider] Already connected');
      return;
    }

    console.log('[MockWebSocketProvider] Connecting...');
    this.connected = true;
    this.startMockUpdates();
    console.log('[MockWebSocketProvider] Connected');
  }

  /**
   * Simulate WebSocket disconnection
   */
  disconnect(): void {
    if (!this.connected) return;

    console.log('[MockWebSocketProvider] Disconnecting...');
    this.stopMockUpdates();
    this.connected = false;
    this.handlers.clear();
    console.log('[MockWebSocketProvider] Disconnected');
  }

  /**
   * Simulate channel subscription
   */
  subscribe(channel: string): void {
    console.log('[MockWebSocketProvider] Subscribed to', channel);
    if (!this.handlers.has(channel)) {
      this.handlers.set(channel, new Set());
    }
  }

  /**
   * Simulate channel unsubscription
   */
  unsubscribe(channel: string): void {
    console.log('[MockWebSocketProvider] Unsubscribed from', channel);
    this.handlers.delete(channel);
  }

  /**
   * Register a message handler
   */
  on(channel: string, handler: MessageHandler): void {
    if (!this.handlers.has(channel)) {
      this.handlers.set(channel, new Set());
    }
    this.handlers.get(channel)?.add(handler);
  }

  /**
   * Unregister a message handler
   */
  off(channel: string, handler: MessageHandler): void {
    this.handlers.get(channel)?.delete(handler);
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.connected;
  }

  /**
   * Start sending mock updates
   */
  private startMockUpdates(): void {
    if (this.updateInterval) return;

    this.updateInterval = setInterval(() => {
      // Simulate market data updates
      if (this.handlers.has('market_data')) {
        const price = this.generatePrice();
        const data = {
          timestamp: new Date().toISOString(),
          symbol: 'USDCOP',
          price,
          volume: Math.floor(Math.random() * 1000000),
          bid: price - 0.5,
          ask: price + 0.5,
          spread: 1.0,
          change_24h: (Math.random() - 0.5) * 100,
          change_percent_24h: (Math.random() - 0.5) * 2,
        };

        this.handlers.get('market_data')?.forEach((handler) => handler(data));
      }

      // Simulate order book updates
      if (this.handlers.has('order_book')) {
        const price = this.generatePrice();
        const data = {
          timestamp: new Date().toISOString(),
          symbol: 'USDCOP',
          bids: this.generateOrderBook(price, 'bid'),
          asks: this.generateOrderBook(price, 'ask'),
          lastPrice: price,
          spread: 1.0,
          spreadPercent: 0.024,
        };

        this.handlers.get('order_book')?.forEach((handler) => handler(data));
      }

      // Simulate signal alerts (less frequent)
      if (Math.random() < 0.1 && this.handlers.has('signals')) {
        const data = {
          signal_id: Math.floor(Math.random() * 10000),
          timestamp: new Date().toISOString(),
          strategy_code: 'MOCK_STRATEGY',
          strategy_name: 'Mock Strategy',
          signal: Math.random() > 0.5 ? 'long' : 'short',
          confidence: Math.random(),
          entry_price: this.generatePrice(),
          reasoning: 'Mock signal for testing',
        };

        this.handlers.get('signals')?.forEach((handler) => handler(data));
      }
    }, 2000); // Update every 2 seconds
  }

  /**
   * Stop sending mock updates
   */
  private stopMockUpdates(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
  }

  /**
   * Generate a realistic price with volatility
   */
  private generatePrice(): number {
    const change = (Math.random() - 0.5) * 2 * 0.001 * this.basePrice;
    this.basePrice += change;
    return Number(this.basePrice.toFixed(4));
  }

  /**
   * Generate mock order book data
   */
  private generateOrderBook(price: number, side: 'bid' | 'ask'): Array<[number, number]> {
    const orders: Array<[number, number]> = [];
    const baseOffset = side === 'bid' ? -0.5 : 0.5;

    for (let i = 0; i < 10; i++) {
      const orderPrice = price + baseOffset + i * (side === 'bid' ? -0.5 : 0.5);
      const orderSize = Math.floor(Math.random() * 100000);
      orders.push([Number(orderPrice.toFixed(4)), orderSize]);
    }

    return orders;
  }
}

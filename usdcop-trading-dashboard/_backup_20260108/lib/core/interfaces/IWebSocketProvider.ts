/**
 * IWebSocketProvider Interface
 * =============================
 *
 * Interface for WebSocket providers that manage real-time connections.
 */

import {
  WebSocketConnectionState,
  ConnectionState,
  WebSocketChannel,
  ChannelSubscription,
  WebSocketStats,
  PerformanceMetrics
} from '@/types/websocket';

/**
 * WebSocket provider interface for real-time data streaming
 */
export interface IWebSocketProvider {
  /**
   * Connect to the WebSocket server
   */
  connect(): Promise<void>;

  /**
   * Disconnect from the WebSocket server
   */
  disconnect(): void;

  /**
   * Subscribe to a channel with a handler
   * @returns Unsubscribe function
   */
  subscribe(channel: string, handler: (data: unknown) => void): () => void;

  /**
   * Unsubscribe from a channel
   */
  unsubscribe(channel: string): void;

  /**
   * Check if connected
   */
  isConnected(): boolean;

  /**
   * Get current connection state
   */
  getConnectionState(): ConnectionState;
}

/**
 * Extended WebSocket provider with advanced features
 */
export interface IExtendedWebSocketProvider extends IWebSocketProvider {
  /**
   * Reconnect to the server
   */
  reconnect(): Promise<void>;

  /**
   * Send a message through the WebSocket
   */
  send(message: unknown): void;

  /**
   * Subscribe to a specific channel with options
   */
  subscribeChannel(
    channel: WebSocketChannel,
    options?: { symbol?: string; filters?: Record<string, unknown> }
  ): void;

  /**
   * Unsubscribe from a specific channel
   */
  unsubscribeChannel(channel: WebSocketChannel): void;

  /**
   * Get all active subscriptions
   */
  getSubscriptions(): ChannelSubscription[];

  /**
   * Get WebSocket statistics
   */
  getStats(): WebSocketStats;

  /**
   * Get performance metrics
   */
  getPerformanceMetrics(): PerformanceMetrics;

  /**
   * Start heartbeat mechanism
   */
  startHeartbeat(): void;

  /**
   * Stop heartbeat mechanism
   */
  stopHeartbeat(): void;

  /**
   * Register event listeners
   */
  on(event: 'connect' | 'disconnect' | 'error' | 'message', handler: (...args: unknown[]) => void): void;

  /**
   * Unregister event listeners
   */
  off(event: 'connect' | 'disconnect' | 'error' | 'message', handler: (...args: unknown[]) => void): void;
}

/**
 * Configuration for WebSocket providers
 */
export interface WebSocketProviderConfig {
  url: string;
  protocols?: string | string[];
  reconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  timeout?: number;
  debug?: boolean;
}

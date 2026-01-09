/**
 * WebSocket Connector - WebSocket Lifecycle Management
 * ====================================================
 *
 * Single Responsibility: WebSocket connection lifecycle
 * - Connect/disconnect WebSocket
 * - Handle reconnection logic
 * - Manage message subscriptions
 *
 * @deprecated This file is deprecated. Use the unified WebSocket manager instead:
 * ```typescript
 * import { getUnifiedWebSocketManager } from '@/lib/services/unified-websocket-manager';
 *
 * const wsManager = getUnifiedWebSocketManager();
 * wsManager.setAuthToken('your-token'); // Optional
 * wsManager.connect();
 * wsManager.subscribe('market_data');
 * wsManager.on('market_data', (data) => console.log(data));
 * ```
 *
 * The unified WebSocket manager provides all features of this connector plus:
 * - Token-based authentication
 * - Message validation with schema
 * - Exponential backoff with jitter for reconnection
 * - Market hours awareness
 * - Multiple fallback strategies
 */

import { createLogger } from '@/lib/utils/logger'
import type { MarketDataPoint } from './types'

const logger = createLogger('WebSocketConnector')

export interface WebSocketConfig {
  url: string
  reconnectInterval?: number
  autoReconnect?: boolean
}

export class WebSocketConnector {
  private websocket: WebSocket | null = null
  private subscribers: Array<(data: MarketDataPoint) => void> = []
  private config: Required<WebSocketConfig>
  private isConnecting: boolean = false
  private reconnectTimeout: NodeJS.Timeout | null = null

  constructor(config: WebSocketConfig) {
    this.config = {
      url: config.url,
      reconnectInterval: config.reconnectInterval ?? 5000,
      autoReconnect: config.autoReconnect ?? true,
    }
  }

  /**
   * Connect to WebSocket server
   */
  connect(symbol: string): WebSocket | null {
    // Only run in browser
    if (typeof window === 'undefined') {
      logger.warn('WebSocket is not available in server-side rendering')
      return null
    }

    if (this.isConnecting) {
      logger.warn('Connection already in progress')
      return this.websocket
    }

    if (this.websocket?.readyState === WebSocket.OPEN) {
      logger.info('WebSocket already connected')
      return this.websocket
    }

    try {
      this.isConnecting = true
      logger.info(`Connecting to WebSocket: ${this.config.url}`)

      this.websocket = new WebSocket(`${this.config.url}/ws`)

      this.websocket.onopen = () => {
        this.isConnecting = false
        logger.info('WebSocket connected successfully')
        this.subscribe(symbol)
      }

      this.websocket.onmessage = (event) => {
        this.handleMessage(event)
      }

      this.websocket.onclose = (event) => {
        this.isConnecting = false
        logger.warn(`WebSocket disconnected (code: ${event.code})`)

        if (this.config.autoReconnect) {
          logger.info(`Attempting to reconnect in ${this.config.reconnectInterval}ms...`)
          this.scheduleReconnect(symbol)
        }
      }

      this.websocket.onerror = () => {
        // Silent when backend unavailable
        this.isConnecting = false
      }

      return this.websocket
    } catch (error) {
      this.isConnecting = false
      logger.error('Failed to connect WebSocket:', error)
      return null
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout)
      this.reconnectTimeout = null
    }

    if (this.websocket) {
      logger.info('Disconnecting WebSocket')
      this.config.autoReconnect = false // Prevent auto-reconnect on manual disconnect
      this.websocket.close()
      this.websocket = null
    }
  }

  /**
   * Subscribe to symbol updates
   */
  private subscribe(symbol: string): void {
    if (this.websocket?.readyState === WebSocket.OPEN) {
      const message = JSON.stringify({
        type: 'subscribe',
        symbol: symbol,
      })
      this.websocket.send(message)
      logger.debug(`Subscribed to symbol: ${symbol}`)
    }
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(symbol: string): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout)
    }

    this.reconnectTimeout = setTimeout(() => {
      logger.info('Attempting to reconnect...')
      this.connect(symbol)
    }, this.config.reconnectInterval)
  }

  /**
   * Handle incoming WebSocket message
   */
  private handleMessage(event: MessageEvent): void {
    try {
      const data = JSON.parse(event.data)
      logger.debug('Received WebSocket data:', data)

      // Handle different message types
      if (data.type === 'price_update' || data.type === 'market_data') {
        const marketData: MarketDataPoint = {
          symbol: data.symbol,
          price: data.price || data.close,
          timestamp: new Date(data.timestamp).getTime(),
          volume: data.volume || 0,
          bid: data.bid,
          ask: data.ask,
          source: data.source || 'websocket',
        }

        // Notify all subscribers
        this.notifySubscribers(marketData)
      } else if (data.type === 'status') {
        logger.info('WebSocket service status:', data)
      }
    } catch (error) {
      logger.error('Error parsing WebSocket message:', error)
    }
  }

  /**
   * Add a subscriber for market data updates
   */
  addSubscriber(callback: (data: MarketDataPoint) => void): void {
    this.subscribers.push(callback)
  }

  /**
   * Remove a subscriber
   */
  removeSubscriber(callback: (data: MarketDataPoint) => void): void {
    const index = this.subscribers.indexOf(callback)
    if (index > -1) {
      this.subscribers.splice(index, 1)
    }
  }

  /**
   * Notify all subscribers of new data
   */
  private notifySubscribers(data: MarketDataPoint): void {
    this.subscribers.forEach((callback) => {
      try {
        callback(data)
      } catch (error) {
        logger.error('Error in subscriber callback:', error)
      }
    })
  }

  /**
   * Get current connection state
   */
  getConnectionState(): number | null {
    return this.websocket?.readyState ?? null
  }

  /**
   * Check if WebSocket is connected
   */
  isConnected(): boolean {
    return this.websocket?.readyState === WebSocket.OPEN
  }
}

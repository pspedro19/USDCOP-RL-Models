/**
 * Basic Usage Example - Elite Trading Platform Architecture
 * Demonstrates how to use the core systems
 */

import {
  initializeEliteTradingPlatform,
  getDataBus,
  getEventManager,
  getWebSocketManager,
  getPerformanceMonitor
} from '@/libs';
import type { MarketTick, OrderEvent, TradingSignal } from '@/core/types';

/**
 * Example: Basic platform initialization and usage
 */
export async function basicUsageExample() {
  try {
    // Initialize the platform
    const platform = await initializeEliteTradingPlatform();
    console.log('Platform initialized:', platform.status);

    // Get core system instances
    const dataBus = getDataBus();
    const eventManager = getEventManager();
    const performanceMonitor = getPerformanceMonitor();

    // Example 1: Subscribe to market data
    const marketDataSubscription = dataBus.subscribe<MarketTick>(
      'market.ticks',
      (tick) => {
        console.log(`Received tick for ${tick.symbol}: ${tick.last}`);
      },
      {
        filter: (tick) => tick.symbol === 'USDCOP',
        includeCache: true
      }
    );

    // Example 2: Subscribe to trading events
    const tradingSubscription = eventManager.subscribe<OrderEvent>(
      { types: ['trading.order.created', 'trading.order.filled'] },
      (event) => {
        console.log(`Trading event: ${event.type}`, event.data);
      }
    );

    // Example 3: Publish market data
    const sampleTick: MarketTick = {
      id: 'tick-1',
      symbol: 'USDCOP',
      timestamp: Date.now(),
      bid: 4150.25,
      ask: 4150.75,
      last: 4150.50,
      volume: 1000,
      change: 5.25,
      changePercent: 0.125,
      high: 4155.00,
      low: 4140.00,
      open: 4145.00,
      source: 'twelvedata',
      quality: 'realtime'
    };

    dataBus.publish('market.ticks', sampleTick, { cache: true });

    // Example 4: Emit trading event
    const orderEvent: OrderEvent = {
      id: 'event-1',
      type: 'trading.order.created',
      timestamp: Date.now(),
      source: 'trading-engine',
      priority: 'high',
      data: {
        id: 'order-123',
        symbol: 'USDCOP',
        side: 'buy',
        type: 'limit',
        timeInForce: 'GTC',
        quantity: 100,
        price: 4150.00,
        status: 'new',
        timestamp: Date.now(),
        lastUpdateTime: Date.now(),
        executedQuantity: 0,
        fills: []
      }
    };

    eventManager.emitEvent(orderEvent);

    // Example 5: Performance monitoring
    const measureTradeExecution = performanceMonitor.measureFunction(
      'trade-execution',
      () => {
        // Simulate trade execution logic
        const start = Date.now();
        while (Date.now() - start < 10) {
          // Simulate processing time
        }
        return 'Trade executed';
      }
    );

    console.log('Trade execution result:', measureTradeExecution);

    // Example 6: Combined data streams
    const combinedStream = dataBus.combineChannels<MarketTick, TradingSignal, any>(
      'market.ticks',
      'trading.signals',
      (tick, signal) => ({
        symbol: tick.symbol,
        price: tick.last,
        signal: signal.type,
        confidence: signal.confidence
      })
    );

    // Subscribe to combined stream
    combinedStream.subscribe(data => {
      console.log('Combined data:', data);
    });

    // Example 7: Get performance metrics
    setTimeout(() => {
      const metrics = performanceMonitor.getCurrentMetrics();
      console.log('Current performance metrics:', {
        memoryUsage: metrics.memoryUsage.percentage,
        fps: metrics.fps,
        eventsPerSecond: metrics.eventsPerSecond
      });

      const dataBusMetrics = dataBus.getMetrics();
      console.log('DataBus metrics:', {
        totalMessages: dataBusMetrics.totalMessages,
        cacheHitRate: dataBusMetrics.cacheHitRate,
        activeChannels: dataBusMetrics.channels.length
      });

      const eventStats = eventManager.getStats();
      console.log('Event manager stats:', {
        totalEvents: eventStats.totalEvents,
        averageLatency: eventStats.averageLatency
      });
    }, 5000);

    return {
      dataBus,
      eventManager,
      performanceMonitor,
      subscriptions: {
        marketData: marketDataSubscription,
        trading: tradingSubscription
      }
    };

  } catch (error) {
    console.error('Error in basic usage example:', error);
    throw error;
  }
}

/**
 * Example: WebSocket connection (when available)
 */
export async function websocketExample() {
  try {
    // This would be used when WebSocket server is available
    const wsManager = getWebSocketManager({
      url: 'wss://api.example.com/ws',
      maxReconnectAttempts: 5,
      reconnectDelay: 3000,
      heartbeatInterval: 30000,
      connectionTimeout: 10000,
      messageTimeout: 5000,
      maxMessageQueue: 500,
      enableCompression: true,
      enableLogging: true,
      autoReconnect: true
    });

    // Connect to WebSocket
    const connectionId = await wsManager.connect();
    console.log('WebSocket connected:', connectionId);

    // Subscribe to connection status
    wsManager.getConnectionStatus$().subscribe(status => {
      console.log('Connection status:', status);
    });

    // Subscribe to messages
    wsManager.getMessages$().subscribe(message => {
      console.log('WebSocket message:', message);
    });

    // Subscribe to specific market data
    wsManager.subscribe({
      id: 'market-data-sub',
      channel: 'market.ticks',
      symbols: ['USDCOP', 'EURUSD']
    });

    return wsManager;

  } catch (error) {
    console.error('WebSocket example error:', error);
    throw error;
  }
}
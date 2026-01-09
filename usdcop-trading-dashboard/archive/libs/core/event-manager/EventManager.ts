/**
 * EventManager - High-Performance Pub/Sub Event System
 * Elite Trading Platform Event Management
 */

import { EventEmitter } from 'eventemitter3';
import type {
  TradingPlatformEvent,
  BaseEvent,
  EventFilter,
  EventSubscription,
  EventStats,
  EventPriority
} from '../types';

export interface EventManagerConfig {
  readonly maxListeners: number;
  readonly enableLogging: boolean;
  readonly logLevel: 'debug' | 'info' | 'warn' | 'error';
  readonly enableMetrics: boolean;
  readonly metricsInterval: number;
  readonly enablePersistence: boolean;
  readonly maxEventHistory: number;
  readonly enableDeduplication: boolean;
  readonly deduplicationWindow: number;
}

export interface EventHandler<T = any> {
  (event: T): void | Promise<void>;
}

export interface EventMiddleware {
  (event: TradingPlatformEvent, next: () => void): void;
}

export class EventManager extends EventEmitter {
  private readonly config: EventManagerConfig;
  private readonly subscriptions = new Map<string, EventSubscription>();
  private readonly eventHistory: TradingPlatformEvent[] = [];
  private readonly middleware: EventMiddleware[] = [];
  private readonly eventQueue: TradingPlatformEvent[] = [];
  private readonly deduplicationCache = new Map<string, number>();

  // Performance tracking
  private readonly stats: EventStats = {
    totalEvents: 0,
    eventsByType: {},
    eventsBySource: {},
    eventsByPriority: {
      critical: 0,
      high: 0,
      normal: 0,
      low: 0
    },
    averageLatency: 0,
    peakEventsPerSecond: 0,
    errorRate: 0,
    lastReset: Date.now()
  };

  private metricsTimer?: NodeJS.Timeout;
  private processingTimer?: NodeJS.Timeout;
  private isProcessing = false;

  constructor(config: Partial<EventManagerConfig> = {}) {
    super();

    this.config = {
      maxListeners: 1000,
      enableLogging: false,
      logLevel: 'info',
      enableMetrics: true,
      metricsInterval: 5000,
      enablePersistence: false,
      maxEventHistory: 10000,
      enableDeduplication: true,
      deduplicationWindow: 1000,
      ...config
    };

    this.initialize();
  }

  private initialize(): void {
    if (this.config.enableMetrics) {
      this.startMetricsCollection();
    }

    this.startEventProcessor();
    this.setupErrorHandling();
  }

  /**
   * Emit an event with high performance processing
   */
  public emitEvent<T extends TradingPlatformEvent>(event: T): void {
    const startTime = performance.now();

    try {
      // Validate event
      if (!this.validateEvent(event)) {
        throw new Error(`Invalid event: ${JSON.stringify(event)}`);
      }

      // Check for duplicates
      if (this.config.enableDeduplication && this.isDuplicate(event)) {
        this.log('debug', `Duplicate event ignored: ${event.id}`);
        return;
      }

      // Add to queue for processing
      this.eventQueue.push(event);

      // Process immediately for critical events
      if (event.priority === 'critical') {
        this.processEvents();
      }

      // Update stats
      this.updateStats(event, performance.now() - startTime);

    } catch (error) {
      this.handleError(error as Error, { event });
    }
  }

  /**
   * Subscribe to events with advanced filtering
   */
  public subscribe<T extends TradingPlatformEvent>(
    filter: EventFilter | string,
    handler: EventHandler<T>,
    options?: {
      once?: boolean;
      priority?: EventPriority;
      context?: any;
    }
  ): string {
    const subscriptionId = this.generateId();

    // Convert string filter to EventFilter
    const eventFilter: EventFilter = typeof filter === 'string'
      ? { types: [filter] }
      : filter;

    const subscription: EventSubscription = {
      id: subscriptionId,
      filter: eventFilter,
      callback: handler as any,
      active: true,
      createdAt: Date.now(),
      eventCount: 0
    };

    this.subscriptions.set(subscriptionId, subscription);

    // Add to EventEmitter for specific event types
    if (eventFilter.types) {
      for (const type of eventFilter.types) {
        if (options?.once) {
          this.once(type, this.createFilteredHandler(subscription, handler));
        } else {
          this.on(type, this.createFilteredHandler(subscription, handler));
        }
      }
    } else {
      // Listen to all events
      this.on('*', this.createFilteredHandler(subscription, handler));
    }

    this.log('debug', `Subscription created: ${subscriptionId}`);
    return subscriptionId;
  }

  /**
   * Unsubscribe from events
   */
  public unsubscribe(subscriptionId: string): boolean {
    const subscription = this.subscriptions.get(subscriptionId);
    if (!subscription) {
      return false;
    }

    // Remove from EventEmitter
    if (subscription.filter.types) {
      for (const type of subscription.filter.types) {
        this.removeAllListeners(type);
      }
    } else {
      this.removeAllListeners('*');
    }

    // Mark as inactive and remove
    (subscription as any).active = false;
    this.subscriptions.delete(subscriptionId);

    this.log('debug', `Subscription removed: ${subscriptionId}`);
    return true;
  }

  /**
   * Add middleware for event processing
   */
  public use(middleware: EventMiddleware): void {
    this.middleware.push(middleware);
  }

  /**
   * Get event statistics
   */
  public getStats(): EventStats {
    return { ...this.stats };
  }

  /**
   * Get event history
   */
  public getEventHistory(filter?: EventFilter): TradingPlatformEvent[] {
    if (!filter) {
      return [...this.eventHistory];
    }

    return this.eventHistory.filter(event => this.matchesFilter(event, filter));
  }

  /**
   * Clear event history
   */
  public clearHistory(): void {
    this.eventHistory.length = 0;
    this.log('info', 'Event history cleared');
  }

  /**
   * Reset statistics
   */
  public resetStats(): void {
    (this.stats as any).totalEvents = 0;
    (this.stats as any).eventsByType = {};
    (this.stats as any).eventsBySource = {};
    (this.stats as any).eventsByPriority = { critical: 0, high: 0, normal: 0, low: 0 };
    (this.stats as any).averageLatency = 0;
    (this.stats as any).peakEventsPerSecond = 0;
    (this.stats as any).errorRate = 0;
    (this.stats as any).lastReset = Date.now();
  }

  /**
   * Destroy event manager and cleanup
   */
  public destroy(): void {
    // Clear timers
    if (this.metricsTimer) clearInterval(this.metricsTimer);
    if (this.processingTimer) clearInterval(this.processingTimer);

    // Clear all subscriptions
    Array.from(this.subscriptions.keys()).forEach(subscriptionId => {
      this.unsubscribe(subscriptionId);
    });

    // Clear arrays and maps
    this.eventHistory.length = 0;
    this.eventQueue.length = 0;
    this.subscriptions.clear();
    this.deduplicationCache.clear();

    // Remove all listeners
    this.removeAllListeners();

    this.log('info', 'EventManager destroyed');
  }

  // Private helper methods
  private createFilteredHandler<T extends TradingPlatformEvent>(
    subscription: EventSubscription,
    handler: EventHandler<T>
  ): (event: T) => void {
    return (event: T) => {
      if (!subscription.active) return;

      if (this.matchesFilter(event, subscription.filter)) {
        try {
          (subscription as any).eventCount++;
          (subscription as any).lastTriggered = Date.now();
          handler(event);
        } catch (error) {
          this.handleError(error as Error, { subscription: subscription.id, event });
        }
      }
    };
  }

  private processEvents(): void {
    if (this.isProcessing || this.eventQueue.length === 0) return;

    this.isProcessing = true;

    try {
      const events = this.eventQueue.splice(0);

      for (const event of events) {
        this.processEvent(event);
      }
    } catch (error) {
      this.handleError(error as Error, { context: 'event_processing' });
    } finally {
      this.isProcessing = false;
    }
  }

  private processEvent(event: TradingPlatformEvent): void {
    // Apply middleware
    let index = 0;
    const next = () => {
      if (index < this.middleware.length) {
        const middleware = this.middleware[index++];
        middleware(event, next);
      } else {
        // Final processing - emit the event
        this.emit(event.type, event);
        this.emit('*', event);
      }
    };

    next();

    // Add to history
    if (this.config.enablePersistence) {
      this.addToHistory(event);
    }
  }

  private addToHistory(event: TradingPlatformEvent): void {
    this.eventHistory.push(event);

    // Trim history if needed
    if (this.eventHistory.length > this.config.maxEventHistory) {
      this.eventHistory.splice(0, this.eventHistory.length - this.config.maxEventHistory);
    }
  }

  private matchesFilter(event: TradingPlatformEvent, filter: EventFilter): boolean {
    // Check event types
    if (filter.types && !filter.types.includes(event.type)) {
      return false;
    }

    // Check sources
    if (filter.sources && !filter.sources.includes(event.source)) {
      return false;
    }

    // Check priority
    if (filter.priority && !filter.priority.includes(event.priority)) {
      return false;
    }

    // Check time range
    if (filter.timeRange) {
      const { from, to } = filter.timeRange;
      if (event.timestamp < from || event.timestamp > to) {
        return false;
      }
    }

    // Check symbols (if applicable)
    if (filter.symbols && event.data && typeof event.data === 'object') {
      const symbol = (event.data as any).symbol;
      if (symbol && !filter.symbols.includes(symbol)) {
        return false;
      }
    }

    return true;
  }

  private validateEvent(event: TradingPlatformEvent): boolean {
    return !!(
      event.id &&
      event.type &&
      event.timestamp &&
      event.source &&
      event.priority &&
      event.data !== undefined
    );
  }

  private isDuplicate(event: TradingPlatformEvent): boolean {
    const key = `${event.type}-${event.source}-${event.id}`;
    const lastSeen = this.deduplicationCache.get(key);
    const now = Date.now();

    if (lastSeen && (now - lastSeen) < this.config.deduplicationWindow) {
      return true;
    }

    this.deduplicationCache.set(key, now);

    // Clean old entries
    Array.from(this.deduplicationCache.entries()).forEach(([cacheKey, timestamp]) => {
      if (now - timestamp > this.config.deduplicationWindow) {
        this.deduplicationCache.delete(cacheKey);
      }
    });

    return false;
  }

  private updateStats(event: TradingPlatformEvent, latency: number): void {
    (this.stats as any).totalEvents++;
    (this.stats as any).eventsByType[event.type] = (this.stats.eventsByType[event.type] || 0) + 1;
    (this.stats as any).eventsBySource[event.source] = (this.stats.eventsBySource[event.source] || 0) + 1;
    (this.stats as any).eventsByPriority[event.priority]++;

    // Update average latency
    (this.stats as any).averageLatency = (this.stats.averageLatency + latency) / 2;
  }

  private startMetricsCollection(): void {
    this.metricsTimer = setInterval(() => {
      const now = Date.now();
      const timeDiff = now - this.stats.lastReset;

      if (timeDiff > 0) {
        const eventsPerSecond = (this.stats.totalEvents / timeDiff) * 1000;
        (this.stats as any).peakEventsPerSecond = Math.max(this.stats.peakEventsPerSecond, eventsPerSecond);
      }

      this.emit('metrics.updated', this.getStats());
    }, this.config.metricsInterval);
  }

  private startEventProcessor(): void {
    this.processingTimer = setInterval(() => {
      this.processEvents();
    }, 10); // Process every 10ms
  }

  private setupErrorHandling(): void {
    this.on('error', (error: Error) => {
      (this.stats as any).errorRate++;
      this.log('error', `EventManager error: ${error.message}`, error);
    });
  }

  private handleError(error: Error, context?: any): void {
    this.emit('error', error);
    this.log('error', `Error in EventManager: ${error.message}`, { error, context });
  }

  private log(level: string, message: string, data?: any): void {
    if (!this.config.enableLogging) return;

    const levels = ['debug', 'info', 'warn', 'error'];
    const configLevel = levels.indexOf(this.config.logLevel);
    const messageLevel = levels.indexOf(level);

    if (messageLevel >= configLevel) {
      const logFn = level === 'error' ? console.error :
                    level === 'warn' ? console.warn :
                    level === 'debug' ? console.debug :
                    console.log;

      logFn(`[EventManager] ${message}`, data || '');
    }
  }

  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substring(2)}`;
  }
}

// Singleton instance
let eventManagerInstance: EventManager | null = null;

export function getEventManager(config?: Partial<EventManagerConfig>): EventManager {
  if (!eventManagerInstance) {
    eventManagerInstance = new EventManager(config);
  }
  return eventManagerInstance;
}

export function resetEventManager(): void {
  if (eventManagerInstance) {
    eventManagerInstance.destroy();
    eventManagerInstance = null;
  }
}
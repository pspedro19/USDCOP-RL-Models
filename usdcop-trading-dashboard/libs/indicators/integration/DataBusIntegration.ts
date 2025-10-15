/**
 * Data Bus Integration
 * ===================
 *
 * Seamless integration between the Indicators Engine and the existing
 * chart engine and data bus systems for real-time indicator updates.
 */

import { EventEmitter } from 'eventemitter3';
import { IndicatorEngine } from '../engine/IndicatorEngine';
import { CandleData, IndicatorConfig, IndicatorValue } from '../types';

// Import from existing data bus (would be actual imports in production)
type DataBusEventType = 'market_data' | 'indicator_update' | 'chart_update' | 'subscription' | 'error';

interface DataBusEvent {
  type: DataBusEventType;
  symbol: string;
  timeframe: string;
  data: any;
  timestamp: number;
}

interface ChartEngine {
  addIndicatorOverlay: (id: string, data: any[], config: any) => void;
  updateIndicatorOverlay: (id: string, data: any[]) => void;
  removeIndicatorOverlay: (id: string) => void;
  subscribeToTimeframeChange: (callback: (timeframe: string) => void) => void;
  subscribeToSymbolChange: (callback: (symbol: string) => void) => void;
}

interface DataBus {
  subscribe: (eventType: DataBusEventType, callback: (event: DataBusEvent) => void) => string;
  unsubscribe: (subscriptionId: string) => void;
  publish: (event: DataBusEvent) => void;
  getLatestData: (symbol: string, timeframe: string, limit?: number) => CandleData[];
}

export interface IndicatorSubscription {
  id: string;
  symbol: string;
  timeframe: string;
  indicator: IndicatorConfig;
  chartOverlayId?: string;
  lastUpdate: number;
  updateInterval: number; // milliseconds
  autoUpdate: boolean;
}

export interface DataBusIntegrationConfig {
  indicatorEngine: IndicatorEngine;
  dataBus: DataBus;
  chartEngine: ChartEngine;
  defaultUpdateInterval: number;
  maxSubscriptions: number;
  enableBatching: boolean;
  batchSize: number;
  batchInterval: number;
}

export class IndicatorDataBusIntegration extends EventEmitter {
  private config: DataBusIntegrationConfig;
  private subscriptions = new Map<string, IndicatorSubscription>();
  private dataBusSubscriptions = new Map<string, string>();
  private batchQueue = new Map<string, { data: CandleData[]; timestamp: number }>();
  private batchTimer: NodeJS.Timeout | null = null;
  private isActive = false;

  constructor(config: DataBusIntegrationConfig) {
    super();
    this.config = config;
    this.setupDataBusListeners();
    this.setupChartEngineListeners();
  }

  /**
   * Initialize the integration and start processing
   */
  public async initialize(): Promise<void> {
    if (this.isActive) return;

    try {
      // Initialize indicator engine if not already done
      await this.config.indicatorEngine.reset();

      // Setup batch processing if enabled
      if (this.config.enableBatching) {
        this.startBatchProcessing();
      }

      this.isActive = true;
      this.emit('initialized');
    } catch (error) {
      this.emit('error', { type: 'initialization', error });
      throw error;
    }
  }

  /**
   * Subscribe to real-time indicator updates
   */
  public async subscribeToIndicator(
    symbol: string,
    timeframe: string,
    indicator: IndicatorConfig,
    options: {
      autoUpdate?: boolean;
      updateInterval?: number;
      chartOverlay?: boolean;
      chartConfig?: any;
    } = {}
  ): Promise<string> {
    const subscriptionId = this.generateSubscriptionId(symbol, timeframe, indicator.name);

    if (this.subscriptions.has(subscriptionId)) {
      throw new Error(`Subscription already exists: ${subscriptionId}`);
    }

    if (this.subscriptions.size >= this.config.maxSubscriptions) {
      throw new Error('Maximum number of subscriptions reached');
    }

    try {
      // Get initial data
      const initialData = this.config.dataBus.getLatestData(symbol, timeframe, 1000);

      if (initialData.length === 0) {
        throw new Error(`No data available for ${symbol} ${timeframe}`);
      }

      // Calculate initial indicator values
      const indicatorResult = await this.config.indicatorEngine.calculateIndicator(
        initialData,
        indicator,
        { useCache: true }
      );

      // Create subscription
      const subscription: IndicatorSubscription = {
        id: subscriptionId,
        symbol,
        timeframe,
        indicator,
        lastUpdate: Date.now(),
        updateInterval: options.updateInterval || this.config.defaultUpdateInterval,
        autoUpdate: options.autoUpdate !== false
      };

      // Add chart overlay if requested
      if (options.chartOverlay) {
        const overlayId = `indicator_${subscriptionId}`;
        this.config.chartEngine.addIndicatorOverlay(overlayId, indicatorResult, {
          ...options.chartConfig,
          name: indicator.name,
          symbol,
          timeframe
        });
        subscription.chartOverlayId = overlayId;
      }

      this.subscriptions.set(subscriptionId, subscription);

      // Subscribe to data bus updates for this symbol/timeframe
      const dataBusKey = `${symbol}_${timeframe}`;
      if (!this.dataBusSubscriptions.has(dataBusKey)) {
        const dataBusSubId = this.config.dataBus.subscribe('market_data', (event) => {
          if (event.symbol === symbol && event.data.timeframe === timeframe) {
            this.handleMarketDataUpdate(event);
          }
        });
        this.dataBusSubscriptions.set(dataBusKey, dataBusSubId);
      }

      this.emit('subscriptionCreated', {
        subscriptionId,
        symbol,
        timeframe,
        indicator: indicator.name
      });

      return subscriptionId;

    } catch (error) {
      this.emit('error', {
        type: 'subscription',
        subscriptionId,
        error
      });
      throw error;
    }
  }

  /**
   * Unsubscribe from indicator updates
   */
  public unsubscribeFromIndicator(subscriptionId: string): void {
    const subscription = this.subscriptions.get(subscriptionId);

    if (!subscription) {
      throw new Error(`Subscription not found: ${subscriptionId}`);
    }

    // Remove chart overlay if exists
    if (subscription.chartOverlayId) {
      this.config.chartEngine.removeIndicatorOverlay(subscription.chartOverlayId);
    }

    // Remove subscription
    this.subscriptions.delete(subscriptionId);

    // Check if we need to unsubscribe from data bus
    const dataBusKey = `${subscription.symbol}_${subscription.timeframe}`;
    const hasOtherSubscriptions = Array.from(this.subscriptions.values()).some(
      sub => sub.symbol === subscription.symbol && sub.timeframe === subscription.timeframe
    );

    if (!hasOtherSubscriptions && this.dataBusSubscriptions.has(dataBusKey)) {
      const dataBusSubId = this.dataBusSubscriptions.get(dataBusKey)!;
      this.config.dataBus.unsubscribe(dataBusSubId);
      this.dataBusSubscriptions.delete(dataBusKey);
    }

    this.emit('subscriptionRemoved', { subscriptionId });
  }

  /**
   * Update indicator configuration for existing subscription
   */
  public async updateIndicatorConfig(
    subscriptionId: string,
    newConfig: Partial<IndicatorConfig>
  ): Promise<void> {
    const subscription = this.subscriptions.get(subscriptionId);

    if (!subscription) {
      throw new Error(`Subscription not found: ${subscriptionId}`);
    }

    // Update indicator configuration
    subscription.indicator = { ...subscription.indicator, ...newConfig };

    // Recalculate with new configuration
    const data = this.config.dataBus.getLatestData(
      subscription.symbol,
      subscription.timeframe,
      1000
    );

    const indicatorResult = await this.config.indicatorEngine.calculateIndicator(
      data,
      subscription.indicator,
      { useCache: false }
    );

    // Update chart overlay if exists
    if (subscription.chartOverlayId) {
      this.config.chartEngine.updateIndicatorOverlay(
        subscription.chartOverlayId,
        indicatorResult
      );
    }

    this.emit('indicatorUpdated', {
      subscriptionId,
      indicator: subscription.indicator.name,
      data: indicatorResult
    });
  }

  /**
   * Get current indicator value for a subscription
   */
  public async getCurrentIndicatorValue(subscriptionId: string): Promise<any> {
    const subscription = this.subscriptions.get(subscriptionId);

    if (!subscription) {
      throw new Error(`Subscription not found: ${subscriptionId}`);
    }

    const data = this.config.dataBus.getLatestData(
      subscription.symbol,
      subscription.timeframe,
      Math.max(subscription.indicator.period || 50, 100)
    );

    return this.config.indicatorEngine.calculateIndicator(
      data,
      subscription.indicator,
      { useCache: true }
    );
  }

  /**
   * Get all active subscriptions
   */
  public getActiveSubscriptions(): IndicatorSubscription[] {
    return Array.from(this.subscriptions.values());
  }

  /**
   * Batch subscribe to multiple indicators
   */
  public async batchSubscribe(
    requests: Array<{
      symbol: string;
      timeframe: string;
      indicator: IndicatorConfig;
      options?: any;
    }>
  ): Promise<string[]> {
    const subscriptionIds: string[] = [];

    try {
      for (const request of requests) {
        const id = await this.subscribeToIndicator(
          request.symbol,
          request.timeframe,
          request.indicator,
          request.options
        );
        subscriptionIds.push(id);
      }

      this.emit('batchSubscriptionCreated', { subscriptionIds });
      return subscriptionIds;

    } catch (error) {
      // Rollback any successful subscriptions
      subscriptionIds.forEach(id => {
        try {
          this.unsubscribeFromIndicator(id);
        } catch (rollbackError) {
          // Log but don't throw
          console.error('Rollback error:', rollbackError);
        }
      });

      throw error;
    }
  }

  /**
   * Stop all subscriptions and cleanup
   */
  public async shutdown(): Promise<void> {
    if (!this.isActive) return;

    // Stop batch processing
    if (this.batchTimer) {
      clearInterval(this.batchTimer);
      this.batchTimer = null;
    }

    // Unsubscribe from all indicators
    const subscriptionIds = Array.from(this.subscriptions.keys());
    subscriptionIds.forEach(id => {
      try {
        this.unsubscribeFromIndicator(id);
      } catch (error) {
        console.error('Error during shutdown:', error);
      }
    });

    // Unsubscribe from all data bus events
    this.dataBusSubscriptions.forEach(subId => {
      this.config.dataBus.unsubscribe(subId);
    });
    this.dataBusSubscriptions.clear();

    this.isActive = false;
    this.emit('shutdown');
  }

  // Private methods

  private setupDataBusListeners(): void {
    // Listen for market data updates
    this.config.dataBus.subscribe('market_data', (event) => {
      this.handleMarketDataUpdate(event);
    });

    // Listen for errors
    this.config.dataBus.subscribe('error', (event) => {
      this.emit('dataBusError', event);
    });
  }

  private setupChartEngineListeners(): void {
    // Listen for timeframe changes
    this.config.chartEngine.subscribeToTimeframeChange((timeframe) => {
      this.handleTimeframeChange(timeframe);
    });

    // Listen for symbol changes
    this.config.chartEngine.subscribeToSymbolChange((symbol) => {
      this.handleSymbolChange(symbol);
    });
  }

  private async handleMarketDataUpdate(event: DataBusEvent): Promise<void> {
    if (!this.isActive) return;

    const { symbol, data } = event;
    const timeframe = data.timeframe;

    // Find relevant subscriptions
    const relevantSubscriptions = Array.from(this.subscriptions.values()).filter(
      sub => sub.symbol === symbol && sub.timeframe === timeframe && sub.autoUpdate
    );

    if (relevantSubscriptions.length === 0) return;

    // Check if enough time has passed since last update
    const now = Date.now();
    const subscriptionsToUpdate = relevantSubscriptions.filter(
      sub => now - sub.lastUpdate >= sub.updateInterval
    );

    if (subscriptionsToUpdate.length === 0) return;

    if (this.config.enableBatching) {
      // Add to batch queue
      const batchKey = `${symbol}_${timeframe}`;
      this.batchQueue.set(batchKey, {
        data: data.candles || [data],
        timestamp: now
      });
    } else {
      // Process immediately
      await this.processIndicatorUpdates(subscriptionsToUpdate, data.candles || [data]);
    }
  }

  private async processIndicatorUpdates(
    subscriptions: IndicatorSubscription[],
    newData: CandleData[]
  ): Promise<void> {
    const batchConfigs = subscriptions.map(sub => sub.indicator);
    const symbol = subscriptions[0].symbol;
    const timeframe = subscriptions[0].timeframe;

    try {
      // Get historical data
      const historicalData = this.config.dataBus.getLatestData(symbol, timeframe, 1000);
      const completeData = [...historicalData, ...newData];

      // Calculate all indicators in batch
      const results = await this.config.indicatorEngine.calculateBatch(
        completeData,
        batchConfigs,
        { parallel: true, useCache: true }
      );

      // Update subscriptions and chart overlays
      subscriptions.forEach(subscription => {
        const indicatorResult = results[subscription.indicator.name];

        if (indicatorResult) {
          subscription.lastUpdate = Date.now();

          // Update chart overlay
          if (subscription.chartOverlayId) {
            this.config.chartEngine.updateIndicatorOverlay(
              subscription.chartOverlayId,
              indicatorResult
            );
          }

          // Emit update event
          this.emit('indicatorUpdated', {
            subscriptionId: subscription.id,
            symbol: subscription.symbol,
            timeframe: subscription.timeframe,
            indicator: subscription.indicator.name,
            data: indicatorResult,
            latestValue: indicatorResult[indicatorResult.length - 1]
          });
        }
      });

    } catch (error) {
      this.emit('error', {
        type: 'indicator_calculation',
        subscriptions: subscriptions.map(s => s.id),
        error
      });
    }
  }

  private startBatchProcessing(): void {
    this.batchTimer = setInterval(async () => {
      if (this.batchQueue.size === 0) return;

      const batches = new Map<string, IndicatorSubscription[]>();

      // Group subscriptions by symbol/timeframe
      this.batchQueue.forEach((queueData, batchKey) => {
        const [symbol, timeframe] = batchKey.split('_');
        const relevantSubs = Array.from(this.subscriptions.values()).filter(
          sub => sub.symbol === symbol && sub.timeframe === timeframe
        );

        if (relevantSubs.length > 0) {
          batches.set(batchKey, relevantSubs);
        }
      });

      // Process each batch
      for (const [batchKey, subscriptions] of batches) {
        const queueData = this.batchQueue.get(batchKey);
        if (queueData) {
          await this.processIndicatorUpdates(subscriptions, queueData.data);
        }
      }

      // Clear queue
      this.batchQueue.clear();
    }, this.config.batchInterval);
  }

  private handleTimeframeChange(newTimeframe: string): void {
    this.emit('timeframeChanged', { timeframe: newTimeframe });

    // Optionally update subscriptions for new timeframe
    // This would depend on the specific requirements
  }

  private handleSymbolChange(newSymbol: string): void {
    this.emit('symbolChanged', { symbol: newSymbol });

    // Optionally update subscriptions for new symbol
    // This would depend on the specific requirements
  }

  private generateSubscriptionId(symbol: string, timeframe: string, indicatorName: string): string {
    return `${symbol}_${timeframe}_${indicatorName}_${Date.now()}`;
  }
}

/**
 * Factory function to create data bus integration
 */
export function createIndicatorDataBusIntegration(
  indicatorEngine: IndicatorEngine,
  dataBus: DataBus,
  chartEngine: ChartEngine,
  config: Partial<DataBusIntegrationConfig> = {}
): IndicatorDataBusIntegration {
  return new IndicatorDataBusIntegration({
    indicatorEngine,
    dataBus,
    chartEngine,
    defaultUpdateInterval: 1000, // 1 second
    maxSubscriptions: 100,
    enableBatching: true,
    batchSize: 10,
    batchInterval: 500, // 500ms
    ...config
  });
}

/**
 * Hook for React components to use indicator data bus
 */
export function useIndicatorDataBus(integration: IndicatorDataBusIntegration) {
  const [subscriptions, setSubscriptions] = React.useState<IndicatorSubscription[]>([]);
  const [isConnected, setIsConnected] = React.useState(false);

  React.useEffect(() => {
    const handleSubscriptionCreated = () => {
      setSubscriptions(integration.getActiveSubscriptions());
    };

    const handleSubscriptionRemoved = () => {
      setSubscriptions(integration.getActiveSubscriptions());
    };

    const handleInitialized = () => {
      setIsConnected(true);
    };

    const handleShutdown = () => {
      setIsConnected(false);
      setSubscriptions([]);
    };

    integration.on('subscriptionCreated', handleSubscriptionCreated);
    integration.on('subscriptionRemoved', handleSubscriptionRemoved);
    integration.on('initialized', handleInitialized);
    integration.on('shutdown', handleShutdown);

    return () => {
      integration.off('subscriptionCreated', handleSubscriptionCreated);
      integration.off('subscriptionRemoved', handleSubscriptionRemoved);
      integration.off('initialized', handleInitialized);
      integration.off('shutdown', handleShutdown);
    };
  }, [integration]);

  return {
    subscriptions,
    isConnected,
    subscribe: integration.subscribeToIndicator.bind(integration),
    unsubscribe: integration.unsubscribeFromIndicator.bind(integration),
    updateConfig: integration.updateIndicatorConfig.bind(integration),
    getCurrentValue: integration.getCurrentIndicatorValue.bind(integration)
  };
}

// For environments without React
declare global {
  namespace React {
    function useState<T>(initialState: T): [T, (newState: T) => void];
    function useEffect(effect: () => void | (() => void), deps?: any[]): void;
  }
}
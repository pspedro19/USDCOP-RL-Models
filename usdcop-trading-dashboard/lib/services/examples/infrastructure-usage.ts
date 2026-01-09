/**
 * Infrastructure Components Usage Examples
 * Complete examples for using cache, logging, and health monitoring
 */

import {
  getSignalCache,
  getMetricsCache,
  type SignalCacheData,
  type MetricsCacheData,
} from '../cache';

import {
  getLogger,
  getAuditLogger,
  getPerformanceLogger,
} from '../logging';

import {
  getHealthChecker,
  getLatencyMonitor,
  getServiceRegistry,
  type ServiceConfig,
} from '../health';

// ============================================================================
// CACHE EXAMPLES
// ============================================================================

export async function cacheExamples() {
  console.log('\n=== Cache Examples ===\n');

  // Signal Cache
  const signalCache = getSignalCache();

  // Store latest signal
  const signal: SignalCacheData = {
    signal_id: 'sig_' + Date.now(),
    timestamp: new Date().toISOString(),
    symbol: 'USDCOP',
    action: 'BUY',
    confidence: 0.85,
    price: 4250.50,
    features: {
      rsi: 45.2,
      macd: 0.15,
      volume_ratio: 1.3,
      price_momentum: 0.02,
    },
    model_version: 'xgboost_v1.2',
    execution_latency_ms: 45,
  };

  await signalCache.setLatest(signal);
  console.log('✓ Latest signal cached');

  // Add to history
  await signalCache.addToHistory('USDCOP', signal);
  console.log('✓ Signal added to history');

  // Retrieve data
  const latest = await signalCache.getLatest();
  console.log('Latest signal:', latest?.signal_id);

  const history = await signalCache.getHistory('USDCOP', 10);
  console.log(`Signal history: ${history.length} entries`);

  const stats = await signalCache.getSignalStats('USDCOP');
  console.log('Signal stats:', stats);

  // Metrics Cache
  const metricsCache = getMetricsCache();

  const metrics: MetricsCacheData = {
    timestamp: new Date().toISOString(),
    total_pnl: 1250.75,
    win_rate: 0.62,
    sharpe_ratio: 1.8,
    max_drawdown: -0.15,
    total_trades: 45,
    active_positions: 3,
    portfolio_value: 50000,
  };

  await metricsCache.setFinancialMetrics(metrics);
  console.log('✓ Financial metrics cached');

  // Custom metrics
  await metricsCache.setCustomMetric('daily_volume', 1500000);
  await metricsCache.setCustomMetric('market_sentiment', 'bullish');
  console.log('✓ Custom metrics cached');

  const cachedMetrics = await metricsCache.getFinancialMetrics();
  console.log('Cached PnL:', cachedMetrics?.total_pnl);
}

// ============================================================================
// LOGGING EXAMPLES
// ============================================================================

export function loggingExamples() {
  console.log('\n=== Logging Examples ===\n');

  // Structured Logging
  const logger = getLogger({ service: 'TradingEngine', env: 'production' });

  logger.debug('Starting signal processing', { signal_id: 'sig_123' });
  logger.info('Signal received', { symbol: 'USDCOP' }, { confidence: 0.85 });
  logger.warn('High latency detected', { service: 'api' }, { latency_ms: 1200 });
  logger.error(
    'Trade execution failed',
    new Error('Connection timeout'),
    { signal_id: 'sig_123' }
  );

  // Child logger with additional context
  const userLogger = logger.child({ user_id: 'user_456', session_id: 'sess_789' });
  userLogger.info('User action', undefined, { action: 'execute_trade' });

  // Measure async operations
  async function exampleOperation() {
    await logger.time('fetch_market_data', async () => {
      // Simulate async operation
      await new Promise(resolve => setTimeout(resolve, 100));
      return { data: 'market_data' };
    });
  }

  console.log('✓ Structured logging examples completed');
}

export async function auditLoggingExamples() {
  console.log('\n=== Audit Logging Examples ===\n');

  const auditLogger = getAuditLogger();

  // Log signal generation
  await auditLogger.logSignalGenerated({
    signal_id: 'sig_' + Date.now(),
    symbol: 'USDCOP',
    action: 'BUY',
    confidence: 0.85,
    price: 4250.50,
    model_version: 'xgboost_v1.2',
  });
  console.log('✓ Signal generation logged');

  // Log position opened
  await auditLogger.logPositionOpened({
    position_id: 'pos_' + Date.now(),
    symbol: 'USDCOP',
    entry_price: 4250.50,
    quantity: 1000,
    entry_time: new Date().toISOString(),
  });
  console.log('✓ Position opened logged');

  // Log position closed
  await auditLogger.logPositionClosed(
    {
      position_id: 'pos_123',
      symbol: 'USDCOP',
      entry_price: 4250.50,
      exit_price: 4275.00,
      quantity: 1000,
    },
    24500, // PnL
    { reason: 'take_profit' }
  );
  console.log('✓ Position closed logged');

  // Log risk alert
  await auditLogger.logRiskAlert({
    type: 'max_drawdown',
    severity: 'high',
    message: 'Maximum drawdown exceeded',
    symbol: 'USDCOP',
    threshold: 0.15,
    current_value: 0.18,
  });
  console.log('✓ Risk alert logged');

  // Get audit trail
  const trail = await auditLogger.getAuditTrail({
    eventType: 'SIGNAL_GENERATED',
    limit: 10,
  });
  console.log(`Audit trail: ${trail.length} entries`);

  const stats = await auditLogger.getAuditStats();
  console.log('Audit stats:', stats);
}

export function performanceLoggingExamples() {
  console.log('\n=== Performance Logging Examples ===\n');

  const perfLogger = getPerformanceLogger();

  // Start and end operation
  const opId = perfLogger.startOperation('model_inference');
  // ... do work ...
  setTimeout(() => {
    perfLogger.endOperation(opId, true, { model: 'xgboost', samples: 1000 });
  }, 100);

  // Log latency directly
  perfLogger.logLatency('api_call', 125, { endpoint: '/api/signals' });
  perfLogger.logLatency('database_query', 45, { table: 'signals' });
  perfLogger.logLatency('model_inference', 230, { model: 'xgboost' });

  // Get metrics
  setTimeout(() => {
    const metrics = perfLogger.getMetrics('api_call');
    console.log('API call metrics:', metrics);

    const summary = perfLogger.getSummary();
    console.log('Performance summary:', summary);

    const slowOps = perfLogger.getSlowOperations(200);
    console.log(`Slow operations: ${slowOps.length}`);
  }, 200);

  console.log('✓ Performance logging examples completed');
}

// ============================================================================
// HEALTH MONITORING EXAMPLES
// ============================================================================

export async function healthMonitoringExamples() {
  console.log('\n=== Health Monitoring Examples ===\n');

  const registry = getServiceRegistry();
  const healthChecker = getHealthChecker();
  const latencyMonitor = getLatencyMonitor();

  // Register services
  const services: ServiceConfig[] = [
    {
      name: 'trading-api',
      url: process.env.TRADING_API_URL || 'http://localhost:8001/health',
      checkInterval: 30000,
      timeout: 5000,
    },
    {
      name: 'postgres',
      checkInterval: 60000,
      healthCheck: async () => {
        // Custom health check
        try {
          // Check database connection
          const isHealthy = true; // Replace with actual check
          return {
            success: isHealthy,
            latency_ms: 25,
            metadata: { connections: 10 },
          };
        } catch (error) {
          return {
            success: false,
            latency_ms: 0,
            error: (error as Error).message,
          };
        }
      },
    },
    {
      name: 'ml-model',
      checkInterval: 30000,
      healthCheck: async () => {
        // Check model availability
        return {
          success: true,
          latency_ms: 15,
          metadata: {
            model_version: 'v1.2',
            loaded: true,
          },
        };
      },
    },
  ];

  for (const service of services) {
    healthChecker.registerService(service);
  }
  console.log(`✓ Registered ${services.length} services`);

  // Check individual service
  const serviceHealth = await healthChecker.checkService('trading-api');
  console.log('Trading API status:', serviceHealth.status);

  // Check all services
  const systemHealth = await healthChecker.checkAllServices();
  console.log('System status:', systemHealth.overall_status);
  console.log('Services:', systemHealth.summary);

  // Record latency
  latencyMonitor.recordLatency('trading-api', 'execute_trade', 125, true, {
    symbol: 'USDCOP',
  });
  latencyMonitor.recordLatency('trading-api', 'fetch_signals', 85, true);
  latencyMonitor.recordLatency('ml-model', 'inference', 230, true);

  // Get latency stats
  const apiStats = latencyMonitor.getLatencyStats('trading-api', 'execute_trade');
  console.log('Trading API latency:', apiStats);

  const avgLatency = latencyMonitor.getAverageLatency('trading-api', 5);
  console.log('Average latency (5 min):', avgLatency);

  const highLatency = latencyMonitor.getHighLatencyServices(5);
  console.log('High latency services:', highLatency);

  // Start monitoring
  healthChecker.startMonitoring();
  console.log('✓ Health monitoring started');

  // Stop after 1 minute
  setTimeout(() => {
    healthChecker.stopMonitoring();
    console.log('✓ Health monitoring stopped');
  }, 60000);
}

// ============================================================================
// INTEGRATED EXAMPLE: COMPLETE TRADING WORKFLOW
// ============================================================================

export class TradingWorkflowExample {
  private signalCache = getSignalCache();
  private metricsCache = getMetricsCache();
  private logger = getLogger({ service: 'TradingWorkflow' });
  private auditLogger = getAuditLogger();
  private perfLogger = getPerformanceLogger();
  private latencyMonitor = getLatencyMonitor();

  async processSignal(rawSignal: any) {
    const opId = this.perfLogger.startOperation('process_signal');
    const startTime = Date.now();

    try {
      this.logger.info('Processing signal', {
        signal_id: rawSignal.signal_id,
        symbol: rawSignal.symbol,
      });

      // 1. Cache the signal
      const signal: SignalCacheData = {
        signal_id: rawSignal.signal_id,
        timestamp: new Date().toISOString(),
        symbol: rawSignal.symbol,
        action: rawSignal.action,
        confidence: rawSignal.confidence,
        price: rawSignal.price,
        features: rawSignal.features,
        model_version: rawSignal.model_version,
        execution_latency_ms: Date.now() - startTime,
      };

      await this.signalCache.setLatest(signal);
      await this.signalCache.addToHistory(signal.symbol, signal);

      // 2. Audit trail
      await this.auditLogger.logSignalGenerated(signal, {
        source: 'ml_model',
        environment: 'production',
      });

      // 3. Execute trade (simulated)
      const executionResult = await this.executeTrade(signal);

      // 4. Log execution
      await this.auditLogger.logSignalExecuted(signal, executionResult, {
        execution_venue: 'exchange_a',
      });

      // 5. Update metrics
      await this.updateMetrics(executionResult);

      // 6. Record latency
      const latency = Date.now() - startTime;
      this.latencyMonitor.recordLatency(
        'trading-workflow',
        'process_signal',
        latency,
        true,
        { symbol: signal.symbol }
      );

      this.perfLogger.endOperation(opId, true, {
        signal_id: signal.signal_id,
        latency_ms: latency,
      });

      this.logger.info('Signal processed successfully', {
        signal_id: signal.signal_id,
        latency_ms: latency,
      });

      return executionResult;
    } catch (error) {
      const latency = Date.now() - startTime;

      this.latencyMonitor.recordLatency(
        'trading-workflow',
        'process_signal',
        latency,
        false,
        { error: (error as Error).message }
      );

      this.perfLogger.endOperation(opId, false, {
        error: (error as Error).message,
      });

      this.logger.error('Failed to process signal', error as Error, {
        signal_id: rawSignal.signal_id,
      });

      throw error;
    }
  }

  private async executeTrade(signal: SignalCacheData) {
    // Simulate trade execution
    await new Promise(resolve => setTimeout(resolve, 50));

    return {
      execution_id: 'exec_' + Date.now(),
      signal_id: signal.signal_id,
      symbol: signal.symbol,
      price: signal.price + (Math.random() - 0.5) * 10,
      quantity: 1000,
      timestamp: new Date().toISOString(),
      status: 'filled',
    };
  }

  private async updateMetrics(execution: any) {
    // Simulate metrics calculation
    const currentMetrics = await this.metricsCache.getFinancialMetrics();

    const updatedMetrics: MetricsCacheData = {
      timestamp: new Date().toISOString(),
      total_pnl: (currentMetrics?.total_pnl || 0) + 125.50,
      win_rate: 0.62,
      sharpe_ratio: 1.8,
      max_drawdown: -0.15,
      total_trades: (currentMetrics?.total_trades || 0) + 1,
      active_positions: (currentMetrics?.active_positions || 0) + 1,
      portfolio_value: 50000,
    };

    await this.metricsCache.setFinancialMetrics(updatedMetrics);
  }

  async getSystemStatus() {
    const healthChecker = getHealthChecker();
    const systemHealth = await healthChecker.checkAllServices();

    const latencyStats = this.latencyMonitor.getLatencyStats('trading-workflow');
    const performanceSummary = this.perfLogger.getSummary();
    const auditStats = await this.auditLogger.getAuditStats();

    return {
      health: systemHealth,
      latency: latencyStats,
      performance: performanceSummary,
      audit: auditStats,
    };
  }
}

// ============================================================================
// RUN ALL EXAMPLES
// ============================================================================

export async function runAllExamples() {
  console.log('\n🚀 Running Infrastructure Examples\n');

  await cacheExamples();
  loggingExamples();
  await auditLoggingExamples();
  performanceLoggingExamples();
  await healthMonitoringExamples();

  console.log('\n=== Integrated Workflow Example ===\n');

  const workflow = new TradingWorkflowExample();

  const testSignal = {
    signal_id: 'sig_' + Date.now(),
    symbol: 'USDCOP',
    action: 'BUY',
    confidence: 0.85,
    price: 4250.50,
    features: { rsi: 45, macd: 0.15 },
    model_version: 'xgboost_v1.2',
  };

  await workflow.processSignal(testSignal);

  const status = await workflow.getSystemStatus();
  console.log('System status:', status);

  console.log('\n✅ All examples completed!\n');
}

// Run if executed directly
if (require.main === module) {
  runAllExamples().catch(console.error);
}

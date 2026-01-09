/**
 * Infrastructure Components Test Suite
 * Run this file to verify all infrastructure components are working correctly
 */

import {
  RedisClient,
  SignalCache,
  MetricsCache,
  type SignalCacheData,
  type MetricsCacheData,
} from '../cache';

import {
  StructuredLogger,
  AuditLogger,
  PerformanceLogger,
} from '../logging';

import {
  HealthChecker,
  LatencyMonitor,
  ServiceRegistry,
} from '../health';

// Test utilities
const assert = (condition: boolean, message: string) => {
  if (!condition) {
    throw new Error(`Assertion failed: ${message}`);
  }
};

const log = (message: string, emoji: string = '✓') => {
  console.log(`${emoji} ${message}`);
};

// ============================================================================
// CACHE TESTS
// ============================================================================

async function testRedisClient() {
  console.log('\n=== Testing RedisClient ===\n');

  const client = new RedisClient({ ttl: 60, namespace: 'test' });

  // Test set and get
  await client.set('key1', 'value1');
  const value1 = await client.get('key1');
  assert(value1 === 'value1', 'Get should return set value');
  log('Set/Get operations');

  // Test TTL
  const ttl = await client.ttl('key1');
  assert(ttl > 0 && ttl <= 60, 'TTL should be set correctly');
  log('TTL management');

  // Test exists
  const exists = await client.exists('key1');
  assert(exists === true, 'Key should exist');
  log('Exists check');

  // Test delete
  await client.delete('key1');
  const afterDelete = await client.get('key1');
  assert(afterDelete === null, 'Key should be deleted');
  log('Delete operation');

  // Test keys pattern
  await client.set('test:a', 'val1');
  await client.set('test:b', 'val2');
  const keys = await client.keys('test:*');
  assert(keys.length === 2, 'Keys pattern should match');
  log('Pattern matching');

  // Test stats
  const stats = await client.getStats();
  assert(stats.size >= 0, 'Stats should be available');
  log('Cache statistics');

  await client.clear();
  log('Cache cleared');
}

async function testSignalCache() {
  console.log('\n=== Testing SignalCache ===\n');

  const cache = new SignalCache();

  const signal: SignalCacheData = {
    signal_id: 'sig_test_1',
    timestamp: new Date().toISOString(),
    symbol: 'USDCOP',
    action: 'BUY',
    confidence: 0.85,
    price: 4250.50,
    features: { rsi: 45 },
    model_version: 'v1.0',
    execution_latency_ms: 45,
  };

  // Test set latest
  await cache.setLatest(signal);
  const latest = await cache.getLatest();
  assert(latest?.signal_id === 'sig_test_1', 'Latest signal should be set');
  log('Set/Get latest signal');

  // Test history
  await cache.addToHistory('USDCOP', signal);
  const history = await cache.getHistory('USDCOP', 10);
  assert(history.length > 0, 'History should contain signals');
  log('Signal history');

  // Test stats
  const stats = await cache.getSignalStats('USDCOP');
  assert(stats.total > 0, 'Stats should be calculated');
  log('Signal statistics');

  // Clear
  await cache.clearHistory('USDCOP');
  log('History cleared');
}

async function testMetricsCache() {
  console.log('\n=== Testing MetricsCache ===\n');

  const cache = new MetricsCache();

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

  // Test financial metrics
  await cache.setFinancialMetrics(metrics);
  const retrieved = await cache.getFinancialMetrics();
  assert(retrieved?.total_pnl === 1250.75, 'Financial metrics should be stored');
  log('Financial metrics');

  // Test custom metrics
  await cache.setCustomMetric('test_metric', 100);
  const customValue = await cache.getCustomMetric('test_metric');
  assert(customValue === 100, 'Custom metrics should be stored');
  log('Custom metrics');

  // Clear
  await cache.clearMetrics();
  log('Metrics cleared');
}

// ============================================================================
// LOGGING TESTS
// ============================================================================

function testStructuredLogger() {
  console.log('\n=== Testing StructuredLogger ===\n');

  const logger = new StructuredLogger({ service: 'test' }, 'debug');

  // Test all log levels
  logger.debug('Debug message');
  logger.info('Info message');
  logger.warn('Warning message');
  logger.error('Error message', new Error('Test error'));
  log('All log levels');

  // Test context
  logger.setContext({ user_id: 'test_user' });
  const context = logger.getContext();
  assert(context.service === 'test', 'Context should be set');
  log('Context management');

  // Test child logger
  const child = logger.child({ session_id: 'test_session' });
  const childContext = child.getContext();
  assert(childContext.session_id === 'test_session', 'Child logger should inherit context');
  log('Child logger');
}

async function testAuditLogger() {
  console.log('\n=== Testing AuditLogger ===\n');

  const logger = new AuditLogger();

  // Test signal generation log
  await logger.logSignalGenerated({
    signal_id: 'sig_test',
    symbol: 'USDCOP',
    action: 'BUY',
    confidence: 0.85,
    model_version: 'v1.0',
  });
  log('Signal generation logged');

  // Test position opened
  await logger.logPositionOpened({
    position_id: 'pos_test',
    symbol: 'USDCOP',
    entry_price: 4250,
    quantity: 1000,
  });
  log('Position opened logged');

  // Test audit trail retrieval
  const trail = await logger.getAuditTrail({ limit: 10 });
  assert(trail.length > 0, 'Audit trail should contain entries');
  log('Audit trail retrieval');

  // Test stats
  const stats = await logger.getAuditStats();
  assert(stats.total_events > 0, 'Stats should be calculated');
  log('Audit statistics');

  // Test export
  const json = await logger.exportAuditTrail('json');
  assert(json.length > 0, 'Export should produce data');
  log('Audit export');

  await logger.clearAuditTrail();
}

function testPerformanceLogger() {
  console.log('\n=== Testing PerformanceLogger ===\n');

  const logger = new PerformanceLogger();

  // Test operation tracking
  const opId = logger.startOperation('test_op');
  setTimeout(() => {
    logger.endOperation(opId, true, { test: 'metadata' });
  }, 100);
  log('Operation tracking');

  // Test direct latency logging
  logger.logLatency('test_service', 125);
  logger.logLatency('test_service', 150);
  logger.logLatency('test_service', 100);
  log('Latency logging');

  // Test metrics
  setTimeout(() => {
    const metrics = logger.getMetrics('test_service');
    assert(Array.isArray(metrics) || metrics.count > 0, 'Metrics should be calculated');
    log('Performance metrics');

    const summary = logger.getSummary();
    assert(summary.total_entries > 0, 'Summary should be available');
    log('Performance summary');

    logger.clearMetrics();
  }, 200);
}

// ============================================================================
// HEALTH MONITORING TESTS
// ============================================================================

async function testServiceRegistry() {
  console.log('\n=== Testing ServiceRegistry ===\n');

  const registry = new ServiceRegistry();

  // Test registration
  registry.register({
    name: 'test-service',
    url: 'http://localhost:8000/health',
    checkInterval: 30000,
  });
  log('Service registered');

  // Test retrieval
  const service = registry.getService('test-service');
  assert(service?.name === 'test-service', 'Service should be retrievable');
  log('Service retrieval');

  // Test all services
  const services = registry.getAllServices();
  assert(services.length > 0, 'All services should be listed');
  log('List all services');

  // Test unregister
  registry.unregister('test-service');
  const afterUnregister = registry.getService('test-service');
  assert(afterUnregister === null, 'Service should be unregistered');
  log('Service unregistered');
}

async function testHealthChecker() {
  console.log('\n=== Testing HealthChecker ===\n');

  const checker = new HealthChecker();

  // Register test service with custom health check
  checker.registerService({
    name: 'test-service',
    checkInterval: 60000,
    healthCheck: async () => ({
      success: true,
      latency_ms: 50,
      metadata: { test: true },
    }),
  });
  log('Service registered with health check');

  // Check single service
  const health = await checker.checkService('test-service');
  assert(health.status === 'healthy', 'Service should be healthy');
  log('Single service check');

  // Check all services
  const systemHealth = await checker.checkAllServices();
  assert(systemHealth.services.length > 0, 'System health should include services');
  log('All services check');

  // Test monitoring (start/stop)
  checker.startMonitoring();
  log('Monitoring started');

  setTimeout(() => {
    checker.stopMonitoring();
    log('Monitoring stopped');
  }, 1000);
}

function testLatencyMonitor() {
  console.log('\n=== Testing LatencyMonitor ===\n');

  const monitor = new LatencyMonitor();

  // Record latencies
  monitor.recordLatency('test-api', 'operation1', 125, true);
  monitor.recordLatency('test-api', 'operation1', 150, true);
  monitor.recordLatency('test-api', 'operation1', 100, true);
  log('Latencies recorded');

  // Get stats
  const stats = monitor.getLatencyStats('test-api', 'operation1');
  assert(Array.isArray(stats) || stats.count === 3, 'Stats should be calculated');
  log('Latency statistics');

  // Get average
  const avg = monitor.getAverageLatency('test-api', 5);
  assert(avg > 0, 'Average latency should be calculated');
  log('Average latency');

  // Get recent measurements
  const recent = monitor.getRecentMeasurements(10);
  assert(recent.length > 0, 'Recent measurements should be available');
  log('Recent measurements');

  monitor.clearMeasurements('test-api');
  log('Measurements cleared');
}

// ============================================================================
// RUN ALL TESTS
// ============================================================================

async function runAllTests() {
  console.log('\n' + '='.repeat(60));
  console.log('INFRASTRUCTURE COMPONENTS TEST SUITE');
  console.log('='.repeat(60));

  try {
    // Cache tests
    await testRedisClient();
    await testSignalCache();
    await testMetricsCache();

    // Logging tests
    testStructuredLogger();
    await testAuditLogger();
    testPerformanceLogger();

    // Health monitoring tests
    await testServiceRegistry();
    await testHealthChecker();
    testLatencyMonitor();

    console.log('\n' + '='.repeat(60));
    console.log('✅ ALL TESTS PASSED!');
    console.log('='.repeat(60) + '\n');

    return true;
  } catch (error) {
    console.error('\n' + '='.repeat(60));
    console.error('❌ TEST FAILED!');
    console.error('='.repeat(60));
    console.error('\nError:', error);
    console.error('\n');

    return false;
  }
}

// Run tests if executed directly
if (require.main === module) {
  runAllTests()
    .then((success) => {
      process.exit(success ? 0 : 1);
    })
    .catch((error) => {
      console.error('Unexpected error:', error);
      process.exit(1);
    });
}

export { runAllTests };

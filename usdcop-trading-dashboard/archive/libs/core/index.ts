/**
 * Core Module Barrel Export
 * Elite Trading Platform Core Architecture
 */

// Type definitions
export * from './types';

// Data Bus
export * from './data-bus';

// Event Management
export * from './event-manager';

// WebSocket Management
export * from './websocket';

// Performance Monitoring
export * from './performance';

// Configuration
export * from './config';

// Core initialization function
export async function initializeTradingPlatform() {
  const { getConfig } = await import('./config');
  const { getDataBus } = await import('./data-bus');
  const { getEventManager } = await import('./event-manager');
  const { getPerformanceMonitor } = await import('./performance');

  const config = getConfig();

  // Initialize core systems
  const dataBus = getDataBus(config.dataBus);
  const eventManager = getEventManager(config.eventManager);
  const performanceMonitor = getPerformanceMonitor(config.performance);

  // Start monitoring
  performanceMonitor.start();

  console.log('🚀 Elite Trading Platform Core Initialized');

  return {
    dataBus,
    eventManager,
    performanceMonitor,
    config
  };
}
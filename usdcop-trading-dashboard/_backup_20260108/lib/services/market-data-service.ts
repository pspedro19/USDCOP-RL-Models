/**
 * Market Data Service - Backwards Compatibility Layer
 * ===================================================
 *
 * This file maintains backwards compatibility with the original API.
 * All functionality has been refactored into separate modules following
 * the Single Responsibility Principle.
 *
 * New modular structure in lib/services/market-data/:
 * - WebSocketConnector.ts - WebSocket lifecycle management
 * - MarketDataFetcher.ts - REST API data retrieval
 * - DataTransformer.ts - Data formatting and transformation
 * - StatisticsCalculator.ts - Market statistics calculations
 * - index.ts - Unified facade and exports
 *
 * @deprecated Import from '@/lib/services/market-data' for better tree-shaking
 */

// Re-export everything from the new modular structure
export * from './market-data'

// Re-export the main service class as default for backwards compatibility
export { MarketDataService as default } from './market-data'

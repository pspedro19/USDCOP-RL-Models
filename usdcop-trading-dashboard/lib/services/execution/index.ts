/**
 * Execution Module Services - SignalBridge Integration
 * =====================================================
 *
 * Central export point for all execution services.
 *
 * @version 1.0.0
 * @lastSync 2026-01-22
 */

// API Client
export { api } from './api';

// Auth Service
export { authService } from './auth.service';

// Exchange Service
export { exchangeService, ExchangeServiceError } from './exchange.service';

// Signal Bridge Service
export { signalBridgeService, SignalBridgeError } from './signal-bridge.service';

// Execution Service
export { executionService } from './execution.service';

// Signal Service
export { signalService } from './signal.service';

// Trading Config Service
export { tradingConfigService } from './trading-config.service';

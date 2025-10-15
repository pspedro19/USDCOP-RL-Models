/**
 * Configuration Module Exports
 */

export {
  ConfigManager,
  configManager,
  getConfig,
  getEnvironmentConfig,
  isFeatureEnabled
} from './TradingPlatformConfig';

export type {
  TradingPlatformConfig,
  FeatureFlags,
  UIConfig,
  TradingConfig,
  RiskLimitsConfig,
  MarketConfig,
  DataProviderConfig
} from './TradingPlatformConfig';
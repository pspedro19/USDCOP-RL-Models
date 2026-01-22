/**
 * SSOT Contract - Single Source of Truth for Frontend
 * ====================================================
 *
 * This file mirrors the backend SSOT values from:
 * - src/core/contracts/feature_contract.py
 * - src/core/contracts/action_contract.py
 * - src/core/constants.py
 * - src/training/config.py
 *
 * IMPORTANT: These values MUST match the backend.
 * Any changes to backend SSOT should be reflected here.
 *
 * The frontend can:
 * 1. Use these constants directly for validation
 * 2. Fetch /api/ssot to verify at runtime
 * 3. Display these values in the UI for transparency
 *
 * @version 2.0.0
 * @lastSync 2026-01-18
 */

// ============================================================================
// FEATURE CONTRACT (from src/core/contracts/feature_contract.py)
// ============================================================================

/**
 * Canonical feature order - IMMUTABLE
 * Index 0-12: Market features
 * Index 13-14: State features (added by environment)
 */
export const FEATURE_ORDER = [
  // Market Features (0-12)
  'log_ret_5m',      // 0: 5-minute log return
  'log_ret_1h',      // 1: 1-hour log return
  'log_ret_4h',      // 2: 4-hour log return
  'rsi_9',           // 3: RSI with period 9
  'atr_pct',         // 4: ATR as percentage of price
  'adx_14',          // 5: ADX with period 14
  'dxy_z',           // 6: Dollar Index z-score
  'dxy_change_1d',   // 7: DXY daily change
  'vix_z',           // 8: VIX z-score
  'embi_z',          // 9: EMBI Colombia z-score
  'brent_change_1d', // 10: Brent oil daily change
  'rate_spread',     // 11: COL-USA rate differential
  'usdmxn_change_1d',// 12: USD/MXN daily change (EM proxy)
  // State Features (13-14) - added by environment
  'position',        // 13: Current position (-1, 0, 1)
  'time_normalized', // 14: Time of day normalized (0-1)
] as const;

export type FeatureName = typeof FEATURE_ORDER[number];

/**
 * Observation dimension - MUST BE 15
 * 13 market features + 2 state features
 */
export const OBSERVATION_DIM = 15 as const;

/**
 * Market features count (without state)
 */
export const MARKET_FEATURES_COUNT = 13 as const;

/**
 * State features count
 */
export const STATE_FEATURES_COUNT = 2 as const;

/**
 * Feature contract version
 */
export const FEATURE_CONTRACT_VERSION = '2.0.0' as const;

// ============================================================================
// ACTION CONTRACT (from src/core/contracts/action_contract.py)
// ============================================================================

/**
 * Action enum values - MUST match backend IntEnum
 */
export const Action = {
  SELL: 0,
  HOLD: 1,
  BUY: 2,
} as const;

export type ActionType = typeof Action[keyof typeof Action];

/**
 * Action names for display
 */
export const ACTION_NAMES: Record<ActionType, string> = {
  [Action.SELL]: 'SELL',
  [Action.HOLD]: 'HOLD',
  [Action.BUY]: 'BUY',
};

/**
 * Action count - MUST BE 3
 */
export const ACTION_COUNT = 3 as const;

/**
 * Valid actions tuple
 */
export const VALID_ACTIONS = [0, 1, 2] as const;

/**
 * Action contract version
 */
export const ACTION_CONTRACT_VERSION = '1.0.0' as const;

// ============================================================================
// INDICATOR PERIODS (from src/core/constants.py)
// ============================================================================

/**
 * RSI period - MUST match backend
 */
export const RSI_PERIOD = 9 as const;

/**
 * ATR period - MUST match backend
 */
export const ATR_PERIOD = 10 as const;

/**
 * ADX period - MUST match backend
 */
export const ADX_PERIOD = 14 as const;

/**
 * Warmup bars required (max of indicator periods)
 */
export const WARMUP_BARS = 14 as const;

// ============================================================================
// NORMALIZATION CONSTANTS (from src/core/constants.py)
// ============================================================================

/**
 * Feature clipping bounds
 */
export const CLIP_MIN = -5.0 as const;
export const CLIP_MAX = 5.0 as const;

// ============================================================================
// ACTION THRESHOLDS (from src/training/config.py)
// ============================================================================

/**
 * Threshold for LONG signal
 * If action > THRESHOLD_LONG → BUY
 */
export const THRESHOLD_LONG = 0.33 as const;

/**
 * Threshold for SHORT signal
 * If action < THRESHOLD_SHORT → SELL
 * Otherwise → HOLD
 */
export const THRESHOLD_SHORT = -0.33 as const;

// ============================================================================
// MARKET HOURS (from src/core/constants.py)
// ============================================================================

export const TRADING_TIMEZONE = 'America/Bogota' as const;
export const TRADING_START_HOUR = 8 as const;
export const TRADING_END_HOUR = 17 as const;
export const UTC_OFFSET_BOGOTA = -5 as const;

// ============================================================================
// RISK MANAGEMENT (from src/core/constants.py)
// ============================================================================

export const MIN_CONFIDENCE_THRESHOLD = 0.6 as const;
export const HIGH_CONFIDENCE_THRESHOLD = 0.8 as const;
export const MAX_POSITION_SIZE = 1.0 as const;
export const DEFAULT_STOP_LOSS_PCT = 0.02 as const;
export const MAX_DRAWDOWN_PCT = 0.15 as const;

// ============================================================================
// TRAINING DEFAULTS (from src/training/config.py)
// ============================================================================

export const PPO_DEFAULTS = {
  learning_rate: 3e-4,
  n_steps: 2048,
  batch_size: 64,
  n_epochs: 10,
  gamma: 0.90,
  gae_lambda: 0.95,
  clip_range: 0.2,
  ent_coef: 0.05,
  total_timesteps: 500_000,
  seed: 42,
} as const;

export const NETWORK_DEFAULTS = {
  policy_layers: [256, 256],
  value_layers: [256, 256],
  activation: 'Tanh',
} as const;

export const ENVIRONMENT_DEFAULTS = {
  initial_capital: 10_000.0,
  transaction_cost_bps: 75.0,
  slippage_bps: 15.0,
  max_episode_steps: 2000,
} as const;

// ============================================================================
// SUPPORTED ALGORITHMS (from backend registry)
// ============================================================================

/**
 * RL algorithms supported by the system
 */
export const RL_ALGORITHMS = ['PPO', 'SAC', 'TD3', 'A2C', 'DQN'] as const;
export type RLAlgorithm = typeof RL_ALGORITHMS[number];

/**
 * ML (non-RL) algorithms supported
 */
export const ML_ALGORITHMS = ['LGBM', 'XGB', 'RF', 'LINEAR'] as const;
export type MLAlgorithm = typeof ML_ALGORITHMS[number];

/**
 * All supported algorithms
 */
export const ALL_ALGORITHMS = [...RL_ALGORITHMS, ...ML_ALGORITHMS] as const;
export type Algorithm = typeof ALL_ALGORITHMS[number];

/**
 * Algorithm display names
 */
export const ALGORITHM_NAMES: Record<Algorithm, string> = {
  PPO: 'Proximal Policy Optimization',
  SAC: 'Soft Actor-Critic',
  TD3: 'Twin Delayed DDPG',
  A2C: 'Advantage Actor-Critic',
  DQN: 'Deep Q-Network',
  LGBM: 'LightGBM',
  XGB: 'XGBoost',
  RF: 'Random Forest',
  LINEAR: 'Linear Model',
};

/**
 * Algorithm colors for UI
 */
export const ALGORITHM_COLORS: Record<Algorithm, string> = {
  PPO: '#10B981',   // Emerald
  SAC: '#8B5CF6',   // Violet
  TD3: '#F59E0B',   // Amber
  A2C: '#EF4444',   // Red
  DQN: '#06B6D4',   // Cyan
  LGBM: '#3B82F6',  // Blue
  XGB: '#EC4899',   // Pink
  RF: '#84CC16',    // Lime
  LINEAR: '#6B7280', // Gray
};

/**
 * Check if algorithm is RL-based
 */
export function isRLAlgorithm(algo: string): algo is RLAlgorithm {
  return RL_ALGORITHMS.includes(algo as RLAlgorithm);
}

/**
 * Check if algorithm is ML-based
 */
export function isMLAlgorithm(algo: string): algo is MLAlgorithm {
  return ML_ALGORITHMS.includes(algo as MLAlgorithm);
}

// ============================================================================
// MODEL STATUS (from database schema)
// ============================================================================

/**
 * Valid model statuses from model_registry table
 */
export const MODEL_STATUSES = ['registered', 'deployed', 'retired'] as const;
export type ModelStatus = typeof MODEL_STATUSES[number];

/**
 * Status display names
 */
export const STATUS_NAMES: Record<ModelStatus, string> = {
  registered: 'Testing',
  deployed: 'Production',
  retired: 'Retired',
};

/**
 * Status colors for UI
 */
export const STATUS_COLORS: Record<ModelStatus, string> = {
  registered: 'text-yellow-500',
  deployed: 'text-green-500',
  retired: 'text-gray-500',
};

// ============================================================================
// TRADE SIDES (from action contract)
// ============================================================================

/**
 * Valid trade sides
 */
export const TRADE_SIDES = ['BUY', 'SELL'] as const;
export type TradeSide = typeof TRADE_SIDES[number];

// ============================================================================
// TIME INTERVALS (from data pipeline)
// ============================================================================

/**
 * Supported OHLCV intervals
 */
export const OHLCV_INTERVALS = ['1m', '5m', '15m', '1h', '4h', '1d'] as const;
export type OHLCVInterval = typeof OHLCV_INTERVALS[number];

/**
 * Primary interval used for trading
 */
export const PRIMARY_INTERVAL = '5m' as const;

/**
 * Bars per time period (for 5m interval)
 */
export const BARS_PER_HOUR = 12 as const;
export const BARS_PER_DAY = 288 as const;
export const BARS_PER_WEEK = 2016 as const;

// ============================================================================
// METRICS (from evaluation)
// ============================================================================

/**
 * Primary metrics for model evaluation
 */
export const PRIMARY_METRICS = [
  'sharpe_ratio',
  'total_return',
  'max_drawdown',
  'win_rate',
  'profit_factor',
  'sortino_ratio',
] as const;
export type PrimaryMetric = typeof PRIMARY_METRICS[number];

/**
 * Metric display names
 */
export const METRIC_NAMES: Record<PrimaryMetric, string> = {
  sharpe_ratio: 'Sharpe Ratio',
  total_return: 'Total Return',
  max_drawdown: 'Max Drawdown',
  win_rate: 'Win Rate',
  profit_factor: 'Profit Factor',
  sortino_ratio: 'Sortino Ratio',
};

/**
 * Metric format hints (for display)
 */
export const METRIC_FORMATS: Record<PrimaryMetric, 'number' | 'percent' | 'ratio'> = {
  sharpe_ratio: 'ratio',
  total_return: 'percent',
  max_drawdown: 'percent',
  win_rate: 'percent',
  profit_factor: 'ratio',
  sortino_ratio: 'ratio',
};

// ============================================================================
// A/B TESTING FEATURE SETS
// ============================================================================

/**
 * Full macro feature set (7 macro features)
 */
export const MACRO_FEATURES_FULL = [
  'dxy_z',
  'dxy_change_1d',
  'vix_z',
  'embi_z',
  'brent_change_1d',
  'rate_spread',
  'usdmxn_change_1d',
] as const;

/**
 * Core macro feature set (4 macro features)
 * Used in reduced experiments
 */
export const MACRO_FEATURES_CORE = [
  'dxy_z',
  'vix_z',
  'embi_z',
  'brent_change_1d',
] as const;

/**
 * Technical features (always included)
 */
export const TECHNICAL_FEATURES = [
  'log_ret_5m',
  'log_ret_1h',
  'log_ret_4h',
  'rsi_9',
  'atr_pct',
  'adx_14',
] as const;

// ============================================================================
// PIPELINE DATE RANGES
// ============================================================================
//
// SSOT HIERARCHY FOR DATES:
// 1. Backend SSOT: config/date_ranges.yaml (AUTHORITATIVE)
// 2. Backend Config: config/trading_config.yaml (reads from date_ranges.yaml)
// 3. Experiment YAMLs: config/experiments/*.yaml (reference date_ranges.yaml)
// 4. Frontend API: /api/pipeline/dates (reads from trading_config.yaml)
// 5. Frontend SSOT: This file (FALLBACK ONLY when API unavailable)
//
// IMPORTANT: Always prefer fetching from /api/pipeline/dates or /api/v1/ssot
// These constants are FALLBACK defaults that should match date_ranges.yaml
// ============================================================================

/**
 * Default pipeline date ranges - FALLBACK VALUES
 *
 * AUTHORITATIVE SOURCE: config/date_ranges.yaml
 *
 * These values are used only when:
 * - API is unavailable
 * - During initial render before API response
 *
 * For production use, always fetch from /api/pipeline/dates
 *
 * @see config/date_ranges.yaml - Backend SSOT for all date ranges
 * @see /api/pipeline/dates - API endpoint that reads from backend config
 */
export const PIPELINE_DATE_RANGES = {
  /** First available data date (from date_ranges.yaml: data.start) */
  DATA_START: '2020-03-01',

  /**
   * Production training period (from date_ranges.yaml: training.*)
   * Full historical data for maximum pattern learning
   * DO NOT backtest here - causes data leakage!
   */
  TRAINING_START: '2020-03-01',
  TRAINING_END: '2024-12-31',

  /**
   * Experiment training period (from date_ranges.yaml: experiment_training.*)
   * Shorter window used for A/B testing experiments
   * Note: Different from production training dates!
   */
  EXPERIMENT_TRAINING_START: '2023-01-01',
  EXPERIMENT_TRAINING_END: '2024-12-31',

  /**
   * Validation period (from date_ranges.yaml: validation.*)
   * Used for hyperparameter tuning - may have bias
   */
  VALIDATION_START: '2025-01-01',
  VALIDATION_END: '2025-06-30',

  /**
   * Test period (from date_ranges.yaml: test.*)
   * True out-of-sample - most realistic performance
   */
  TEST_START: '2025-07-01',
  // TEST_END is dynamic (today's date)
} as const;

/**
 * Get current test end date (today)
 */
export function getTestEndDate(): string {
  return new Date().toISOString().split('T')[0];
}

/**
 * Backtest preset types
 */
export const BACKTEST_PRESETS = ['validation', 'test', 'both', 'custom'] as const;
export type BacktestPreset = typeof BACKTEST_PRESETS[number];

/**
 * Backtest preset configurations
 */
export const BACKTEST_PRESET_CONFIG: Record<BacktestPreset, {
  label: string;
  labelEs: string;
  description: string;
  descriptionEs: string;
}> = {
  validation: {
    label: 'Validation',
    labelEs: 'Validación',
    description: 'Validation period (may have tuning bias)',
    descriptionEs: 'Período de validación (puede tener sesgo de tuning)',
  },
  test: {
    label: 'Test',
    labelEs: 'Test',
    description: 'Out-of-sample test period (most realistic)',
    descriptionEs: 'Período de prueba fuera de muestra (más realista)',
  },
  both: {
    label: 'Both',
    labelEs: 'Ambos',
    description: 'Validation + Test combined',
    descriptionEs: 'Validación + Test combinados',
  },
  custom: {
    label: 'Custom',
    labelEs: 'Personalizado',
    description: 'Custom date range',
    descriptionEs: 'Rango de fechas personalizado',
  },
};

/**
 * Get date range for a preset
 */
export function getPresetDateRange(preset: BacktestPreset): { startDate: string; endDate: string } {
  const today = getTestEndDate();

  switch (preset) {
    case 'validation':
      return {
        startDate: PIPELINE_DATE_RANGES.VALIDATION_START,
        endDate: PIPELINE_DATE_RANGES.VALIDATION_END,
      };
    case 'test':
      return {
        startDate: PIPELINE_DATE_RANGES.TEST_START,
        endDate: today,
      };
    case 'both':
      return {
        startDate: PIPELINE_DATE_RANGES.VALIDATION_START,
        endDate: today,
      };
    case 'custom':
    default:
      return {
        startDate: PIPELINE_DATE_RANGES.VALIDATION_START,
        endDate: today,
      };
  }
}

// ============================================================================
// HASH COMPUTATION
// ============================================================================

/**
 * Compute a simple hash for feature order validation
 * Note: For production, this should match the backend's hash computation
 * Backend uses: hashlib.sha256(json.dumps(feature_order).encode()).hexdigest()[:16]
 */
export function computeFeatureOrderHash(features: readonly string[]): string {
  // Simple implementation - join features and create a hash-like string
  // For full compatibility with backend, you would need a proper SHA256 implementation
  return features.join('|');
}

/**
 * Get the pre-computed feature order hash
 * This should be updated whenever FEATURE_ORDER changes
 */
export const FEATURE_ORDER_HASH = computeFeatureOrderHash(FEATURE_ORDER);

// ============================================================================
// VALIDATION FUNCTIONS
// ============================================================================

/**
 * Validate observation dimension
 */
export function validateObservationDim(dim: number): boolean {
  return dim === OBSERVATION_DIM;
}

/**
 * Validate action value
 */
export function validateAction(action: number): action is ActionType {
  return VALID_ACTIONS.includes(action as typeof VALID_ACTIONS[number]);
}

/**
 * Validate feature order matches SSOT
 */
export function validateFeatureOrder(features: string[]): boolean {
  if (features.length !== FEATURE_ORDER.length) return false;
  return features.every((f, i) => f === FEATURE_ORDER[i]);
}

/**
 * Get action name from value
 */
export function getActionName(action: ActionType): string {
  return ACTION_NAMES[action] ?? 'UNKNOWN';
}

/**
 * Determine action from continuous value
 */
export function discretizeAction(value: number): ActionType {
  if (value > THRESHOLD_LONG) return Action.BUY;
  if (value < THRESHOLD_SHORT) return Action.SELL;
  return Action.HOLD;
}

// ============================================================================
// SSOT SUMMARY (for API endpoint)
// ============================================================================

export const SSOT_SUMMARY = {
  feature_contract: {
    version: FEATURE_CONTRACT_VERSION,
    feature_order: FEATURE_ORDER,
    observation_dim: OBSERVATION_DIM,
    market_features_count: MARKET_FEATURES_COUNT,
    state_features_count: STATE_FEATURES_COUNT,
  },
  action_contract: {
    version: ACTION_CONTRACT_VERSION,
    actions: Action,
    action_count: ACTION_COUNT,
    action_names: ACTION_NAMES,
  },
  indicators: {
    rsi_period: RSI_PERIOD,
    atr_period: ATR_PERIOD,
    adx_period: ADX_PERIOD,
    warmup_bars: WARMUP_BARS,
  },
  normalization: {
    clip_min: CLIP_MIN,
    clip_max: CLIP_MAX,
  },
  thresholds: {
    long: THRESHOLD_LONG,
    short: THRESHOLD_SHORT,
  },
  risk: {
    min_confidence: MIN_CONFIDENCE_THRESHOLD,
    high_confidence: HIGH_CONFIDENCE_THRESHOLD,
    max_position_size: MAX_POSITION_SIZE,
    default_stop_loss_pct: DEFAULT_STOP_LOSS_PCT,
    max_drawdown_pct: MAX_DRAWDOWN_PCT,
  },
  market_hours: {
    timezone: TRADING_TIMEZONE,
    start_hour: TRADING_START_HOUR,
    end_hour: TRADING_END_HOUR,
    utc_offset: UTC_OFFSET_BOGOTA,
  },
  algorithms: {
    rl: RL_ALGORITHMS,
    ml: ML_ALGORITHMS,
    all: ALL_ALGORITHMS,
    names: ALGORITHM_NAMES,
    colors: ALGORITHM_COLORS,
  },
  model_statuses: {
    values: MODEL_STATUSES,
    names: STATUS_NAMES,
    colors: STATUS_COLORS,
  },
  intervals: {
    supported: OHLCV_INTERVALS,
    primary: PRIMARY_INTERVAL,
    bars_per_hour: BARS_PER_HOUR,
    bars_per_day: BARS_PER_DAY,
  },
  metrics: {
    primary: PRIMARY_METRICS,
    names: METRIC_NAMES,
    formats: METRIC_FORMATS,
  },
  trade_sides: TRADE_SIDES,
  pipeline_dates: {
    data_start: PIPELINE_DATE_RANGES.DATA_START,
    training_start: PIPELINE_DATE_RANGES.TRAINING_START,
    training_end: PIPELINE_DATE_RANGES.TRAINING_END,
    validation_start: PIPELINE_DATE_RANGES.VALIDATION_START,
    validation_end: PIPELINE_DATE_RANGES.VALIDATION_END,
    test_start: PIPELINE_DATE_RANGES.TEST_START,
  },
  backtest_presets: {
    presets: BACKTEST_PRESETS,
    config: BACKTEST_PRESET_CONFIG,
  },
} as const;

export type SSOTSummary = typeof SSOT_SUMMARY;

// ============================================================================
// EXPORTS
// ============================================================================

export default SSOT_SUMMARY;

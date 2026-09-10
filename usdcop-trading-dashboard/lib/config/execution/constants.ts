/**
 * Execution Module Constants - SignalBridge Integration
 * ======================================================
 *
 * Constants for the execution module. Integrates with dashboard SSOT.
 * Uses Next.js environment variables.
 *
 * @version 1.0.0
 * @lastSync 2026-01-22
 */

// Re-export from SSOT for consistency
export {
  Action as TRADING_ACTIONS,
  ACTION_NAMES,
  TRADE_SIDES,
} from '@/lib/contracts/ssot.contract';

// Re-export from exchange contract
export {
  SUPPORTED_EXCHANGES,
  EXCHANGE_METADATA,
  getExchangeDisplayName,
  type SupportedExchange,
} from '@/lib/contracts/execution/exchange.contract';

// Re-export from auth contract
export {
  SUBSCRIPTION_TIERS,
  SUBSCRIPTION_TIER_LIMITS,
  RISK_PROFILES,
  RISK_PROFILE_MULTIPLIERS,
} from '@/lib/contracts/execution/auth.contract';

// Re-export from execution contract
export {
  ORDER_STATUSES,
  ORDER_STATUS_COLORS,
  ORDER_STATUS_NAMES,
} from '@/lib/contracts/execution/execution.contract';

// ============================================================================
// APPLICATION CONFIG
// ============================================================================

export const APP_NAME = 'SignalBridge';

/**
 * API Base URL - uses Next.js environment variables
 */
export const API_BASE_URL = process.env.NEXT_PUBLIC_SIGNALBRIDGE_API_URL || '/api/execution';

/**
 * WebSocket URL — el fallback es el puerto que el compose PUBLICA EN EL HOST, no el
 * interno del contenedor.
 *
 * Medido el 2026-08-05: `docker-compose.yml` mapea signalbridge `8085:8000` y **no
 * declara `NEXT_PUBLIC_SIGNALBRIDGE_WS_URL` para el dashboard**, así que este fallback
 * era el valor efectivo. Apuntaba a `ws://localhost:8000`, un puerto que nadie publica
 * en el host — y esta URL la resuelve el NAVEGADOR, donde `localhost` es la máquina del
 * usuario, no la red de compose. Resultado: `ERR_CONNECTION_REFUSED` garantizado,
 * observado en la corrida E2E de BL-05.
 *
 * 8000 es el puerto INTERNO del contenedor; sólo sería correcto para un cliente que
 * viva dentro de la red de compose, y este no lo es.
 */
export const WS_URL = process.env.NEXT_PUBLIC_SIGNALBRIDGE_WS_URL || 'ws://localhost:8085/ws';

/**
 * Mock mode flag
 */
export const MOCK_MODE = process.env.NEXT_PUBLIC_MOCK_MODE === 'true';

// ============================================================================
// TRADING DEFAULTS
// ============================================================================

export const DEFAULT_TRADING_CONFIG = {
  symbol: 'USD/COP',
  is_enabled: false,
  stop_loss_pct: 0.02,
  take_profit_pct: 0.05,
  min_confidence: 0.70,
  execute_on_exchanges: ['mexc', 'binance'] as const,
};

// ============================================================================
// ROUTES (for execution module, prefixed with /execution)
// ============================================================================

export const EXECUTION_ROUTES = {
  HOME: '/execution',
  LOGIN: '/execution/login',
  REGISTER: '/execution/register',
  FORGOT_PASSWORD: '/execution/forgot-password',
  RESET_PASSWORD: '/execution/reset-password/[token]',
  DASHBOARD: '/execution/dashboard',
  EXCHANGES: '/execution/exchanges',
  CONNECT_EXCHANGE: '/execution/exchanges/connect/[exchange]',
  TRADING: '/execution/trading',
  SIGNALS: '/execution/signals',
  SIGNAL_DETAIL: '/execution/signals/[id]',
  EXECUTIONS: '/execution/executions',
  EXECUTION_DETAIL: '/execution/executions/[id]',
  PORTFOLIO: '/execution/portfolio',
  SETTINGS: '/execution/settings',
  SETTINGS_PROFILE: '/execution/settings/profile',
  SETTINGS_SECURITY: '/execution/settings/security',
  SETTINGS_NOTIFICATIONS: '/execution/settings/notifications',
} as const;

/**
 * @deprecated Use EXECUTION_ROUTES instead
 */
export const ROUTES = EXECUTION_ROUTES;

// ============================================================================
// UI HELPERS
// ============================================================================

export const ACTION_ICONS: Record<number, string> = {
  0: '🔴', // SELL
  1: '⚪', // HOLD
  2: '🟢', // BUY
};

export const RISK_LIMITS: Record<string, { maxPositionPct: number; description: string }> = {
  conservative: { maxPositionPct: 0.01, description: 'Max 1% per trade' },
  moderate: { maxPositionPct: 0.02, description: 'Max 2% per trade' },
  aggressive: { maxPositionPct: 0.05, description: 'Max 5% per trade' },
};

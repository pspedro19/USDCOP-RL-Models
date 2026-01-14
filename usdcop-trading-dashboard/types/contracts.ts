// types/contracts.ts
// =============================================================================
// SHARED API CONTRACTS - FRONTEND IMPLEMENTATION
// Consolidated types based on audit report
//
// NOTE: For new code, prefer importing from '@/types/schemas' which provides
// Zod-validated types. This file is maintained for backwards compatibility.
// =============================================================================

// Re-export existing compatible types from trading.ts to avoid duplication
import { MarketStatus, OrderSide } from './trading';
export { MarketStatus };
export { OrderSide as TradeSide }; // Alias for contract compatibility

// Re-export interfaces that match or can be mapped
export type { Position, Trade } from './trading';

// Import canonical SignalType from schemas (BUY/SELL/HOLD - matches backend)
// Note: Legacy code may use LONG/SHORT, prefer BUY/SELL for new code
import { SignalType as CanonicalSignalType } from './schemas';
export type { CanonicalSignalType };

// Legacy SignalType for backwards compatibility
// Deprecated: Use 'BUY' | 'SELL' | 'HOLD' from schemas.ts instead
export type SignalType = 'LONG' | 'SHORT' | 'HOLD' | 'CLOSE' | 'BUY' | 'SELL';

// -------------------- Signals API --------------------

export interface StrategySignal {
    strategy_code: string;
    strategy_name: string;
    signal: SignalType;
    side: OrderSide;
    confidence: number;         // 0.0 - 1.0
    size: number;               // 0.0 - 1.0
    entry_price?: number;
    stop_loss?: number;
    take_profit?: number;
    risk_usd: number;
    reasoning: string;
    timestamp: string;          // ISO 8601
    age_seconds: number;
}

export interface LatestSignalsResponse {
    timestamp: string;
    market_price: number;
    market_status: MarketStatus;
    signals: StrategySignal[];
}

// -------------------- Performance API --------------------

export interface StrategyPerformance {
    strategy_code: string;
    strategy_name: string;
    sharpe_ratio: number;
    sortino_ratio: number;
    calmar_ratio: number;
    total_return_pct: number;
    max_drawdown_pct: number;
    current_drawdown_pct: number;
    win_rate: number;
    profit_factor: number;
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    avg_win_pct: number;
    avg_loss_pct: number;
    current_equity: number;
    initial_equity: number;
}

export interface PerformanceResponse {
    period: string;
    start_date: string;
    end_date: string;
    strategies: StrategyPerformance[];
}

// -------------------- Risk Status API (NEW) --------------------

export interface RiskLimits {
    max_drawdown_pct: number;
    max_daily_loss_pct: number;
    max_trades_per_day: number;
}

export interface RiskStatusResponse {
    is_paper_trading: boolean;
    kill_switch_active: boolean;
    daily_blocked: boolean;
    cooldown_active: boolean;
    cooldown_remaining_minutes: number;

    // Metrics
    trade_count_today: number;
    trades_remaining: number;
    daily_pnl_pct: number;
    consecutive_losses: number;
    daily_loss_remaining_pct: number;
    current_drawdown_pct: number;

    limits: RiskLimits;
    current_day: string;
    last_updated: string;
}

// -------------------- Model Health API (NEW) --------------------

export type HealthStatus = 'healthy' | 'warning' | 'critical';

export interface ModelHealth {
    model_id: string;
    action_drift_kl: number;
    stuck_behavior: boolean;
    rolling_sharpe: number;
    status: HealthStatus;
    last_inference: string;
}

export interface ModelHealthResponse {
    models: ModelHealth[];
    overall_status: HealthStatus;
}

// -------------------- Common Errors --------------------

export interface APIError {
    error: string;
    message: string;
    details?: Record<string, unknown>;
    timestamp: string;
}

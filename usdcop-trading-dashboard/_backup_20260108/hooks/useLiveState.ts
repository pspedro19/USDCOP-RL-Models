import useSWR from 'swr';
import { fetcher } from '@/lib/utils';

export interface LiveState {
    model_id: string;
    position: 'LONG' | 'SHORT' | 'FLAT';
    entry_price: number;
    entry_time: string | null;
    current_price: number;
    unrealized_pnl: number;
    unrealized_pnl_pct: number;
    bars_in_position: number;
    equity: number;
    drawdown_pct: number;
    peak_equity: number;
    market_status: 'OPEN' | 'CLOSED' | 'PRE_MARKET';
    last_signal: 'BUY' | 'SELL' | 'HOLD' | 'FLAT';
    last_updated: string;
}

// API Response wrapper type
interface ApiResponse<T> {
    success: boolean;
    data: T;
    metadata?: {
        dataSource: string;
        timestamp: string;
    };
}

interface PositionsResponse {
    positions: Array<{
        strategy_code: string;
        strategy_name: string;
        side: 'long' | 'short' | 'flat';
        entry_price: number | null;
        current_price: number;
        unrealized_pnl: number;
        unrealized_pnl_pct: number;
        entry_time: string | null;
        duration_minutes: number;
    }>;
    timestamp: string;
}

// Transform API positions to LiveState format
function transformToLiveState(positions: PositionsResponse['positions']): LiveState | null {
    if (!positions || positions.length === 0) {
        // Return default flat state
        return {
            model_id: 'PORTFOLIO',
            position: 'FLAT',
            entry_price: 0,
            entry_time: null,
            current_price: 4285.50,
            unrealized_pnl: 0,
            unrealized_pnl_pct: 0,
            bars_in_position: 0,
            equity: 10000,
            drawdown_pct: 0,
            peak_equity: 10000,
            market_status: 'OPEN',
            last_signal: 'FLAT',
            last_updated: new Date().toISOString()
        };
    }

    // Find first active position or use first position
    const activePosition = positions.find(p => p.side !== 'flat') || positions[0];

    return {
        model_id: activePosition.strategy_code || 'PORTFOLIO',
        position: (activePosition.side?.toUpperCase() as 'LONG' | 'SHORT' | 'FLAT') || 'FLAT',
        entry_price: activePosition.entry_price ?? 0,
        entry_time: activePosition.entry_time || null,
        current_price: activePosition.current_price ?? 4285.50,
        unrealized_pnl: activePosition.unrealized_pnl ?? 0,
        unrealized_pnl_pct: activePosition.unrealized_pnl_pct ?? 0,
        bars_in_position: Math.floor((activePosition.duration_minutes ?? 0) / 5),
        equity: 10000 + (activePosition.unrealized_pnl ?? 0),
        drawdown_pct: Math.max(0, -(activePosition.unrealized_pnl_pct ?? 0)),
        peak_equity: 10000,
        market_status: 'OPEN',
        last_signal: activePosition.side === 'long' ? 'BUY' : activePosition.side === 'short' ? 'SELL' : 'FLAT',
        last_updated: new Date().toISOString()
    };
}

export function useLiveState() {
    const { data: rawData, error, isLoading } = useSWR<ApiResponse<PositionsResponse>>(
        `/api/models/positions/current`,
        fetcher,
        {
            refreshInterval: 5000, // 5 seconds
            revalidateOnFocus: true
        }
    );

    // Transform API response to LiveState
    const state = rawData?.data?.positions ? transformToLiveState(rawData.data.positions) : null;

    return {
        state,
        isLoading,
        isError: !!error,
        error
    };
}

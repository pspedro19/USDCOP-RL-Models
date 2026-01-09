import useSWR from 'swr';
import { fetcher } from '@/lib/utils';

export interface TradeHistoryItem {
    trade_id: number;
    model_id: string;
    side: 'LONG' | 'SHORT';
    entry_price: number;
    exit_price: number | null;
    entry_time: string;
    exit_time: string | null;
    duration_bars: number;
    pnl_usd: number;
    pnl_pct: number;
    exit_reason: string;
}

export interface TradesHistoryResponse {
    trades: TradeHistoryItem[];
    summary: {
        total_trades: number;
        winning: number;
        losing: number;
        win_rate: number;
    };
}

// API Response wrapper type
interface ApiResponse<T> {
    success: boolean;
    data: T;
    metadata?: {
        dataSource: string;
        timestamp: string;
        isRealData: boolean;
    };
}

export function useTradesHistory(limit: number = 50, modelId: string = 'ppo_v1') {
    const { data: rawData, error, isLoading } = useSWR<ApiResponse<TradesHistoryResponse>>(
        `/api/trading/trades/history?limit=${limit}&model_id=${modelId}`,
        fetcher,
        {
            refreshInterval: 30000, // 30 seconds (reduced frequency)
            revalidateOnFocus: false, // Don't refetch on window focus
            dedupingInterval: 5000, // Dedupe requests within 5 seconds
            keepPreviousData: true, // Show previous data while loading new
        }
    );

    // Extract trades from API response wrapper
    const trades = rawData?.data?.trades || [];
    const summary = rawData?.data?.summary || { total_trades: 0, winning: 0, losing: 0, win_rate: 0 };

    return {
        data: trades.length > 0 ? { trades, summary } : null,
        isLoading,
        isError: !!error,
        error
    };
}

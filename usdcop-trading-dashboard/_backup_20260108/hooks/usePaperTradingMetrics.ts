import useSWR from 'swr';
import { APIError } from '@/types/contracts';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface PaperTrade {
    trade_id: number;
    model_id: string;
    signal: string;
    entry_price: number;
    exit_price: number;
    pnl: number;
    pnl_pct: number;
    entry_time: string;
    exit_time: string;
}

export interface PaperTradingMetricsResponse {
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    total_pnl: number;
    win_rate: number;
    avg_win: number;
    avg_loss: number;
    current_equity: number;
    trades: PaperTrade[];
}

interface UsePaperTradingMetricsReturn {
    metrics: PaperTradingMetricsResponse | null;
    trades: PaperTrade[];
    isLoading: boolean;
    isError: boolean;
    error: Error | null;
    mutate: () => void;
}

const fetcher = async (url: string): Promise<PaperTradingMetricsResponse> => {
    const res = await fetch(url);
    if (!res.ok) {
        let errorData: any = {};
        try {
            errorData = await res.json();
        } catch { }
        throw new Error(errorData.detail || 'Failed to fetch paper trading metrics');
    }
    return res.json();
};

export function usePaperTradingMetrics(enabled: boolean = true): UsePaperTradingMetricsReturn {
    const { data, error, isLoading, mutate } = useSWR<PaperTradingMetricsResponse>(
        enabled ? `${API_BASE}/api/paper-trading/metrics` : null,
        fetcher,
        {
            refreshInterval: 10000,
            revalidateOnFocus: true
        }
    );

    return {
        metrics: data ?? null,
        trades: data?.trades ?? [],
        isLoading,
        isError: !!error,
        error: error ?? null,
        mutate
    };
}

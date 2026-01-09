// hooks/useRiskStatus.ts
'use client';

import useSWR from 'swr';
import { RiskStatusResponse, RiskLimits } from '@/types/contracts';

const fetcher = async (url: string): Promise<RiskStatusResponse> => {
    const res = await fetch(url);
    if (!res.ok) {
        throw new Error('Failed to fetch risk status');
    }
    return res.json();
};

interface UseRiskStatusReturn {
    // Estado actual
    isPaperTrading: boolean;
    killSwitchActive: boolean;
    currentDrawdownPct: number;
    dailyPnlPct: number;
    tradesToday: number;
    consecutiveLosses: number;
    cooldownActive: boolean;
    cooldownUntil: Date | null;

    // Limites configurados
    limits: RiskLimits | null;

    // Estados derivados para UI
    riskLevel: 'safe' | 'warning' | 'danger' | 'critical';
    tradingAllowed: boolean;
    warningMessages: string[];

    // SWR states
    isLoading: boolean;
    isError: boolean;
    error: Error | null;
    mutate: () => void;
}

export function useRiskStatus(): UseRiskStatusReturn {
    const { data, error, isLoading, mutate } = useSWR<RiskStatusResponse>(
        `/api/risk/status`,
        fetcher,
        {
            refreshInterval: 10000, // 10s - frequent checks for risk monitoring
            revalidateOnFocus: true,
            shouldRetryOnError: true,
        }
    );

    // Calcular risk level
    const calculateRiskLevel = (): 'safe' | 'warning' | 'danger' | 'critical' => {
        if (!data) return 'safe';
        if (data.kill_switch_active || data.daily_blocked) return 'critical';
        if (data.current_drawdown_pct > 10 || data.cooldown_active) return 'danger';
        if (
            (data.limits && data.current_drawdown_pct > data.limits.max_drawdown_pct * 0.5) ||
            data.consecutive_losses >= 2
        ) return 'warning';
        return 'safe';
    };

    // Generar mensajes de advertencia del sistema
    const getWarningMessages = (): string[] => {
        if (!data) return [];
        const messages: string[] = [];

        if (data.kill_switch_active) {
            messages.push('KILL SWITCH ACTIVADO - Trading detenido automáticamente por seguridad');
        }
        if (data.daily_blocked) {
            messages.push('Trading bloqueado por exceso de pérdidas diarias');
        }
        if (data.cooldown_active) {
            const mins = data.cooldown_remaining_minutes || 0;
            messages.push(`Cooldown activo: ${mins.toFixed(1)} min restantes`);
        }
        if (data.limits && data.current_drawdown_pct > data.limits.max_drawdown_pct * 0.8) {
            messages.push(`CRÍTICO: Drawdown (${data.current_drawdown_pct.toFixed(1)}%) cerca del límite (${data.limits.max_drawdown_pct}%)`);
        } else if (data.current_drawdown_pct > 5) {
            messages.push(`Atención: Drawdown moderado detectado (${data.current_drawdown_pct.toFixed(1)}%)`);
        }
        if (data.limits && data.trade_count_today >= data.limits.max_trades_per_day * 0.8) {
            messages.push(`Cerca del límite de trades diarios (${data.trade_count_today}/${data.limits.max_trades_per_day})`);
        }

        return messages;
    };

    return {
        isPaperTrading: data?.is_paper_trading ?? true, // Default to true for safety
        killSwitchActive: data?.kill_switch_active ?? false,
        currentDrawdownPct: data?.current_drawdown_pct ?? 0,
        dailyPnlPct: data?.daily_pnl_pct ?? 0,
        tradesToday: data?.trade_count_today ?? 0,
        consecutiveLosses: data?.consecutive_losses ?? 0,
        cooldownActive: data?.cooldown_active ?? false,
        cooldownUntil: data?.cooldown_remaining_minutes
            ? new Date(Date.now() + data.cooldown_remaining_minutes * 60000)
            : null,
        limits: data?.limits ?? null,

        riskLevel: calculateRiskLevel(),
        tradingAllowed: data ? (!data.kill_switch_active && !data.daily_blocked && !data.cooldown_active) : true,
        warningMessages: getWarningMessages(),

        isLoading,
        isError: !!error,
        error: error ?? null,
        mutate,
    };
}

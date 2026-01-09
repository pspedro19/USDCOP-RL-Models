'use client';

import { useMemo } from 'react';
import { useEquityCurveStream } from '@/hooks/useEquityCurveStream';
import EquityCurveChartBase from '@/components/views/EquityCurveChart';

/**
 * Wrapper component that fetches equity curve data and renders the chart
 * This connects the chart to real data from the API
 */
export function EquityCurveChart() {
    const { data, isConnected, connectionType, error, reconnect } = useEquityCurveStream();

    // Transform the hook data format to the chart component format
    const chartData = useMemo(() => {
        if (!data || Object.keys(data).length === 0) return [];

        // Get all unique timestamps across all strategies
        const allTimestamps = new Set<string>();
        Object.values(data).forEach(points => {
            points.forEach(p => allTimestamps.add(p.timestamp));
        });

        // Sort timestamps
        const sortedTimestamps = Array.from(allTimestamps).sort();

        // Map data to chart format
        return sortedTimestamps.map(timestamp => {
            const rlData = data['RL_PPO']?.find(p => p.timestamp === timestamp);
            const lgbmData = data['ML_LGBM']?.find(p => p.timestamp === timestamp);
            const xgbData = data['ML_XGB']?.find(p => p.timestamp === timestamp);
            const llmData = data['LLM_CLAUDE']?.find(p => p.timestamp === timestamp);

            const rlEquity = rlData?.equity_value ?? 10000;
            const lgbmEquity = lgbmData?.equity_value ?? 10000;
            const xgbEquity = xgbData?.equity_value ?? null;
            const llmEquity = llmData?.equity_value ?? null;

            return {
                timestamp,
                RL_PPO: rlEquity,
                ML_LGBM: lgbmEquity,
                ML_XGB: xgbEquity,
                LLM_CLAUDE: llmEquity,
                PORTFOLIO: rlEquity + lgbmEquity + (xgbEquity || 10000) + (llmEquity || 10000),
                CAPITAL_INICIAL: 10000
            };
        });
    }, [data]);

    const isLoading = connectionType === 'disconnected' && !isConnected && chartData.length === 0;

    return (
        <EquityCurveChartBase
            data={chartData}
            isLoading={isLoading}
            error={error}
            onRetry={reconnect}
        />
    );
}

export default EquityCurveChart;

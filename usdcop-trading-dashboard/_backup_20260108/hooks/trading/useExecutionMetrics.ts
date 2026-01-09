import useSWR from 'swr';

const fetcher = (url: string) => fetch(url).then((res) => res.json());

export const useExecutionMetrics = () => {
  const { data, error } = useSWR(
    '/api/analytics/execution-metrics?symbol=USDCOP&days=7',
    fetcher,
    { refreshInterval: 30000 }
  );

  return {
    vwap: data?.metrics?.vwap || 0,
    effectiveSpread: data?.metrics?.effective_spread_bps || 0,
    avgSlippage: data?.metrics?.avg_slippage_bps || 0,
    turnoverCost: data?.metrics?.turnover_cost_bps || 0,
    fillRatio: data?.metrics?.fill_ratio_pct || 0,
    isLoading: !error && !data,
    error
  };
};

'use client';

/**
 * useEquityCurveStream Hook
 * =========================
 * SSE-based real-time equity curve updates.
 * Falls back to polling if SSE is not available.
 */

import { useState, useEffect, useCallback, useRef } from 'react';

interface EquityPoint {
  timestamp: string;
  equity_value: number;
  return_pct: number;
  drawdown_pct: number;
}

interface StrategyEquity {
  strategy_name: string;
  timestamp: string;
  equity_value: number;
  return_pct: number;
  drawdown_pct: number;
}

interface EquityCurveData {
  [strategyCode: string]: EquityPoint[];
}

interface UseEquityCurveStreamOptions {
  strategies?: string[];
  enabled?: boolean;
  fallbackInterval?: number;
}

interface UseEquityCurveStreamReturn {
  data: EquityCurveData;
  latestValues: Record<string, StrategyEquity>;
  isConnected: boolean;
  connectionType: 'sse' | 'polling' | 'disconnected';
  lastUpdate: string | null;
  error: string | null;
  reconnect: () => void;
}

const MULTI_MODEL_API_URL = process.env.NEXT_PUBLIC_MULTI_MODEL_API_URL || 'http://localhost:8006';

export function useEquityCurveStream(
  options: UseEquityCurveStreamOptions = {}
): UseEquityCurveStreamReturn {
  const {
    strategies = [],
    enabled = true,
    fallbackInterval = 10000
  } = options;

  const [data, setData] = useState<EquityCurveData>({});
  const [latestValues, setLatestValues] = useState<Record<string, StrategyEquity>>({});
  const [isConnected, setIsConnected] = useState(false);
  const [connectionType, setConnectionType] = useState<'sse' | 'polling' | 'disconnected'>('disconnected');
  const [lastUpdate, setLastUpdate] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const eventSourceRef = useRef<EventSource | null>(null);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch historical data for initial load
  const fetchHistoricalData = useCallback(async () => {
    try {
      const params = new URLSearchParams({ hours: '24', resolution: '5m' });
      if (strategies.length > 0) params.append('strategies', strategies.join(','));
      const response = await fetch(`/api/trading/equity-curves/multi-strategy?${params.toString()}`);

      if (response.ok) {
        const result = await response.json();

        // Handle pivoted format (equity_curves array)
        const equityCurves = result.data?.equity_curves || result.equity_curves || [];

        if (equityCurves.length > 0) {
          // Transform pivoted format to by-strategy format
          const newData: EquityCurveData = {
            RL_PPO: [],
            ML_LGBM: [],
            ML_XGB: [],
            LLM_CLAUDE: [],
            PORTFOLIO: []
          };
          const newLatest: Record<string, StrategyEquity> = {};

          equityCurves.forEach((point: any) => {
            const timestamp = point.timestamp;

            if (point.RL_PPO !== undefined) {
              newData.RL_PPO.push({
                timestamp,
                equity_value: point.RL_PPO,
                return_pct: ((point.RL_PPO - 10000) / 10000) * 100,
                drawdown_pct: 0
              });
            }
            if (point.ML_LGBM !== undefined) {
              newData.ML_LGBM.push({
                timestamp,
                equity_value: point.ML_LGBM,
                return_pct: ((point.ML_LGBM - 10000) / 10000) * 100,
                drawdown_pct: 0
              });
            }
            if (point.ML_XGB !== undefined) {
              newData.ML_XGB.push({
                timestamp,
                equity_value: point.ML_XGB,
                return_pct: ((point.ML_XGB - 10000) / 10000) * 100,
                drawdown_pct: 0
              });
            }
            if (point.PORTFOLIO !== undefined) {
              newData.PORTFOLIO.push({
                timestamp,
                equity_value: point.PORTFOLIO,
                return_pct: ((point.PORTFOLIO - 10000) / 10000) * 100,
                drawdown_pct: 0
              });
            }
          });

          // Set latest values
          const strategyNames: Record<string, string> = {
            RL_PPO: 'PPO USDCOP V1',
            ML_LGBM: 'LightGBM Ensemble',
            ML_XGB: 'XGBoost Classifier',
            LLM_CLAUDE: 'LLM Claude',
            PORTFOLIO: 'Portfolio'
          };

          Object.keys(newData).forEach(code => {
            const points = newData[code];
            if (points.length > 0) {
              const latest = points[points.length - 1];
              newLatest[code] = {
                strategy_name: strategyNames[code] || code,
                ...latest
              };
            }
          });

          setData(newData);
          setLatestValues(newLatest);
          setLastUpdate(new Date().toISOString());
          return;
        }

        // Fallback: Handle by-strategy format (curves array)
        const curves = result.curves || result.data?.curves || [];
        if (curves.length > 0) {
          const newData: EquityCurveData = {};
          const newLatest: Record<string, StrategyEquity> = {};

          curves.forEach((curve: any) => {
            const code = curve.strategy_code || curve.strategyCode;
            newData[code] = (curve.data || []).map((p: any) => ({
              timestamp: p.timestamp,
              equity_value: p.equity_value || p.equityValue || 10000,
              return_pct: p.return_pct || p.returnPct || 0,
              drawdown_pct: p.drawdown_pct || p.drawdownPct || 0
            }));

            if (newData[code].length > 0) {
              const latest = newData[code][newData[code].length - 1];
              newLatest[code] = {
                strategy_name: curve.strategy_name || curve.strategyName || code,
                ...latest
              };
            }
          });

          setData(newData);
          setLatestValues(newLatest);
          setLastUpdate(new Date().toISOString());
        }
      }
    } catch (err) {
      console.error('Error fetching historical equity data:', err);
    }
  }, [strategies]);

  // Polling fallback - defined before connectSSE to avoid circular dependency
  const startPolling = useCallback(() => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }

    setConnectionType('polling');

    const poll = async () => {
      await fetchHistoricalData();
      setIsConnected(true);
    };

    poll(); // Initial fetch
    pollingIntervalRef.current = setInterval(poll, fallbackInterval);
  }, [fetchHistoricalData, fallbackInterval]);

  // Connect to SSE stream (only if backend is available)
  const connectSSE = useCallback(async () => {
    if (!enabled) return;

    // Close existing connection
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    // First check if backend is available by testing the API proxy
    try {
      const testResponse = await fetch('/api/trading/equity-curves/multi-strategy?hours=1', {
        signal: AbortSignal.timeout(3000)
      });
      const testData = await testResponse.json();

      // If data source is 'demo', skip SSE and use polling
      if (testData.metadata?.dataSource === 'demo' || testData.data?.source === 'demo') {
        console.log('[EquityCurveStream] Backend unavailable, using polling mode with demo data');
        startPolling();
        return;
      }
    } catch (err) {
      console.log('[EquityCurveStream] Backend check failed, using polling mode');
      startPolling();
      return;
    }

    try {
      const strategiesParam = strategies.length > 0 ? `?strategies=${strategies.join(',')}` : '';
      const sseUrl = `${MULTI_MODEL_API_URL}/api/stream/equity-curves${strategiesParam}`;

      const eventSource = new EventSource(sseUrl);
      eventSourceRef.current = eventSource;

      eventSource.onopen = () => {
        setIsConnected(true);
        setConnectionType('sse');
        setError(null);
        console.log('[EquityCurveStream] SSE connected');
      };

      eventSource.addEventListener('equity_update', (event) => {
        try {
          const payload = JSON.parse(event.data);
          const updates = payload.strategies || {};

          setLatestValues(prev => {
            const newValues = { ...prev };
            Object.entries(updates).forEach(([code, value]: [string, any]) => {
              newValues[code] = value;
            });
            return newValues;
          });

          // Append to historical data
          setData(prev => {
            const newData = { ...prev };
            Object.entries(updates).forEach(([code, value]: [string, any]) => {
              if (!newData[code]) {
                newData[code] = [];
              }
              newData[code] = [
                ...newData[code],
                {
                  timestamp: value.timestamp,
                  equity_value: value.equity_value,
                  return_pct: value.return_pct,
                  drawdown_pct: value.drawdown_pct
                }
              ].slice(-500); // Keep last 500 points
            });
            return newData;
          });

          setLastUpdate(payload.timestamp);
        } catch (err) {
          console.error('Error parsing SSE equity update:', err);
        }
      });

      eventSource.addEventListener('heartbeat', () => {
        // Keep connection alive
      });

      eventSource.addEventListener('error', () => {
        console.log('[EquityCurveStream] SSE error, falling back to polling');
        setIsConnected(false);
        setConnectionType('disconnected');
        eventSource.close();

        // Fallback to polling
        startPolling();
      });

      eventSource.onerror = () => {
        if (eventSource.readyState === EventSource.CLOSED) {
          setIsConnected(false);
          setConnectionType('disconnected');

          // Don't retry SSE, just use polling
          startPolling();
        }
      };

    } catch (err) {
      console.log('[EquityCurveStream] Error creating SSE connection, using polling');
      startPolling();
    }
  }, [enabled, strategies, startPolling]);

  // Manual reconnect
  const reconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    setIsConnected(false);
    setConnectionType('disconnected');

    // Try SSE first, then fall back to polling
    connectSSE();
  }, [connectSSE]);

  // Initialize
  useEffect(() => {
    if (!enabled) return;

    // Fetch initial historical data
    fetchHistoricalData();

    // Try to connect via SSE
    connectSSE();

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [enabled, connectSSE, fetchHistoricalData]);

  return {
    data,
    latestValues,
    isConnected,
    connectionType,
    lastUpdate,
    error,
    reconnect
  };
}

export default useEquityCurveStream;

'use client';

/**
 * useLiveProduction — Adaptive polling hook for production monitoring.
 *
 * Polls /api/production/live with adaptive frequency:
 *   - Market open:  every 30 seconds
 *   - Market closed: every 5 minutes
 *
 * Falls back to file-based data when DB is unavailable.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  LiveProductionResponse,
  CurrentSignal,
  ActivePosition,
  LiveEquityCurve,
  LiveStats,
  Guardrails,
  MarketState,
  LiveDataSource,
  LiveTrade,
} from '@/lib/contracts/production-monitor.contract';
import type {
  ApprovalState,
  ProductionSummary,
  ProductionTradeFile,
} from '@/lib/contracts/production-approval.contract';
import type { StrategyTrade, StrategyStats } from '@/lib/contracts/strategy.contract';

const POLL_MARKET_OPEN_MS = 30_000;   // 30 seconds
const POLL_MARKET_CLOSED_MS = 300_000; // 5 minutes

export interface UseLiveProductionResult {
  // Live data (from DB when available)
  currentSignal: CurrentSignal | null;
  activePosition: ActivePosition | null;
  equityCurve: LiveEquityCurve | null;
  guardrails: Guardrails | null;
  market: MarketState | null;

  // Merged data (live or file-based fallback)
  trades: StrategyTrade[];
  stats: StrategyStats | undefined;
  summary: ProductionSummary | null;
  approval: ApprovalState | null;

  // Meta
  isLive: boolean;
  dataSource: LiveDataSource;
  loading: boolean;
  error: string | null;
  lastFetchTime: Date | null;
  nextRefreshIn: number; // seconds
  strategyId: string;
  strategyName: string;

  // Actions
  refresh: () => void;

  // New trade detection
  newTradeCount: number;
  newTradeIds: Set<number>;
  dismissNewTrades: () => void;
}

function liveTradesToStrategyTrades(liveTrades: LiveTrade[]): StrategyTrade[] {
  return liveTrades.map(t => ({
    trade_id: t.trade_id,
    timestamp: t.timestamp,
    exit_timestamp: t.exit_timestamp ?? undefined,
    side: t.side,
    entry_price: t.entry_price,
    exit_price: t.exit_price,
    pnl_usd: t.pnl_usd,
    pnl_pct: t.pnl_pct,
    exit_reason: t.exit_reason,
    equity_at_entry: t.equity_at_entry,
    equity_at_exit: t.equity_at_exit,
    leverage: t.leverage,
    confidence_tier: t.confidence_tier,
    hard_stop_pct: t.hard_stop_pct,
    take_profit_pct: t.take_profit_pct,
  }));
}

function liveStatsToStrategyStats(stats: LiveStats, equityCurve: LiveEquityCurve): StrategyStats {
  return {
    final_equity: equityCurve.current_equity,
    total_return_pct: stats.total_return_pct,
    sharpe: stats.sharpe,
    max_dd_pct: stats.max_dd_pct,
    win_rate_pct: stats.win_rate_pct,
    profit_factor: stats.profit_factor,
    exit_reasons: stats.exit_reasons,
    n_long: stats.n_long,
    n_short: stats.n_short,
  };
}

export function useLiveProduction(): UseLiveProductionResult {
  const [currentSignal, setCurrentSignal] = useState<CurrentSignal | null>(null);
  const [activePosition, setActivePosition] = useState<ActivePosition | null>(null);
  const [equityCurve, setEquityCurve] = useState<LiveEquityCurve | null>(null);
  const [guardrails, setGuardrails] = useState<Guardrails | null>(null);
  const [market, setMarket] = useState<MarketState | null>(null);
  const [trades, setTrades] = useState<StrategyTrade[]>([]);
  const [stats, setStats] = useState<StrategyStats | undefined>(undefined);
  const [summary, setSummary] = useState<ProductionSummary | null>(null);
  const [approval, setApproval] = useState<ApprovalState | null>(null);
  const [isLive, setIsLive] = useState(false);
  const [dataSource, setDataSource] = useState<LiveDataSource>('file');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastFetchTime, setLastFetchTime] = useState<Date | null>(null);
  const [nextRefreshIn, setNextRefreshIn] = useState(0);
  const [strategyId, setStrategyId] = useState('smart_simple_v11');
  const [strategyName, setStrategyName] = useState('Smart Simple v1.1');
  const [newTradeCount, setNewTradeCount] = useState(0);
  const [newTradeIds, setNewTradeIds] = useState<Set<number>>(new Set());

  const prevTradeCountRef = useRef(0);
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const marketOpenRef = useRef(false);

  const dismissNewTrades = useCallback(() => {
    setNewTradeCount(0);
    setNewTradeIds(new Set());
  }, []);

  // File-based fallback fetcher
  const fetchFileBased = useCallback(async () => {
    const [summaryRes, approvalRes] = await Promise.all([
      fetch('/data/production/summary.json'),
      fetch('/api/production/status'),
    ]);

    let summaryData: ProductionSummary | null = null;
    let approvalData: ApprovalState | null = null;

    if (summaryRes.ok) summaryData = await summaryRes.json();
    if (approvalRes.ok) approvalData = await approvalRes.json();

    const sid = summaryData?.strategy_id || 'smart_simple_v11';
    const tradesRes = await fetch(`/data/production/trades/${sid}.json`);
    let fileTrades: StrategyTrade[] = [];
    if (tradesRes.ok) {
      const data: ProductionTradeFile = await tradesRes.json();
      fileTrades = data.trades || [];
    }

    return { summaryData, approvalData, fileTrades, sid };
  }, []);

  // Main fetch function
  const fetchData = useCallback(async (isPolling = false) => {
    try {
      if (!isPolling) setLoading(true);

      // Always try live endpoint first
      let liveData: LiveProductionResponse | null = null;
      try {
        const res = await fetch('/api/production/live');
        if (res.ok) {
          liveData = await res.json();
        }
      } catch {
        // DB unavailable, will fall back to files
      }

      // Also fetch approval state (file-based) and summary (for additional context)
      const { summaryData, approvalData } = await fetchFileBased();
      setApproval(approvalData);

      // Use DB data only if it actually has trades; otherwise fall back to files
      const dbHasData = liveData && liveData.data_source === 'db'
        && (liveData.trades.length > 0 || liveData.active_position != null);

      if (dbHasData && liveData) {
        // Use live DB data
        setCurrentSignal(liveData.current_signal);
        setActivePosition(liveData.active_position);
        setEquityCurve(liveData.equity_curve);
        setGuardrails(liveData.guardrails);
        setMarket(liveData.market);
        setStrategyId(liveData.strategy_id);
        setStrategyName(liveData.strategy_name);
        setIsLive(true);
        setDataSource('db');

        const convertedTrades = liveTradesToStrategyTrades(liveData.trades);
        const convertedStats = liveStatsToStrategyStats(liveData.stats, liveData.equity_curve);

        // Merge active position as an open trade so it appears in the trade table
        const allTrades = [...convertedTrades];
        if (liveData.active_position) {
          const ap = liveData.active_position;
          const alreadyInList = convertedTrades.some(
            t => t.entry_price === ap.entry_price && t.timestamp === ap.entry_timestamp
          );
          if (!alreadyInList) {
            allTrades.push({
              trade_id: convertedTrades.length + 1,
              timestamp: ap.entry_timestamp,
              exit_timestamp: null,
              side: ap.direction === -1 ? 'SHORT' : 'LONG',
              entry_price: ap.entry_price,
              exit_price: null,
              pnl_usd: null,
              pnl_pct: null,
              exit_reason: null,
              equity_at_entry: liveData.equity_curve.current_equity,
              equity_at_exit: null,
              leverage: ap.leverage,
            });
          }
        }

        // Detect new trades
        if (isPolling && allTrades.length > prevTradeCountRef.current) {
          const diff = allTrades.length - prevTradeCountRef.current;
          setNewTradeCount(diff);
          const ids = new Set(
            allTrades.slice(prevTradeCountRef.current).map(t => t.trade_id)
          );
          setNewTradeIds(ids);
          setTimeout(() => setNewTradeIds(new Set()), 10000);
        }
        prevTradeCountRef.current = allTrades.length;

        setTrades(allTrades);
        setStats(convertedStats);
        setSummary(summaryData); // Keep summary for year/initial_capital/etc.
        marketOpenRef.current = liveData.market.is_open;
      } else {
        // File-based fallback
        const { fileTrades, sid } = await fetchFileBased();
        setCurrentSignal(null);
        setActivePosition(null);
        setEquityCurve(null);
        setGuardrails(null);
        setMarket(null);
        setIsLive(false);
        setDataSource(liveData?.data_source === 'unavailable' ? 'unavailable' : 'file');
        setStrategyId(sid);
        setStrategyName(summaryData?.strategy_name ?? 'Smart Simple v1.1');

        // Detect new trades on file fallback
        if (isPolling && fileTrades.length > prevTradeCountRef.current) {
          const diff = fileTrades.length - prevTradeCountRef.current;
          setNewTradeCount(diff);
          const ids = new Set(
            fileTrades.slice(prevTradeCountRef.current).map(t => t.trade_id)
          );
          setNewTradeIds(ids);
          setTimeout(() => setNewTradeIds(new Set()), 10000);
        }
        prevTradeCountRef.current = fileTrades.length;

        setTrades(fileTrades);
        setSummary(summaryData);
        if (summaryData) {
          const stratStats = summaryData.strategies[summaryData.strategy_id];
          setStats(stratStats);
        }
        marketOpenRef.current = false;
      }

      setLastFetchTime(new Date());
      const interval = marketOpenRef.current ? POLL_MARKET_OPEN_MS : POLL_MARKET_CLOSED_MS;
      setNextRefreshIn(interval / 1000);
      setError(null);
    } catch (e) {
      if (!isPolling) setError(e instanceof Error ? e.message : 'Error loading data');
    } finally {
      if (!isPolling) setLoading(false);
    }
  }, [fetchFileBased]);

  // Initial fetch
  useEffect(() => {
    fetchData(false);
  }, [fetchData]);

  // Adaptive polling
  useEffect(() => {
    const scheduleNext = () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
      const interval = marketOpenRef.current ? POLL_MARKET_OPEN_MS : POLL_MARKET_CLOSED_MS;
      pollIntervalRef.current = setInterval(() => {
        fetchData(true);
      }, interval);
    };

    scheduleNext();
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, [fetchData]);

  // Countdown timer
  useEffect(() => {
    countdownRef.current = setInterval(() => {
      setNextRefreshIn(prev => Math.max(0, prev - 1));
    }, 1000);
    return () => {
      if (countdownRef.current) clearInterval(countdownRef.current);
    };
  }, []);

  const refresh = useCallback(() => {
    fetchData(true);
  }, [fetchData]);

  return {
    currentSignal,
    activePosition,
    equityCurve,
    guardrails,
    market,
    trades,
    stats,
    summary,
    approval,
    isLive,
    dataSource,
    loading,
    error,
    lastFetchTime,
    nextRefreshIn,
    strategyId,
    strategyName,
    refresh,
    newTradeCount,
    newTradeIds,
    dismissNewTrades,
  };
}

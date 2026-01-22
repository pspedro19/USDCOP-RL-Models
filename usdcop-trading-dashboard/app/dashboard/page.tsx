'use client';

/**
 * Dashboard Page - Enhanced PowerBI Style
 * =======================================
 * Mobile-first, scrollable layout with:
 * - Dropdown model selector
 * - Large KPI cards
 * - Candlestick price chart (real data)
 * - Equity curve (redesigned with line chart + drawdown)
 * - Profit summary
 * - Recent trades table
 */

import { Suspense, useState, useEffect, useMemo, useRef, useCallback, lazy } from "react";
import { ModelProvider, useModel } from "@/contexts/ModelContext";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { GlobalNavbar } from "@/components/navigation/GlobalNavbar";
import { Badge } from "@/components/ui/badge";
import {
    TrendingUp,
    TrendingDown,
    Activity,
    Target,
    BarChart3,
    Clock,
    ChevronDown,
    RefreshCw,
    AlertCircle,
    LineChart,
    Table2,
    Percent,
    DollarSign,
    Play,
    Calendar
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
    ResponsiveContainer,
    AreaChart,
    Area,
    XAxis,
    YAxis,
    Tooltip,
    ReferenceLine,
    CartesianGrid,
    ComposedChart,
} from 'recharts';

// Lazy load heavy components
const TradingChartWithSignals = lazy(() => import("@/components/charts/TradingChartWithSignals"));
import { TradingSummaryCard } from "@/components/trading/TradingSummaryCard";
import { TradesTable } from "@/components/trading/TradesTable";

// Replay system
import { useReplay } from "@/hooks/useReplay";
import { ReplayControlBarCollapsible } from "@/components/trading/ReplayControlBar";
import { BacktestControlPanel } from "@/components/trading/BacktestControlPanel";
import { BacktestResult, BacktestTradeEvent } from "@/lib/contracts/backtest.contract";

// ============================================================================
// Types
// ============================================================================
interface BacktestMetrics {
    sharpe: number;
    maxDrawdown: number;
    winRate: number;
    holdPercent: number;
    totalTrades: number;
}

// ============================================================================
// Model Dropdown Selector with Promote Button
// ============================================================================
function ModelDropdown() {
    const { models, selectedModelId, selectedModel, setSelectedModel, isLoading, refreshModels } = useModel();
    const [isOpen, setIsOpen] = useState(false);
    const [isPromoting, setIsPromoting] = useState(false);

    // Promote a model to deployed status
    const handlePromote = async (modelId: string, e: React.MouseEvent) => {
        e.stopPropagation(); // Prevent dropdown from selecting the model

        if (isPromoting) return;

        const confirmed = window.confirm(
            `¿Promover "${modelId}" a producción?\n\nEsto demotará el modelo actualmente desplegado.`
        );

        if (!confirmed) return;

        setIsPromoting(true);
        try {
            const response = await fetch(`/api/models/${modelId}/promote`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
            });

            const data = await response.json();

            if (data.success) {
                alert(`✅ ${data.message}`);
                // Refresh models list to reflect new status
                await refreshModels();
            } else {
                alert(`❌ Error: ${data.error || 'Failed to promote model'}`);
            }
        } catch (error) {
            console.error('Error promoting model:', error);
            alert('❌ Error de conexión al promover modelo');
        } finally {
            setIsPromoting(false);
        }
    };

    if (isLoading) {
        return (
            <div className="h-12 w-full sm:w-64 animate-pulse rounded-xl bg-slate-800" />
        );
    }

    if (models.length === 0) {
        return (
            <div className="flex items-center gap-2 text-amber-400 text-sm">
                <AlertCircle className="w-4 h-4" />
                <span>No models available</span>
            </div>
        );
    }

    // Check if selected model can be promoted (dbStatus = 'registered')
    const canPromote = selectedModel?.dbStatus === 'registered';

    return (
        <div className="relative w-full sm:w-auto sm:min-w-[280px]">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={cn(
                    "w-full flex items-center justify-between gap-2 px-3 py-2.5 rounded-xl",
                    "bg-slate-800/80 border border-slate-700 hover:border-slate-600",
                    "transition-all duration-200"
                )}
            >
                <div className="flex items-center gap-3">
                    <span
                        className="w-3 h-3 rounded-full flex-shrink-0"
                        style={{ backgroundColor: selectedModel?.color || '#6B7280' }}
                    />
                    <div className="text-left">
                        <div className="font-semibold text-white text-sm">
                            {selectedModel?.name || 'Select Model'}
                        </div>
                        <div className="text-xs text-slate-400">
                            {selectedModel?.dbStatus === 'deployed' ? 'En Producción' :
                             selectedModel?.dbStatus === 'registered' ? 'Registrado (Testing)' :
                             selectedModel?.isRealData ? 'Production Data' : 'Demo Data'}
                        </div>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    <Badge
                        variant="outline"
                        className={cn(
                            "text-[10px] font-bold",
                            selectedModel?.dbStatus === 'deployed'
                                ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                                : selectedModel?.dbStatus === 'registered'
                                ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
                                : "bg-slate-500/10 text-slate-400 border-slate-500/30"
                        )}
                    >
                        {selectedModel?.dbStatus === 'deployed' ? 'DEPLOYED' :
                         selectedModel?.dbStatus === 'registered' ? 'TESTING' :
                         selectedModel?.isRealData ? 'REAL' : 'DEMO'}
                    </Badge>
                    <ChevronDown className={cn(
                        "w-4 h-4 text-slate-400 transition-transform",
                        isOpen && "rotate-180"
                    )} />
                </div>
            </button>

            {isOpen && (
                <div className="absolute top-full left-0 right-0 mt-2 z-50 bg-slate-800 border border-slate-700 rounded-xl overflow-hidden shadow-2xl">
                    {models.map((model) => (
                        <div
                            key={model.id}
                            className={cn(
                                "w-full flex items-center justify-between px-4 py-3",
                                "hover:bg-slate-700/50 transition-colors cursor-pointer",
                                model.id === selectedModelId && "bg-slate-700/30"
                            )}
                            onClick={() => {
                                setSelectedModel(model.id);
                                setIsOpen(false);
                            }}
                        >
                            <div className="flex items-center gap-3">
                                <span
                                    className="w-2.5 h-2.5 rounded-full"
                                    style={{ backgroundColor: model.color }}
                                />
                                <span className="text-white font-medium text-sm">{model.name}</span>
                            </div>
                            <div className="flex items-center gap-2">
                                {/* Show promote button for registered models */}
                                {model.dbStatus === 'registered' && (
                                    <button
                                        onClick={(e) => handlePromote(model.id, e)}
                                        disabled={isPromoting}
                                        className={cn(
                                            "px-2 py-1 text-[10px] font-medium rounded",
                                            "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30",
                                            "hover:bg-cyan-500/30 transition-colors",
                                            isPromoting && "opacity-50 cursor-not-allowed"
                                        )}
                                        title="Promover a producción"
                                    >
                                        {isPromoting ? '...' : 'PROMOVER'}
                                    </button>
                                )}
                                <Badge
                                    variant="outline"
                                    className={cn(
                                        "text-[10px]",
                                        model.dbStatus === 'deployed'
                                            ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                                            : model.dbStatus === 'registered'
                                            ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
                                            : "bg-slate-500/10 text-slate-400 border-slate-500/30"
                                    )}
                                >
                                    {model.dbStatus === 'deployed' ? 'PROD' :
                                     model.dbStatus === 'registered' ? 'TEST' :
                                     model.isRealData ? 'REAL' : 'DEMO'}
                                </Badge>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

// ============================================================================
// KPI Card Component
// ============================================================================
interface KPICardProps {
    title: string;
    value: string | number;
    subtitle?: string;
    icon: React.ReactNode;
    trend?: 'up' | 'down' | 'neutral';
    color?: string;
    isLoading?: boolean;
}

function KPICard({ title, value, subtitle, icon, trend, color = '#10B981', isLoading }: KPICardProps) {
    if (isLoading) {
        return (
            <Card className="bg-slate-900/50 border-slate-800">
                <CardContent className="p-4 sm:p-6">
                    <div className="animate-pulse space-y-2">
                        <div className="h-3 w-16 bg-slate-700 rounded" />
                        <div className="h-8 w-24 bg-slate-700 rounded" />
                        <div className="h-2 w-20 bg-slate-700 rounded" />
                    </div>
                </CardContent>
            </Card>
        );
    }

    return (
        <Card className="bg-slate-900/50 border-slate-800 hover:border-slate-600 transition-all duration-300 hover:shadow-lg hover:shadow-cyan-500/5">
            <CardContent className="p-5 sm:p-6 flex flex-col items-center text-center">
                {/* Icon */}
                <div
                    className="p-2.5 sm:p-3 rounded-xl mb-4"
                    style={{ backgroundColor: `${color}15` }}
                >
                    <div style={{ color }}>{icon}</div>
                </div>
                {/* Title */}
                <span className="text-xs sm:text-sm font-medium text-slate-400 uppercase tracking-wider mb-2">
                    {title}
                </span>
                {/* Value */}
                <div
                    className="text-2xl sm:text-3xl font-bold tracking-tight mb-1"
                    style={{ color }}
                >
                    {value}
                </div>
                {/* Subtitle with trend */}
                {subtitle && (
                    <div className="flex items-center justify-center gap-1.5 text-xs sm:text-sm text-slate-400">
                        {trend === 'up' && <TrendingUp className="w-3 h-3 sm:w-4 sm:h-4 text-emerald-400" />}
                        {trend === 'down' && <TrendingDown className="w-3 h-3 sm:w-4 sm:h-4 text-red-400" />}
                        <span>{subtitle}</span>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}

// ============================================================================
// Backtest Metrics Panel - Connected to Real API Data
// ============================================================================
interface BacktestMetricsPanelProps {
    isReplayMode?: boolean;
    /** Force showing empty/default values (used when dashboard is cleared) */
    forceEmpty?: boolean;
    replayVisibleTrades?: Array<{
        pnl?: number;
        pnl_usd?: number;
        pnl_percent?: number;
        pnl_pct?: number;
        duration_minutes?: number;
        hold_time_minutes?: number;
        equity_at_entry?: number;
        equity_at_exit?: number;
        current_equity?: number;  // Real-time equity from streaming trades
    }>;
}

function BacktestMetricsPanel({ isReplayMode = false, forceEmpty = false, replayVisibleTrades = [] }: BacktestMetricsPanelProps) {
    const { selectedModel } = useModel();
    const [metricsData, setMetricsData] = useState<any>(null);
    const [equitySummary, setEquitySummary] = useState<any>(null);
    const [isLoading, setIsLoading] = useState(true);

    // Fetch real metrics from API (both metrics and equity curve for accurate max DD)
    useEffect(() => {
        const fetchMetrics = async () => {
            setIsLoading(true);
            try {
                const modelId = selectedModel?.id || 'RL_PPO';

                // Fetch both metrics and equity curve in parallel
                const [metricsResponse, equityResponse] = await Promise.all([
                    fetch(`/api/models/${modelId}/metrics?period=all`),
                    fetch(`/api/models/${modelId}/equity-curve?days=90`)
                ]);

                const metricsResult = await metricsResponse.json();
                const equityResult = await equityResponse.json();

                if (metricsResult.success && metricsResult.data) {
                    setMetricsData(metricsResult.data);
                }

                // Get max drawdown from equity curve (more accurate - bar-by-bar)
                if (equityResult.success && equityResult.data?.summary) {
                    setEquitySummary(equityResult.data.summary);
                }
            } catch (error) {
                console.error('Error fetching backtest metrics:', error);
            } finally {
                setIsLoading(false);
            }
        };

        fetchMetrics();
    }, [selectedModel?.id]);

    // Extract metrics from API response
    const apiMetrics = metricsData?.metrics || {};
    const liveMetrics = metricsData?.live || {};

    // Use equity curve max drawdown (more accurate - 5min bar granularity)
    // Fall back to trade-based max drawdown if equity curve not available
    const accurateMaxDD = equitySummary?.max_drawdown_pct ?? apiMetrics.max_drawdown_pct ?? 0;

    // Calculate replay metrics from visible trades (progressive)
    const calculateReplayMetrics = (): BacktestMetrics => {
        if (replayVisibleTrades.length === 0) {
            return {
                sharpe: 0,
                maxDrawdown: 0,
                winRate: 0,
                holdPercent: 0,
                totalTrades: 0
            };
        }

        const trades = replayVisibleTrades;
        const pnls = trades.map(t => t.pnl ?? t.pnl_usd ?? 0);
        const wins = pnls.filter(p => p > 0).length;
        const totalPnl = pnls.reduce((sum, p) => sum + p, 0);

        // Calculate max drawdown using equity_at_entry/exit if available
        const initialEquity = 10000;
        let peak = initialEquity;
        let maxDD = 0;
        let lastEquity = initialEquity;

        for (const trade of trades) {
            const pnl = trade.pnl ?? trade.pnl_usd ?? 0;
            // Priority: current_equity > equity_at_entry/exit > fallback calculation
            const hasCurrentEquity = trade.current_equity != null;
            const hasEquityFields = trade.equity_at_entry != null && trade.equity_at_exit != null;

            let entryEquity: number;
            let exitEquity: number;

            if (hasCurrentEquity) {
                // Real-time streaming: current_equity is the equity AFTER this trade
                exitEquity = trade.current_equity as number;
                entryEquity = exitEquity - pnl;
            } else if (hasEquityFields) {
                entryEquity = trade.equity_at_entry as number;
                exitEquity = trade.equity_at_exit as number;
            } else {
                // Fallback: use running total
                entryEquity = lastEquity;
                exitEquity = lastEquity + pnl;
            }

            // Update peak and calculate drawdown at entry and exit
            peak = Math.max(peak, entryEquity, exitEquity);
            const entryDD = peak > 0 ? ((peak - entryEquity) / peak) * 100 : 0;
            const exitDD = peak > 0 ? ((peak - exitEquity) / peak) * 100 : 0;
            maxDD = Math.max(maxDD, entryDD, exitDD);

            lastEquity = exitEquity;
        }

        // Calculate Sharpe (simplified - annualized assuming daily returns)
        const avgReturn = pnls.length > 0 ? totalPnl / pnls.length : 0;
        const variance = pnls.length > 1
            ? pnls.reduce((sum, p) => sum + Math.pow(p - avgReturn, 2), 0) / (pnls.length - 1)
            : 0;
        const stdDev = Math.sqrt(variance);
        const sharpe = stdDev > 0 ? (avgReturn / stdDev) * Math.sqrt(252) : 0;

        // Calculate average hold time
        const holdTimes = trades.map(t => t.duration_minutes ?? t.hold_time_minutes ?? 0).filter(h => h > 0);
        const avgHoldTime = holdTimes.length > 0 ? holdTimes.reduce((a, b) => a + b, 0) / holdTimes.length : 0;
        const holdPercent = Math.min(avgHoldTime / 60, 1); // Normalize to percentage (cap at 100%)

        return {
            sharpe: Math.max(-5, Math.min(5, sharpe)), // Clamp to reasonable range
            maxDrawdown: -maxDD,
            winRate: trades.length > 0 ? wins / trades.length : 0,
            holdPercent: holdPercent,
            totalTrades: trades.length
        };
    };

    // Check if model is in testing mode (not yet promoted to production)
    const isTestingMode = selectedModel?.dbStatus === 'registered';

    // Use replay metrics when in replay mode, otherwise use API metrics
    // Show zeros when model is in testing mode and no real data exists
    // Or when forceEmpty is true (dashboard cleared)
    const hasRealData = (liveMetrics.totalTrades ?? apiMetrics.total_trades ?? 0) > 0;

    const emptyMetrics: BacktestMetrics = {
        sharpe: 0,
        maxDrawdown: 0,
        winRate: 0,
        holdPercent: 0,
        totalTrades: 0
    };

    const metrics: BacktestMetrics = forceEmpty
        ? emptyMetrics  // Dashboard cleared - show zeros
        : isReplayMode
            ? calculateReplayMetrics()
            : (isTestingMode && !hasRealData)
                ? emptyMetrics  // Model in testing, no backtest run yet - show zeros
                : {
                    sharpe: liveMetrics.sharpe ?? apiMetrics.sharpe_ratio ?? 0,
                    maxDrawdown: -accurateMaxDD, // Use equity-curve-based max DD (negative value)
                    winRate: (liveMetrics.winRate ?? apiMetrics.win_rate ?? 0) / 100, // Convert to decimal
                    holdPercent: (apiMetrics.hold_pct ?? 0) / 100, // Default to 0, not 40%
                    totalTrades: liveMetrics.totalTrades ?? apiMetrics.total_trades ?? 0
                };

    const modelColor = selectedModel?.color || '#10B981';

    // Show loading skeleton
    if (isLoading) {
        return (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 sm:gap-6 lg:gap-8">
                {[1, 2, 3, 4, 5].map((i) => (
                    <div key={i} className="bg-slate-800/50 rounded-lg p-4 animate-pulse">
                        <div className="h-4 w-16 bg-slate-700 rounded mb-2" />
                        <div className="h-8 w-20 bg-slate-700 rounded" />
                    </div>
                ))}
            </div>
        );
    }

    return (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 sm:gap-6 lg:gap-8">
            <KPICard
                title="Sharpe"
                value={metrics.sharpe.toFixed(2)}
                subtitle="Risk-adjusted"
                icon={<BarChart3 className="w-4 h-4 sm:w-5 sm:h-5" />}
                color={metrics.sharpe > 1.5 ? '#10B981' : metrics.sharpe > 1 ? '#F59E0B' : '#EF4444'}
                trend={metrics.sharpe > 1.5 ? 'up' : 'neutral'}
            />
            <KPICard
                title="Max DD"
                value={`${metrics.maxDrawdown.toFixed(1)}%`}
                subtitle="Peak-to-trough"
                icon={<TrendingDown className="w-4 h-4 sm:w-5 sm:h-5" />}
                color={metrics.maxDrawdown > -10 ? '#10B981' : metrics.maxDrawdown > -15 ? '#F59E0B' : '#EF4444'}
                trend="down"
            />
            <KPICard
                title="Win Rate"
                value={`${(metrics.winRate * 100).toFixed(0)}%`}
                subtitle="Profitable"
                icon={<Target className="w-4 h-4 sm:w-5 sm:h-5" />}
                color={metrics.winRate > 0.50 ? '#10B981' : metrics.winRate > 0.40 ? '#F59E0B' : '#EF4444'}
                trend={metrics.winRate > 0.50 ? 'up' : 'down'}
            />
            <KPICard
                title="Hold Time"
                value={`${(metrics.holdPercent * 100).toFixed(0)}%`}
                subtitle="In market"
                icon={<Clock className="w-4 h-4 sm:w-5 sm:h-5" />}
                color="#8B5CF6"
            />
            <KPICard
                title="Trades"
                value={metrics.totalTrades.toLocaleString()}
                subtitle={isReplayMode ? "Replay Progress" : (hasRealData ? "Real Data" : "Sin backtest")}
                icon={<Activity className="w-4 h-4 sm:w-5 sm:h-5" />}
                color={isReplayMode ? '#06B6D4' : (hasRealData ? modelColor : '#6B7280')}
            />
        </div>
    );
}

// ============================================================================
// Price Display Card - Real-time with Market Hours Logic
// ============================================================================

interface PriceData {
    price: number;
    change: number;
    changePct: number;
    dayHigh: number | null;
    dayLow: number | null;
    week52High: number | null;
    week52Low: number | null;
    source: string;
    lastUpdate: string;
    isMarketOpen: boolean;
    marketStatus: 'open' | 'closed' | 'pre_market' | 'after_hours';
    nextUpdateMinutes: number;
}

/**
 * Format time to Colombia Time (COT = UTC-5)
 */
function formatTimeCOT(isoString: string): string {
    const date = new Date(isoString);
    // Convert to COT
    const cotDate = new Date(date.getTime() - 5 * 60 * 60 * 1000);
    const hours = cotDate.getUTCHours();
    const minutes = cotDate.getUTCMinutes().toString().padStart(2, '0');
    const ampm = hours >= 12 ? 'PM' : 'AM';
    const displayHours = hours % 12 || 12;
    return `${displayHours}:${minutes} ${ampm}`;
}

function PriceCard() {
    const [priceData, setPriceData] = useState<PriceData | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [refreshInterval, setRefreshInterval] = useState<number>(5 * 60 * 1000); // 5 min default

    useEffect(() => {
        const fetchPrice = async () => {
            try {
                setError(null);
                const response = await fetch('/api/market/realtime-price');
                const result = await response.json();

                if (result.success && result.data) {
                    const data = result.data;
                    setPriceData({
                        price: data.price,
                        change: data.change || 0,
                        changePct: data.changePct || 0,
                        dayHigh: data.dayHigh,
                        dayLow: data.dayLow,
                        week52High: data.week52High,
                        week52Low: data.week52Low,
                        source: data.originalSource || data.source,
                        lastUpdate: data.lastUpdate,
                        isMarketOpen: data.isMarketOpen,
                        marketStatus: data.marketStatus,
                        nextUpdateMinutes: data.nextUpdateMinutes || 5,
                    });

                    // Adjust refresh interval based on market status
                    const interval = data.isMarketOpen ? 5 * 60 * 1000 : 30 * 60 * 1000;
                    setRefreshInterval(interval);
                } else {
                    throw new Error(result.error || 'Failed to fetch price');
                }
            } catch (err) {
                console.error('Error fetching price:', err);
                setError('Unable to fetch live data');
                // Keep last known price if available
                if (!priceData) {
                    setPriceData({
                        price: 4200,
                        change: 0,
                        changePct: 0,
                        dayHigh: null,
                        dayLow: null,
                        week52High: null,
                        week52Low: null,
                        source: 'fallback',
                        lastUpdate: new Date().toISOString(),
                        isMarketOpen: false,
                        marketStatus: 'closed',
                        nextUpdateMinutes: 30,
                    });
                }
            } finally {
                setIsLoading(false);
            }
        };

        fetchPrice();
        const interval = setInterval(fetchPrice, refreshInterval);
        return () => clearInterval(interval);
    }, [refreshInterval]);

    if (isLoading) {
        return (
            <Card className="bg-gradient-to-br from-slate-900 to-slate-800 border-slate-700">
                <CardContent className="p-4 sm:p-6 text-center">
                    <div className="animate-pulse space-y-3">
                        <div className="h-4 w-24 mx-auto bg-slate-700 rounded" />
                        <div className="h-10 w-40 mx-auto bg-slate-700 rounded" />
                        <div className="h-4 w-16 mx-auto bg-slate-700 rounded" />
                    </div>
                </CardContent>
            </Card>
        );
    }

    const isPositive = (priceData?.changePct || 0) >= 0;

    // Market status label and color
    const marketStatusConfig = {
        open: { label: 'Mercado Abierto', color: 'bg-emerald-500', textColor: 'text-emerald-400' },
        closed: { label: 'Mercado Cerrado', color: 'bg-slate-500', textColor: 'text-slate-400' },
        pre_market: { label: 'Pre-Market', color: 'bg-amber-500', textColor: 'text-amber-400' },
        after_hours: { label: 'After Hours', color: 'bg-purple-500', textColor: 'text-purple-400' },
    };

    const statusConfig = marketStatusConfig[priceData?.marketStatus || 'closed'];

    // Source label
    const sourceLabel = {
        twelvedata: 'TwelveData',
        investing: 'Investing.com',
        cache: 'Caché',
        fallback: 'Estimado',
    }[priceData?.source || 'fallback'] || priceData?.source;

    return (
        <Card className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border-slate-700">
            <CardContent className="p-4 sm:p-6 text-center">
                {/* Header with market status */}
                <div className="flex items-center justify-center gap-2 mb-2">
                    <span className="text-xs sm:text-sm font-medium text-slate-400 uppercase tracking-widest">
                        USD/COP
                    </span>
                    <span
                        className={cn(
                            "w-2 h-2 rounded-full",
                            priceData?.isMarketOpen ? "bg-emerald-500 animate-pulse" : statusConfig.color
                        )}
                        title={statusConfig.label}
                    />
                </div>

                {/* Price */}
                <div className="text-3xl sm:text-4xl lg:text-5xl font-bold text-white tracking-tight mb-2">
                    ${priceData?.price?.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </div>

                {/* Change - absolute and percentage */}
                <div className={cn(
                    "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm sm:text-base font-semibold",
                    isPositive ? "bg-emerald-500/10 text-emerald-400" : "bg-red-500/10 text-red-400"
                )}>
                    {isPositive ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                    <span>{isPositive ? '+' : ''}{(priceData?.change || 0).toFixed(2)}</span>
                    <span className="text-xs opacity-80">({isPositive ? '+' : ''}{(priceData?.changePct || 0).toFixed(2)}%)</span>
                </div>

                {/* Day Range and 52-week Range */}
                {(priceData?.dayLow || priceData?.week52Low) && (
                    <div className="mt-3 w-full grid grid-cols-2 gap-3 text-xs">
                        {/* Day Range */}
                        {priceData?.dayLow && priceData?.dayHigh && (
                            <div className="bg-slate-800/50 rounded-lg p-2">
                                <div className="text-slate-500 mb-1">Rango día</div>
                                <div className="flex items-center justify-between gap-1">
                                    <span className="text-red-400 font-mono">{priceData.dayLow.toLocaleString('en-US', { minimumFractionDigits: 2 })}</span>
                                    <div className="flex-1 h-1 mx-1 bg-gradient-to-r from-red-500 via-slate-500 to-emerald-500 rounded-full relative">
                                        {/* Price position indicator */}
                                        <div
                                            className="absolute top-1/2 -translate-y-1/2 w-2 h-2 bg-white rounded-full shadow-lg"
                                            style={{
                                                left: `${Math.min(100, Math.max(0, ((priceData.price - priceData.dayLow) / (priceData.dayHigh - priceData.dayLow)) * 100))}%`,
                                                transform: 'translate(-50%, -50%)'
                                            }}
                                        />
                                    </div>
                                    <span className="text-emerald-400 font-mono">{priceData.dayHigh.toLocaleString('en-US', { minimumFractionDigits: 2 })}</span>
                                </div>
                            </div>
                        )}
                        {/* 52-week Range */}
                        {priceData?.week52Low && priceData?.week52High && (
                            <div className="bg-slate-800/50 rounded-lg p-2">
                                <div className="text-slate-500 mb-1">52 semanas</div>
                                <div className="flex items-center justify-between gap-1">
                                    <span className="text-red-400 font-mono">{priceData.week52Low.toLocaleString('en-US', { minimumFractionDigits: 2 })}</span>
                                    <div className="flex-1 h-1 mx-1 bg-gradient-to-r from-red-500 via-slate-500 to-emerald-500 rounded-full relative">
                                        <div
                                            className="absolute top-1/2 -translate-y-1/2 w-2 h-2 bg-cyan-400 rounded-full shadow-lg"
                                            style={{
                                                left: `${Math.min(100, Math.max(0, ((priceData.price - priceData.week52Low) / (priceData.week52High - priceData.week52Low)) * 100))}%`,
                                                transform: 'translate(-50%, -50%)'
                                            }}
                                        />
                                    </div>
                                    <span className="text-emerald-400 font-mono">{priceData.week52High.toLocaleString('en-US', { minimumFractionDigits: 2 })}</span>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {/* Status bar */}
                <div className="mt-3 flex flex-col items-center gap-1">
                    {/* Last update time (in COT) */}
                    <div className="text-xs text-slate-500">
                        Last: {priceData?.lastUpdate ? formatTimeCOT(priceData.lastUpdate) : '--:--'} COT
                    </div>

                    {/* Market status + source */}
                    <div className="flex items-center gap-2 text-[10px]">
                        <span className={cn("px-1.5 py-0.5 rounded", statusConfig.textColor, "bg-slate-800")}>
                            {statusConfig.label}
                        </span>
                        <span className="text-slate-600">•</span>
                        <span className="text-slate-500">
                            vía {sourceLabel}
                        </span>
                    </div>

                    {/* Next update indicator */}
                    <div className="text-[10px] text-slate-600 mt-1">
                        Próxima actualización: {priceData?.nextUpdateMinutes || 5} min
                    </div>
                </div>

                {/* Error indicator */}
                {error && (
                    <div className="mt-2 text-xs text-amber-400 flex items-center justify-center gap-1">
                        <AlertCircle className="w-3 h-3" />
                        {error}
                    </div>
                )}
            </CardContent>
        </Card>
    );
}

// ============================================================================
// Signal Card (Compact)
// ============================================================================
function SignalCard() {
    const { selectedModel } = useModel();
    const [signal, setSignal] = useState<'BUY' | 'SELL' | 'HOLD'>('HOLD');
    const [confidence, setConfidence] = useState<number>(0);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const timer = setTimeout(() => {
            const signals: ('BUY' | 'SELL' | 'HOLD')[] = ['BUY', 'SELL', 'HOLD'];
            setSignal(signals[Math.floor(Math.random() * 3)]);
            setConfidence(50 + Math.random() * 40);
            setIsLoading(false);
        }, 1500);
        return () => clearTimeout(timer);
    }, [selectedModel?.id]);

    if (isLoading) {
        return (
            <Card className="bg-slate-900/50 border-slate-800">
                <CardContent className="p-4 sm:p-6 text-center">
                    <div className="animate-pulse space-y-3">
                        <div className="h-3 w-20 mx-auto bg-slate-700 rounded" />
                        <div className="h-12 w-24 mx-auto bg-slate-700 rounded" />
                    </div>
                </CardContent>
            </Card>
        );
    }

    const signalConfig = {
        BUY: { color: '#10B981', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', icon: TrendingUp },
        SELL: { color: '#EF4444', bg: 'bg-red-500/10', border: 'border-red-500/30', icon: TrendingDown },
        HOLD: { color: '#6B7280', bg: 'bg-slate-500/10', border: 'border-slate-500/30', icon: Activity }
    };

    const config = signalConfig[signal];
    const Icon = config.icon;

    return (
        <Card className={cn("border-2", config.bg, config.border)}>
            <CardContent className="p-4 sm:p-6 text-center">
                <div className="text-xs sm:text-sm font-medium text-slate-400 uppercase tracking-wider mb-3">
                    Signal
                </div>
                <div className="flex flex-col items-center gap-2">
                    <div
                        className="p-2.5 sm:p-3 rounded-xl"
                        style={{ backgroundColor: `${config.color}15` }}
                    >
                        <Icon className="w-6 h-6 sm:w-8 sm:h-8" style={{ color: config.color }} />
                    </div>
                    <div
                        className="text-2xl sm:text-3xl font-black tracking-tight"
                        style={{ color: config.color }}
                    >
                        {signal}
                    </div>
                    <div className="text-xs sm:text-sm text-slate-400">
                        Conf: <span className="font-semibold text-white">{confidence.toFixed(0)}%</span>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}

// ============================================================================
// Market Status Badge
// ============================================================================
function MarketStatusBadge() {
    const [isOpen, setIsOpen] = useState(false);

    useEffect(() => {
        const now = new Date();
        const hour = now.getUTCHours();
        const day = now.getUTCDay();
        setIsOpen(day >= 1 && day <= 5 && hour >= 13 && hour < 18);
    }, []);

    return (
        <Badge
            variant="outline"
            className={cn(
                "text-xs font-semibold px-2 py-1",
                isOpen
                    ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                    : "bg-slate-500/10 text-slate-400 border-slate-500/30"
            )}
        >
            <span className={cn(
                "w-1.5 h-1.5 rounded-full mr-1.5",
                isOpen ? "bg-emerald-500 animate-pulse" : "bg-slate-500"
            )} />
            {isOpen ? 'Open' : 'Closed'}
        </Badge>
    );
}

// ============================================================================
// Equity Curve Section - Professional Line Chart with Drawdown
// ============================================================================

interface EquityCurveData {
    timestamp: string;
    value: number;
    drawdown_pct: number;
    position?: string;
    price?: number;
}

interface EquityCurveSummary {
    start_equity: number;
    end_equity: number;
    total_return: number;
    total_return_pct: number;
    max_drawdown_pct: number;
    total_points?: number;
}

/**
 * Format timestamp to Colombia Time (COT = UTC-5)
 */
function formatTimestampCOT(timestamp: string, format: 'short' | 'full' = 'short'): string {
    const date = new Date(timestamp);
    // Convert to COT (UTC-5)
    const cotDate = new Date(date.getTime() - 5 * 60 * 60 * 1000);

    const hours = cotDate.getUTCHours().toString().padStart(2, '0');
    const minutes = cotDate.getUTCMinutes().toString().padStart(2, '0');
    const day = cotDate.getUTCDate().toString().padStart(2, '0');
    const months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'];
    const month = months[cotDate.getUTCMonth()];

    if (format === 'short') {
        // Short format: "06 Ene 08:00" - includes time for chart X-axis
        return `${day} ${month} ${hours}:${minutes}`;
    }

    // Full format: same as short for now
    return `${day} ${month} ${hours}:${minutes}`;
}

/**
 * Calculate additional metrics from equity curve data
 */
function calculateMetrics(data: EquityCurveData[], summary: EquityCurveSummary) {
    if (data.length < 2) {
        return {
            sharpeRatio: 0,
            profitFactor: 0,
            avgReturn: 0,
            volatility: 0,
        };
    }

    // Calculate daily returns
    const returns: number[] = [];
    for (let i = 1; i < data.length; i++) {
        const prevValue = data[i - 1].value;
        const currValue = data[i].value;
        if (prevValue > 0) {
            returns.push((currValue - prevValue) / prevValue);
        }
    }

    if (returns.length === 0) {
        return { sharpeRatio: 0, profitFactor: 0, avgReturn: 0, volatility: 0 };
    }

    // Average return
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;

    // Volatility (standard deviation)
    const squaredDiffs = returns.map(r => Math.pow(r - avgReturn, 2));
    const volatility = Math.sqrt(squaredDiffs.reduce((a, b) => a + b, 0) / returns.length);

    // Sharpe Ratio (annualized, assuming daily data and 0% risk-free rate)
    // Multiply by sqrt(252) for annualization
    const sharpeRatio = volatility > 0 ? (avgReturn / volatility) * Math.sqrt(252) : 0;

    // Profit Factor
    const gains = returns.filter(r => r > 0).reduce((a, b) => a + b, 0);
    const losses = Math.abs(returns.filter(r => r < 0).reduce((a, b) => a + b, 0));
    const profitFactor = losses > 0 ? gains / losses : gains > 0 ? Infinity : 0;

    return {
        sharpeRatio: Math.min(5, Math.max(-5, sharpeRatio)), // Clamp to reasonable range
        profitFactor: Math.min(10, profitFactor), // Clamp to reasonable range
        avgReturn: avgReturn * 100, // Convert to percentage
        volatility: volatility * 100 * Math.sqrt(252), // Annualized volatility %
    };
}

/**
 * Custom tooltip for equity curve chart
 */
function EquityCurveTooltip({ active, payload, label }: any) {
    if (!active || !payload || !payload.length) return null;

    const data = payload[0].payload;

    return (
        <div className="bg-slate-800/95 border border-slate-600 rounded-lg p-3 shadow-xl backdrop-blur-sm">
            <p className="text-xs text-slate-400 mb-2 font-medium">
                {formatTimestampCOT(data.timestamp, 'full')} COT
            </p>
            <div className="space-y-1">
                <div className="flex items-center justify-between gap-4">
                    <span className="text-xs text-slate-400">Equity:</span>
                    <span className="font-mono text-sm text-white font-bold">
                        ${data.value?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                    </span>
                </div>
                <div className="flex items-center justify-between gap-4">
                    <span className="text-xs text-slate-400">Drawdown:</span>
                    <span className={cn(
                        "font-mono text-sm font-medium",
                        data.drawdown_pct <= -5 ? "text-red-400" : data.drawdown_pct < 0 ? "text-amber-400" : "text-green-400"
                    )}>
                        {data.drawdown_pct?.toFixed(2)}%
                    </span>
                </div>
                {data.position && (
                    <div className="flex items-center justify-between gap-4">
                        <span className="text-xs text-slate-400">Posición:</span>
                        <span className={cn(
                            "text-xs px-1.5 py-0.5 rounded font-medium",
                            data.position === 'LONG' ? "bg-green-500/20 text-green-400" :
                                data.position === 'SHORT' ? "bg-red-500/20 text-red-400" :
                                    "bg-slate-500/20 text-slate-400"
                        )}>
                            {data.position}
                        </span>
                    </div>
                )}
            </div>
        </div>
    );
}

/**
 * Mini metric card for equity curve section
 */
function MiniMetricCard({
    label,
    value,
    suffix = '',
    icon: Icon,
    isPositive
}: {
    label: string;
    value: string | number;
    suffix?: string;
    icon?: any;
    isPositive?: boolean | null;
}) {
    const colorClass = isPositive === true ? "text-green-400" :
        isPositive === false ? "text-red-400" : "text-slate-300";

    return (
        <div className="flex flex-col items-center p-2 sm:p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
            {Icon && (
                <Icon className={cn("w-3.5 h-3.5 mb-1", colorClass)} />
            )}
            <span className="text-[10px] sm:text-xs text-slate-500 uppercase tracking-wider">{label}</span>
            <span className={cn("font-mono text-sm sm:text-base font-bold", colorClass)}>
                {typeof value === 'number' ? value.toFixed(2) : value}{suffix}
            </span>
        </div>
    );
}

interface EquityCurveSectionProps {
    isReplayMode?: boolean;
    replayVisibleTrades?: Array<{
        pnl?: number;
        pnl_usd?: number;
        timestamp?: string;
        entry_time?: string;
        duration_minutes?: number;
        side?: string;
        equity_at_entry?: number;
        equity_at_exit?: number;
        current_equity?: number;  // Real-time equity from streaming trades
    }>;
}

function EquityCurveSection({ isReplayMode = false, replayVisibleTrades = [] }: EquityCurveSectionProps) {
    const { selectedModel } = useModel();
    const [data, setData] = useState<EquityCurveData[]>([]);
    const [summary, setSummary] = useState<EquityCurveSummary | null>(null);
    const [apiMetrics, setApiMetrics] = useState<any>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Fetch equity curve data AND metrics from API
    useEffect(() => {
        const fetchData = async () => {
            setIsLoading(true);
            setError(null);

            try {
                const modelId = selectedModel?.id || 'RL_PPO';

                // Fetch both equity curve and metrics in parallel
                const [equityResponse, metricsResponse] = await Promise.all([
                    fetch(`/api/models/${modelId}/equity-curve?days=90`),
                    fetch(`/api/models/${modelId}/metrics?period=all`)
                ]);

                const equityResult = await equityResponse.json();
                const metricsResult = await metricsResponse.json();

                if (equityResult.data?.points && Array.isArray(equityResult.data.points)) {
                    setData(equityResult.data.points);
                    setSummary(equityResult.data.summary);
                } else {
                    throw new Error('Invalid equity data format');
                }

                // Store API metrics (sharpe, profit_factor, etc.)
                if (metricsResult.data?.metrics) {
                    setApiMetrics(metricsResult.data.metrics);
                }
            } catch (err) {
                console.error('Error fetching equity curve:', err);
                setError('Error cargando datos');
                // Generate demo data with realistic drawdown
                const demoData: EquityCurveData[] = [];
                let equity = 10000;
                let peak = equity;

                for (let i = 0; i < 60; i++) {
                    const change = (Math.random() - 0.48) * 100;
                    equity += change;
                    equity = Math.max(8000, Math.min(12000, equity));
                    peak = Math.max(peak, equity);
                    const drawdown = ((equity - peak) / peak) * 100;

                    demoData.push({
                        timestamp: new Date(Date.now() - (59 - i) * 4 * 60 * 60 * 1000).toISOString(),
                        value: equity,
                        drawdown_pct: drawdown,
                        position: Math.random() > 0.7 ? 'LONG' : Math.random() > 0.5 ? 'SHORT' : 'FLAT'
                    });
                }

                setData(demoData);
                setSummary({
                    start_equity: demoData[0].value,
                    end_equity: demoData[demoData.length - 1].value,
                    total_return: demoData[demoData.length - 1].value - demoData[0].value,
                    total_return_pct: ((demoData[demoData.length - 1].value - demoData[0].value) / demoData[0].value) * 100,
                    max_drawdown_pct: Math.min(...demoData.map(d => d.drawdown_pct))
                });
            } finally {
                setIsLoading(false);
            }
        };

        fetchData();
    }, [selectedModel?.id]);

    // Use API metrics if available, otherwise calculate locally
    const metrics = useMemo(() => {
        // Prefer API metrics (calculated from actual trades)
        if (apiMetrics) {
            return {
                sharpeRatio: apiMetrics.sharpe_ratio ?? 0,
                profitFactor: apiMetrics.profit_factor ?? 0,
                avgReturn: apiMetrics.total_return_pct ?? 0,
                volatility: 0, // Not provided by API, will show as N/A
            };
        }
        // Fallback to local calculation
        if (!summary || data.length < 2) return null;
        return calculateMetrics(data, summary);
    }, [data, summary, apiMetrics]);

    // Calculate replay equity curve from visible trades
    // Creates TWO points per trade (entry + exit) for smooth real-time updates
    const replayChartData = useMemo(() => {
        if (!isReplayMode || replayVisibleTrades.length === 0) return [];

        const initialEquity = 10000;
        let peak = initialEquity;
        const chartPoints: Array<{
            date: string;
            equity: number;
            value: number;
            drawdown: number;
            drawdown_pct: number;
            originalDate: string;
            timestamp: string;
            position?: string;
        }> = [];

        // Add initial point
        const firstTradeTime = replayVisibleTrades[0]?.timestamp || new Date().toISOString();
        chartPoints.push({
            date: new Date(firstTradeTime).toLocaleDateString('es-CO', { day: '2-digit', month: 'short' }),
            equity: initialEquity,
            value: initialEquity,
            drawdown: 0,
            drawdown_pct: 0,
            originalDate: firstTradeTime,
            timestamp: firstTradeTime,
            position: 'flat',
        });

        // Process each trade - create entry AND exit points
        for (const trade of replayVisibleTrades) {
            const entryTime = trade.entry_time || trade.timestamp || new Date().toISOString();
            const pnl = trade.pnl ?? trade.pnl_usd ?? 0;

            // Priority for equity values:
            // 1. current_equity (from real-time streaming)
            // 2. equity_at_entry/exit (from backtest results)
            // 3. Calculate from running total (fallback)
            const hasCurrentEquity = trade.current_equity != null;
            const hasEquityFields = trade.equity_at_entry != null && trade.equity_at_exit != null;

            let entryEquity: number;
            let exitEquity: number;

            if (hasCurrentEquity) {
                // Real-time streaming: current_equity is the equity AFTER this trade
                exitEquity = trade.current_equity as number;
                entryEquity = exitEquity - pnl;
            } else if (hasEquityFields) {
                entryEquity = trade.equity_at_entry as number;
                exitEquity = trade.equity_at_exit as number;
            } else {
                // Fallback: use last point's equity as entry, add pnl for exit
                const lastEquity = chartPoints.length > 0 ? chartPoints[chartPoints.length - 1].equity : initialEquity;
                entryEquity = lastEquity;
                exitEquity = lastEquity + pnl;
            }

            // Calculate exit time
            const exitTime = trade.duration_minutes
                ? new Date(new Date(entryTime).getTime() + (trade.duration_minutes * 60000)).toISOString()
                : entryTime;

            // Update peak for drawdown
            peak = Math.max(peak, entryEquity, exitEquity);

            // Entry point
            const entryDrawdown = peak > 0 ? ((peak - entryEquity) / peak) * 100 : 0;
            chartPoints.push({
                date: new Date(entryTime).toLocaleDateString('es-CO', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }),
                equity: entryEquity,
                value: entryEquity,
                drawdown: -entryDrawdown,
                drawdown_pct: -entryDrawdown,
                originalDate: entryTime,
                timestamp: entryTime,
                position: trade.side,
            });

            // Exit point (only if different from entry time)
            if (exitTime !== entryTime) {
                const exitDrawdown = peak > 0 ? ((peak - exitEquity) / peak) * 100 : 0;
                chartPoints.push({
                    date: new Date(exitTime).toLocaleDateString('es-CO', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }),
                    equity: exitEquity,
                    value: exitEquity,
                    drawdown: -exitDrawdown,
                    drawdown_pct: -exitDrawdown,
                    originalDate: exitTime,
                    timestamp: exitTime,
                    position: 'flat',
                });
            }
        }

        return chartPoints;
    }, [isReplayMode, replayVisibleTrades]);

    // Calculate replay summary using equity_at_entry/exit for accuracy
    const replaySummary = useMemo(() => {
        if (!isReplayMode) return null;

        const initialEquity = 10000;
        if (replayVisibleTrades.length === 0) {
            return {
                start_equity: initialEquity,
                end_equity: initialEquity,
                total_return: 0,
                total_return_pct: 0,
                max_drawdown_pct: 0,
            };
        }

        // Calculate max drawdown using equity_at_entry/exit if available
        let peak = initialEquity;
        let maxDD = 0;
        let lastEquity = initialEquity;

        for (const trade of replayVisibleTrades) {
            const pnl = trade.pnl ?? trade.pnl_usd ?? 0;
            // Priority: current_equity > equity_at_entry/exit > fallback calculation
            const hasCurrentEquity = trade.current_equity != null;
            const hasEquityFields = trade.equity_at_entry != null && trade.equity_at_exit != null;

            let entryEquity: number;
            let exitEquity: number;

            if (hasCurrentEquity) {
                // Real-time streaming: current_equity is the equity AFTER this trade
                exitEquity = trade.current_equity as number;
                entryEquity = exitEquity - pnl;
            } else if (hasEquityFields) {
                entryEquity = trade.equity_at_entry as number;
                exitEquity = trade.equity_at_exit as number;
            } else {
                entryEquity = lastEquity;
                exitEquity = lastEquity + pnl;
            }

            // Check drawdown at entry and exit
            peak = Math.max(peak, entryEquity, exitEquity);
            const entryDD = peak > 0 ? ((peak - entryEquity) / peak) * 100 : 0;
            const exitDD = peak > 0 ? ((peak - exitEquity) / peak) * 100 : 0;
            maxDD = Math.max(maxDD, entryDD, exitDD);

            lastEquity = exitEquity;
        }

        const totalPnl = lastEquity - initialEquity;

        return {
            start_equity: initialEquity,
            end_equity: lastEquity,
            total_return: totalPnl,
            total_return_pct: (totalPnl / initialEquity) * 100,
            max_drawdown_pct: -maxDD,
        };
    }, [isReplayMode, replayVisibleTrades]);

    // Replay metrics
    const replayMetrics = useMemo(() => {
        if (!isReplayMode || replayVisibleTrades.length < 2) {
            return {
                sharpeRatio: 0,
                profitFactor: 0,
                avgReturn: 0,
                volatility: 0,
            };
        }

        const pnls = replayVisibleTrades.map(t => t.pnl ?? t.pnl_usd ?? 0);
        const wins = pnls.filter(p => p > 0);
        const losses = pnls.filter(p => p < 0);

        const totalWins = wins.reduce((sum, p) => sum + p, 0);
        const totalLosses = Math.abs(losses.reduce((sum, p) => sum + p, 0));
        const profitFactor = totalLosses > 0 ? totalWins / totalLosses : totalWins > 0 ? 999 : 0;

        const avgReturn = pnls.reduce((sum, p) => sum + p, 0) / pnls.length;
        const variance = pnls.reduce((sum, p) => sum + Math.pow(p - avgReturn, 2), 0) / (pnls.length - 1);
        const stdDev = Math.sqrt(variance);
        const sharpe = stdDev > 0 ? (avgReturn / stdDev) * Math.sqrt(252) : 0;

        return {
            sharpeRatio: Math.max(-5, Math.min(5, sharpe)),
            profitFactor: Math.min(profitFactor, 99),
            avgReturn: replaySummary?.total_return_pct ?? 0,
            volatility: stdDev,
        };
    }, [isReplayMode, replayVisibleTrades, replaySummary]);

    // Prepare chart data with proper formatting
    const chartData = useMemo(() => {
        // Use replay data when in replay mode
        if (isReplayMode) return replayChartData;

        if (data.length === 0) return [];

        // Sample data if too many points (for performance)
        const maxPoints = 100;
        const step = data.length > maxPoints ? Math.ceil(data.length / maxPoints) : 1;
        const sampled = data.filter((_, i) => i % step === 0 || i === data.length - 1);

        return sampled.map(point => ({
            ...point,
            // Invert drawdown for area chart (positive values for shading)
            drawdown_area: Math.abs(point.drawdown_pct),
        }));
    }, [data, isReplayMode, replayChartData]);

    // Calculate Y-axis domain with 20% padding
    const yDomain = useMemo(() => {
        if (chartData.length === 0) return [9000, 11000];

        const values = chartData.map(d => d.value);
        const min = Math.min(...values);
        const max = Math.max(...values);
        const range = max - min || 1000;
        const padding = range * 0.2;

        return [Math.floor(min - padding), Math.ceil(max + padding)];
    }, [chartData]);

    const modelColor = selectedModel?.color || '#10B981';

    // Use replay summary when in replay mode, otherwise use API summary
    const displaySummary = isReplayMode ? replaySummary : summary;
    const displayMetrics = isReplayMode ? replayMetrics : metrics;

    const isPositiveReturn = (displaySummary?.total_return_pct || 0) >= 0;
    const gradientId = `equity-gradient-${selectedModel?.id || 'default'}`;
    const drawdownGradientId = `drawdown-gradient-${selectedModel?.id || 'default'}`;

    // Loading state - but skip if in replay mode with streaming trades
    // This allows real-time updates even while API data is still loading
    if (isLoading && !isReplayMode && replayVisibleTrades.length === 0) {
        return (
            <Card className="bg-slate-900/40 border-slate-800">
                <CardHeader className="py-3 px-4 border-b border-slate-800/50">
                    <CardTitle className="text-sm font-medium text-slate-300 flex items-center gap-2">
                        <LineChart className="w-4 h-4" />
                        Equity Curve
                    </CardTitle>
                </CardHeader>
                <CardContent className="p-4">
                    <div className="h-[280px] sm:h-[320px] flex items-center justify-center">
                        <RefreshCw className="w-8 h-8 text-cyan-500 animate-spin" />
                    </div>
                </CardContent>
            </Card>
        );
    }

    return (
        <Card className="bg-slate-900/40 border-slate-800">
            <CardHeader className="py-3 px-4 border-b border-slate-800/50">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                    <CardTitle className="text-sm font-medium text-slate-300 flex items-center gap-2">
                        <LineChart className="w-4 h-4" style={{ color: modelColor }} />
                        Equity Curve
                        {selectedModel && (
                            <span
                                className="text-xs px-2 py-0.5 rounded"
                                style={{ backgroundColor: `${modelColor}20`, color: modelColor }}
                            >
                                {selectedModel.name}
                            </span>
                        )}
                        <Badge variant="outline" className="text-[10px] bg-cyan-500/10 text-cyan-400 border-cyan-500/30 ml-1">
                            COT
                        </Badge>
                    </CardTitle>

                    {/* Return Badge */}
                    <div className="flex items-center gap-3">
                        <div className="flex items-center gap-2">
                            <span className="text-xs text-slate-500">Equity:</span>
                            <span className="font-mono text-sm text-white font-bold">
                                ${(displaySummary?.end_equity || 10000).toLocaleString(undefined, { minimumFractionDigits: 0 })}
                            </span>
                            {isReplayMode && (
                                <span className="text-xs px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-400">REPLAY</span>
                            )}
                        </div>
                        <div className={cn(
                            "flex items-center gap-1 px-2 py-1 rounded-md text-sm font-bold",
                            isPositiveReturn ? "bg-green-500/10 text-green-400" : "bg-red-500/10 text-red-400"
                        )}>
                            {isPositiveReturn ? <TrendingUp className="w-3.5 h-3.5" /> : <TrendingDown className="w-3.5 h-3.5" />}
                            {isPositiveReturn ? '+' : ''}{(displaySummary?.total_return_pct || 0).toFixed(2)}%
                        </div>
                    </div>
                </div>
            </CardHeader>

            <CardContent className="p-2 sm:p-4">
                {/* Chart */}
                <div className="h-[200px] sm:h-[240px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                            <defs>
                                {/* Gradient for equity line */}
                                <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                                    <stop
                                        offset="5%"
                                        stopColor={isPositiveReturn ? '#10B981' : '#EF4444'}
                                        stopOpacity={0.4}
                                    />
                                    <stop
                                        offset="95%"
                                        stopColor={isPositiveReturn ? '#10B981' : '#EF4444'}
                                        stopOpacity={0.05}
                                    />
                                </linearGradient>
                                {/* Gradient for drawdown */}
                                <linearGradient id={drawdownGradientId} x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#EF4444" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#EF4444" stopOpacity={0.02} />
                                </linearGradient>
                            </defs>

                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} vertical={false} />

                            {/* X-Axis with COT timestamps */}
                            <XAxis
                                dataKey="timestamp"
                                tickFormatter={(ts) => formatTimestampCOT(ts, 'short')}
                                stroke="#475569"
                                fontSize={10}
                                tickLine={false}
                                axisLine={false}
                                minTickGap={40}
                            />

                            {/* Y-Axis with 20% padding */}
                            <YAxis
                                domain={yDomain}
                                stroke="#475569"
                                fontSize={10}
                                tickLine={false}
                                axisLine={false}
                                tickFormatter={(value) => `$${(value / 1000).toFixed(1)}k`}
                                width={50}
                            />

                            <Tooltip content={<EquityCurveTooltip />} />

                            {/* Reference line at initial capital */}
                            <ReferenceLine
                                y={displaySummary?.start_equity || 10000}
                                stroke="#64748b"
                                strokeDasharray="5 5"
                                strokeWidth={1}
                            />

                            {/* Equity Area */}
                            <Area
                                type="monotone"
                                dataKey="value"
                                stroke={isPositiveReturn ? '#10B981' : '#EF4444'}
                                strokeWidth={2}
                                fill={`url(#${gradientId})`}
                                dot={false}
                                activeDot={{
                                    r: 5,
                                    fill: isPositiveReturn ? '#10B981' : '#EF4444',
                                    stroke: '#0f172a',
                                    strokeWidth: 2
                                }}
                            />
                        </ComposedChart>
                    </ResponsiveContainer>
                </div>

                {/* Metrics Row */}
                <div className="mt-4 grid grid-cols-4 gap-2 sm:gap-3">
                    <MiniMetricCard
                        label="Max DD"
                        value={displaySummary?.max_drawdown_pct || 0}
                        suffix="%"
                        icon={TrendingDown}
                        isPositive={(displaySummary?.max_drawdown_pct || 0) > -10}
                    />
                    <MiniMetricCard
                        label="Sharpe"
                        value={displayMetrics?.sharpeRatio || 0}
                        icon={BarChart3}
                        isPositive={displayMetrics?.sharpeRatio ? displayMetrics.sharpeRatio > 1 : null}
                    />
                    <MiniMetricCard
                        label="P. Factor"
                        value={displayMetrics?.profitFactor || 0}
                        icon={Target}
                        isPositive={displayMetrics?.profitFactor ? displayMetrics.profitFactor > 1 : null}
                    />
                    <MiniMetricCard
                        label="Volatility"
                        value={displayMetrics?.volatility || 0}
                        suffix="%"
                        icon={Activity}
                        isPositive={null}
                    />
                </div>

                {/* Demo data indicator */}
                {error && (
                    <div className="mt-3 px-2 py-1.5 rounded bg-amber-500/10 border border-amber-500/20 text-center">
                        <span className="text-xs text-amber-400">
                            Demo data - Datos simulados para ilustración
                        </span>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}

// ============================================================================
// Chart Loading Fallback
// ============================================================================
function ChartLoadingFallback() {
    return (
        <Card className="bg-slate-900/40 border-slate-800">
            <CardHeader className="py-3 px-4 border-b border-slate-800/50">
                <CardTitle className="text-sm font-medium text-slate-300">
                    USD/COP Price Chart
                </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
                <div className="h-[300px] sm:h-[400px] flex items-center justify-center">
                    <div className="text-center">
                        <RefreshCw className="w-10 h-10 text-cyan-400 animate-spin mx-auto mb-3" />
                        <p className="text-slate-400 text-sm">Loading chart...</p>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}

// ============================================================================
// Section Header Component - Centered with decorative line
// ============================================================================
interface SectionHeaderProps {
    title: string;
    subtitle?: string;
    icon?: React.ReactNode;
}

function SectionHeader({ title, subtitle, icon }: SectionHeaderProps) {
    return (
        <div className="mb-8 sm:mb-10 lg:mb-12 text-center flex flex-col items-center">
            <h2 className="text-lg sm:text-xl lg:text-2xl font-bold text-white flex items-center justify-center gap-3">
                {icon && <span className="text-cyan-400">{icon}</span>}
                {title}
            </h2>
            {subtitle && (
                <p className="mt-3 text-sm sm:text-base text-slate-400 max-w-2xl">
                    {subtitle}
                </p>
            )}
            {/* Decorative line */}
            <div className="mt-4 flex items-center justify-center gap-1">
                <div className="h-0.5 w-8 rounded-full bg-gradient-to-r from-transparent to-cyan-500/50" />
                <div className="h-0.5 w-16 rounded-full bg-gradient-to-r from-cyan-500/50 to-blue-500/50" />
                <div className="h-0.5 w-8 rounded-full bg-gradient-to-r from-blue-500/50 to-transparent" />
            </div>
        </div>
    );
}

// ============================================================================
// Dashboard Content
// ============================================================================
function DashboardContent() {
    const { selectedModel, models, setSelectedModel } = useModel();
    const [lastUpdate, setLastUpdate] = useState<string>('--:--:--');
    const [isReplayMode, setIsReplayMode] = useState(false);
    const [isBacktestPanelExpanded, setIsBacktestPanelExpanded] = useState(true);
    const [backtestDateRange, setBacktestDateRange] = useState<{ start: string; end: string } | null>(null);
    const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);

    // Real-time streaming trades (accumulates as backtest runs)
    const [streamingTrades, setStreamingTrades] = useState<BacktestTradeEvent[]>([]);

    // Animation state for animated backtest replay
    const [isAnimating, setIsAnimating] = useState(false);
    const [animationBarIndex, setAnimationBarIndex] = useState(0);
    const [animationSpeed, setAnimationSpeed] = useState(1); // bars per second
    const [totalBars, setTotalBars] = useState(0);
    const animationRef = useRef<NodeJS.Timeout | null>(null);

    // Clear key - changes when dashboard is cleared to force components to remount
    const [clearKey, setClearKey] = useState(0);

    // Track if dashboard was explicitly cleared (forces empty state until next backtest)
    const [isDashboardCleared, setIsDashboardCleared] = useState(false);

    // Map frontend model ID to backend model ID for inference
    const inferenceModelId = useMemo(() => {
        if (!selectedModel?.id) return 'ppo_v20';
        // Map frontend model IDs to inference service model IDs
        const modelMap: Record<string, string> = {
            'ppo_v19_prod': 'ppo_v19',
            'ppo_v20_prod': 'ppo_v20',
        };
        return modelMap[selectedModel.id] || 'ppo_v20';
    }, [selectedModel?.id]);

    // Replay system hook with hybrid timeline enabled
    const replay = useReplay({
        initialMode: 'validation',
        autoLoad: false,
        enableTimeline: true, // Enable hybrid replay with timeline-based navigation
        modelId: inferenceModelId, // Use selected model for inference
        forceRegenerate: true, // Force regeneration to ensure complete data
        onTradeAppear: (trade) => {
            console.log('[Replay] New trade appeared:', trade.trade_id);
        },
        onComplete: () => {
            console.log('[Replay] Replay completed');
        },
    });

    // Set time only on client to avoid hydration mismatch
    useEffect(() => {
        setLastUpdate(new Date().toLocaleTimeString());
        const interval = setInterval(() => {
            setLastUpdate(new Date().toLocaleTimeString());
        }, 60000);
        return () => clearInterval(interval);
    }, []);

    // Toggle replay mode
    const toggleReplayMode = () => {
        if (isReplayMode) {
            replay.stop();
            replay.reset();
        }
        setIsReplayMode(!isReplayMode);
    };

    // Clear dashboard - reset all state to clean slate
    const handleClearDashboard = useCallback(() => {
        console.log('[Dashboard] Clearing dashboard');
        // Stop any running animation
        if (animationRef.current) {
            clearInterval(animationRef.current);
            animationRef.current = null;
        }
        // Reset all state
        setIsAnimating(false);
        setAnimationBarIndex(0);
        setTotalBars(0);
        setBacktestResult(null);
        setBacktestDateRange(null);
        setIsReplayMode(false);
        setStreamingTrades([]); // Clear streaming trades
        replay.stop();
        replay.reset();
        // Increment clear key to force components to remount with fresh state
        setClearKey(prev => prev + 1);
        // Mark dashboard as explicitly cleared (forces empty state)
        setIsDashboardCleared(true);
        console.log('[Dashboard] Dashboard cleared');
    }, [replay]);

    // Start animated replay
    const startAnimatedReplay = useCallback((numBars: number) => {
        console.log(`[Dashboard] Starting animated replay with ${numBars} bars`);
        setTotalBars(numBars);
        setAnimationBarIndex(0);
        setIsAnimating(true);

        // Clear any existing animation
        if (animationRef.current) {
            clearInterval(animationRef.current);
        }

        // Calculate step size to complete animation in ~30 seconds
        // With 60fps updates (16ms interval), we need numBars / (30 * 60) bars per tick
        const TARGET_DURATION_SECONDS = 30;
        const TICK_INTERVAL_MS = 50; // 20 updates per second for smooth animation
        const totalTicks = (TARGET_DURATION_SECONDS * 1000) / TICK_INTERVAL_MS;
        const barsPerTick = Math.max(1, Math.ceil(numBars / totalTicks));

        console.log(`[Dashboard] Animation: ${numBars} bars, ${barsPerTick} bars/tick, ~${TARGET_DURATION_SECONDS}s duration`);

        // Start animation timer - advance multiple bars per tick for faster playback
        animationRef.current = setInterval(() => {
            setAnimationBarIndex(prev => {
                const next = prev + (barsPerTick * animationSpeed);
                if (next >= numBars) {
                    // Animation complete
                    if (animationRef.current) {
                        clearInterval(animationRef.current);
                        animationRef.current = null;
                    }
                    setIsAnimating(false);
                    console.log('[Dashboard] Animated replay completed');
                    return numBars;
                }
                return next;
            });
        }, TICK_INTERVAL_MS);
    }, [animationSpeed]);

    // Stop animation
    const stopAnimation = useCallback(() => {
        if (animationRef.current) {
            clearInterval(animationRef.current);
            animationRef.current = null;
        }
        setIsAnimating(false);
    }, []);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (animationRef.current) {
                clearInterval(animationRef.current);
            }
        };
    }, []);

    // Handle backtest start - clear all previous state for fresh start
    const handleBacktestStart = useCallback(() => {
        console.log('[Dashboard] Backtest starting - clearing all previous state');
        // Clear streaming trades from any previous run
        setStreamingTrades([]);
        // Clear previous backtest result to avoid stale data interference
        setBacktestResult(null);
        setBacktestDateRange(null);
        // Reset replay mode and manual replay state
        setIsReplayMode(false);
        setIsDashboardCleared(false);
        // Stop and reset the replay system to clear any stale replay.visibleTrades
        replay.stop();
        replay.reset();
        // Stop any running animation
        if (animationRef.current) {
            clearInterval(animationRef.current);
            animationRef.current = null;
        }
        setIsAnimating(false);
        setAnimationBarIndex(0);
    }, [replay]);

    // Handle real-time trade events from backtest streaming
    // This updates the equity curve live as trades are generated
    const handleTradeGenerated = useCallback((trade: BacktestTradeEvent) => {
        console.log(`[Dashboard] Real-time trade #${trade.trade_id}: ${trade.side} @ ${trade.entry_price}, equity=${trade.current_equity}`);

        // Clear dashboard cleared flag since we have new data
        setIsDashboardCleared(false);

        // Activate replay mode to show live updates
        setIsReplayMode(true);

        // Accumulate trades for real-time equity curve updates
        setStreamingTrades(prev => [...prev, trade]);
    }, []);

    // Handle backtest completion - show final results
    // NOTE: Don't call handleClearDashboard here - it would clear streaming trades
    // The streaming trades are already cleared when a NEW backtest starts (handleBacktestStart)
    const handleBacktestComplete = useCallback((result: BacktestResult, startDate: string, endDate: string) => {
        console.log(`[Dashboard] Backtest completed: ${result.trade_count} trades from ${startDate} to ${endDate}`);
        console.log('[Dashboard] Result summary:', result.summary);
        console.log('[Dashboard] First trade:', result.trades?.[0]);

        // Stop any running animation first
        if (animationRef.current) {
            clearInterval(animationRef.current);
            animationRef.current = null;
        }
        setIsAnimating(false);
        setAnimationBarIndex(0);

        // Clear streaming trades - we now have the final result
        // The final result contains all trades with equity_at_entry/exit
        console.log('[Dashboard] Clearing streamingTrades, setting backtestResult');
        setStreamingTrades([]);

        // Store the date range for chart
        setBacktestDateRange({ start: startDate, end: endDate });

        // Store backtest result for display in other components
        setBacktestResult(result);

        // Clear the "dashboard cleared" flag since we have new data
        setIsDashboardCleared(false);

        // Activate replay mode to show all trades immediately (no animation)
        // The user already saw the trades streaming in real-time, no need to animate again
        if (result.trade_count > 0) {
            setIsReplayMode(true);
            console.log(`[Dashboard] Backtest complete - isReplayMode=true, ${result.trades?.length || 0} trades available`);
            // Don't start animation - just show all trades at once
        }
    }, []);

    // Compute animation date based on current bar index
    // Each bar is 5 minutes, starting from the backtest start date
    const animationEndDate = useMemo(() => {
        if (!isAnimating || !backtestDateRange) return null;
        const startDate = new Date(backtestDateRange.start + 'T08:00:00');
        // Each bar = 5 minutes
        const msPerBar = 5 * 60 * 1000;
        const animationMs = animationBarIndex * msPerBar;
        return new Date(startDate.getTime() + animationMs);
    }, [isAnimating, backtestDateRange, animationBarIndex]);

    // Filter trades to show only those up to the current animation point
    const visibleAnimationTrades = useMemo(() => {
        if (!backtestResult?.trades || !isAnimating || !animationEndDate) {
            return backtestResult?.trades || [];
        }
        return backtestResult.trades.filter(trade => {
            const tradeTime = new Date(trade.entry_time || trade.timestamp);
            return tradeTime <= animationEndDate;
        });
    }, [backtestResult?.trades, isAnimating, animationEndDate]);

    // Animation progress percentage
    const animationProgress = totalBars > 0 ? Math.round((animationBarIndex / totalBars) * 100) : 0;

    // Debug: Log equity curve data source on every render (when data changes)
    const equityCurveDataSource = useMemo(() => {
        const source = streamingTrades.length > 0
            ? 'streamingTrades'
            : isAnimating
                ? 'visibleAnimationTrades'
                : replay.visibleTrades.length > 0
                    ? 'replay.visibleTrades'
                    : backtestResult?.trades?.length
                        ? 'backtestResult.trades'
                        : 'none';
        const count = streamingTrades.length > 0
            ? streamingTrades.length
            : isAnimating
                ? visibleAnimationTrades.length
                : replay.visibleTrades.length > 0
                    ? replay.visibleTrades.length
                    : backtestResult?.trades?.length || 0;
        console.log(`[Dashboard] Equity curve source: ${source} (${count} trades), isReplayMode=${isReplayMode}, isAnimating=${isAnimating}`);
        return { source, count };
    }, [streamingTrades.length, isAnimating, visibleAnimationTrades.length, replay.visibleTrades.length, backtestResult?.trades?.length, isReplayMode]);

    return (
        <div className="min-h-screen bg-gradient-to-b from-[#030712] via-[#0f172a] to-[#030712] overflow-x-hidden">
            {/* Global Navigation (fixed position) */}
            <GlobalNavbar currentPage="dashboard" />

            {/* Sticky Header - Compact single row */}
            <header className="sticky top-16 z-40 bg-[#030712]/95 backdrop-blur-xl border-b border-slate-800/50">
                <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex items-center justify-between gap-3 py-3 sm:py-4">
                        {/* Left: Title + Badges */}
                        <div className="flex items-center gap-3 sm:gap-4 min-w-0">
                            <h1 className="text-lg sm:text-xl lg:text-2xl font-bold bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 bg-clip-text text-transparent whitespace-nowrap">
                                USD/COP Trading
                            </h1>
                            <div className="hidden sm:flex items-center gap-2">
                                <Badge variant="outline" className="bg-purple-500/10 text-purple-400 border-purple-500/30 text-[10px] px-2 py-0.5">
                                    PAPER
                                </Badge>
                                <MarketStatusBadge />
                            </div>
                        </div>

                        {/* Center: Backtest Toggle */}
                        <button
                            onClick={() => setIsBacktestPanelExpanded(!isBacktestPanelExpanded)}
                            className={cn(
                                "flex items-center gap-2 px-3 py-2 rounded-lg transition-all duration-200",
                                "border text-xs sm:text-sm font-medium",
                                isBacktestPanelExpanded
                                    ? "bg-cyan-500/20 border-cyan-500/50 text-cyan-400"
                                    : "bg-slate-800/80 border-slate-700 text-slate-400 hover:border-slate-600 hover:text-white"
                            )}
                        >
                            <Calendar className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                            <span className="hidden xs:inline">Backtest</span>
                            <ChevronDown className={cn(
                                "w-3 h-3 transition-transform",
                                isBacktestPanelExpanded && "rotate-180"
                            )} />
                        </button>

                        {/* Right: Model Selector */}
                        <div className="flex items-center gap-2 sm:gap-3">
                            <ModelDropdown />
                            {/* Replay button removed - backtest handles replay automatically */}
                        </div>
                    </div>
                </div>
            </header>

            {/* Backtest Control Panel - Only rendered when expanded */}
            {isBacktestPanelExpanded && (
                <div className="sticky top-[7.5rem] z-30">
                    <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-2">
                        <BacktestControlPanel
                            models={models.map(m => ({ id: m.id, name: m.name }))}
                            selectedModelId={selectedModel?.id || 'ppo_v20'}
                            onModelChange={(modelId) => setSelectedModel(modelId)}
                            onBacktestComplete={handleBacktestComplete}
                            onTradeGenerated={handleTradeGenerated}
                            onBacktestStart={handleBacktestStart}
                            onClearDashboard={handleClearDashboard}
                            enableAnimatedReplay={true}
                            expanded={true}
                            onToggleExpand={() => setIsBacktestPanelExpanded(false)}
                            hideHeader={true}
                        />
                    </div>
                </div>
            )}

            {/* Animation Progress Bar - Shows during animated replay (Mobile-first) */}
            {isAnimating && (
                <div className="sticky top-[4rem] sm:top-[7.5rem] z-30 bg-slate-900/95 backdrop-blur border-b border-slate-700/50">
                    <div className="w-full max-w-6xl mx-auto px-3 sm:px-6 lg:px-8 py-2 sm:py-3">
                        {/* Mobile: 2 rows, Desktop: 1 row */}
                        <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4">
                            {/* Row 1: Label + Progress bar + Stop button */}
                            <div className="flex items-center gap-2 sm:gap-3 flex-1 min-w-0">
                                <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
                                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                                    <span className="text-xs sm:text-sm font-medium text-white whitespace-nowrap">Replay</span>
                                </div>
                                {/* Progress bar - takes remaining space */}
                                <div className="flex-1 min-w-0">
                                    <div className="relative h-1.5 sm:h-2 bg-slate-800 rounded-full overflow-hidden">
                                        <div
                                            className="absolute inset-y-0 left-0 bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-300"
                                            style={{ width: `${animationProgress}%` }}
                                        />
                                    </div>
                                </div>
                                {/* Percentage - always visible */}
                                <span className="text-xs font-mono text-cyan-400 shrink-0">{animationProgress}%</span>
                                {/* Stop button */}
                                <button
                                    onClick={stopAnimation}
                                    className="px-2 sm:px-3 py-1 sm:py-1.5 text-xs font-medium text-red-400 border border-red-500/30 rounded-full hover:bg-red-500/10 transition-colors shrink-0"
                                >
                                    Detener
                                </button>
                            </div>
                            {/* Row 2 (mobile) / Inline (desktop): Stats */}
                            <div className="flex items-center justify-between sm:justify-end gap-2 sm:gap-3 text-[10px] sm:text-xs text-slate-400">
                                <span className="font-mono text-cyan-400">
                                    {animationEndDate ? animationEndDate.toLocaleDateString('es-CO', { day: '2-digit', month: 'short' }) : '-'}
                                </span>
                                <span className="hidden sm:inline">|</span>
                                <span>{visibleAnimationTrades.length} trades</span>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Minimal spacer for fixed navbar */}
            <div className="h-16" aria-hidden="true" />

            {/* Main Content */}
            <main className="w-full overflow-x-hidden">

                {/* ════════════════════════════════════════════════════════════════
                    Section 1: KPIs - Backtest Metrics
                ════════════════════════════════════════════════════════════════ */}
                <section className="w-full py-8 sm:py-10 lg:py-12 flex flex-col items-center">
                    <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
                        <SectionHeader
                            title="Backtest Metrics"
                            subtitle="Key performance indicators from historical backtesting"
                            icon={<BarChart3 className="w-5 h-5 sm:w-6 sm:h-6" />}
                        />
                        <BacktestMetricsPanel
                            key={`metrics-${clearKey}`}
                            isReplayMode={isReplayMode || !!backtestResult || isAnimating || streamingTrades.length > 0}
                            forceEmpty={isDashboardCleared}
                            replayVisibleTrades={
                                // Priority: streaming trades > animation > replay > backtest result
                                streamingTrades.length > 0
                                    ? streamingTrades  // Real-time streaming during backtest
                                    : isAnimating
                                        ? visibleAnimationTrades
                                        : (replay.visibleTrades.length > 0 ? replay.visibleTrades : (backtestResult?.trades || []))
                            }
                        />
                    </div>
                </section>

                {/* ════════════════════════════════════════════════════════════════
                    Section 2: Live Price + Signal
                ════════════════════════════════════════════════════════════════ */}
                <section className="w-full py-12 sm:py-16 lg:py-20 bg-slate-900/30 flex flex-col items-center">
                    <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
                        <SectionHeader
                            title="Market Overview"
                            subtitle="Real-time price and ML-powered trading signal"
                            icon={<Activity className="w-5 h-5 sm:w-6 sm:h-6" />}
                        />
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8 lg:gap-10">
                            <div className="sm:col-span-1 lg:col-span-2">
                                <PriceCard />
                            </div>
                            <div className="flex flex-col items-center">
                                <SignalCard />
                            </div>
                        </div>
                    </div>
                </section>

                {/* ════════════════════════════════════════════════════════════════
                    Section 3: Candlestick Chart
                ════════════════════════════════════════════════════════════════ */}
                <section className="w-full py-12 sm:py-16 lg:py-20 flex flex-col items-center">
                    <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
                        <SectionHeader
                            title="Price Chart"
                            subtitle="Japanese candlestick chart with trading signals overlay"
                            icon={<BarChart3 className="w-5 h-5 sm:w-6 sm:h-6" />}
                        />
                        <Suspense fallback={<ChartLoadingFallback />}>
                            <div className="rounded-2xl overflow-hidden border border-slate-700/50 shadow-2xl">
                                <TradingChartWithSignals
                                    key={`chart-${selectedModel?.id || 'default'}-${clearKey}${isReplayMode || backtestResult || streamingTrades.length > 0 ? '-backtest' : ''}${backtestDateRange?.start || ''}`}
                                    symbol="USDCOP"
                                    timeframe="5m"
                                    height={400}
                                    showSignals={true}
                                    showPositions={false}
                                    enableRealTime={!isReplayMode && !backtestResult && streamingTrades.length === 0}
                                    modelId={selectedModel?.id || 'ppo_v19_prod'}
                                    startDate={backtestDateRange ? backtestDateRange.start : undefined}
                                    endDate={
                                        // Priority: streaming > animation > replay > backtest result
                                        streamingTrades.length > 0
                                            // During streaming, use the latest trade's exit_time to progressively reveal candles
                                            ? (() => {
                                                const lastTrade = streamingTrades[streamingTrades.length - 1];
                                                // Use exit_time if available, otherwise entry_time + duration
                                                const exitTime = lastTrade.exit_time
                                                    ? new Date(lastTrade.exit_time)
                                                    : new Date(new Date(lastTrade.entry_time || lastTrade.timestamp).getTime() + (lastTrade.duration_minutes || 0) * 60000);
                                                console.log(`[Dashboard] Price chart endDate: ${exitTime.toISOString()} (streaming trade #${lastTrade.trade_id})`);
                                                return exitTime;
                                            })()
                                            : isAnimating && animationEndDate
                                                ? animationEndDate
                                                : replay.isPlaying && replay.state.currentDate
                                                    ? replay.state.currentDate
                                                    : backtestResult
                                                        ? new Date(backtestDateRange?.end + 'T23:59:59')
                                                        : undefined
                                    }
                                    isReplayMode={isReplayMode || !!backtestResult || streamingTrades.length > 0}
                                    replayVisibleTradeIds={
                                        // Priority: streaming > animation > replay > backtest result
                                        streamingTrades.length > 0
                                            ? new Set(streamingTrades.map(t => String(t.trade_id)))
                                            : isAnimating
                                                ? new Set(visibleAnimationTrades.map(t => String(t.trade_id)))
                                                : isReplayMode
                                                    ? new Set(replay.visibleTrades.map(t => String(t.trade_id)))
                                                    : (backtestResult ? new Set(backtestResult.trades.map(t => String(t.trade_id))) : undefined)
                                    }
                                    replayTrades={
                                        // Priority: streaming > animation > replay > backtest result
                                        (streamingTrades.length > 0
                                            ? streamingTrades
                                            : isAnimating
                                                ? visibleAnimationTrades
                                                : (replay.visibleTrades.length > 0 ? replay.visibleTrades : backtestResult?.trades || [])
                                        ).map(t => ({
                                            trade_id: t.trade_id,
                                            timestamp: t.timestamp || t.entry_time,
                                            entry_time: t.entry_time,
                                            side: t.side || 'buy',
                                            entry_price: t.entry_price,
                                            pnl: t.pnl ?? t.pnl_usd,
                                            status: t.status,
                                        }))
                                    }
                                />
                            </div>
                        </Suspense>
                    </div>
                </section>

                {/* ════════════════════════════════════════════════════════════════
                    Section 4: Equity Curve + Profit Summary
                ════════════════════════════════════════════════════════════════ */}
                <section className="w-full py-12 sm:py-16 lg:py-20 bg-slate-900/30 flex flex-col items-center">
                    <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
                        <SectionHeader
                            title="Performance"
                            subtitle="Portfolio equity curve and trading summary"
                            icon={<LineChart className="w-5 h-5 sm:w-6 sm:h-6" />}
                        />
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 sm:gap-8 lg:gap-10">
                            <div className="lg:col-span-2">
                                <EquityCurveSection
                                    isReplayMode={isReplayMode || !!backtestResult || streamingTrades.length > 0}
                                    replayVisibleTrades={
                                        // Priority: streaming trades > animation > replay > backtest result
                                        streamingTrades.length > 0
                                            ? streamingTrades  // Real-time streaming during backtest
                                            : isAnimating
                                                ? visibleAnimationTrades
                                                : (replay.visibleTrades.length > 0 ? replay.visibleTrades : (backtestResult?.trades || []))
                                    }
                                />
                            </div>
                            <div className="flex flex-col items-center lg:items-stretch">
                                <TradingSummaryCard
                                    isReplayMode={isReplayMode || isAnimating || streamingTrades.length > 0}
                                    forceEmpty={isDashboardCleared}
                                    replayVisibleTrades={
                                        // Priority: streaming trades > animation > replay
                                        streamingTrades.length > 0
                                            ? streamingTrades
                                            : (isAnimating ? visibleAnimationTrades : replay.visibleTrades)
                                    }
                                    backtestSummary={
                                        // Real-time streaming trades - compute summary on the fly
                                        streamingTrades.length > 0
                                            ? (() => {
                                                const trades = streamingTrades;
                                                const wins = trades.filter(t => (t.pnl || t.pnl_usd || 0) > 0).length;
                                                const totalPnl = trades.reduce((sum, t) => sum + (t.pnl || t.pnl_usd || 0), 0);
                                                return {
                                                    totalTrades: trades.length,
                                                    winRate: trades.length > 0 ? (wins / trades.length) * 100 : 0,
                                                    totalPnl,
                                                    totalPnlPct: (totalPnl / 10000) * 100,
                                                    tradingDays: 0,
                                                };
                                            })()
                                        // During animation, compute from visible trades
                                        : isAnimating
                                            ? (() => {
                                                const trades = visibleAnimationTrades;
                                                const wins = trades.filter(t => (t.pnl || 0) > 0).length;
                                                const totalPnl = trades.reduce((sum, t) => sum + (t.pnl || 0), 0);
                                                return {
                                                    totalTrades: trades.length,
                                                    winRate: trades.length > 0 ? (wins / trades.length) * 100 : 0,
                                                    totalPnl,
                                                    totalPnlPct: (totalPnl / 10000) * 100,
                                                    tradingDays: Math.floor(animationBarIndex / 60),
                                                };
                                            })()
                                            : backtestResult?.summary ? {
                                                totalTrades: backtestResult.summary.total_trades,
                                                winRate: backtestResult.summary.win_rate,
                                                totalPnl: backtestResult.summary.total_pnl,
                                                totalPnlPct: backtestResult.summary.total_return_pct,
                                                tradingDays: Math.ceil((new Date(backtestDateRange?.end || '').getTime() - new Date(backtestDateRange?.start || '').getTime()) / (1000 * 60 * 60 * 24)) || 0,
                                            } : null
                                    }
                                />
                            </div>
                        </div>
                    </div>
                </section>

                {/* ════════════════════════════════════════════════════════════════
                    Section 5: Historial de Trades
                ════════════════════════════════════════════════════════════════ */}
                <section className="w-full py-12 sm:py-16 lg:py-20 flex flex-col items-center">
                    <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
                        <SectionHeader
                            title="Historial de Trades"
                            subtitle="Actividad de trading en horario Colombia (COT)"
                            icon={<Table2 className="w-5 h-5 sm:w-6 sm:h-6" />}
                        />
                        <TradesTable
                            key={`trades-${selectedModel?.id || 'default'}-${clearKey}${isReplayMode || backtestResult || streamingTrades.length > 0 ? '-backtest' : ''}`}
                            initialLimit={isReplayMode || backtestResult || isAnimating || streamingTrades.length > 0 ? 50 : 10}
                            compact={false}
                            forceEmpty={isDashboardCleared}
                            highlightedTradeIds={
                                // Priority: streaming > animation > replay
                                streamingTrades.length > 0
                                    ? new Set(streamingTrades.slice(-3).map(t => String(t.trade_id)))
                                    : isAnimating
                                        ? new Set(visibleAnimationTrades.slice(-3).map(t => String(t.trade_id)))
                                        : (isReplayMode ? replay.highlightedTradeIds : undefined)
                            }
                            replayEndDate={
                                streamingTrades.length > 0
                                    ? undefined
                                    : isAnimating
                                        ? animationEndDate
                                        : (isReplayMode ? replay.state.currentDate : undefined)
                            }
                            isReplayMode={isReplayMode || !!backtestResult || isAnimating || streamingTrades.length > 0}
                            replayVisibleTradeIds={
                                // Priority: streaming > animation > replay > backtest result
                                streamingTrades.length > 0
                                    ? new Set(streamingTrades.map(t => String(t.trade_id)))
                                    : isAnimating
                                        ? new Set(visibleAnimationTrades.map(t => String(t.trade_id)))
                                        : (isReplayMode ? new Set(replay.visibleTrades.map(t => String(t.trade_id))) : (backtestResult ? new Set(backtestResult.trades.map(t => String(t.trade_id))) : undefined))
                            }
                            replayTrades={
                                // Priority: streaming > animation > replay > backtest result
                                streamingTrades.length > 0
                                    ? streamingTrades
                                    : isAnimating
                                        ? visibleAnimationTrades
                                        : (replay.visibleTrades.length > 0 ? replay.visibleTrades : (backtestResult?.trades || undefined))
                            }
                        />
                    </div>
                </section>

                {/* ════════════════════════════════════════════════════════════════
                    Footer
                ════════════════════════════════════════════════════════════════ */}
                <footer className="w-full py-12 sm:py-16 border-t border-slate-800/50 flex flex-col items-center">
                    <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
                        <p className="text-sm sm:text-base text-slate-400 font-medium">
                            USDCOP RL Trading System
                        </p>
                        <p className="mt-2 text-xs sm:text-sm text-slate-500">
                            Paper Trading Mode • Last update: {lastUpdate}
                        </p>
                        {/* Decorative badge */}
                        <div className="mt-6 flex justify-center">
                            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-slate-800/50 border border-slate-700/50">
                                <span className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse" />
                                <span className="text-xs text-slate-400">
                                    Powered by Reinforcement Learning
                                </span>
                            </div>
                        </div>
                    </div>
                </footer>
            </main>
        </div>
    );
}

// ============================================================================
// Loading State
// ============================================================================
function DashboardSkeleton() {
    return (
        <div className="min-h-screen bg-[#030712] flex items-center justify-center">
            <div className="text-center">
                <RefreshCw className="w-10 h-10 text-cyan-400 animate-spin mx-auto mb-4" />
                <p className="text-slate-400">Loading Dashboard...</p>
            </div>
        </div>
    );
}

// ============================================================================
// Page Export
// ============================================================================
export default function DashboardPage() {
    return (
        <ModelProvider>
            <Suspense fallback={<DashboardSkeleton />}>
                <DashboardContent />
            </Suspense>
        </ModelProvider>
    );
}

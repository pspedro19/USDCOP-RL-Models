'use client';

/**
 * Dashboard Page
 * ==============
 * Shows ForecastingBacktestSection (2025 OOS + approval)
 * with KPIs, candlestick chart, gates, and trade table.
 * Strategy selector lives inside ForecastingBacktestSection.
 */

import { Suspense, useState, useEffect, useRef } from "react";
import { GlobalNavbar } from "@/components/navigation/GlobalNavbar";
import { Badge } from "@/components/ui/badge";
import { RefreshCw, DollarSign, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

// ============================================================================
// Strategy Dropdown — visible in header
// ============================================================================
interface StrategyInfo {
  strategy_id: string;
  strategy_name: string;
  pipeline: string;
  status: string;
  return_pct: number;
  sharpe: number;
  p_value: number;
}

function HeaderStrategyDropdown() {
  const [isOpen, setIsOpen] = useState(false);
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [selected, setSelected] = useState<StrategyInfo | null>(null);

  useEffect(() => {
    fetch('/data/production/strategies.json')
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data?.strategies) {
          setStrategies(data.strategies);
          const def = data.strategies.find((s: StrategyInfo) => s.strategy_id === data.default_strategy) || data.strategies[0];
          setSelected(def);
        }
      })
      .catch(() => {});
  }, []);

  if (!selected) return null;

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "flex items-center gap-2 px-4 py-2 rounded-xl",
          "bg-slate-800/90 border-2 border-cyan-500/40 hover:border-cyan-400/70",
          "transition-all duration-200 shadow-lg shadow-cyan-500/10"
        )}
      >
        <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
        <div className="text-left">
          <div className="font-bold text-white text-sm">{selected.strategy_name}</div>
          <div className="text-[10px] text-slate-400">
            {selected.pipeline} | +{selected.return_pct.toFixed(1)}% | Sharpe {selected.sharpe.toFixed(2)}
          </div>
        </div>
        <Badge
          variant="outline"
          className={cn(
            "text-[9px] font-bold ml-1",
            selected.status === 'APPROVED'
              ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
              : "bg-amber-500/10 text-amber-400 border-amber-500/30"
          )}
        >
          {selected.status}
        </Badge>
        <ChevronDown className={cn("w-4 h-4 text-cyan-400 transition-transform", isOpen && "rotate-180")} />
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 mt-2 z-50 bg-slate-800 border-2 border-cyan-500/30 rounded-xl overflow-hidden shadow-2xl min-w-[320px]">
          <div className="px-4 py-2 bg-slate-900/80 border-b border-slate-700/50">
            <span className="text-[10px] text-cyan-400 font-bold uppercase tracking-wider">Estrategia Activa</span>
          </div>
          {strategies.map((s) => (
            <button
              key={s.strategy_id}
              className={cn(
                "w-full flex items-center justify-between px-4 py-3",
                "hover:bg-cyan-500/10 transition-colors text-left",
                s.strategy_id === selected.strategy_id && "bg-cyan-500/5 border-l-2 border-cyan-400"
              )}
              onClick={() => { setSelected(s); setIsOpen(false); }}
            >
              <div>
                <div className="text-white font-semibold text-sm">{s.strategy_name}</div>
                <div className="text-[10px] text-slate-400 mt-0.5">
                  {s.pipeline} | +{s.return_pct.toFixed(1)}% | Sharpe {s.sharpe.toFixed(2)} | p={s.p_value.toFixed(4)}
                </div>
              </div>
              <Badge
                variant="outline"
                className={cn(
                  "text-[9px] font-bold ml-3 shrink-0",
                  s.status === 'APPROVED'
                    ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                    : "bg-amber-500/10 text-amber-400 border-amber-500/30"
                )}
              >
                {s.status}
              </Badge>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// Forecasting backtest section (self-contained, includes strategy selector)
import { ForecastingBacktestSection } from "@/components/production/ForecastingBacktestSection";

// ============================================================================
// Live Price Display — polls /api/market/realtime-price adaptively
// ============================================================================
function LivePriceDisplay() {
  const [price, setPrice] = useState<number | null>(null);
  const [change, setChange] = useState<number | null>(null);
  const [changePct, setChangePct] = useState<number | null>(null);
  const [isMarketOpen, setIsMarketOpen] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    let mounted = true;
    const fetchPrice = async () => {
      try {
        const res = await fetch('/api/market/realtime-price');
        if (res.ok && mounted) {
          const json = await res.json();
          const d = json.data ?? json; // handle nested response
          setPrice(d.price);
          setChange(d.change ?? null);
          setChangePct(d.changePct ?? null);
          setIsMarketOpen(d.isMarketOpen ?? false);
          // Adaptive polling: 5min market open, 30min closed
          const pollMs = d.isMarketOpen ? 5 * 60 * 1000 : 30 * 60 * 1000;
          if (intervalRef.current) clearInterval(intervalRef.current);
          intervalRef.current = setInterval(fetchPrice, pollMs);
        }
      } catch {
        // Silently fail — price is optional
      }
    };
    fetchPrice();
    return () => {
      mounted = false;
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  if (price == null) return null;

  const isPositive = (change ?? 0) >= 0;

  return (
    <div className="flex items-center gap-2.5">
      <div className="flex items-center gap-1.5">
        <DollarSign className="w-3.5 h-3.5 text-slate-400" />
        <span className="font-mono text-sm font-bold text-white">
          {price.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 })}
        </span>
      </div>
      {change !== null && (
        <span className={cn(
          'font-mono text-xs font-semibold',
          isPositive ? 'text-emerald-400' : 'text-red-400'
        )}>
          {isPositive ? '+' : ''}{change.toFixed(1)} ({isPositive ? '+' : ''}{changePct?.toFixed(2)}%)
        </span>
      )}
      <Badge
        variant="outline"
        className={cn(
          "text-[9px] font-semibold px-1.5 py-0.5",
          isMarketOpen
            ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
            : "bg-slate-500/10 text-slate-400 border-slate-500/30"
        )}
      >
        <span className={cn(
          "w-1.5 h-1.5 rounded-full mr-1 inline-block",
          isMarketOpen ? "bg-emerald-500 animate-pulse" : "bg-slate-500"
        )} />
        {isMarketOpen ? 'Open' : 'Closed'}
      </Badge>
    </div>
  );
}

// ============================================================================
// Dashboard Content
// ============================================================================
function DashboardContent() {
  const [lastUpdate, setLastUpdate] = useState<string>('--:--:--');

  // Set time only on client to avoid hydration mismatch
  useEffect(() => {
    setLastUpdate(new Date().toLocaleTimeString());
    const interval = setInterval(() => {
      setLastUpdate(new Date().toLocaleTimeString());
    }, 60000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#030712] via-[#0f172a] to-[#030712] overflow-x-hidden">
      {/* Global Navigation (fixed position) */}
      <GlobalNavbar currentPage="dashboard" />

      {/* Sticky Header - Compact single row */}
      <header className="sticky top-16 z-40 bg-[#030712]/95 backdrop-blur-xl border-b border-slate-800/50">
        <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between gap-3 py-3 sm:py-4">
            {/* Left: Title + Strategy Dropdown */}
            <div className="flex items-center gap-3 sm:gap-4 min-w-0">
              <h1 className="text-lg sm:text-xl lg:text-2xl font-bold bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 bg-clip-text text-transparent whitespace-nowrap">
                Dashboard
              </h1>
              <HeaderStrategyDropdown />
            </div>

            {/* Right: Live Price */}
            <div className="flex items-center gap-3">
              <div className="hidden sm:block">
                <LivePriceDisplay />
              </div>
              <Badge variant="outline" className="bg-purple-500/10 text-purple-400 border-purple-500/30 text-[10px] px-2 py-0.5">
                PAPER
              </Badge>
            </div>
          </div>
        </div>
      </header>

      {/* Minimal spacer for fixed navbar */}
      <div className="h-16" aria-hidden="true" />

      {/* Main Content */}
      <main className="w-full overflow-x-hidden">

        {/* ================================================================
            Forecasting Backtest (2025 OOS) + Approval
            Self-contained component with strategy selector
        ================================================================ */}
        <ForecastingBacktestSection />

        {/* ================================================================
            Footer
        ================================================================ */}
        <footer className="w-full py-12 sm:py-16 border-t border-slate-800/50 flex flex-col items-center pb-32">
          <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <p className="text-sm sm:text-base text-slate-400 font-medium">
              USDCOP Trading System
            </p>
            <p className="mt-2 text-xs sm:text-sm text-slate-500">
              Paper Trading Mode | Last update: {lastUpdate}
            </p>
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
    <Suspense fallback={<DashboardSkeleton />}>
      <DashboardContent />
    </Suspense>
  );
}

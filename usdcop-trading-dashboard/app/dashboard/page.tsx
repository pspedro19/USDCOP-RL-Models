'use client';

/**
 * Dashboard Page
 * ==============
 * Shows ForecastingBacktestSection (2025 OOS + approval)
 * with KPIs, candlestick chart, gates, and trade table.
 * Strategy selector lives inside ForecastingBacktestSection.
 */

import { Suspense, useState, useEffect } from "react";
import { GlobalNavbar } from "@/components/navigation/GlobalNavbar";
import { Badge } from "@/components/ui/badge";
import { RefreshCw, DollarSign } from "lucide-react";
import { cn } from "@/lib/utils";

// Forecasting backtest section (self-contained, includes strategy selector)
import { ForecastingBacktestSection } from "@/components/production/ForecastingBacktestSection";

// ============================================================================
// Live Price Display — polls /api/market/realtime-price every 60s
// ============================================================================
function LivePriceDisplay() {
  const [price, setPrice] = useState<number | null>(null);
  const [change, setChange] = useState<number | null>(null);
  const [changePct, setChangePct] = useState<number | null>(null);
  const [isMarketOpen, setIsMarketOpen] = useState(false);
  const [source, setSource] = useState<string>('');
  const [lastUpdate, setLastUpdate] = useState<string>('');

  useEffect(() => {
    const fetchPrice = async () => {
      try {
        const res = await fetch('/api/market/realtime-price');
        if (res.ok) {
          const data = await res.json();
          setPrice(data.price);
          setChange(data.change);
          setChangePct(data.changePct);
          setIsMarketOpen(data.isMarketOpen);
          setSource(data.source);
          setLastUpdate(new Date(data.lastUpdate).toLocaleTimeString());
        }
      } catch {
        // Silently fail — price is optional
      }
    };
    fetchPrice();
    const interval = setInterval(fetchPrice, 60000);
    return () => clearInterval(interval);
  }, []);

  if (price === null) return null;

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
            {/* Left: Title + Badges */}
            <div className="flex items-center gap-3 sm:gap-4 min-w-0">
              <h1 className="text-lg sm:text-xl lg:text-2xl font-bold bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 bg-clip-text text-transparent whitespace-nowrap">
                USD/COP Trading
              </h1>
              <div className="hidden sm:flex items-center gap-2">
                <Badge variant="outline" className="bg-purple-500/10 text-purple-400 border-purple-500/30 text-[10px] px-2 py-0.5">
                  PAPER
                </Badge>
              </div>
            </div>

            {/* Right: Live Price */}
            <div className="hidden sm:block">
              <LivePriceDisplay />
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

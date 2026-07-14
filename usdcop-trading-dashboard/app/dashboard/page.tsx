'use client';

/**
 * Dashboard Page — vista BACKTEST del GlobalMarkets Terminal
 * ==========================================================
 * GM re-skin (CTR-GM-UI-001, prototipo Var B líneas 363-484): GmPageHeader único
 * (sin duplicados) + selector de estrategia con dot pulse + badge "OOS 2025 ·
 * BACKTEST". El contenido es ForecastingBacktestSection (KPIs, candle+replay,
 * equity, resumen, razones de salida, tabla, gates Voto 1 y aprobación Voto 2/2
 * — lógica de aprobación INTACTA, ver approval-gates.md).
 */

import { Suspense, useState, useEffect, useRef } from "react";
import { TerminalShell } from "@/components/gm/TerminalShell";
import { GmBadge, GmPageHeader } from "@/components/gm";
import { defineGmDict, useGmT } from "@/lib/i18n/gm-core";
import { GM, GMT } from "@/lib/ui/gm-tokens";
import { RefreshCw, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

// ── Diccionario local ES/EN (gm-core) ──────────────────────────────────────
const DICT = defineGmDict({
  es: {
    title: 'Backtest',
    kicker: 'Estrategias · OOS',
    oosBadge: 'OOS 2025 · BACKTEST',
    paper: 'PAPER',
    open: 'Abierto',
    closed: 'Cerrado',
    loading: 'Cargando dashboard…',
    footerTitle: 'USDCOP Trading System',
    footerMode: 'Modo paper trading · Última actualización',
  },
  en: {
    title: 'Backtest',
    kicker: 'Strategies · OOS',
    oosBadge: 'OOS 2025 · BACKTEST',
    paper: 'PAPER',
    open: 'Open',
    closed: 'Closed',
    loading: 'Loading dashboard…',
    footerTitle: 'USDCOP Trading System',
    footerMode: 'Paper trading mode · Last update',
  },
});

// ============================================================================
// Strategy Dropdown — visible in header
// ============================================================================
interface StrategyInfo {
  strategy_id: string;
  strategy_name: string;
  asset_id: string;
  asset_name: string;
  pipeline: string;
  status: string;
  return_pct: number;
  sharpe: number;
  p_value: number;
}

/**
 * Header strategy selector — reads the SSOT registry (/api/registry) so EVERY trained
 * strategy (USD/COP, Gold, BTC) is selectable, grouped by asset. Controlled: the parent
 * owns the selected id and passes it to ForecastingBacktestSection, so picking a strategy
 * here drives that section's backtest replay. Adding a strategy/asset to the registry makes
 * it appear here automatically — no code change (DRY / open-closed).
 */
function HeaderStrategyDropdown({
  selectedId,
  onSelect,
}: {
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);

  useEffect(() => {
    fetch('/api/registry')
      .then(r => (r.ok ? r.json() : null))
      .then(idx => {
        if (!idx?.strategies) return;
        const assetName: Record<string, string> = {};
        (idx.assets ?? []).forEach((a: { asset_id: string; display_name?: string }) => {
          assetName[a.asset_id] = a.display_name ?? a.asset_id;
        });
        const list: StrategyInfo[] = idx.strategies.map((s: {
          strategy_id: string; asset_id: string; display_name?: string; pipeline_type?: string;
          status?: string; return_pct?: number; sharpe?: number; p_value?: number;
        }) => ({
          strategy_id: s.strategy_id,
          strategy_name: s.display_name ?? s.strategy_id,
          asset_id: s.asset_id,
          asset_name: assetName[s.asset_id] ?? s.asset_id,
          pipeline: s.pipeline_type ?? 'rule_based',
          status: s.status === 'production' ? 'APPROVED' : (s.status ?? 'experimental'),
          return_pct: s.return_pct ?? 0,
          sharpe: s.sharpe ?? 0,
          p_value: s.p_value ?? 0,
        }));
        setStrategies(list);
        // Initialise the shared selection to the registry default (once, if uncontrolled).
        if (!selectedId && list.length) {
          onSelect(idx.default?.strategy_id ?? list[0].strategy_id);
        }
      })
      .catch(() => {});
    // Registry is immutable per session; load once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selected = strategies.find(s => s.strategy_id === selectedId) ?? null;
  if (!selected) return null;

  // Group by asset so the user scans USD/COP, Gold and BTC strategies separately.
  const byAsset = strategies.reduce<Record<string, StrategyInfo[]>>((acc, s) => {
    (acc[s.asset_id] ??= []).push(s);
    return acc;
  }, {});

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`${GM.ctaGhost} ${GM.focus} flex items-center gap-2.5 min-h-[44px] px-3.5 py-1.5`}
        aria-expanded={isOpen}
      >
        <span className="w-2 h-2 rounded-full bg-[var(--gm-pos)] motion-safe:animate-pulse shrink-0" aria-hidden />
        <span className="text-left">
          <span className={`block text-[0.8125rem] font-bold leading-tight ${GM.text}`}>{selected.strategy_name}</span>
          <span className={`block ${GMT.micro} ${GM.textMuted} font-mono leading-tight mt-0.5`}>
            {selected.asset_name} · +{selected.return_pct.toFixed(1)}% · Sharpe {selected.sharpe.toFixed(2)}
          </span>
        </span>
        <GmBadge tone={selected.status === 'APPROVED' ? 'pos' : 'warn'} className="shrink-0">
          {selected.status}
        </GmBadge>
        <ChevronDown className={cn(`w-4 h-4 ${GM.accent} transition-transform`, isOpen && "rotate-180")} aria-hidden />
      </button>

      {isOpen && (
        <div className={`absolute top-full left-0 mt-1.5 z-40 min-w-[340px] max-h-[70vh] overflow-y-auto ${GM.popover} p-1.5 flex flex-col gap-0.5`}>
          {Object.entries(byAsset).map(([assetId, list]) => (
            <div key={assetId}>
              <div className={`px-3 py-2 border-b border-[rgba(148,163,184,.10)] ${GMT.label} ${GM.accent}`}>
                {list[0].asset_name}
              </div>
              {list.map((s) => (
                <button
                  key={s.strategy_id}
                  className={cn(
                    `${GM.rowHover} ${GM.focus} w-full flex items-center justify-between gap-3 px-3 py-2.5 rounded-[9px] text-left`,
                    s.strategy_id === selected.strategy_id && 'bg-[rgba(34,211,238,.07)]'
                  )}
                  onClick={() => { onSelect(s.strategy_id); setIsOpen(false); }}
                >
                  <span className="min-w-0">
                    <span className={`block text-[0.8125rem] font-bold ${GM.text}`}>{s.strategy_name}</span>
                    <span className={`block ${GMT.micro} ${GM.textMuted} font-mono mt-0.5`}>
                      {s.pipeline} · +{s.return_pct.toFixed(1)}% · Sharpe {s.sharpe.toFixed(2)} · p={s.p_value.toFixed(4)}
                    </span>
                  </span>
                  <GmBadge tone={s.status === 'APPROVED' ? 'pos' : 'warn'} className="shrink-0">
                    {s.status}
                  </GmBadge>
                </button>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Forecasting backtest section (self-contained; el selector del header lo controla)
import { ForecastingBacktestSection } from "@/components/production/ForecastingBacktestSection";

// ============================================================================
// Live Price Display — polls /api/market/realtime-price adaptively
// ============================================================================
function LivePriceDisplay() {
  const t = useGmT(DICT);
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
      <span className={`${GMT.mono} ${GMT.body} font-bold ${GM.textStrong}`}>
        ${price.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 })}
      </span>
      {change !== null && (
        <span className={`${GMT.mono} ${GMT.micro} font-semibold ${isPositive ? GM.pos : GM.neg}`}>
          {isPositive ? '+' : ''}{change.toFixed(1)} ({isPositive ? '+' : ''}{changePct?.toFixed(2)}%)
        </span>
      )}
      <GmBadge tone={isMarketOpen ? 'pos' : 'neutral'}>
        <span
          className={cn(
            'w-1.5 h-1.5 rounded-full inline-block',
            isMarketOpen ? 'bg-[var(--gm-pos)] motion-safe:animate-pulse' : 'bg-[var(--gm-text-faint)]'
          )}
          aria-hidden
        />
        {isMarketOpen ? t('open') : t('closed')}
      </GmBadge>
    </div>
  );
}

// ============================================================================
// Dashboard Content
// ============================================================================
function DashboardContent() {
  const t = useGmT(DICT);
  const [lastUpdate, setLastUpdate] = useState<string>('--:--:--');
  // Shared strategy selection: the header selector and the backtest section stay in sync,
  // so choosing a USD/COP, Gold or BTC strategy up here replays it below.
  const [selectedStrategyId, setSelectedStrategyId] = useState<string>('');

  // Set time only on client to avoid hydration mismatch
  useEffect(() => {
    setLastUpdate(new Date().toLocaleTimeString());
    const interval = setInterval(() => {
      setLastUpdate(new Date().toLocaleTimeString());
    }, 60000);
    return () => clearInterval(interval);
  }, []);

  return (
    <TerminalShell active="dashboard" width="wide">
      {/* Header único (prototipo: título + selector con dot pulse + badge OOS) */}
      <GmPageHeader
        kicker={t('kicker')}
        title={t('title')}
        actions={
          <div className="flex flex-wrap items-center gap-2.5">
            <HeaderStrategyDropdown selectedId={selectedStrategyId} onSelect={setSelectedStrategyId} />
            <div className="hidden sm:block">
              <LivePriceDisplay />
            </div>
            <GmBadge tone="info">{t('oosBadge')}</GmBadge>
            <GmBadge tone="neutral">{t('paper')}</GmBadge>
          </div>
        }
      />

      {/* ================================================================
          Forecasting Backtest (2025 OOS) + Approval — Vote 2/2 flow intacto
      ================================================================ */}
      <ForecastingBacktestSection controlledStrategyId={selectedStrategyId} onStrategyChange={setSelectedStrategyId} />

      {/* ================================================================
          Footer
      ================================================================ */}
      <footer className="w-full pt-10 mt-8 border-t border-[rgba(148,163,184,.10)] text-center">
        <p className={`${GMT.body} font-medium ${GM.textSec} m-0`}>{t('footerTitle')}</p>
        <p className={`mt-2 ${GMT.micro} ${GMT.mono} ${GM.textMuted}`}>
          {t('footerMode')}: {lastUpdate}
        </p>
      </footer>
    </TerminalShell>
  );
}

// ============================================================================
// Loading State
// ============================================================================
function DashboardSkeleton() {
  const t = useGmT(DICT);
  return (
    <div className={`min-h-screen ${GM.page} flex items-center justify-center`}>
      <div className="text-center">
        <RefreshCw className={`w-10 h-10 ${GM.accent} motion-safe:animate-spin mx-auto mb-4`} aria-hidden />
        <p className={`${GMT.body} ${GM.textMuted}`}>{t('loading')}</p>
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

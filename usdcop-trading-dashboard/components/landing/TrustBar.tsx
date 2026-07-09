'use client';

/**
 * S3 · Trust bar — ONLY ● LIVE metrics, straight from the published bundle via
 * /api/public/live-stats (ux-navigation §2). Hard rule: no backtest number ever
 * renders here. Fails soft: if stats are unavailable the bar simply doesn't render
 * (never a broken-looking landing).
 */
import { useEffect, useState } from 'react';
import { MetricBadge } from '@/components/ui/MetricBadge';

interface LiveStats {
  unavailable?: boolean;
  strategy_name?: string;
  year?: number;
  return_ytd_pct?: number | null;
  max_dd_pct?: number | null;
  weeks_live?: number | null;
  n_trades?: number | null;
  bundle_date?: string | null;
}

export default function TrustBar() {
  const [stats, setStats] = useState<LiveStats | null>(null);

  useEffect(() => {
    fetch('/api/public/live-stats')
      .then((r) => (r.ok ? r.json() : null))
      .then(setStats)
      .catch(() => setStats(null));
  }, []);

  if (!stats || stats.unavailable || stats.return_ytd_pct == null) return null;

  const items: Array<{ label: string; value: string }> = [
    { label: 'en producción', value: `${stats.weeks_live ?? '—'} semanas` },
    {
      label: `${stats.year} YTD`,
      value: `${stats.return_ytd_pct >= 0 ? '+' : ''}${stats.return_ytd_pct.toFixed(2)}%`,
    },
  ];
  if (stats.max_dd_pct != null) {
    items.push({ label: 'Max DD', value: `−${Math.abs(stats.max_dd_pct).toFixed(1)}%` });
  }
  if (stats.n_trades != null) {
    items.push({ label: 'operaciones', value: String(stats.n_trades) });
  }

  return (
    <section
      aria-label="Métricas de producción en vivo"
      className="w-full border-y border-slate-800/80 bg-slate-900/40 backdrop-blur"
    >
      <div className="max-w-6xl mx-auto px-4 py-4 flex flex-wrap items-center justify-center gap-x-8 gap-y-2">
        <MetricBadge
          phase="live"
          provenance={{
            strategyId: stats.strategy_name ?? 'estrategia activa',
            bundleDate: stats.bundle_date ?? undefined,
          }}
        />
        {items.map((it) => (
          <div key={it.label} className="flex items-baseline gap-2">
            <span className="text-lg font-semibold text-slate-100 tabular-nums">{it.value}</span>
            <span className="text-xs text-slate-400">{it.label}</span>
          </div>
        ))}
        <span className="text-[11px] text-slate-500">
          señales publicadas antes del hecho · sin edición retroactiva
        </span>
      </div>
    </section>
  );
}

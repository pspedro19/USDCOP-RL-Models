'use client';

/**
 * S5 · Track record — tabs ● LIVE (producción forward, el titular) / ◆ BACKTEST (OOS,
 * visible pero nunca líder). Data: /api/public/live-stats (bundle-derived aggregates).
 * S7 · Methodology mini-cards live below the tabs (the moat, translated for clients).
 */
import { useEffect, useState } from 'react';
import { MetricBadge } from '@/components/ui/MetricBadge';
import { canShowRatios } from '@/lib/contracts/ui.contract';

interface Stats {
  unavailable?: boolean;
  strategy_name?: string;
  year?: number;
  return_ytd_pct?: number | null;
  max_dd_pct?: number | null;
  n_trades?: number | null;
  sharpe?: number | null;
  bundle_date?: string | null;
  backtest?: { year: number; return_pct: number | null; sharpe: number | null } | null;
}

const PILLARS = [
  { t: 'Pre-registro', d: 'Los parámetros se fijan antes de ver resultados. Nada se ajusta mirando el examen.' },
  { t: 'Validación estadística', d: 'Corregimos por intentos (Deflated Sharpe). Un backtest bonito no promueve nada.' },
  { t: 'Paper → Producción', d: 'Simulado primero, con reglas congeladas. Solo el forward cuenta como prueba.' },
  { t: 'Protocolo de retiro', d: 'Condiciones de retiro firmadas de antemano. Si deja de funcionar, se retira — publicado.' },
];

export default function TrackRecord() {
  const [tab, setTab] = useState<'live' | 'backtest'>('live');
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    fetch('/api/public/live-stats').then((r) => (r.ok ? r.json() : null)).then(setStats).catch(() => null);
  }, []);

  const live = stats && !stats.unavailable ? stats : null;
  const bt = live?.backtest ?? null;

  return (
    <section id="track-record" className="w-full py-16 sm:py-20 px-4 flex flex-col items-center">
      <div className="w-full max-w-4xl mx-auto">
        <h2 className="text-2xl sm:text-3xl font-bold text-center text-white">Track record</h2>
        <p className="mt-2 text-center text-sm text-slate-300 max-w-xl mx-auto">
          LIVE = señales publicadas antes del hecho, sin edición retroactiva.{' '}
          <a href="/metodologia" className="text-cyan-400 hover:underline">Por qué mostramos ambos →</a>
        </p>

        {/* tabs */}
        <div className="mt-8 flex justify-center gap-2" role="tablist" aria-label="Track record">
          <button
            role="tab" aria-selected={tab === 'live'}
            onClick={() => setTab('live')}
            className={`px-4 py-2 rounded-lg text-sm font-semibold border transition ${
              tab === 'live'
                ? 'border-emerald-500/60 bg-emerald-500/10 text-emerald-300'
                : 'border-slate-700 text-slate-400 hover:text-slate-200'
            }`}
          >
            ● Producción {live?.year ?? ''} (LIVE)
          </button>
          <button
            role="tab" aria-selected={tab === 'backtest'}
            onClick={() => setTab('backtest')}
            className={`px-4 py-2 rounded-lg text-sm font-semibold border transition ${
              tab === 'backtest'
                ? 'border-blue-500/60 bg-blue-500/10 text-blue-300'
                : 'border-slate-700 text-slate-400 hover:text-slate-200'
            }`}
          >
            ◆ Backtest OOS {bt?.year ?? 2025}
          </button>
        </div>

        {/* panel */}
        <div className="mt-6 rounded-2xl border border-slate-700/60 bg-slate-900/40 p-6 sm:p-8">
          {tab === 'live' ? (
            live ? (
              <div className="flex flex-wrap items-center justify-center gap-x-10 gap-y-4">
                <MetricBadge phase="live" provenance={{
                  strategyId: live.strategy_name ?? '', bundleDate: live.bundle_date ?? undefined }} />
                <Metric label={`Retorno ${live.year} YTD`}
                        value={fmtPct(live.return_ytd_pct)} strong />
                <Metric label="Max Drawdown" value={live.max_dd_pct != null ? `−${Math.abs(live.max_dd_pct).toFixed(1)}%` : '—'} />
                <Metric label="Operaciones" value={String(live.n_trades ?? '—')} />
                {canShowRatios(live.n_trades) && live.sharpe != null && (
                  <Metric label="Sharpe" value={live.sharpe.toFixed(2)} />
                )}
                {!canShowRatios(live.n_trades) && (
                  <span className="text-[11px] text-slate-500 basis-full text-center">
                    Con pocas operaciones publicamos solo conteo y P&L — sin ratios (honestidad estadística).
                  </span>
                )}
              </div>
            ) : (
              <p className="text-center text-sm text-slate-400">Métricas en vivo no disponibles en este momento.</p>
            )
          ) : bt ? (
            <div className="flex flex-wrap items-center justify-center gap-x-10 gap-y-4">
              <MetricBadge phase="backtest" provenance={{
                strategyId: live?.strategy_name ?? '', version: 'OOS', bundleDate: live?.bundle_date ?? undefined }} />
              <Metric label={`Retorno ${bt.year} (fuera de muestra)`} value={fmtPct(bt.return_pct)} strong />
              {bt.sharpe != null && <Metric label="Sharpe" value={bt.sharpe.toFixed(2)} />}
              <span className="text-[11px] text-slate-500 basis-full text-center">
                El backtest es evidencia secundaria: hereda riesgo de sobreajuste y por eso nunca es el titular.
              </span>
            </div>
          ) : (
            <p className="text-center text-sm text-slate-400">Backtest no disponible.</p>
          )}
        </div>

        {/* S7 — methodology mini-cards */}
        <div className="mt-14">
          <p className="text-center text-slate-200 font-medium max-w-2xl mx-auto">
            “No vendemos backtests bonitos. Cada estrategia se pre-registra, pasa control de
            significancia, corre en simulado antes de producción y tiene reglas de retiro
            firmadas de antemano.”
          </p>
          <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {PILLARS.map((p) => (
              <article key={p.t} className="rounded-xl border border-slate-700/60 bg-slate-900/40 p-4">
                <h3 className="text-sm font-semibold text-white">{p.t}</h3>
                <p className="mt-1.5 text-xs leading-relaxed text-slate-300">{p.d}</p>
              </article>
            ))}
          </div>
          <div className="mt-6 text-center">
            <a href="/metodologia"
               className="inline-block px-6 py-3 rounded-lg border border-slate-600 text-sm font-semibold text-white hover:bg-white/5 transition">
              Leer la metodología completa →
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}

function Metric({ label, value, strong }: { label: string; value: string; strong?: boolean }) {
  return (
    <div className="text-center">
      <div className={`tabular-nums ${strong ? 'text-3xl font-bold text-white' : 'text-xl font-semibold text-slate-100'}`}>
        {value}
      </div>
      <div className="text-xs text-slate-400 mt-0.5">{label}</div>
    </div>
  );
}

function fmtPct(v: number | null | undefined): string {
  return v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;
}

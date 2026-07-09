'use client';

/**
 * WeeklyInferenceView — per-asset (Gold / BTC) whole-year weekly inference.
 * ========================================================================
 * USD/COP uses the 9-model ML model-zoo (ForecastingDashboard). Gold & BTC are
 * rule-based daily science stacks with NO ML forecast — this view shows their
 * honest weekly POSITIONING inference (direction / exposure / regime) vs what
 * actually happened (strategy realized return vs buy&hold), for the whole year.
 *
 * Methodology (all pairs): trained on history ≤ Dec-2024, 2025 = backtest (OOS,
 * default), 2026 = production (YTD). Data:
 *   /forecasting/<asset>/index.json  +  /forecasting/<asset>/weekly_inference_<year>.json
 * produced by scripts/pipeline/generate_asset_weekly_forecast.py.
 */

import { useEffect, useMemo, useState } from 'react';
import { useSession } from 'next-auth/react';
import Link from 'next/link';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip, ResponsiveContainer, Legend,
} from 'recharts';
import {
  Loader2, AlertCircle, TrendingUp, Target, Activity, Layers, Info,
} from 'lucide-react';
import type {
  AssetWeeklyInference, WeeklyInferenceIndex, WeeklyInferenceStrategy,
} from './types';

const DIRECTION_STYLE: Record<string, string> = {
  LONG: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
  SHORT: 'text-red-400 bg-red-500/10 border-red-500/30',
  FLAT: 'text-slate-400 bg-slate-500/10 border-slate-500/30',
};

// Regime → colour (Gold: compression/trend/stretched/event · BTC: accumulation/markup/distribution/markdown)
const REGIME_STYLE: Record<string, string> = {
  trend: 'text-emerald-300 bg-emerald-500/10', markup: 'text-emerald-300 bg-emerald-500/10',
  compression: 'text-sky-300 bg-sky-500/10', accumulation: 'text-sky-300 bg-sky-500/10',
  stretched: 'text-amber-300 bg-amber-500/10', distribution: 'text-amber-300 bg-amber-500/10',
  event: 'text-red-300 bg-red-500/10', markdown: 'text-red-300 bg-red-500/10',
};

const num = (v: number | null | undefined, d = 2) =>
  v === null || v === undefined || isNaN(Number(v)) ? '—' : Number(v).toFixed(d);

function KPI({ label, value, color, hint }: { label: string; value: string; color: string; hint?: string }) {
  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-4 sm:p-5 text-center">
      <div className="text-[10px] sm:text-xs font-medium text-slate-400 uppercase tracking-wider mb-1.5">{label}</div>
      <div className="text-lg sm:text-2xl font-bold tracking-tight" style={{ color }}>{value}</div>
      {hint && <div className="text-[10px] text-slate-500 mt-1">{hint}</div>}
    </div>
  );
}

export function WeeklyInferenceView({ assetId }: { assetId: string }) {
  const { data: __session } = useSession();
  const __role = (__session?.user as { role?: string } | undefined)?.role ?? 'free';
  const __isInternal = __role === 'admin' || __role === 'developer';
  const [index, setIndex] = useState<WeeklyInferenceIndex | null>(null);
  const [year, setYear] = useState<number | null>(null);
  const [strategyId, setStrategyId] = useState<string>('');
  const [data, setData] = useState<AssetWeeklyInference | null>(null);
  // Forward forecast (imagen + horizontes, como USD/COP — bandas de vol + posicionamiento)
  const [forward, setForward] = useState<{
    image: string; direction: string; exposure: number; last_close: number;
    vol_daily_pct: number; generated_at: string; methodology: string;
    horizons: Array<{ h_days: number; exp_move_pct: number; ci95_pct: [number, number];
      da_2025_pct: number | null; n_2025: number }>;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load the per-asset index (years + strategies). Default to 2025 (backtest) if present.
  useEffect(() => {
    let alive = true;
    setLoading(true); setError(null); setData(null); setIndex(null);
    // Gated route enforces the plan (asset scope + forecast delay). 403 ⇒ locked asset.
    fetch(`/api/forecasting/${assetId}/index.json`)
      .then(async r => {
        if (r.status === 403) throw new Error('locked');
        if (!r.ok) throw new Error(`index ${r.status}`); return r.json();
      })
      .then((idx: WeeklyInferenceIndex) => {
        if (!alive) return;
        setIndex(idx);
        const preferBacktest = idx.years.includes(2025) ? 2025 : idx.years[idx.years.length - 1];
        setYear(preferBacktest ?? null);
        setStrategyId(idx.primary_strategy_id || idx.strategies[0]?.strategy_id || '');
      })
      .catch((e) => { if (alive) {
        setError(e?.message === 'locked'
          ? 'Este activo requiere un plan superior. Actualiza tu suscripción para ver Oro y Bitcoin.'
          : 'Sin datos de inferencia semanal para este activo.');
        setLoading(false);
      } });
    return () => { alive = false; };
  }, [assetId]);

  useEffect(() => {
    let alive = true;
    fetch(`/api/forecasting/${assetId}/forward.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (alive) setForward(d); })
      .catch(() => { if (alive) setForward(null); });
    return () => { alive = false; };
  }, [assetId]);

  // Load the selected year's payload
  useEffect(() => {
    if (year == null) return;
    let alive = true;
    setLoading(true);
    fetch(`/api/forecasting/${assetId}/weekly_inference_${year}.json`)
      .then(async r => {
        if (r.status === 403) throw new Error('locked');
        if (!r.ok) throw new Error(`data ${r.status}`); return r.json();
      })
      .then((d: AssetWeeklyInference) => { if (alive) { setData(d); setError(null); } })
      .catch((e) => { if (alive) setError(e?.message === 'locked'
        ? 'Este activo requiere un plan superior. Actualiza tu suscripción para ver Oro y Bitcoin.'
        : 'No se pudo cargar la inferencia semanal.'); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [assetId, year]);

  const strategy: WeeklyInferenceStrategy | null = useMemo(() => {
    if (!data) return null;
    return data.strategies.find(s => s.strategy_id === strategyId) || data.strategies[0] || null;
  }, [data, strategyId]);

  // Cumulative curve (strategy vs buy&hold) from weekly returns
  const chartData = useMemo(() => {
    if (!strategy) return [];
    let cs = 1, cb = 1;
    return strategy.weeks.map(w => {
      cs *= 1 + (w.realized_return_pct ?? 0) / 100;
      cb *= 1 + (w.buyhold_return_pct ?? 0) / 100;
      return { week: w.iso_week.replace(/^\d+-/, ''), Estrategia: +((cs - 1) * 100).toFixed(2), BuyHold: +((cb - 1) * 100).toFixed(2) };
    });
  }, [strategy]);

  if (loading && !data) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex flex-col items-center gap-3 text-slate-400">
          <Loader2 className="w-8 h-8 animate-spin text-purple-500" />
          <span className="text-sm">Cargando inferencia semanal…</span>
        </div>
      </div>
    );
  }
  if (error || !data || !strategy || !index) {
    const locked = /plan superior|suscripci/i.test(error ?? '');
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center text-slate-400 max-w-md">
          <AlertCircle className="w-10 h-10 mx-auto mb-3 text-amber-400" />
          <p className="text-sm">{error || 'Sin datos de inferencia semanal.'}</p>
          {locked && (
            <Link href="/pricing" className="inline-block mt-3 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-black font-semibold px-4 py-2 text-sm">
              Ver planes
            </Link>
          )}
          {/* Internal-only regeneration hint — never shown to clients (RBAC §8). */}
          {!locked && __isInternal && (
            <p className="text-xs text-slate-600 mt-2">
              Ejecuta: <code className="bg-slate-800 rounded px-1.5 py-0.5 text-purple-300">
                python -m scripts.pipeline.generate_asset_weekly_forecast --asset {assetId}
              </code>
            </p>
          )}
        </div>
      </div>
    );
  }

  const s = strategy.summary;
  const isBacktest = year === 2025;

  return (
    <div className="space-y-8">
      {/* Forward Forecast — mismo lenguaje que USD/COP: imagen + horizontes (honesto:
          bandas de vol realizada + posicionamiento de la campeona + DA 2025 medida) */}
      {forward && (
        <div className="rounded-2xl border border-slate-700/50 bg-slate-900/40 p-4 sm:p-6 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-lg font-bold text-white">Forward Forecast</h3>
            <div className="flex items-center gap-2 text-xs">
              <span className={`px-2 py-1 rounded-lg font-bold ${forward.direction === 'LONG'
                ? 'bg-emerald-500/15 text-emerald-400' : 'bg-slate-600/30 text-slate-300'}`}>
                {forward.direction}
              </span>
              <span className="text-slate-400">exposición {forward.exposure}x · σ diaria {forward.vol_daily_pct}%</span>
            </div>
          </div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={`/api/${forward.image}`} alt="Forward forecast"
            className="w-full rounded-xl border border-slate-800"
            onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }} />
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
            {forward.horizons.map((h) => (
              <div key={h.h_days} className="rounded-xl bg-slate-800/60 border border-slate-700/50 p-3 text-center">
                <div className="text-[10px] text-slate-400 font-semibold">H = {h.h_days}d</div>
                <div className="text-sm font-bold text-cyan-300">±{h.exp_move_pct}%</div>
                <div className="text-[10px] text-slate-500">IC95 {h.ci95_pct[0]}% / +{h.ci95_pct[1]}%</div>
                <div className={`text-[11px] font-semibold mt-1 ${
                  (h.da_2025_pct ?? 0) >= 55 ? 'text-emerald-400'
                    : (h.da_2025_pct ?? 0) >= 50 ? 'text-amber-300' : 'text-red-400'}`}>
                  DA 2025: {h.da_2025_pct ?? '—'}%
                </div>
              </div>
            ))}
          </div>
          <p className="text-[11px] text-slate-500">{forward.methodology}</p>
        </div>
      )}

      {/* Controls */}
      <div className="flex flex-wrap items-end justify-center gap-4">
        <div className="min-w-[240px]">
          <label className="flex items-center justify-center gap-2 text-xs font-medium text-slate-400 mb-2">
            <Layers className="w-4 h-4 text-purple-400" /> Estrategia
          </label>
          <select value={strategyId} onChange={e => setStrategyId(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-purple-500">
            {data.strategies.map(st => (
              <option key={st.strategy_id} value={st.strategy_id}>
                {st.strategy_name}{st.is_primary ? ' ★' : ''}
              </option>
            ))}
          </select>
        </div>
        <div className="min-w-[200px]">
          <label className="flex items-center justify-center gap-2 text-xs font-medium text-slate-400 mb-2">
            <Target className="w-4 h-4 text-purple-400" /> Periodo
          </label>
          <select value={year ?? ''} onChange={e => setYear(Number(e.target.value))}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-purple-500">
            {index.years.map(y => (
              <option key={y} value={y}>{y === 2025 ? 'Backtest 2025 (OOS)' : y === 2026 ? 'Producción 2026 (YTD)' : String(y)}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Methodology note */}
      <div className="flex items-start gap-2 max-w-3xl mx-auto text-xs text-slate-500 bg-slate-900/40 border border-slate-800/60 rounded-lg px-4 py-3">
        <Info className="w-4 h-4 text-purple-400/70 shrink-0 mt-0.5" />
        <p>
          Inferencia semanal <span className="text-slate-300">basada en reglas</span> (no ML): posicionamiento causal
          de la estrategia por semana (dirección · exposición · régimen) frente al resultado real.
          Entrenado con histórico ≤ Dic-2024 · <span className="text-emerald-300/80">2025 = backtest OOS</span> ·
          <span className="text-purple-300/80"> 2026 = producción</span>. "Esperado" es un proxy de sesgo, no una predicción ML.
        </p>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <KPI label={isBacktest ? 'Retorno Estrategia 2025' : 'Retorno Estrategia YTD'}
          value={`${num(s.ytd_strategy_return_pct, 1)}%`} color="#10B981" />
        <KPI label="Buy & Hold" value={`${num(s.ytd_buyhold_return_pct, 1)}%`} color="#64748B" />
        <KPI label="Acierto Direccional" value={`${num(s.hit_rate_pct, 1)}%`} color="#A855F7" />
        <KPI label="Semanas en Mercado" value={`${s.weeks_in_market}/${s.weeks_total}`} color="#EC4899"
          hint={`Exp. prom. ${num((s.avg_exposure ?? 0) * 100, 0)}%`} />
      </div>

      {/* Cumulative curve */}
      <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-4 sm:p-6">
        <h3 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-purple-400" /> Curva Acumulada — Estrategia vs Buy&amp;Hold
        </h3>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="week" tick={{ fontSize: 10, fill: '#64748b' }} interval="preserveStartEnd" minTickGap={24} />
              <YAxis tick={{ fontSize: 10, fill: '#64748b' }} tickFormatter={(v) => `${v}%`} />
              <RTooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }}
                formatter={(v: number) => `${v}%`} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="Estrategia" stroke="#10B981" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="BuyHold" stroke="#64748B" dot={false} strokeWidth={1.5} strokeDasharray="4 3" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Weekly table */}
      <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-4 sm:p-5">
        <h3 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
          <Activity className="w-4 h-4 text-purple-400" /> Inferencia Semanal — {data.display_name} {year}
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-500">
                <th className="text-left py-2 px-2 font-medium">Semana</th>
                <th className="text-left py-2 px-2 font-medium">Dirección</th>
                <th className="text-left py-2 px-2 font-medium">Exposición</th>
                <th className="text-left py-2 px-2 font-medium">Régimen</th>
                <th className="text-right py-2 px-2 font-medium">Esperado</th>
                <th className="text-right py-2 px-2 font-medium">Estrategia</th>
                <th className="text-right py-2 px-2 font-medium">Buy&amp;Hold</th>
                <th className="text-center py-2 px-2 font-medium">✓</th>
              </tr>
            </thead>
            <tbody>
              {strategy.weeks.map((w) => {
                const exp = Math.round((w.exposure ?? 0) * 100);
                const rColor = REGIME_STYLE[w.regime] || 'text-slate-400 bg-slate-500/10';
                return (
                  <tr key={w.iso_week} className="border-b border-slate-800/40 hover:bg-slate-800/20">
                    <td className="py-2 px-2 text-slate-300 whitespace-nowrap">{w.iso_week}</td>
                    <td className="py-2 px-2">
                      <span className={`px-2 py-0.5 rounded border text-[10px] font-semibold ${DIRECTION_STYLE[w.direction] || DIRECTION_STYLE.FLAT}`}>
                        {w.direction}
                      </span>
                    </td>
                    <td className="py-2 px-2">
                      <div className="flex items-center gap-2 min-w-[90px]">
                        <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                          <div className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full" style={{ width: `${exp}%` }} />
                        </div>
                        <span className="text-slate-400 tabular-nums w-8 text-right">{exp}%</span>
                      </div>
                    </td>
                    <td className="py-2 px-2">
                      <span className={`px-2 py-0.5 rounded text-[10px] ${rColor}`}>{w.regime}</span>
                    </td>
                    <td className="py-2 px-2 text-right text-slate-500 tabular-nums">{num(w.expected_return_pct, 1)}%</td>
                    <td className={`py-2 px-2 text-right tabular-nums font-medium ${(w.realized_return_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {num(w.realized_return_pct, 1)}%
                    </td>
                    <td className={`py-2 px-2 text-right tabular-nums ${(w.buyhold_return_pct ?? 0) >= 0 ? 'text-slate-300' : 'text-slate-500'}`}>
                      {num(w.buyhold_return_pct, 1)}%
                    </td>
                    <td className="py-2 px-2 text-center">{w.hit ? '✅' : '·'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

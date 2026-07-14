'use client';

/**
 * ForecastingView — vista FORECASTING del GlobalMarkets Terminal (CTR-GM-UI-001,
 * prototipo Var B líneas 615–720): 4 filtros dropdown (Activo · Periodo · Modelo ·
 * Horizonte) con estado en la URL, cuerpo por activo con piel GM.
 *
 * Contratos que NO se rompen (CTR-FE-BE-001 §5):
 *  • Datos e imágenes por las rutas existentes con delay por plan server-side:
 *    - USD/COP (model zoo): /api/forecasting/bi_dashboard_unified.csv + PNGs raíz
 *      (backtest_<model>_h<n>.png / forward_*_<YYYY>_W<NN>.png) vía /api/forecasting/<file>.
 *    - Oro/BTC (weekly inference): /api/forecasting/<asset>/{index,forward,weekly_inference_<year>}.json.
 *  • El selector de activos deriva de ANALYSIS_ASSETS (SSOT compartida con /analysis).
 *
 * Metodología (todas las parejas): entrenado ≤ Dic-2024 · 2025 = backtest OOS · 2026 = producción.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { useSession } from 'next-auth/react';
import Link from 'next/link';
import Papa from 'papaparse';
import { ChevronDown, ImageOff, Info, Lock } from 'lucide-react';
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer,
  Tooltip as RTooltip, XAxis, YAxis,
} from 'recharts';

import {
  AsyncBoundary, GmBadge, GmDelta, GmKpi, GmPageHeader, GmPanel, useGmQuery,
  type AsyncState,
} from '@/components/gm';
import { ClientApiError } from '@/lib/api/gm-client';
import { defineGmDict, useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT, toneOf, GM_HEX, type GmTone } from '@/lib/ui/gm-tokens';
import { ANALYSIS_ASSETS, resolveAnalysisAsset } from '@/lib/contracts/analysis-assets';
import type {
  AssetWeeklyInference, EnsembleVariant, ForecastRecord, ViewType,
  WeeklyInferenceIndex, WeeklyInferenceStrategy,
} from '@/components/forecasting/types';

// ───────────────────────────────────────────────────────────── helpers

const num = (v: number | null | undefined, d = 2) =>
  v === null || v === undefined || isNaN(Number(v)) ? '—' : Number(v).toFixed(d);

const uniq = <T,>(arr: T[], key: keyof T): string[] =>
  Array.from(new Set(
    arr.map((r) => String(r[key])).filter((v) => v && v !== 'null' && v !== 'undefined'),
  )).sort();

const ENSEMBLE_VARIANTS: EnsembleVariant[] = [
  { value: 'ENSEMBLE_BEST_OF_BREED', label: 'Best of Breed', imageKey: 'best_of_breed' },
  { value: 'ENSEMBLE_TOP_3', label: 'Top 3 Average', imageKey: 'top_3' },
  { value: 'ENSEMBLE_TOP_6_MEAN', label: 'Top 6 Average', imageKey: 'top_6_mean' },
];

const DIRECTION_TONE: Record<string, GmTone> = { LONG: 'pos', SHORT: 'neg', FLAT: 'neutral' };

// Régimen → tono (Oro: compression/trend/stretched/event · BTC: accumulation/markup/distribution/markdown)
const REGIME_TONE: Record<string, GmTone> = {
  trend: 'pos', markup: 'pos',
  compression: 'info', accumulation: 'info',
  stretched: 'warn', distribution: 'warn',
  event: 'neg', markdown: 'neg',
};

interface ForwardDoc {
  image: string;
  direction: string;
  exposure: number;
  last_close: number;
  vol_daily_pct: number;
  generated_at: string;
  methodology?: string;
  horizons: Array<{
    h_days: number; exp_move_pct: number; ci95_pct: [number, number];
    da_2025_pct: number | null; n_2025: number;
  }>;
}

function isLocked(err: Error | null): boolean {
  return err instanceof ClientApiError && (err.status === 403 || err.code === 'FORBIDDEN');
}

// ───────────────────────────────────────────────────────────── CSV hook (model zoo)

/** Igual que useGmQuery pero para el CSV del zoo (delay por plan aplicado server-side). */
function useForecastCsv(path: string | null): AsyncState<ForecastRecord[]> {
  const [data, setData] = useState<ForecastRecord[] | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(!!path);
  const abortRef = useRef<AbortController | null>(null);

  const reload = useCallback(() => {
    if (!path) return;
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setLoading(true);
    fetch(path, { signal: ac.signal, cache: 'no-store', credentials: 'same-origin' })
      .then(async (res) => {
        if (!res.ok) {
          throw new ClientApiError(
            res.status === 401 ? 'UNAUTHENTICATED'
              : res.status === 403 ? 'FORBIDDEN'
              : res.status >= 500 ? 'UPSTREAM_ERROR' : 'UPSTREAM_BAD_REQUEST',
            `No se pudo cargar el CSV de forecasting (HTTP ${res.status}).`,
            res.status,
          );
        }
        return res.text();
      })
      .then((text) => new Promise<ForecastRecord[]>((resolve, reject) => {
        Papa.parse(text, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results) => resolve(results.data as ForecastRecord[]),
          error: (e: Error) => reject(e),
        });
      }))
      .then((rows) => { setData(rows); setError(null); })
      .catch((e: unknown) => {
        if ((e as Error)?.name === 'AbortError') return;
        setError(e instanceof Error ? e : new Error(String(e)));
      })
      .finally(() => setLoading(false));
  }, [path]);

  useEffect(() => { reload(); return () => abortRef.current?.abort(); }, [reload]);

  return { data, error, loading, reload };
}

// ───────────────────────────────────────────────────────────── UI atoms

/** Dropdown GM del prototipo (label micro + botón + popover). */
function GmDropdown({ label, value, options, onChange, disabled = false, minW = 180, mono = false }: {
  label: string;
  value: string;
  options: Array<{ value: string; label: string; disabled?: boolean }>;
  onChange: (v: string) => void;
  disabled?: boolean;
  minW?: number;
  mono?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, [open]);

  const current = options.find((o) => o.value === value);

  return (
    <div className="relative" ref={ref}>
      <div className={`${GMT.label} ${GM.textMuted} mb-1.5`}>{label}</div>
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className={`${GM.input} ${GM.focus} h-10 flex items-center justify-between gap-2.5 font-semibold
          ${mono ? 'font-mono' : ''} ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
        style={{ minWidth: minW }}
      >
        <span className="truncate">{current?.label ?? value}</span>
        <ChevronDown className={`w-4 h-4 shrink-0 ${GM.accent}`} aria-hidden />
      </button>
      {open && (
        <div
          role="listbox"
          className={`absolute top-[70px] left-0 z-40 p-1.5 flex flex-col gap-0.5 max-h-72 overflow-y-auto ${GM.popover}`}
          style={{ minWidth: Math.max(minW, 200) }}
        >
          {options.map((o) => (
            <button
              key={o.value}
              type="button"
              role="option"
              aria-selected={o.value === value}
              disabled={o.disabled}
              onClick={() => { onChange(o.value); setOpen(false); }}
              className={`text-left px-3 py-2 rounded-[9px] text-[12.5px] font-semibold ${GM.focus}
                ${o.disabled ? `${GM.textFaint} cursor-default` : o.value === value ? GM.navActive : GM.navIdle}`}
            >
              {o.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/** PNG del pipeline con onError→ocultar (deja el placeholder punteado del prototipo). */
function GmForecastImage({ src, alt }: { src: string; alt: string }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => { setFailed(false); }, [src]);

  if (failed) {
    return (
      <div className={`${GM.panelInner} border-dashed min-h-[300px] flex flex-col items-center justify-center gap-2.5 p-6`}>
        <ImageOff className={`w-9 h-9 ${GM.textFaint}`} aria-hidden />
        <span className={`${GMT.body} font-semibold ${GM.textSec}`}>Gráfico no disponible</span>
        <code className={`${GMT.micro} ${GM.textMuted} font-mono ${GM.panelSoft} px-2.5 py-1`}>{src}</code>
      </div>
    );
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={src}
      alt={alt}
      className={`w-full h-auto ${GM.panelInner}`}
      onError={() => setFailed(true)}
    />
  );
}

// ─────────────────────────────────────────────── i18n (panel bloqueado por plan)

const FC_LOCK_DICT = defineGmDict({
  es: {
    lockedTitleSuffix: 'no está incluido en tu plan',
    lockedBody: 'El plan free incluye USD/COP. Desbloquea este activo con un plan superior o como add-on por activo — mismos datos publicados que ven los suscriptores, sin recomputar nada.',
    lockedFeatures: 'Inferencia semanal|Dirección · exposición · régimen|Forward forecast con IC95',
    seePlans: 'Ver planes',
    lockedHint: 'Tu sesión y tu selección se conservan — vuelve aquí tras actualizar tu plan.',
  },
  en: {
    lockedTitleSuffix: 'is not included in your plan',
    lockedBody: 'The free plan includes USD/COP. Unlock this asset with a higher plan or as a per-asset add-on — the same published data subscribers see, nothing recomputed.',
    lockedFeatures: 'Weekly inference|Direction · exposure · regime|Forward forecast with 95% CI',
    seePlans: 'See plans',
    lockedHint: 'Your session and selection are kept — come back after upgrading your plan.',
  },
});

/** Activo bloqueado por plan (403 del gate server-side) — panel rico del prototipo:
 *  candado + copy + chips de lo que incluye + CTA "Ver planes" (upsell → /pricing). */
function LockedAsset({ assetName }: { assetName: string }) {
  const t = useGmT(FC_LOCK_DICT);
  return (
    <GmPanel>
      <div
        className="flex flex-col items-center justify-center text-center min-h-[320px] p-5"
        data-testid="forecasting-locked-asset"
      >
        <span className={`w-16 h-16 rounded-[18px] ${GM.warnBadge} flex items-center justify-center mb-4`}>
          <Lock className="w-7 h-7" aria-hidden />
        </span>
        <div className={`text-lg font-bold ${GM.text} mb-1.5`}>
          {assetName} {t('lockedTitleSuffix')}
        </div>
        <p className={`${GMT.body} ${GM.textSec} max-w-[420px] leading-relaxed mb-4`}>
          {t('lockedBody')}
        </p>
        <div className="flex flex-wrap justify-center gap-1.5 mb-5">
          {t('lockedFeatures').split('|').map((f) => (
            <span key={f} className={`${GM.neutralBadge} rounded-[7px] px-[9px] py-1 text-[11px] font-medium`}>
              {f}
            </span>
          ))}
        </div>
        <Link
          href="/pricing"
          data-testid="forecasting-locked-cta"
          className={`${GM.ctaSoft} ${GM.focus} inline-flex items-center h-11 px-5 text-[13px]`}
        >
          {t('seePlans')} →
        </Link>
        <p className={`${GMT.micro} ${GM.textMuted} mt-3.5 m-0`}>{t('lockedHint')}</p>
      </div>
    </GmPanel>
  );
}

// ─────────────────────────────────────────────── model zoo (USD/COP · BTC/USDT)

function AssetModelZoo({ rows, view, week, model, horizon, pngBase, forecastLabel }: {
  rows: ForecastRecord[];
  view: ViewType;
  week: string;
  model: string;
  horizon: string;
  /** Base path for PNGs served through the plan gate ('/api/forecasting/' or '.../btcusdt/'). */
  pngBase: string;
  /** Human label for empty/alt text (e.g. 'USD/COP', 'Bitcoin'). */
  forecastLabel: string;
}) {
  const isEnsemble = ENSEMBLE_VARIANTS.some((v) => v.value === model);
  const ensembleKey = ENSEMBLE_VARIANTS.find((v) => v.value === model)?.imageKey ?? null;
  const modelLabel = model === 'ALL'
    ? 'Consensus (todos los modelos)'
    : ENSEMBLE_VARIANTS.find((v) => v.value === model)?.label ?? model;

  // Misma lógica de filtrado que la página legacy (view → semana → modelo → horizonte).
  const filtered = useMemo(() => {
    let res = rows.filter((r) => {
      if (r.view_type !== view) return false;
      if (view === 'forward_forecast' && week && String(r.inference_week) !== week) return false;
      return true;
    });
    if (model !== 'ALL') res = res.filter((r) => r.model_name === model);
    if (horizon !== 'ALL') res = res.filter((r) => String(r.horizon_days) === horizon);
    return res;
  }, [rows, view, week, model, horizon]);

  const weekSuffix = week ? week.replace('-', '_') : '';

  // Imagen + métricas (misma resolución de casos que la página legacy).
  let imageFile: string | null = null;
  let caption = '';
  let kpis: Array<{ label: string; value: string; tone: GmTone }> = [];

  if (view === 'forward_forecast' && model === 'ALL') {
    imageFile = weekSuffix ? `forward_consensus_${weekSuffix}.png` : 'forward_consensus.png';
    caption = 'Consensus Forecast (todos los modelos + promedio)';
    if (filtered.length > 0) {
      const row = filtered[0];
      kpis = [
        { label: 'DA promedio', value: `${num(row.model_avg_direction_accuracy)}%`, tone: (row.model_avg_direction_accuracy ?? 0) >= 55 ? 'pos' : 'warn' },
        { label: 'RMSE promedio', value: num(row.model_avg_rmse, 4), tone: 'neutral' },
        { label: 'Semana', value: week || '—', tone: 'accent' },
        { label: 'Registros', value: String(filtered.length), tone: 'neutral' },
      ];
    }
  } else if (isEnsemble && ensembleKey) {
    imageFile = weekSuffix
      ? `forward_ensemble_${ensembleKey}_${weekSuffix}.png`
      : `forward_ensemble_${ensembleKey}.png`;
    caption = `${modelLabel} — Ensemble Forecast`;
    kpis = [
      { label: 'Método', value: modelLabel, tone: 'accent' },
      { label: 'Tipo', value: ensembleKey === 'best_of_breed' ? 'Best/horizonte' : ensembleKey === 'top_3' ? 'Top 3 Avg' : 'Top 6 Avg', tone: 'info' },
      { label: 'Semana', value: week || '—', tone: 'accent' },
      { label: 'Registros', value: String(filtered.length), tone: 'neutral' },
    ];
  } else if (filtered.length >= 1) {
    const row = filtered[0];
    if (view === 'backtest') {
      imageFile = row.image_backtest;
      caption = `${row.model_name} — Backtest (H=${row.horizon_days})`;
      kpis = [
        { label: 'Direction Accuracy', value: `${num(row.direction_accuracy)}%`, tone: (row.direction_accuracy ?? 0) >= 55 ? 'pos' : 'warn' },
        { label: 'RMSE', value: num(row.rmse, 4), tone: 'neutral' },
        { label: 'MAE', value: num(row.mae, 4), tone: 'info' },
        { label: 'R²', value: num(row.r2, 4), tone: (row.r2 ?? 0) > 0 ? 'pos' : 'neg' },
      ];
    } else {
      imageFile = row.image_forecast || row.image_path;
      caption = `${row.model_name} — Forecast (H=${row.horizon_days})`;
      kpis = [
        { label: 'WF Direction Accuracy', value: `${num(row.wf_direction_accuracy)}%`, tone: (row.wf_direction_accuracy ?? 0) >= 55 ? 'pos' : 'warn' },
        { label: 'Sharpe', value: num(row.sharpe, 2), tone: toneOf(row.sharpe) },
        { label: 'Profit Factor', value: num(row.profit_factor, 2), tone: (row.profit_factor ?? 0) >= 1 ? 'pos' : 'neg' },
        { label: 'Max Drawdown', value: `${num(row.max_drawdown != null ? row.max_drawdown * 100 : null, 1)}%`, tone: 'neg' },
      ];
    }
  }

  // Ranking de modelos (solo backtest) — misma agregación que MetricsRankingPanel legacy.
  const rankings = useMemo(() => {
    if (view !== 'backtest') return [];
    const bt = rows.filter((r) =>
      r.view_type === 'backtest' && (horizon === 'ALL' || String(r.horizon_days) === horizon));
    const acc: Record<string, { da: number[]; sharpe: number[]; pf: number[]; mdd: number[] }> = {};
    bt.forEach((r) => {
      const m = (acc[r.model_id] ??= { da: [], sharpe: [], pf: [], mdd: [] });
      const da = r.wf_direction_accuracy || r.direction_accuracy;
      if (da != null && !isNaN(da)) m.da.push(da);
      if (r.sharpe != null && !isNaN(r.sharpe)) m.sharpe.push(r.sharpe);
      if (r.profit_factor != null && !isNaN(r.profit_factor)) m.pf.push(r.profit_factor);
      if (r.max_drawdown != null && !isNaN(r.max_drawdown)) m.mdd.push(r.max_drawdown);
    });
    const avg = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
    return Object.entries(acc)
      .map(([model_id, m]) => ({
        model_id, da: avg(m.da), sharpe: avg(m.sharpe), pf: avg(m.pf), mdd: avg(m.mdd),
      }))
      .filter((m) => m.da !== null)
      .sort((a, b) => (b.da ?? 0) - (a.da ?? 0));
  }, [rows, view, horizon]);

  const fmtDa = (da: number | null) => {
    if (da == null || isNaN(da)) return '—';
    return `${(da < 1 ? da * 100 : da).toFixed(1)}%`;
  };
  const prettyModel = (id: string) => id.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase());
  const normalize = (s: string) => s.toLowerCase().replace(/[_ ]/g, '');

  const imageSrc = imageFile ? `${pngBase}${imageFile.split('/').pop()}` : null;

  if (!imageSrc && filtered.length === 0) {
    return (
      <GmPanel>
        <div className={`min-h-[200px] flex items-center justify-center ${GMT.body} ${GM.textMuted}`}>
          No hay datos para esta selección.
        </div>
      </GmPanel>
    );
  }

  return (
    <div className="flex flex-col gap-4" data-testid="forecasting-cop-zoo">
      <div className={`grid gap-4 items-start ${view === 'backtest' ? 'lg:grid-cols-[1.6fr_1fr]' : ''}`}>
        {/* Gráfico del forecast (PNG servido por la ruta gateada) */}
        <GmPanel title={caption || 'Forecast'} meta={imageFile ? imageFile.split('/').pop() : undefined}>
          {imageSrc
            ? <GmForecastImage src={imageSrc} alt={caption || `Forecast ${forecastLabel}`} />
            : (
              <div className={`min-h-[200px] flex items-center justify-center ${GMT.body} ${GM.textMuted}`}>
                Selecciona un modelo y horizonte para ver el gráfico.
              </div>
            )}
          <p className={`mt-3 mb-0 ${GMT.micro} ${GM.textMuted} leading-relaxed`}>
            Walk-forward, sin re-etiquetado: entrenado ≤ Dic-2024 · 2025 = backtest OOS · forward = inferencia semanal.
          </p>
        </GmPanel>

        {/* Ranking (solo en vista backtest, como el prototipo fcHasModels) */}
        {view === 'backtest' && (
          <GmPanel title={`Ranking de modelos${horizon !== 'ALL' ? ` (H=${horizon})` : ''}`} meta="walk-forward">
            {rankings.length === 0 ? (
              <div className={`${GMT.body} ${GM.textMuted}`}>Sin datos de backtest para este horizonte.</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-[12px]">
                  <thead>
                    <tr className={`${GMT.micro} ${GM.textMuted} uppercase tracking-[.4px]`}>
                      <th className="text-left py-1.5 pr-2 font-bold">#</th>
                      <th className="text-left py-1.5 pr-2 font-bold">Modelo</th>
                      <th className="text-right py-1.5 pr-2 font-bold">DA</th>
                      <th className="text-right py-1.5 pr-2 font-bold">Sharpe</th>
                      <th className="text-right py-1.5 font-bold">PF</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rankings.map((m, idx) => {
                      const selected = model !== 'ALL' && normalize(m.model_id) === normalize(model);
                      return (
                        <tr
                          key={m.model_id}
                          className={`border-t border-[rgba(148,163,184,.07)] ${GM.rowHover} ${selected ? GM.navActive : ''}`}
                        >
                          <td className={`py-2 pr-2 font-mono ${idx === 0 ? GM.pos : GM.textMuted}`}>{idx + 1}</td>
                          <td className={`py-2 pr-2 font-semibold ${selected ? GM.accent : GM.textStrong}`}>
                            {prettyModel(m.model_id)}
                          </td>
                          <td className={`py-2 pr-2 text-right font-mono font-bold ${((m.da ?? 0) > 55 || ((m.da ?? 0) < 1 && (m.da ?? 0) > 0.55)) ? GM.pos : GM.warn}`}>
                            {fmtDa(m.da)}
                          </td>
                          <td className={`py-2 pr-2 text-right font-mono ${(m.sharpe ?? 0) >= 0 ? GM.textSec : GM.neg}`}>{num(m.sharpe)}</td>
                          <td className={`py-2 text-right font-mono ${(m.pf ?? 0) >= 1 ? GM.textSec : GM.neg}`}>{num(m.pf)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </GmPanel>
        )}
      </div>

      {/* KPIs del modelo/selección */}
      {kpis.length > 0 && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {kpis.map((k) => <GmKpi key={k.label} label={k.label} value={k.value} tone={k.tone} />)}
        </div>
      )}
    </div>
  );
}

// ───────────────────────────────────────────────────────────── Oro/BTC · weekly inference

function AssetWeeklyBody({ data, strategyId, forward }: {
  data: AssetWeeklyInference;
  strategyId: string;
  forward: ForwardDoc | null;
}) {
  const strategy: WeeklyInferenceStrategy | null =
    data.strategies.find((s) => s.strategy_id === strategyId) || data.strategies[0] || null;

  // Curva acumulada (estrategia vs buy&hold) — misma composición que la vista legacy.
  const chartData = useMemo(() => {
    if (!strategy) return [];
    let cs = 1, cb = 1;
    return strategy.weeks.map((w) => {
      cs *= 1 + (w.realized_return_pct ?? 0) / 100;
      cb *= 1 + (w.buyhold_return_pct ?? 0) / 100;
      return {
        week: w.iso_week.replace(/^\d+-/, ''),
        Estrategia: +((cs - 1) * 100).toFixed(2),
        BuyHold: +((cb - 1) * 100).toFixed(2),
      };
    });
  }, [strategy]);

  if (!strategy) return null;
  const s = strategy.summary;
  const isBacktest = data.year === 2025;

  return (
    <div className="flex flex-col gap-4" data-testid="forecasting-weekly-inference">
      {/* Forward forecast — imagen + horizontes (bandas de vol + posicionamiento de la campeona) */}
      {forward && (
        <GmPanel
          title="Forward Forecast"
          meta={forward.generated_at?.slice(0, 10)}
          actions={
            <span className="flex items-center gap-2">
              <GmBadge tone={DIRECTION_TONE[forward.direction] ?? 'neutral'}>{forward.direction}</GmBadge>
              <span className={`${GMT.micro} ${GM.textSec} font-mono`}>
                exposición {forward.exposure}x · σ diaria {forward.vol_daily_pct}%
              </span>
            </span>
          }
        >
          <GmForecastImage src={`/api/${forward.image}`} alt={`Forward forecast ${data.display_name}`} />
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5 mt-3.5">
            {forward.horizons.map((h) => (
              <div key={h.h_days} className={`${GM.panelInner} p-3 text-center`}>
                <div className={`${GMT.micro} ${GM.textMuted} font-bold font-mono`}>H = {h.h_days}d</div>
                <div className={`text-[14px] font-bold font-mono ${GM.accent}`}>±{h.exp_move_pct}%</div>
                <div className={`${GMT.micro} ${GM.textMuted} font-mono`}>IC95 {h.ci95_pct[0]}% / +{h.ci95_pct[1]}%</div>
                <div className={`text-[11px] font-bold mt-1 ${
                  (h.da_2025_pct ?? 0) >= 55 ? GM.pos : (h.da_2025_pct ?? 0) >= 50 ? GM.warn : GM.neg}`}>
                  DA 2025: {h.da_2025_pct ?? '—'}%
                </div>
              </div>
            ))}
          </div>
          {forward.methodology && (
            <p className={`mt-3 mb-0 ${GMT.micro} ${GM.textMuted} leading-relaxed`}>{forward.methodology}</p>
          )}
        </GmPanel>
      )}

      {/* Nota metodológica */}
      <div className={`${GM.panelSoft} flex items-start gap-2.5 px-4 py-3`}>
        <Info className={`w-4 h-4 shrink-0 mt-0.5 ${GM.accent}`} aria-hidden />
        <p className={`m-0 ${GMT.meta} ${GM.textSec} leading-relaxed`}>
          Inferencia semanal <span className={GM.textStrong}>basada en reglas</span> (no ML): posicionamiento causal
          por semana (dirección · exposición · régimen) frente al resultado real. Entrenado ≤ Dic-2024 ·{' '}
          <span className={GM.pos}>2025 = backtest OOS</span> · <span className={GM.accent}>2026 = producción</span>.
          &quot;Esperado&quot; es un proxy de sesgo, no una predicción ML.
        </p>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <GmKpi
          label={isBacktest ? 'Retorno estrategia 2025' : 'Retorno estrategia YTD'}
          value={`${num(s.ytd_strategy_return_pct, 1)}%`}
          tone={toneOf(s.ytd_strategy_return_pct)}
        />
        <GmKpi label="Buy & Hold" value={`${num(s.ytd_buyhold_return_pct, 1)}%`} tone="neutral" />
        <GmKpi label="Acierto direccional" value={`${num(s.hit_rate_pct, 1)}%`} tone="accent" />
        <GmKpi
          label="Semanas en mercado"
          value={`${s.weeks_in_market}/${s.weeks_total}`}
          tone="info"
          sub={`Exposición promedio ${num((s.avg_exposure ?? 0) * 100, 0)}%`}
        />
      </div>

      {/* Curva acumulada */}
      <GmPanel title="Curva acumulada — Estrategia vs Buy&Hold" meta={`${data.display_name} · ${data.year}`}>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,.12)" />
              <XAxis dataKey="week" tick={{ fontSize: 10, fill: GM_HEX.tick }} interval="preserveStartEnd" minTickGap={24} />
              <YAxis tick={{ fontSize: 10, fill: GM_HEX.tick }} tickFormatter={(v) => `${v}%`} />
              <RTooltip
                contentStyle={{ background: GM_HEX.tooltipBg, border: `1px solid ${GM_HEX.gridStroke}`, borderRadius: 10, fontSize: 12 }}
                formatter={(v: number) => `${v}%`}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="Estrategia" stroke={GM_HEX.pos} dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="BuyHold" stroke={GM_HEX.tick} dot={false} strokeWidth={1.5} strokeDasharray="4 3" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </GmPanel>

      {/* Tabla semanal */}
      <GmPanel title={`Inferencia semanal — ${data.display_name} ${data.year}`} meta={strategy.strategy_name}>
        <div className="overflow-x-auto">
          <table className="w-full text-[11.5px]">
            <thead>
              <tr className={`${GMT.micro} ${GM.textMuted} uppercase tracking-[.4px]`}>
                <th className="text-left py-2 px-2 font-bold">Semana</th>
                <th className="text-left py-2 px-2 font-bold">Dirección</th>
                <th className="text-left py-2 px-2 font-bold">Exposición</th>
                <th className="text-left py-2 px-2 font-bold">Régimen</th>
                <th className="text-right py-2 px-2 font-bold">Esperado</th>
                <th className="text-right py-2 px-2 font-bold">Estrategia</th>
                <th className="text-right py-2 px-2 font-bold">Buy&amp;Hold</th>
                <th className="text-center py-2 px-2 font-bold">✓</th>
              </tr>
            </thead>
            <tbody>
              {strategy.weeks.map((w) => {
                const exp = Math.round((w.exposure ?? 0) * 100);
                return (
                  <tr key={w.iso_week} className={`border-t border-[rgba(148,163,184,.07)] ${GM.rowHover}`}>
                    <td className={`py-2 px-2 whitespace-nowrap font-mono ${GM.textSec}`}>{w.iso_week}</td>
                    <td className="py-2 px-2">
                      <GmBadge tone={DIRECTION_TONE[w.direction] ?? 'neutral'}>{w.direction}</GmBadge>
                    </td>
                    <td className="py-2 px-2">
                      <div className="flex items-center gap-2 min-w-[90px]">
                        <div className="flex-1 h-1.5 rounded-full overflow-hidden bg-[rgba(148,163,184,.12)]">
                          <div className={`h-full rounded-full ${GM.brandGradient}`} style={{ width: `${exp}%` }} />
                        </div>
                        <span className={`${GM.textSec} font-mono tabular-nums w-9 text-right`}>{exp}%</span>
                      </div>
                    </td>
                    <td className="py-2 px-2">
                      <GmBadge tone={REGIME_TONE[w.regime] ?? 'neutral'}>{w.regime}</GmBadge>
                    </td>
                    <td className={`py-2 px-2 text-right font-mono tabular-nums ${GM.textMuted}`}>
                      {num(w.expected_return_pct, 1)}%
                    </td>
                    <td className="py-2 px-2 text-right"><GmDelta value={w.realized_return_pct} digits={1} /></td>
                    <td className="py-2 px-2 text-right"><GmDelta value={w.buyhold_return_pct} digits={1} /></td>
                    <td className={`py-2 px-2 text-center font-bold ${w.hit ? GM.pos : GM.textFaint}`}>
                      {w.hit ? '✓' : '·'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </GmPanel>
    </div>
  );
}

// ───────────────────────────────────────────────────────────── vista principal

export function ForecastingView() {
  const router = useRouter();
  const sp = useSearchParams();
  const { data: session } = useSession();
  const role = (session?.user as { role?: string } | undefined)?.role ?? 'free';
  const isFree = role === 'free';
  const isInternal = role === 'admin' || role === 'developer';

  // ── estado en la URL (querystring) ──
  const asset = resolveAnalysisAsset(sp.get('asset'));
  const assetMeta = ANALYSIS_ASSETS.find((a) => a.asset_id === asset) ?? ANALYSIS_ASSETS[0];
  // Model-zoo assets (USD/COP + BTC) share the 9-model ML surface; Gold = weekly inference.
  const isModelZoo = assetMeta.forecast_mode === 'model_zoo';
  // USD/COP keeps root paths (470 files unmoved); other zoo assets are namespaced by asset_id.
  const csvPath = isModelZoo
    ? (asset === 'usdcop'
      ? '/api/forecasting/bi_dashboard_unified.csv'
      : `/api/forecasting/${asset}/bi_dashboard_unified.csv`)
    : null;
  const pngBase = asset === 'usdcop' ? '/api/forecasting/' : `/api/forecasting/${asset}/`;

  const setParams = useCallback((patch: Record<string, string | null>) => {
    const q = new URLSearchParams(sp.toString());
    Object.entries(patch).forEach(([k, v]) => {
      if (v === null || v === '') q.delete(k); else q.set(k, v);
    });
    const qs = q.toString();
    router.replace(qs ? `/forecasting?${qs}` : '/forecasting', { scroll: false });
  }, [router, sp]);

  // ── datos (por activo; hooks incondicionales con path null) ──
  const csv = useForecastCsv(csvPath);
  const index = useGmQuery<WeeklyInferenceIndex>(!isModelZoo ? `/api/forecasting/${asset}/index.json` : null);
  const forward = useGmQuery<ForwardDoc>(!isModelZoo ? `/api/forecasting/${asset}/forward.json` : null);

  // ── model zoo: derivar opciones del CSV ──
  const rows = useMemo(() => csv.data ?? [], [csv.data]);
  const weeks = useMemo(
    () => uniq(rows.filter((r) => r.view_type === 'forward_forecast'), 'inference_week'),
    [rows],
  );
  const models = useMemo(() => uniq(rows, 'model_name'), [rows]);

  const periodParam = sp.get('period');
  const copView: ViewType = periodParam === 'backtest' ? 'backtest' : 'forward_forecast';
  const copWeek = copView === 'forward_forecast'
    ? (periodParam && weeks.includes(periodParam) ? periodParam : weeks[weeks.length - 1] ?? '')
    : '';

  const modelParam = sp.get('model') ?? 'ALL';
  const copModel = modelParam !== 'ALL'
    && (models.includes(modelParam) || ENSEMBLE_VARIANTS.some((v) => v.value === modelParam))
    ? modelParam : 'ALL';
  const copHorizons = useMemo(() => {
    const hs = uniq(rows.filter((r) => r.view_type === copView), 'horizon_days');
    return hs.sort((a, b) => Number(a) - Number(b));
  }, [rows, copView]);
  const hParam = sp.get('h') ?? 'ALL';
  const copHorizon = hParam !== 'ALL' && copHorizons.includes(hParam) ? hParam : 'ALL';
  const horizonEnabled = isModelZoo && copView === 'backtest' && copModel !== 'ALL';

  // ── Oro/BTC: derivar opciones del index ──
  const yearsAvail = useMemo(
    () => [...(index.data?.years ?? [])].sort((a, b) => b - a),
    [index.data],
  );
  const year = !isModelZoo
    ? (periodParam && yearsAvail.includes(Number(periodParam)) ? Number(periodParam)
      : yearsAvail.includes(2025) ? 2025 : yearsAvail[0] ?? null)
    : null;
  const weekly = useGmQuery<AssetWeeklyInference>(
    !isModelZoo && year != null ? `/api/forecasting/${asset}/weekly_inference_${year}.json` : null,
  );
  const strategies = index.data?.strategies ?? [];
  const strategyId = strategies.some((st) => st.strategy_id === modelParam)
    ? modelParam
    : (index.data?.primary_strategy_id || strategies[0]?.strategy_id || '');

  // ── opciones de los 4 dropdowns del prototipo ──
  const assetOptions = ANALYSIS_ASSETS.map((a) => ({ value: a.asset_id, label: a.display_name }));

  const periodOptions = isModelZoo
    ? [
      ...[...weeks].reverse().map((w) => ({ value: w, label: `Semana ${w}` })),
      { value: 'backtest', label: 'Backtest 2025 (OOS)' },
    ]
    : yearsAvail.map((y) => ({
      value: String(y),
      label: y === 2025 ? 'Backtest 2025 (OOS)' : y === 2026 ? 'Producción 2026 (YTD)' : String(y),
    }));

  const modelOptions = isModelZoo
    ? [
      { value: 'ALL', label: 'Consensus (todos)' },
      ...models
        .filter((m) => !m.includes('ENSEMBLE') && m !== 'CONSENSUS')
        .map((m) => ({ value: m, label: m })),
      { value: '__sep__', label: '— Ensembles —', disabled: true },
      ...ENSEMBLE_VARIANTS.map((v) => ({ value: v.value, label: v.label })),
    ]
    : strategies.map((st) => ({
      value: st.strategy_id,
      label: `${st.strategy_name}${st.strategy_id === index.data?.primary_strategy_id ? ' ★' : ''}`,
    }));

  const horizonOptions = horizonEnabled
    ? [
      { value: 'ALL', label: 'Overview (prom.)' },
      ...copHorizons.map((h) => ({ value: h, label: `H=${h} días` })),
    ]
    : [{ value: 'ALL', label: isModelZoo ? 'H=1…30 días' : 'Semanal (H=5d)' }];

  // Bloqueado por plan: cualquiera de los fetches del activo devuelve el 403 del gate
  // (asset not in plan / upgrade:true). El panel rico reemplaza al empty genérico.
  // USD/COP nunca se bloquea (incluido en el plan free); resto: CSV (zoo) o index/weekly/forward.
  const locked = asset !== 'usdcop' && (
    isModelZoo
      ? isLocked(csv.error)
      : (isLocked(index.error) || isLocked(weekly.error) || isLocked(forward.error))
  );

  return (
    <div data-testid="forecasting-view">
      <GmPageHeader
        kicker="Predicción semanal"
        title="Forecasting"
        subtitle={isModelZoo
          ? `${assetMeta.display_name} · 9 modelos de Machine Learning (walk-forward) con consensus y ensembles`
          : `${assetMeta.display_name} · inferencia semanal basada en reglas: dirección, exposición y régimen para todo el año`}
        actions={
          <span className="flex items-center gap-2">
            {isFree && <GmBadge tone="warn">T-1 semana · plan free</GmBadge>}
            <GmBadge tone="accent">{isModelZoo ? 'ML Model Zoo' : 'Weekly Inference'}</GmBadge>
          </span>
        }
      />

      {/* Filtros del prototipo: Activo · Periodo · Modelo · Horizonte (estado en URL) */}
      <div className="flex flex-wrap items-end gap-3.5 mb-5" data-testid="forecasting-filters">
        <GmDropdown
          label="Activos"
          value={asset}
          options={assetOptions}
          minW={220}
          onChange={(v) => setParams({ asset: v, period: null, model: null, h: null })}
        />
        <GmDropdown
          label="Periodo"
          value={isModelZoo ? (copView === 'backtest' ? 'backtest' : copWeek) : String(year ?? '')}
          options={periodOptions}
          minW={210}
          disabled={periodOptions.length === 0}
          onChange={(v) => setParams({ period: v, ...(v === 'backtest' ? {} : { h: null }) })}
        />
        <GmDropdown
          label={isModelZoo ? 'Modelo' : 'Estrategia'}
          value={isModelZoo ? copModel : strategyId}
          options={modelOptions}
          minW={200}
          mono={isModelZoo}
          disabled={modelOptions.length === 0}
          onChange={(v) => setParams({ model: v === 'ALL' ? null : v, ...(v === 'ALL' ? { h: null } : {}) })}
        />
        <GmDropdown
          label="Horizonte"
          value={copHorizon}
          options={horizonOptions}
          minW={140}
          mono
          disabled={!horizonEnabled}
          onChange={(v) => setParams({ h: v === 'ALL' ? null : v })}
        />
      </div>

      {/* Cuerpo por activo */}
      {locked ? (
        <LockedAsset assetName={assetMeta.display_name} />
      ) : isModelZoo ? (
        <AsyncBoundary
          state={csv}
          empty={(d) => d.length === 0}
          emptyProps={{
            title: 'Sin datos de forecasting',
            body: isInternal
              ? `Regenera el CSV con: python scripts/pipeline/generate_weekly_forecasts.py --asset ${asset} --num-weeks 30`
              : `El pipeline semanal aún no ha publicado datos para ${assetMeta.display_name}.`,
          }}
        >
          {(data) => (
            <AssetModelZoo
              rows={data} view={copView} week={copWeek} model={copModel} horizon={copHorizon}
              pngBase={pngBase} forecastLabel={assetMeta.display_name}
            />
          )}
        </AsyncBoundary>
      ) : (
        <AsyncBoundary
          state={{
            data: weekly.data,
            error: index.error ?? weekly.error,
            loading: index.loading || weekly.loading,
            reload: () => { index.reload(); weekly.reload(); },
          }}
          empty={(d) => !d.strategies?.length}
          emptyProps={{
            title: 'Sin inferencia semanal',
            body: isInternal
              ? `Regenera con: python -m scripts.pipeline.generate_asset_weekly_forecast --asset ${asset}`
              : 'Este activo aún no tiene inferencia semanal publicada.',
          }}
        >
          {(data) => (
            <AssetWeeklyBody data={data} strategyId={strategyId} forward={forward.data} />
          )}
        </AsyncBoundary>
      )}
    </div>
  );
}

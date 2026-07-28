'use client';

/**
 * Interpretabilidad (BL-20 · FABRIC A.7 · CTR-ADMIN-CONSOLE-001). Selector
 * superficie → activo → modelo → versión sobre los artefactos publicados en
 * `<repo>/data/interpretability/**` por `scripts/analysis/generate_interpretability.py`
 * (fuera de public/ — C-006: solo accesibles vía la API con gate admin:all).
 *
 * HONESTIDAD constitucional (quant-constitution + A.7):
 *  - Header fijo: "SHAP explica el modelo, no el mercado". Diagnóstico 0 trials.
 *  - La UI RENDERIZA los artefactos — jamás recomputa SHAP, kill-flags ni PnL
 *    (strategy-engines I-7: el frontend no re-evalúa condiciones).
 *  - Cero verde/rojo direccional sobre valores SHAP/PnL: los signos van en tinta
 *    neutra; el único acento es el warn ámbar de los kill-flags (inestabilidad).
 *  - Sirve para RECHAZAR modelos absurdos (p.ej. ridge con signo inestable entre
 *    años ⇒ confirma R²<0), no para probar verdades ni "importancia para operar".
 *  - Read-only: sin acciones; sin Sharpe/p-value (no es una superficie de claims).
 */
import { Microscope, ScanSearch } from 'lucide-react';

import type {
  InterpIndexEntry, InterpIndexResponse, InterpLinearSummary, InterpRuleSummary,
  InterpRuleYear, InterpSummary,
} from '@/lib/contracts/admin-console.contract';
import { COLOR, CTA, SURFACE, TYPE } from '@/lib/ui/tokens';

import { useMemo, useState } from 'react';

import { REFRESH, useAdminWidget } from './useAdminWidget';
import { Badge, Card, EmptyState, SkeletonRows, fmtRelative, useNow } from './ui';

export type { InterpLinearSummary, InterpRuleSummary, InterpSummary };

const DASH = '—';
/** Signo tipográfico (U+2212) — consistente con tabular-nums, sin tono semántico. */
const MINUS = '−';

// ─────────────────────────────────────────────── formatters (tinta neutra siempre)

/** Fracción → % con 1 decimal y minus tipográfico. 1.553 → "155.3%", −0.461 → "−46.1%". */
export function fmtSignedPct(v: number | null | undefined, digits = 1): string {
  if (v == null || !Number.isFinite(v)) return DASH;
  const s = (Math.abs(v) * 100).toFixed(digits);
  return `${v < 0 ? MINUS : ''}${s}%`;
}

/** Aporte |φ| en puntos porcentuales del retorno 5d predicho. 0.0056 → "0.56 pp". */
function fmtPp(v: number): string {
  return Number.isFinite(v) ? `${(v * 100).toFixed(2)} pp` : DASH;
}

function fmtCoef(v: number): string {
  return Number.isFinite(v) ? `${v < 0 ? MINUS : ''}${Math.abs(v).toFixed(4)}` : DASH;
}

// ─────────────────────────────────────────────── helpers puros (formateo del artefacto)

export interface SignMatrix {
  years: string[];
  rows: Array<{ feature: string; flagged: boolean; signs: Array<-1 | 0 | 1> }>;
}

/**
 * Matriz de signos por año a partir del `by_year` del artefacto. SOLO formatea:
 * el kill-flag viene de `kill_flags_sign_change_by_year` (lo computó el generador),
 * nunca se rederiva aquí. Feature ausente en un año ⇒ 0 (nunca NaN).
 */
export function buildSignMatrix(summary: InterpLinearSummary, topN = 12): SignMatrix {
  const years = Object.keys(summary.by_year).sort();
  const flagged = new Set(summary.kill_flags_sign_change_by_year);
  const rows = summary.top_features.slice(0, topN).map((f) => ({
    feature: f.feature,
    flagged: flagged.has(f.feature),
    signs: years.map<-1 | 0 | 1>((y) => {
      const hit = (summary.by_year[y] ?? []).find((r) => r.feature === f.feature);
      const v = hit?.mean_shap ?? 0;
      if (!Number.isFinite(v) || v === 0) return 0;
      return v > 0 ? 1 : -1;
    }),
  }));
  return { years, rows };
}

// ─────────────────────────────────────────────── banner constitucional (header fijo)

function ConstitutionBanner({ nota }: { nota: string }) {
  return (
    <p className={`border-l-2 border-[var(--gm-accent)] pl-3 ${TYPE.body} ${COLOR.textSecondary}`}>
      {nota}
    </p>
  );
}

// ─────────────────────────────────────────────── panel SHAP lineal (zoo ridge/BR)

export function LinearShapPanel({ summary }: { summary: InterpLinearSummary }) {
  const matrix = buildSignMatrix(summary);
  const nFlags = summary.kill_flags_sign_change_by_year.length;
  const maxAbs = Math.max(...summary.top_features.map((f) => f.mean_abs_shap), 1e-12);

  return (
    <div className="space-y-4" data-testid="interp-linear-panel">
      <ConstitutionBanner nota={summary.nota} />

      <Card
        title={`Kill-flags A.7 — ${summary.model_id} · ${summary.asset}`}
        icon={<ScanSearch className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
        info="Flag calculado por el generador (signo del aporte medio que cambia entre años con magnitud material). La decisión de rechazo es humana; la UI no recomputa nada."
        badge={<Badge tone={nFlags > 0 ? 'warn' : 'neutral'}>{nFlags}/{summary.n_features}</Badge>}
      >
        <p className={`${TYPE.body} ${COLOR.textPrimary}`}>
          Diagnóstico del predictor débil: {nFlags} de {summary.n_features} features cambian el
          signo de su aporte medio entre años. Esto <strong>no es importancia para operar</strong> —
          una atribución inestable confirma que el modelo no sostiene una relación estable
          (coherente con R²&lt;0); sirve para rechazar modelos absurdos, no para probar verdades.
        </p>
        {nFlags > 0 && (
          <ul className="mt-3 flex flex-wrap gap-1.5" aria-label="features con signo inestable">
            {summary.kill_flags_sign_change_by_year.map((f) => (
              <li key={f}><Badge tone="warn">{f}</Badge></li>
            ))}
          </ul>
        )}
        <p className={`mt-3 ${TYPE.meta}`}>
          {summary.scope} · fit: {summary.fit.scheme} · n_train {summary.fit.n_train} ·
          origen {summary.fit.origin} · {summary.fit.scaler} · H={summary.fit.horizon}
        </p>
      </Card>

      <Card
        title="Aporte por feature (φ, SHAP lineal cerrado)"
        icon={<Microscope className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
        info="phi_j = coef_j·(x_j−mu_j)/sigma_j del último fit congelado. Magnitud = mean|φ| en pp del retorno 5d predicho. Sin colores direccionales: el signo es dato, no veredicto."
      >
        <div role="region" aria-label="aporte por feature" tabIndex={0} className="overflow-x-auto">
          <table className="w-full min-w-[560px] text-xs">
            <caption className="sr-only">Aporte por feature (φ, SHAP lineal cerrado)</caption>
            <thead>
              <tr className={`text-left ${COLOR.textSecondary} border-b border-[var(--gm-border)]`}>
                <th scope="col" className="py-2 pr-3">#</th>
                <th scope="col" className="pr-3">Feature</th>
                <th scope="col" className="pr-3 text-right">coef</th>
                <th scope="col" className="pr-3 text-right">mean |φ|</th>
                <th scope="col" className="pr-3 w-[30%]">magnitud relativa</th>
                <th scope="col" className="pr-3">estabilidad</th>
              </tr>
            </thead>
            <tbody>
              {summary.top_features.map((f) => (
                <tr key={f.feature} className="h-9 border-b border-[rgba(148,163,184,.08)]">
                  <td className={`pr-3 ${COLOR.textSecondary}`}>{f.rank}</td>
                  <td className={`pr-3 font-medium ${COLOR.textPrimary}`}>{f.feature}</td>
                  <td className={`pr-3 text-right ${TYPE.mono} ${COLOR.textPrimary}`}>{fmtCoef(f.coef)}</td>
                  <td className={`pr-3 text-right ${TYPE.mono} ${COLOR.textPrimary}`}>{fmtPp(f.mean_abs_shap)}</td>
                  <td className="pr-3">
                    <div className="h-1.5 w-full rounded-full bg-[rgba(148,163,184,.12)] overflow-hidden" aria-hidden>
                      <div
                        className="h-full rounded-full bg-[rgba(148,163,184,.55)]"
                        style={{ width: `${Math.min(100, (f.mean_abs_shap / maxAbs) * 100)}%` }}
                      />
                    </div>
                  </td>
                  <td className="pr-3">
                    {summary.kill_flags_sign_change_by_year.includes(f.feature)
                      ? <Badge tone="warn">signo inestable</Badge>
                      : <span className={COLOR.textSecondary}>{DASH}</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card
        title="Signo del aporte medio por año"
        icon={<ScanSearch className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
        info="Signo de mean(φ) por año del artefacto (+ / − / ·). Tinta neutra a propósito: un signo no es una señal de trading."
      >
        <div role="region" aria-label="signo del aporte medio por año" tabIndex={0} className="overflow-x-auto">
          <table className="w-full min-w-[480px] text-xs" aria-label="Signo del aporte medio por año">
            <caption className="sr-only">Signo del aporte medio por año</caption>
            <thead>
              <tr className={`text-left ${COLOR.textSecondary} border-b border-[var(--gm-border)]`}>
                <th scope="col" className="py-2 pr-3">Feature</th>
                {matrix.years.map((y) => (
                  <th key={y} scope="col" className={`pr-2 text-center ${TYPE.mono}`}>{y}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.rows.map((r) => (
                <tr key={r.feature} className="h-8 border-b border-[rgba(148,163,184,.08)]">
                  <td className={`pr-3 font-medium ${r.flagged ? COLOR.warn.text : COLOR.textPrimary}`}>
                    {r.feature}
                  </td>
                  {r.signs.map((s, i) => (
                    <td key={matrix.years[i]} className={`pr-2 text-center ${TYPE.mono} ${COLOR.textSecondary}`}>
                      {s > 0 ? '+' : s < 0 ? MINUS : '·'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

// ─────────────────────────────────────────────── panel atribución de reglas (NO SHAP)

function RuleYearCells({ d }: { d: InterpRuleYear }) {
  return (
    <>
      <td className={`pr-3 text-right ${TYPE.mono} ${COLOR.textSecondary}`}>{d.n_days}</td>
      <td className={`pr-3 text-right ${TYPE.mono} ${COLOR.textPrimary}`}>{fmtSignedPct(d.pnl_beta)}</td>
      <td className={`pr-3 text-right ${TYPE.mono} ${COLOR.textPrimary}`}>{fmtSignedPct(d.pnl_timing_cov_pos_ret)}</td>
      <td className={`pr-3 text-right ${TYPE.mono} ${COLOR.textSecondary}`}>{fmtSignedPct(d.costs)}</td>
      <td className={`pr-3 text-right ${TYPE.mono} ${COLOR.textPrimary}`}>{fmtSignedPct(d.pnl_net)}</td>
      <td className={`pr-3 text-right ${TYPE.mono} ${COLOR.textSecondary}`}>
        {d.pct_days_trend_on != null ? fmtSignedPct(d.pct_days_trend_on, 0) : DASH}
      </td>
      <td className={`pr-3 text-right ${TYPE.mono} ${COLOR.textSecondary}`}>
        {d.avg_exposure != null ? `${d.avg_exposure.toFixed(2)}×` : DASH}
      </td>
    </>
  );
}

export function RuleAttributionPanel({ summary }: { summary: InterpRuleSummary }) {
  const years = Object.keys(summary.by_year).sort();
  const t = summary.pnl_decomposition;

  return (
    <div className="space-y-4" data-testid="interp-rule-panel">
      <ConstitutionBanner nota={summary.nota} />

      <Card
        title={`Atribución de reglas — ${summary.model_id} · ${summary.asset}`}
        icon={<ScanSearch className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
        info="Política determinista: aquí no hay SHAP — se atribuye qué regla decidió y cuánto del PnL es beta vs timing. Diagnóstico sobre la misma serie del adapter publicado, no claim de edge."
        badge={<Badge tone="neutral">atribución, no SHAP</Badge>}
      >
        <p className={`${TYPE.body} ${COLOR.textPrimary}`}>
          Gate: <span className={TYPE.mono}>{summary.rules.gate}</span>
        </p>
        <dl className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <dt className={TYPE.sectionTitle}>% días trend on</dt>
            <dd className={`${TYPE.mono} ${COLOR.textPrimary} mt-0.5`}>{fmtSignedPct(summary.rules.pct_days_trend_on, 1)}</dd>
          </div>
          <div>
            <dt className={TYPE.sectionTitle}>% días con posición</dt>
            <dd className={`${TYPE.mono} ${COLOR.textPrimary} mt-0.5`}>{fmtSignedPct(summary.rules.pct_days_position_active, 1)}</dd>
          </div>
          <div>
            <dt className={TYPE.sectionTitle}>Exposición media</dt>
            <dd className={`${TYPE.mono} ${COLOR.textPrimary} mt-0.5`}>{summary.rules.avg_exposure.toFixed(2)}×</dd>
          </div>
          <div>
            <dt className={TYPE.sectionTitle}>Trades</dt>
            <dd className={`${TYPE.mono} ${COLOR.textPrimary} mt-0.5`}>{summary.rules.n_trades}</dd>
          </div>
        </dl>
        <p className={`mt-3 ${TYPE.meta}`}>{summary.scope}</p>
      </Card>

      <Card
        title="Descomposición del PnL (beta vs timing)"
        icon={<Microscope className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
        info="pnl_gross = beta + timing; timing = n·cov(pos, ret). Σ de retornos diarios (×100). Tinta neutra: es una descomposición diagnóstica, no un ranking de mérito."
      >
        <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          {([
            ['PnL bruto', t.pnl_gross],
            ['Beta', t.pnl_beta],
            ['Timing (cov)', t.pnl_timing_cov_pos_ret],
            ['Costes', t.costs],
            ['PnL neto', t.pnl_net],
          ] as const).map(([label, v]) => (
            <div key={label}>
              <dt className={TYPE.sectionTitle}>{label}</dt>
              <dd className={`${TYPE.mono} ${COLOR.textPrimary} mt-0.5`}>{fmtSignedPct(v)}</dd>
            </div>
          ))}
        </dl>
        <p className={`mt-3 ${TYPE.meta}`}>
          {t.n_days} días · el timing negativo en la mayoría de años es el hallazgo honesto:
          el gate aporta control de exposición, no predicción.
        </p>
      </Card>

      <Card
        title="Atribución por año"
        icon={<ScanSearch className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
        info="Misma descomposición, cortada por año del artefacto. Sin Sharpe ni p-values: no es una superficie de claims."
      >
        <div role="region" aria-label="atribución por año" tabIndex={0} className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-xs">
            <caption className="sr-only">Atribución de reglas por año</caption>
            <thead>
              <tr className={`text-left ${COLOR.textSecondary} border-b border-[var(--gm-border)]`}>
                <th scope="col" className="py-2 pr-3">Año</th>
                <th scope="col" className="pr-3 text-right">Días</th>
                <th scope="col" className="pr-3 text-right">Beta</th>
                <th scope="col" className="pr-3 text-right">Timing</th>
                <th scope="col" className="pr-3 text-right">Costes</th>
                <th scope="col" className="pr-3 text-right">Neto</th>
                <th scope="col" className="pr-3 text-right">% trend on</th>
                <th scope="col" className="pr-3 text-right">Expo</th>
              </tr>
            </thead>
            <tbody>
              {years.map((y) => (
                <tr key={y} className="h-8 border-b border-[rgba(148,163,184,.08)]">
                  <td className={`pr-3 ${TYPE.mono} ${COLOR.textPrimary}`}>{y}</td>
                  <RuleYearCells d={summary.by_year[y]} />
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

// ─────────────────────────────────────────────── carga del summary seleccionado

function SummaryLoader({ surface, asset, modelId, version }: {
  surface: string; asset: string; modelId: string; version: string;
}) {
  const url = `/api/admin/interpretability/summary?surface=${encodeURIComponent(surface)}` +
    `&asset=${encodeURIComponent(asset)}&model_id=${encodeURIComponent(modelId)}` +
    `&version=${encodeURIComponent(version)}`;
  const w = useAdminWidget<InterpSummary>(url);

  if (w.error && !w.data) {
    return (
      <EmptyState
        icon={<Microscope className="w-8 h-8" aria-hidden />}
        cause={<>No se pudo cargar el artefacto: {w.error}</>}
        action={<button onClick={w.reload} className={`${CTA.primary} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Reintentar</button>}
      />
    );
  }
  if (!w.data) return <SkeletonRows rows={5} cols={4} />;

  if (w.data.attribution_not_shap === true) return <RuleAttributionPanel summary={w.data} />;
  if (w.data.method === 'linear_shap_closed_form') return <LinearShapPanel summary={w.data} />;
  return (
    <EmptyState
      icon={<Microscope className="w-8 h-8" aria-hidden />}
      cause={`Método "${(w.data as { method?: string }).method}" aún sin renderer (fase 2: TreeSHAP).`}
    />
  );
}

// ─────────────────────────────────────────────── sección (selector 4 niveles)

interface Selection { surface: string; asset: string; model_id: string; version: string }

function firstSelection(entries: InterpIndexEntry[]): Selection | null {
  const e = entries[0];
  return e ? { surface: e.surface, asset: e.asset, model_id: e.model_id, version: e.versions[0] } : null;
}

export function InterpretabilitySection() {
  const index = useAdminWidget<InterpIndexResponse>('/api/admin/interpretability', { refreshMs: REFRESH.models });
  const [manual, setManual] = useState<Selection | null>(null);
  const now = useNow(30_000);

  const entries = useMemo(() => index.data?.entries ?? [], [index.data]);
  const sel = manual ?? firstSelection(entries);

  // Opciones dependientes: superficie → activo → modelo → versión.
  const surfaces = [...new Set(entries.map((e) => e.surface))];
  const assets = [...new Set(entries.filter((e) => e.surface === sel?.surface).map((e) => e.asset))];
  const models = entries.filter((e) => e.surface === sel?.surface && e.asset === sel?.asset);
  const entry = entries.find((e) => e.surface === sel?.surface && e.asset === sel?.asset && e.model_id === sel?.model_id);

  const pick = (patch: Partial<Selection>) => {
    // Al cambiar un nivel, re-resuelve los inferiores al primer valor válido.
    const surface = patch.surface ?? sel?.surface ?? '';
    const pool = entries.filter((e) => e.surface === surface);
    const asset = patch.asset && pool.some((e) => e.asset === patch.asset)
      ? patch.asset : (patch.surface ? pool[0]?.asset : sel?.asset) ?? pool[0]?.asset ?? '';
    const pool2 = pool.filter((e) => e.asset === asset);
    const model_id = patch.model_id && pool2.some((e) => e.model_id === patch.model_id)
      ? patch.model_id : (patch.surface || patch.asset ? pool2[0]?.model_id : sel?.model_id) ?? pool2[0]?.model_id ?? '';
    const found = pool2.find((e) => e.model_id === model_id);
    const version = patch.version && found?.versions.includes(patch.version)
      ? patch.version : found?.versions[0] ?? '';
    setManual({ surface, asset, model_id, version });
  };

  const meta = index.updatedAt
    ? <span title={new Date(index.updatedAt).toISOString()}>{fmtRelative(new Date(index.updatedAt).toISOString(), now)}</span>
    : null;

  return (
    <div className="space-y-4" data-testid="admin-section-interpretabilidad">
      {/* Header fijo (BL-20 impacto frontend) — visible siempre, con o sin datos. */}
      <p className={`${TYPE.body} font-semibold ${COLOR.textPrimary}`}>
        SHAP explica el modelo, no el mercado
        <span className={`ml-2 font-normal ${COLOR.textSecondary}`}>
          — diagnóstico 0 trials sobre modelos congelados; sirve para rechazar modelos absurdos, no para probar verdades (A.7).
        </span>
      </p>

      <Card
        title="Artefactos de interpretabilidad"
        icon={<Microscope className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
        info="Generados por scripts/analysis/generate_interpretability.py (solo test-folds). La consola solo los lee: regenerar = correr el script, no un botón."
        badge={index.data ? <Badge tone="neutral">{entries.length}</Badge> : null}
        meta={meta} stale={index.stale}
      >
        {index.error && !index.data && (
          <EmptyState
            icon={<Microscope className="w-8 h-8" aria-hidden />}
            cause={<>No se pudo cargar el índice: {index.error}</>}
            action={<button onClick={index.reload} className={`${CTA.primary} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Reintentar</button>}
          />
        )}
        {index.loading && !index.data && <SkeletonRows rows={1} cols={4} />}
        {index.data && entries.length === 0 && (
          <EmptyState
            icon={<Microscope className="w-8 h-8" aria-hidden />}
            cause="Sin artefactos — corre scripts/analysis/generate_interpretability.py para publicarlos."
          />
        )}
        {sel && (
          <div className="flex flex-wrap items-end gap-3">
            {([
              ['Superficie', 'surface', surfaces, sel.surface],
              ['Activo', 'asset', assets, sel.asset],
              ['Modelo', 'model_id', models.map((m) => m.model_id), sel.model_id],
              ['Versión', 'version', entry?.versions ?? [], sel.version],
            ] as const).map(([label, key, options, value]) => (
              <label key={key} className="flex flex-col gap-1">
                <span className={TYPE.sectionTitle}>{label}</span>
                <select
                  value={value}
                  onChange={(ev) => pick({ [key]: ev.target.value } as Partial<Selection>)}
                  className={`${SURFACE.input} ${CTA.focusRing} min-w-36`}
                  data-testid={`interp-select-${key}`}
                >
                  {options.map((o) => <option key={o} value={o}>{o}</option>)}
                </select>
              </label>
            ))}
          </div>
        )}
      </Card>

      {sel && sel.version && (
        <SummaryLoader
          key={`${sel.surface}/${sel.asset}/${sel.model_id}/${sel.version}`}
          surface={sel.surface} asset={sel.asset} modelId={sel.model_id} version={sel.version}
        />
      )}
    </div>
  );
}

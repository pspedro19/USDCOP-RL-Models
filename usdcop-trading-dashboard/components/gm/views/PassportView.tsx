'use client';

/**
 * PassportView — Control Tower (§24.5) + Strategy Passport (§24.4). BL-32.
 * ========================================================================
 *
 * "La portada que responde en diez segundos": LIBRO / SLEEVES / DATOS, and below it
 * the per-strategy Passport (identity · governance · lineage · 5 environments · live
 * · risk).
 *
 * Two rules shape every pixel here:
 *
 *  1. **The frontend does not calculate** (§24.1). Everything rendered comes from
 *     `/api/passport/*`, which composes published artifacts. This file contains no
 *     arithmetic on performance figures — only formatting.
 *  2. **DIAGNOSTIC surface** (plan 00, approval-gates §3). There is not a single
 *     approve / promote / deploy / kill control, for any role, ever. Vote 2 lives
 *     exclusively on /dashboard. The only interactive element is the strategy
 *     selector.
 *
 * Unavailable fields render as `—` with a "pendiente: BL-xx" hint instead of a zero.
 * That is the point: the reader can tell "we have no data" apart from "the value is
 * zero", which is precisely what a control tower has to get right.
 */

import { useState } from 'react';
import { Info, Layers, ShieldQuestion, Database, FileClock } from 'lucide-react';

import { AsyncBoundary, GmBadge, GmPageHeader, GmPanel, useGmQuery } from '@/components/gm';
import { GM, GMT, GM_TONE_TEXT, toneOf, type GmTone } from '@/lib/ui/gm-tokens';
import {
  BOOK_STATES, DSR_BAR, N_MAX_TRIALS, PASSPORT_ENVS,
  type BookState, type ControlTowerSnapshot, type EnvPerformance, type PassportEnv,
  type PendingInterface, type Sourced, type StrategyPassport, type TowerSleeve,
} from '@/lib/contracts/passport.contract';

// ────────────────────────────────────────────────────────────── formatting

/** Sourced → display string. NEVER renders NaN/Infinity, never invents a zero. */
function fmt(field: Sourced<number> | undefined, opts: { digits?: number; suffix?: string; signed?: boolean } = {}): string {
  const v = field?.value;
  if (v == null || !Number.isFinite(v)) return '—';
  const { digits = 2, suffix = '', signed = false } = opts;
  return `${signed && v >= 0 ? '+' : ''}${v.toFixed(digits)}${suffix}`;
}

function fmtInt(field: Sourced<number> | undefined): string {
  const v = field?.value;
  return v == null || !Number.isFinite(v) ? '—' : String(v);
}

function fmtStr(field: Sourced<string> | undefined): string {
  return field?.value ?? '—';
}

/** The hint shown under an unavailable field: what is missing and who supplies it. */
function pendingOf(field: Sourced<unknown> | undefined): string | null {
  if (!field || field.source?.status !== 'unavailable') return null;
  return field.source.pending ?? null;
}

/** One figure + its provenance. The provenance line is not decoration: it is the
 *  contract's promise that no number is published without a source. */
function Figure({ label, field, display, tone = 'neutral' }: {
  label: string; field?: Sourced<unknown>; display: string; tone?: GmTone;
}) {
  const pending = pendingOf(field as Sourced<unknown>);
  const note = field?.source?.note;
  return (
    <div className={`${GM.panelInner} p-3`}>
      <div className={`${GMT.label} ${GM.textMuted} mb-1`}>{label}</div>
      <div className={`${GMT.kpi} ${pending ? GM.textFaint : GM_TONE_TEXT[tone]}`}>{display}</div>
      {pending
        ? <div className={`${GMT.micro} ${GM.textFaint} mt-1`} title={pending}>pendiente: {pending}</div>
        : note
          ? <div className={`${GMT.micro} ${GM.textMuted} mt-1`}>{note}</div>
          : field?.source?.path
            ? <div className={`${GMT.micro} ${GM.textFaint} mt-1 truncate`} title={field.source.path}>
                fuente: {field.source.path.split('/').slice(-2).join('/')}
              </div>
            : null}
    </div>
  );
}

/** Cell inside a dense table: value + a title-attr carrying its provenance. */
function Cell({ field, display, className = '' }: {
  field?: Sourced<unknown>; display: string; className?: string;
}) {
  const pending = pendingOf(field as Sourced<unknown>);
  const title = pending
    ? `sin dato — pendiente de ${pending}`
    : field?.source?.path
      ? `fuente: ${field.source.path}${field.source.note ? ` · ${field.source.note}` : ''}`
      : undefined;
  return (
    <td className={`px-3 py-2 ${GMT.mono} text-right ${pending ? GM.textFaint : GM.textStrong} ${className}`} title={title}>
      {display}
    </td>
  );
}

const DIAGNOSTIC_BANNER = 'SUPERFICIE DIAGNÓSTICA — muestra, no aprueba ni ejecuta. '
  + 'El Voto 2/2 y el deploy viven exclusivamente en /dashboard.';

// ───────────────────────────────────────────────────────────────── LIBRO

function BookSection({ tower }: { tower: ControlTowerSnapshot }) {
  const b = tower.book;
  const counts = b.state_counts ?? ({} as Record<BookState, number | null>);
  return (
    <GmPanel
      title="LIBRO"
      meta="capital · PnL · vol · DD · exposición · CVaR · estados"
      className="mb-4"
    >
      <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-4">
        <Figure label="Capital" field={b.capital} display={fmt(b.capital, { digits: 0 })} />
        <Figure label="PnL día" field={b.pnl_d_pct} display={fmt(b.pnl_d_pct, { suffix: '%', signed: true })} tone={toneOf(b.pnl_d_pct.value)} />
        <Figure label="PnL mes" field={b.pnl_m_pct} display={fmt(b.pnl_m_pct, { suffix: '%', signed: true })} tone={toneOf(b.pnl_m_pct.value)} />
        <Figure label="PnL año" field={b.pnl_y_pct} display={fmt(b.pnl_y_pct, { suffix: '%', signed: true })} tone={toneOf(b.pnl_y_pct.value)} />
        <Figure label="Vol prevista" field={b.vol_forecast_pct} display={fmt(b.vol_forecast_pct, { suffix: '%' })} />
        <Figure label="Vol objetivo" field={b.vol_target_pct} display={fmt(b.vol_target_pct, { suffix: '%' })} />
        <Figure label="Max DD libro" field={b.max_dd_pct} display={fmt(b.max_dd_pct, { suffix: '%' })} tone="neg" />
        <Figure label="CVaR" field={b.cvar_pct} display={fmt(b.cvar_pct, { suffix: '%' })} tone="neg" />
        <Figure label="Gross" field={b.gross_exposure} display={fmt(b.gross_exposure)} />
        <Figure label="Net" field={b.net_exposure} display={fmt(b.net_exposure)} />
        <Figure label="Ratio diversificación" field={b.diversification_ratio} display={fmt(b.diversification_ratio)} />
        <Figure label="Atribución PnL" field={b.pnl_attribution} display="—" />
      </div>

      <div className={`${GM.panelInner} mt-3 p-3`} data-testid="tower-state-counts">
        <div className={`${GMT.label} ${GM.textMuted} mb-2`}>Conteo por estado</div>
        <div className="flex flex-wrap gap-2">
          {BOOK_STATES.map((state) => {
            const n = counts[state];
            return (
              <span key={state} className={`${GM.neutralBadge} rounded-[7px] px-2.5 py-1 text-[11px] font-semibold`}
                title={n == null ? 'sin productor que publique este estado (BL-27/BL-30)' : undefined}>
                {state}: <span className={GMT.mono}>{n == null ? '—' : n}</span>
              </span>
            );
          })}
        </div>
        <p className={`${GMT.micro} ${GM.textFaint} m-0 mt-2`}>
          {b.state_counts_source.note}
        </p>
      </div>
    </GmPanel>
  );
}

// ──────────────────────────────────────────────────────────────── SLEEVES

function SleevesSection({ tower, onSelect, selected }: {
  tower: ControlTowerSnapshot; onSelect: (id: string) => void; selected: string | null;
}) {
  const th = `px-3 py-2.5 text-left ${GMT.label} ${GM.textMuted}`;
  const RETIREMENT_TONE: Record<string, GmTone> = { green: 'pos', yellow: 'warn', red: 'neg', unknown: 'neutral' };
  return (
    <GmPanel
      title="SLEEVES"
      meta={`${tower.sleeves.length} estrategias no archivadas · DSR bar ${DSR_BAR}`}
      className="mb-4"
    >
      <div role="region" aria-label="Sleeves — tabla de gobierno y desempeño" tabIndex={0}
        className={`overflow-x-auto -mx-[18px] ${GM.focus}`}>
        <table className="w-full min-w-[1080px] text-[0.78125rem]">
          <caption className="sr-only">
            Una fila por sleeve: estado de investigación, entornos con dato publicado, Sharpe,
            DSR y los tres N, días al juez y semáforo de retiro. Solo lectura.
          </caption>
          <thead>
            <tr className="border-b border-[rgba(148,163,184,.1)]">
              <th scope="col" className={th}>Sleeve</th>
              <th scope="col" className={th}>Research</th>
              <th scope="col" className={th}>Entornos con dato</th>
              <th scope="col" className={`${th} text-right`}>Retorno</th>
              <th scope="col" className={`${th} text-right`}>Trades</th>
              <th scope="col" className={`${th} text-right`}>Sharpe</th>
              <th scope="col" className={`${th} text-right`}>DSR</th>
              <th scope="col" className={`${th} text-right`}>N fam/clu/glob</th>
              <th scope="col" className={`${th} text-right`}>timing_ratio</th>
              <th scope="col" className={`${th} text-right`}>Días al juez</th>
              <th scope="col" className={th}>Semáforo retiro</th>
            </tr>
          </thead>
          <tbody>
            {tower.sleeves.map((s: TowerSleeve) => (
              <tr key={s.strategy_id}
                className={`border-t border-[rgba(148,163,184,.07)] ${GM.rowHover} ${selected === s.strategy_id ? 'bg-[rgba(34,211,238,.06)]' : ''}`}>
                <th scope="row" className="px-3 py-2 text-left font-normal">
                  <button onClick={() => onSelect(s.strategy_id)}
                    data-testid={`sleeve-${s.strategy_id}`}
                    className={`${GM.focus} text-left ${GMT.mono} font-bold ${GM.accent} hover:underline`}>
                    {s.strategy_id}
                  </button>
                  <span className={`block ${GMT.micro} ${GM.textMuted}`}>{s.asset_id} · {s.display_name}</span>
                </th>
                <td className={`px-3 py-2 ${GMT.micro} ${GM.textSec}`}>{s.research_state ?? '—'}</td>
                <td className="px-3 py-2">
                  <div className="flex flex-wrap gap-1">
                    {s.envs_with_data.length === 0
                      ? <span className={`${GMT.micro} ${GM.textFaint}`}>—</span>
                      : s.envs_with_data.map((e) => (
                        <span key={e} className={`${GM.neutralBadge} rounded px-1.5 py-0.5 text-[10px]`}>{e}</span>
                      ))}
                  </div>
                </td>
                <Cell field={s.return_pct} display={fmt(s.return_pct, { suffix: '%', signed: true })} />
                <Cell field={s.n_trades} display={fmtInt(s.n_trades)} />
                <Cell field={s.sharpe} display={fmt(s.sharpe)} />
                <Cell field={s.dsr_family} display={fmt(s.dsr_family, { digits: 4 })} />
                <td className={`px-3 py-2 ${GMT.mono} ${GM.textSec} text-right whitespace-nowrap`}>
                  {fmtInt(s.n_family)}/{fmtInt(s.n_cluster)}/{fmtInt(s.n_global)}
                </td>
                <Cell field={s.timing_ratio} display={fmt(s.timing_ratio, { digits: 4 })} />
                <Cell field={s.judge_days} display={fmtInt(s.judge_days)} />
                <td className="px-3 py-2">
                  <GmBadge tone={RETIREMENT_TONE[s.retirement_signal] ?? 'neutral'}>
                    {s.retirement_signal.toUpperCase()}
                  </GmBadge>
                  {s.retirement_reason && (
                    <span className={`block ${GMT.micro} ${GM.textFaint} max-w-[220px]`}>{s.retirement_reason}</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {tower.paired_tests.length > 0 && (
        <div className={`${GM.panelInner} mt-3 p-3`} data-testid="tower-paired-tests">
          <div className={`${GMT.label} ${GM.textMuted} mb-2`}>
            Test pareado — campeona viva vs candidatas paper
          </div>
          <ul className="m-0 p-0 list-none flex flex-col gap-1.5">
            {tower.paired_tests.map((t) => (
              <li key={t.candidate_id} className={`${GMT.micro} ${GM.textSec} flex flex-wrap items-baseline gap-2`}>
                <span className={`${GMT.mono} ${GM.textStrong}`}>{t.baseline_id}</span>
                <span>{fmt(t.baseline_return_pct, { suffix: '%', signed: true })}</span>
                <span className={GM.textFaint}>vs</span>
                <span className={`${GMT.mono} ${GM.textStrong}`}>{t.candidate_id}</span>
                <span>{fmt(t.candidate_return_pct, { suffix: '%', signed: true })}</span>
                <span className={GM.textFaint}>
                  · N pareado {fmtInt(t.n_paired)} · p {fmt(t.p_value)} · e {fmt(t.e_value)}
                </span>
              </li>
            ))}
          </ul>
          <p className={`${GMT.micro} ${GM.textFaint} m-0 mt-2`}>
            {tower.paired_tests[0]?.note} · el p/e-value del test pareado no se calcula aquí:
            se publica cuando exista el motor único de métricas (BL-18).
          </p>
        </div>
      )}
    </GmPanel>
  );
}

// ───────────────────────────────────────────────────────────────── DATOS

function DataSection({ tower }: { tower: ControlTowerSnapshot }) {
  const d = tower.data;
  const CLOCK_TONE: Record<string, GmTone> = { green: 'pos', yellow: 'warn', red: 'neg' };
  const families = d.trials_by_family.value ?? [];
  return (
    <GmPanel title="DATOS" meta="frescura · relojes · trials por familia · N_global vs N_MAX" className="mb-4">
      <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-4">
        {Object.entries(d.clocks).map(([clock, field]) => {
          const v = field.value;
          const pending = pendingOf(field);
          return (
            <div key={clock} className={`${GM.panelInner} p-3`}>
              <div className={`${GMT.label} ${GM.textMuted} mb-1`}>reloj {clock}</div>
              <div className={`text-[15px] font-bold ${v ? GM_TONE_TEXT[CLOCK_TONE[v.signal] ?? 'neutral'] : GM.textFaint}`}>
                {v ? v.signal.toUpperCase() : '—'}
              </div>
              <div className={`${GMT.micro} ${GM.textFaint} mt-1`}>
                {pending ? `pendiente: ${pending}` : (v?.actions.join(', ') || 'sin acciones')}
              </div>
            </div>
          );
        })}
        <Figure label="Sondas no-OK" field={d.stale_probes} display={fmtInt(d.stale_probes)} />
        <Figure label="Paridad replay" field={d.replay_parity} display={fmtStr(d.replay_parity)} />
        <Figure label="Última revisión vintage" field={d.last_vintage_revision} display={fmtStr(d.last_vintage_revision)} />
        <div className={`${GM.panelInner} p-3`}>
          <div className={`${GMT.label} ${GM.textMuted} mb-1`}>N_global vs N_MAX</div>
          <div className={`${GMT.kpi} ${GM.textStrong}`}>
            {fmtInt(d.n_global)} <span className={`${GMT.meta} ${GM.textFaint}`}>/ {N_MAX_TRIALS}</span>
          </div>
          <div className={`${GMT.micro} ${GM.textFaint} mt-1`}>
            cota de GASTO (§9.7) — jamás entra en el DSR
          </div>
        </div>
      </div>

      <div className="grid gap-2.5 sm:grid-cols-2 mt-3">
        <div className={`${GM.panelInner} p-3`}>
          <div className={`${GMT.label} ${GM.textMuted} mb-1`}>Promociones congeladas</div>
          <div className={`text-[15px] font-bold ${d.promotions_frozen.value ? GM.warn : GM.textSec}`}>
            {d.promotions_frozen.value == null ? '—' : d.promotions_frozen.value ? 'SÍ' : 'NO'}
          </div>
        </div>
        <div className={`${GM.panelInner} p-3`}>
          <div className={`${GMT.label} ${GM.textMuted} mb-1`}>Retiro disparado (global)</div>
          <div className={`text-[15px] font-bold ${d.withdrawal_triggered.value ? GM.neg : GM.textSec}`}>
            {d.withdrawal_triggered.value == null ? '—' : d.withdrawal_triggered.value ? 'SÍ' : 'NO'}
          </div>
          <div className={`${GMT.micro} ${GM.textFaint} mt-1`}>{d.withdrawal_triggered.source.note}</div>
        </div>
      </div>

      <div className={`${GM.panelInner} p-3 mt-3`} data-testid="tower-families">
        <div className={`${GMT.label} ${GM.textMuted} mb-2`}>Trials por familia</div>
        {families.length === 0 ? (
          <p className={`${GMT.micro} ${GM.textFaint} m-0`}>
            {pendingOf(d.trials_by_family) ?? 'sin familias publicadas'}
          </p>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {families.map((f) => (
              <span key={f.family_id} className={`${GM.neutralBadge} rounded-[7px] px-2 py-1 text-[11px]`}
                title={`${f.kind} · cluster ${f.cluster_id ?? '—'}${f.closed ? ' · CERRADA' : ''}`}>
                {f.family_id} <span className={GMT.mono}>{f.n_trials}</span>
                {f.closed && <span className={GM.textFaint}> · cerrada</span>}
              </span>
            ))}
          </div>
        )}
      </div>
    </GmPanel>
  );
}

// ───────────────────────────────────────────────────────── passport (5 envs)

function PassportDetail({ strategyId }: { strategyId: string }) {
  const state = useGmQuery<StrategyPassport>(`/api/passport/${strategyId}`);
  return (
    <AsyncBoundary
      state={state}
      emptyProps={{ title: 'Sin passport', body: 'Esta estrategia no tiene registry ni manifiesto publicado.' }}
    >
      {(p) => (
        <>
          <GmPanel
            title={`Passport · ${p.identity.display_name}`}
            meta={`${p.identity.asset_id} · ${p.identity.engine_type ?? 'motor n/d'} · superficie ${p.identity.surface ?? 'n/d'}`}
            className="mb-4"
          >
            <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-4 mb-3">
              <Figure label="Versión activa" field={p.identity.active_version} display={fmtStr(p.identity.active_version)} />
              <Figure label="Estado aprobación" field={p.governance.approval_status} display={fmtStr(p.governance.approval_status)} />
              <Figure label={`DSR (bar ${p.governance.dsr_bar})`} field={p.governance.dsr_family} display={fmt(p.governance.dsr_family, { digits: 4 })} />
              <Figure label="Trials totales (FT+AT)" field={p.governance.n_trials_total} display={fmtInt(p.governance.n_trials_total)} />
              <Figure label="FT (predictivos)" field={p.governance.n_trials_forecast} display={fmtInt(p.governance.n_trials_forecast)} />
              <Figure label="AT (económicos)" field={p.governance.n_trials_action} display={fmtInt(p.governance.n_trials_action)} />
              <Figure label="N familia / cluster" field={p.governance.n_family}
                display={`${fmtInt(p.governance.n_family)} / ${fmtInt(p.governance.n_cluster)}`} />
              <Figure label="N global" field={p.governance.n_global} display={fmtInt(p.governance.n_global)} />
            </div>

            <div className={`${GM.panelInner} p-3 mb-3`}>
              <div className={`${GMT.label} ${GM.textMuted} mb-1`}>Protocolo de retiro</div>
              <div className={`${GMT.mono} text-[12px] ${p.governance.withdrawal_protocol.value ? GM.textStrong : GM.textFaint}`}>
                {fmtStr(p.governance.withdrawal_protocol)}
              </div>
              <div className={`${GMT.micro} ${GM.textFaint} mt-1`}>
                semáforo: {p.governance.retirement_signal.toUpperCase()} — {p.governance.retirement_reason}
              </div>
            </div>

            {/* Gates: the numbers Vote 2 decides on — shown READ-ONLY here. */}
            {p.governance.gates.value && (
              <div className={`${GM.panelInner} p-3`}>
                <div className={`${GMT.label} ${GM.textMuted} mb-2`}>Gates del bundle publicado (lectura)</div>
                <div className="flex flex-wrap gap-1.5">
                  {p.governance.gates.value.map((g) => (
                    <span key={g.gate} className={`rounded-[7px] px-2 py-1 text-[11px] border
                      ${g.passed ? 'border-[rgba(52,211,153,.3)] text-[var(--gm-pos)]' : 'border-[rgba(251,113,133,.24)] text-[var(--gm-neg)]'}`}>
                      {g.label}: <span className={GMT.mono}>{g.value ?? '—'}</span>
                    </span>
                  ))}
                </div>
                <p className={`${GMT.micro} ${GM.textFaint} m-0 mt-2`}>
                  El Voto 2/2 se emite sobre estos mismos números en <strong>/dashboard</strong>; aquí solo se leen.
                </p>
              </div>
            )}
          </GmPanel>

          <GmPanel
            title="Desempeño por entorno"
            meta="backtest · held-out · paper · canary · live"
            className="mb-4"
          >
            <div className={`${GM.panelSoft} p-2.5 mb-3 flex items-start gap-2`}>
              <Info className={`w-4 h-4 shrink-0 mt-0.5 ${GM.warn}`} aria-hidden />
              <p className={`m-0 ${GMT.micro} ${GM.textSec}`}>
                {p.metric_engine.source.note} (pendiente: {p.metric_engine.source.pending}).
              </p>
            </div>
            <div role="region" aria-label="Desempeño por entorno" tabIndex={0}
              className={`overflow-x-auto -mx-[18px] ${GM.focus}`}>
              <table className="w-full min-w-[820px] text-[0.78125rem]">
                <caption className="sr-only">
                  Cinco entornos; cada celda declara el artefacto del que sale. Con N&lt;20 trades
                  solo se publican conteo y PnL.
                </caption>
                <thead>
                  <tr className="border-b border-[rgba(148,163,184,.1)]">
                    <th scope="col" className={`px-3 py-2.5 text-left ${GMT.label} ${GM.textMuted}`}>Entorno</th>
                    {['Periodo', 'Retorno', 'Trades', 'Max DD', 'Win rate', 'Sharpe', 'Calmar', 'p-value'].map((h) => (
                      <th key={h} scope="col" className={`px-3 py-2.5 text-right ${GMT.label} ${GM.textMuted}`}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {PASSPORT_ENVS.map((env: PassportEnv) => {
                    const e: EnvPerformance = p.performance[env];
                    return (
                      <tr key={env} className={`border-t border-[rgba(148,163,184,.07)] ${GM.rowHover}`}>
                        <th scope="row" className="px-3 py-2 text-left font-normal">
                          <span className={`${GMT.mono} font-bold ${GM.textStrong}`}>{env}</span>
                          {e.insufficient_trades && (
                            <span className={`block ${GMT.micro} ${GM.warn}`}>N&lt;20 · solo conteo y PnL</span>
                          )}
                        </th>
                        <Cell field={e.period_label} display={fmtStr(e.period_label)} className="text-left" />
                        <Cell field={e.return_pct} display={fmt(e.return_pct, { suffix: '%', signed: true })} />
                        <Cell field={e.n_trades} display={fmtInt(e.n_trades)} />
                        <Cell field={e.max_dd_pct} display={fmt(e.max_dd_pct, { suffix: '%' })} />
                        <Cell field={e.win_rate_pct} display={fmt(e.win_rate_pct, { suffix: '%' })} />
                        <Cell field={e.sharpe} display={fmt(e.sharpe)} />
                        <Cell field={e.calmar} display={fmt(e.calmar)} />
                        <Cell field={e.p_value} display={fmt(e.p_value, { digits: 4 })} />
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </GmPanel>

          <div className="grid gap-4 lg:grid-cols-2 mb-4">
            <GmPanel title="Estado live (no materializado)" meta="§24.4 v_strategy_passport_live">
              <div className="grid gap-2.5 sm:grid-cols-2">
                <Figure label="Órdenes abiertas" field={p.live.open_orders} display={fmtInt(p.live.open_orders)} />
                <Figure label="Último fill" field={p.live.last_fill_at} display={fmtStr(p.live.last_fill_at)} />
                <Figure label="Deploy" field={p.live.deploy_status} display={fmtStr(p.live.deploy_status)} />
                <Figure label="Última señal" field={p.live.last_signal_at} display={fmtStr(p.live.last_signal_at)} />
              </div>
            </GmPanel>
            <GmPanel title="Riesgo" meta="exposición · vol objetivo · multiplicadores">
              <div className="grid gap-2.5 sm:grid-cols-2">
                <Figure label="Exposición actual" field={p.risk.current_exposure} display={fmt(p.risk.current_exposure)} />
                <Figure label="Vol objetivo" field={p.risk.vol_target_pct} display={fmt(p.risk.vol_target_pct, { suffix: '%' })} />
                <Figure label="m_forward" field={p.risk.m_forward} display={fmt(p.risk.m_forward)} />
                <Figure label="m_dd" field={p.risk.m_dd} display={fmt(p.risk.m_dd)} />
              </div>
            </GmPanel>
          </div>

          <GmPanel title="Linaje" meta="§24.4 snapshots + fingerprints">
            <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-4 mb-3">
              <Figure label="spec_fingerprint" field={p.lineage.spec_fingerprint} display={fmtStr(p.lineage.spec_fingerprint)} />
              <Figure label="feature_set_hash" field={p.lineage.feature_set_hash} display={fmtStr(p.lineage.feature_set_hash)} />
              <Figure label="policy_hash" field={p.lineage.policy_hash} display={fmtStr(p.lineage.policy_hash)} />
              <Figure label="Grafo de linaje" field={p.lineage.lineage_graph} display="—" />
            </div>
            {p.lineage.model_versions.value && (
              <div role="region" aria-label="Snapshots del modelo" tabIndex={0} className={`overflow-x-auto ${GM.focus}`}>
                <table className="w-full min-w-[560px] text-[0.78125rem]">
                  <caption className="sr-only">Snapshots del modelo por versión.</caption>
                  <thead>
                    <tr className="border-b border-[rgba(148,163,184,.1)]">
                      {['Versión', 'Activa', 'Entrenada', 'feature_hash'].map((h) => (
                        <th key={h} scope="col" className={`px-3 py-2 text-left ${GMT.label} ${GM.textMuted}`}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {p.lineage.model_versions.value.map((mv) => (
                      <tr key={mv.version} className="border-t border-[rgba(148,163,184,.07)]">
                        <td className={`px-3 py-2 ${GMT.mono} ${GM.textStrong}`}>{mv.version}</td>
                        <td className="px-3 py-2">{mv.active ? <GmBadge tone="accent">activa</GmBadge> : ''}</td>
                        <td className={`px-3 py-2 ${GMT.mono} ${GM.textSec}`}>{mv.trained_at ?? '—'}</td>
                        <td className={`px-3 py-2 ${GMT.mono} ${GM.textFaint}`}>{mv.feature_hash ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </GmPanel>
        </>
      )}
    </AsyncBoundary>
  );
}

// ─────────────────────────────────────────────────── pendientes (acople)

function PendingSection({ items }: { items: PendingInterface[] }) {
  return (
    <GmPanel
      title="Interfaces pendientes"
      meta="lo que la torre declara y todavía no puede leer"
      className="mb-4"
    >
      <p className={`${GMT.micro} ${GM.textMuted} m-0 mb-3`}>
        Cada fila es un punto de acople documentado: cuando su productor exista, solo cambia
        el compositor (<span className={GMT.mono}>lib/passport/compose.ts</span>) — ni el
        contrato ni esta vista.
      </p>
      <ul className="m-0 p-0 list-none flex flex-col gap-2">
        {items.map((i) => (
          <li key={`${i.backlog_id}-${i.field}`} className={`${GM.panelInner} p-3`}>
            <div className="flex items-center gap-2 flex-wrap">
              <GmBadge tone="info">{i.backlog_id}</GmBadge>
              <span className={`${GMT.mono} text-[11.5px] ${GM.textStrong}`}>{i.field}</span>
              <span className={`${GMT.micro} ${GM.textMuted}`}>← {i.produced_by}</span>
            </div>
            <p className={`m-0 mt-1.5 ${GMT.micro} ${GM.textSec}`}>{i.note}</p>
          </li>
        ))}
      </ul>
    </GmPanel>
  );
}

// ──────────────────────────────────────────────────────────────────── view

interface TowerResponse {
  tower: ControlTowerSnapshot;
  strategies: Array<{ strategy_id: string; asset_id: string; display_name: string; status: string | null }>;
}

export function PassportView() {
  const state = useGmQuery<TowerResponse>('/api/passport/tower');
  const [selected, setSelected] = useState<string | null>(null);

  return (
    <div className="motion-safe:animate-in motion-safe:fade-in">
      <GmPageHeader
        kicker="Control Tower · Strategy Passport"
        title="Torre de control"
        subtitle="Libro, sleeves y datos en una sola lectura. Cada cifra declara el artefacto del que sale."
      />

      {/* The invariant, stated on the surface itself — not buried in a spec. */}
      <div className={`${GM.panelSoft} p-3 mb-4 flex items-start gap-2.5`} data-testid="passport-diagnostic-banner">
        <ShieldQuestion className={`w-[18px] h-[18px] shrink-0 mt-0.5 ${GM.warn}`} aria-hidden />
        <p className={`m-0 ${GMT.meta} ${GM.textSec}`}>{DIAGNOSTIC_BANNER}</p>
      </div>

      <AsyncBoundary
        state={state}
        emptyProps={{
          title: 'Sin artefactos publicados',
          body: 'La torre lee proyecciones publicadas. Corre el pipeline de producción y '
            + 'scripts/pipeline/export_control_tower.py.',
        }}
      >
        {(r) => (
          <>
            <BookSection tower={r.tower} />
            <SleevesSection tower={r.tower} onSelect={setSelected} selected={selected} />
            <DataSection tower={r.tower} />

            <GmPanel title="Passport por estrategia" meta="§24.4 · un SELECT, cinco entornos" className="mb-4">
              <label htmlFor="passport-strategy" className={`${GMT.label} ${GM.textMuted} block mb-1.5`}>
                <Layers className="w-3.5 h-3.5 inline mr-1" aria-hidden /> Estrategia
              </label>
              <select
                id="passport-strategy"
                data-testid="passport-strategy-select"
                value={selected ?? ''}
                onChange={(e) => setSelected(e.target.value || null)}
                className={`${GM.input} ${GM.focus} w-full max-w-[420px]`}
              >
                <option value="">— elige una estrategia —</option>
                {r.strategies.map((s) => (
                  <option key={s.strategy_id} value={s.strategy_id}>
                    {s.strategy_id} · {s.asset_id} ({s.status ?? 'n/d'})
                  </option>
                ))}
              </select>
              {!selected && (
                <p className={`${GMT.micro} ${GM.textFaint} m-0 mt-2`}>
                  <FileClock className="w-3 h-3 inline mr-1" aria-hidden />
                  Elige una sleeve arriba o en el desplegable para abrir su passport completo.
                </p>
              )}
            </GmPanel>

            {selected && <PassportDetail key={selected} strategyId={selected} />}

            <PendingSection items={r.tower.pending_interfaces} />

            <p className={`${GMT.micro} ${GM.textFaint} flex items-center gap-1.5`}>
              <Database className="w-3 h-3" aria-hidden />
              Proyecciones regenerables (FABRIC §24.6): estos archivos no son fuente de verdad.
              Generado {r.tower.generated_at}.
            </p>
          </>
        )}
      </AsyncBoundary>
    </div>
  );
}

export default PassportView;

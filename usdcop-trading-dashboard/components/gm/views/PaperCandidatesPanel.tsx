'use client';

/**
 * BL-05 — Panel READ-ONLY del paper ledger A/B (v11 producción vs v12/v14 paper).
 * Shape de public/data/production/paper/candidates_ledger_2026.json (read-only).
 *
 * Solo números del JSON publicado; cero botones/acciones (mismo patrón display-only
 * que ApprovalPanel). Con N<20 el ledger manda note_n/note y SIEMPRE se muestran
 * (quant-constitution §6: solo conteo y PnL).
 */

// Import directo de primitives (no el barrel): el barrel arrastra TerminalShell
// → next/font, que no carga fuera del runtime de Next (vitest/jsdom).
import { GmBadge, GmPanel } from '@/components/gm/primitives';
import { GM, GMT } from '@/lib/ui/gm-tokens';

// ─────────────────────────────────────────────────────────── tipos

export interface PaperJudgeWindow {
  starts_after: string;
  n_trades: number | null;
  pnl_pct_compound: number | null;
  note?: string | null;
}

export interface PaperCandidate {
  ret_2026_ytd_pct: number | null;
  n_trades: number | null;
  note_n?: string | null;
  judge_window: PaperJudgeWindow | null;
}

export interface PaperCandidatesLedger {
  contract: string;
  anchor: string;
  labels: Record<string, string>;
  judge_note: string;
  generated_at: string;
  strategies: Record<string, PaperCandidate>;
}

// ─────────────────────────────────────────────────────────── helpers

/** null/NaN-safe % con signo (nunca renderiza Infinity/NaN — strategy-contract §2). */
export function fmtSignedPct(n: number | null | undefined, digits = 2): string {
  if (n == null || !Number.isFinite(n)) return '—';
  return `${n >= 0 ? '+' : ''}${n.toFixed(digits)}%`;
}

/**
 * Días transcurridos desde una fecha del bundle (starts_after / anchor) hasta el
 * reloj del cliente, a granularidad de día calendario. Acepta el prefijo ISO
 * "YYYY-MM-DD" aunque el campo traiga sufijo textual (p.ej. el anchor del ledger:
 * "2026-01-01 (directiva operador 2026-07-22)").
 *
 * NO es una métrica: es aritmética de fechas sobre datos ya publicados
 * (constitución §7 — nada se recomputa en el frontend). Negativo ⇒ el juez aún
 * no arranca; null ⇒ fecha ausente/no parseable (la UI muestra '—', nunca NaN).
 */
export function judgeElapsedDays(dateIso: string | null | undefined, now: Date): number | null {
  if (!dateIso) return null;
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(dateIso).trim());
  if (!m) return null;
  const startUtc = Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
  // Fecha calendario local del cliente (día, no horas) comparada en el mismo eje UTC.
  const todayUtc = Date.UTC(now.getFullYear(), now.getMonth(), now.getDate());
  return Math.round((todayUtc - startUtc) / 86_400_000);
}

/** Render de "días al juez": 'N d' corriendo, 'faltan N d' si aún no arranca, '—' sin dato. */
export function fmtJudgeDays(days: number | null): string {
  if (days == null) return '—';
  return days < 0 ? `faltan ${-days} d` : `${days} d`;
}

// ─────────────────────────────────────────────────────────── panel

export function PaperCandidatesPanel({ ledger }: { ledger: PaperCandidatesLedger }) {
  const rows = Object.entries(ledger.strategies);
  // Etiquetas sin fila de datos (p.ej. v13 EXCLUIDA hasta freeze) — se listan como nota.
  const labelOnly = Object.entries(ledger.labels).filter(([key]) => !(key in ledger.strategies));
  const th = `px-3 py-2.5 text-left ${GMT.label} ${GM.textMuted}`;
  // Reloj del cliente contra fechas ya publicadas en el bundle (nunca métricas
  // recomputadas). El panel solo se monta client-side (post-fetch) → sin riesgo
  // de hydration mismatch.
  const now = new Date();
  return (
    <GmPanel
      title="Candidatas A/B (paper, ancla ene-2026)"
      meta={`Ancla ${ledger.anchor} · generado ${ledger.generated_at}`}
      className="mb-4"
    >
      <p className={`${GMT.micro} ${GM.textMuted} m-0 mb-3`}>{ledger.judge_note}</p>
      {/* Región de scroll horizontal focusable (axe scrollable-region-focusable):
          en móvil la tabla scrollea DENTRO de este contenedor, jamás desborda la página. */}
      <div
        role="region"
        aria-label="Candidatas A/B — tabla comparativa"
        tabIndex={0}
        className={`overflow-x-auto -mx-[18px] ${GM.focus}`}
      >
        <table className="w-full min-w-[760px] text-[12.5px]">
          <caption className="sr-only">
            Candidatas A/B del paper ledger: v11 en producción (forward real 2026) frente a
            candidatas paper con juez sellado post-freeze. Solo lectura del JSON publicado.
          </caption>
          <thead>
            <tr className="border-b border-[rgba(148,163,184,.1)]">
              <th scope="col" className={th}>Estrategia</th>
              <th scope="col" className={`${th} text-right`}>Ret. 2026 YTD</th>
              <th scope="col" className={`${th} text-right`}>Trades</th>
              <th scope="col" className={th}>Juez desde</th>
              <th scope="col" className={`${th} text-right`}>Días al juez</th>
              <th scope="col" className={`${th} text-right`}>Trades juez</th>
              <th scope="col" className={`${th} text-right`}>PnL juez (comp.)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([sid, c]) => {
              const isProd = c.judge_window == null;
              // Días al juez: paper ⇒ desde judge_window.starts_after (juez sellado
              // post-freeze); producción (v11) ⇒ desde el anchor del ledger (su juez
              // ES el forward 2026 completo). Ambas fechas vienen del bundle.
              const judgeDays = judgeElapsedDays(
                c.judge_window ? c.judge_window.starts_after : ledger.anchor,
                now,
              );
              return (
                <tr key={sid} className={`border-t border-[rgba(148,163,184,.07)] ${GM.rowHover}`}>
                  <td className="px-3 py-2.5">
                    <div className="flex items-center gap-2">
                      <span className={`${GMT.mono} font-bold ${GM.textStrong}`}>{sid}</span>
                      <GmBadge tone={isProd ? 'accent' : 'neutral'}>
                        {isProd ? 'PRODUCCIÓN' : 'PAPER · JUEZ SELLADO'}
                      </GmBadge>
                    </div>
                    {ledger.labels[sid] && (
                      <span className={`block ${GMT.micro} ${GM.textMuted} mt-0.5`}>{ledger.labels[sid]}</span>
                    )}
                  </td>
                  <td className={`px-3 py-2.5 ${GMT.mono} font-bold text-right ${
                    c.ret_2026_ytd_pct == null ? GM.textMuted : c.ret_2026_ytd_pct >= 0 ? GM.pos : GM.neg
                  }`}>
                    {fmtSignedPct(c.ret_2026_ytd_pct)}
                  </td>
                  <td className={`px-3 py-2.5 text-right`}>
                    <span className={`${GMT.mono} ${GM.textStrong}`}>{c.n_trades ?? '—'}</span>
                    {c.note_n && (
                      <span className={`block ${GMT.micro} ${GM.textMuted}`}>{c.note_n}</span>
                    )}
                  </td>
                  <td className={`px-3 py-2.5 ${GMT.mono} ${GM.textSec} whitespace-nowrap`}>
                    {c.judge_window ? c.judge_window.starts_after : '—'}
                  </td>
                  <td className={`px-3 py-2.5 text-right whitespace-nowrap`}>
                    <span className={`${GMT.mono} ${judgeDays == null ? GM.textMuted : GM.textStrong}`}>
                      {fmtJudgeDays(judgeDays)}
                    </span>
                    <span className={`block ${GMT.micro} ${GM.textMuted}`}>
                      {isProd ? 'forward 2026 (ancla)' : 'juez sellado'}
                    </span>
                  </td>
                  <td className={`px-3 py-2.5 ${GMT.mono} ${GM.textStrong} text-right`}>
                    {c.judge_window ? c.judge_window.n_trades ?? '—' : '—'}
                  </td>
                  <td className="px-3 py-2.5 text-right">
                    <span className={`${GMT.mono} ${
                      c.judge_window?.pnl_pct_compound == null ? GM.textMuted
                        : c.judge_window.pnl_pct_compound >= 0 ? GM.pos : GM.neg
                    }`}>
                      {c.judge_window ? fmtSignedPct(c.judge_window.pnl_pct_compound) : '—'}
                    </span>
                    {c.judge_window?.note && (
                      <span className={`block ${GMT.micro} ${GM.textMuted}`}>{c.judge_window.note}</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {labelOnly.length > 0 && (
        <p className={`${GMT.micro} ${GM.textMuted} m-0 mt-3 pt-3 border-t border-dashed border-[rgba(148,163,184,.16)]`}>
          {labelOnly.map(([key, label]) => `${key}: ${label}`).join(' · ')}
        </p>
      )}
    </GmPanel>
  );
}

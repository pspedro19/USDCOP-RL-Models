'use client';

/**
 * Admin base components (CTR-ADMIN-UI-001 §1.4). All color/typography via
 * `lib/ui/tokens.ts` — zero hardcoded palette classes here beyond structure.
 */
import { useEffect, useState, type ReactNode } from 'react';
import { Info } from 'lucide-react';

import { COLOR, CTA, HOVER_ACCENT_BORDER, SURFACE, TYPE, type SemanticTone } from '@/lib/ui/tokens';

// ─────────────────────────────────────────────── StatusDot (pulse only on error)

export function StatusDot({ tone, label }: { tone: SemanticTone; label?: string }) {
  return (
    <span
      role="status"
      aria-label={label ?? tone}
      className={`inline-block w-2.5 h-2.5 rounded-full shrink-0 ${COLOR[tone].dot} ${tone === 'error' ? 'motion-safe:animate-pulse' : ''}`}
    />
  );
}

// ─────────────────────────────────────────────── Badge

export function Badge({ tone = 'neutral', className = '', children, 'aria-label': ariaLabel }: {
  tone?: SemanticTone; className?: string; children: ReactNode; 'aria-label'?: string;
}) {
  return (
    <span aria-label={ariaLabel} className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold ${COLOR[tone].badge} ${className}`}>
      {children}
    </span>
  );
}

export function TestBadge() {
  return <Badge tone="warn" className="uppercase" aria-label="cuenta de test">test</Badge>;
}

// ─────────────────────────────────────────────── Card (header with live meta)

export function Card({ title, icon, meta, info, badge, children, stale = false, testId }: {
  title: string;
  icon?: ReactNode;
  /** Right-side meta, e.g. "hace 12 s". */
  meta?: ReactNode;
  /** Explanatory prose lives in an ⓘ tooltip, never in the body (§1.4). */
  info?: string;
  badge?: ReactNode;
  children: ReactNode;
  /** stale-while-error: dim the card, keep last data (§3.4). */
  stale?: boolean;
  testId?: string;
}) {
  return (
    <section className={`${SURFACE.card} p-4 ${stale ? SURFACE.cardStale : ''}`} data-testid={testId}>
      <header className="flex items-center gap-2 mb-3">
        {icon}
        <h2 className={TYPE.sectionTitle}>{title}</h2>
        {badge}
        {info && (
          <span title={info} aria-label={info} className="cursor-help">
            <Info className={`w-3.5 h-3.5 ${COLOR.textSecondary}`} aria-hidden />
          </span>
        )}
        <span className={`ml-auto ${TYPE.meta}`}>{meta}</span>
      </header>
      {children}
    </section>
  );
}

// ─────────────────────────────────────────────── KpiTile (§1.4)

export function KpiTile({ label, value, delta, deltaTone = 'neutral', note, onClick, dimmed = false, testId }: {
  label: string;
  /** "—" = not applicable yet (pair with a phase note); "0" = measured zero. */
  value: string;
  delta?: string;
  deltaTone?: SemanticTone;
  note?: ReactNode;
  onClick?: () => void;
  /** Phase-gated tiles render dimmed (§2.2). */
  dimmed?: boolean;
  testId?: string;
}) {
  const Tag = onClick ? 'button' : 'div';
  return (
    <Tag
      onClick={onClick}
      data-testid={testId}
      className={`text-left rounded-xl border border-[rgba(148,163,184,.12)] bg-slate-900/50 px-4 py-3 min-w-0
        ${dimmed ? 'opacity-50' : ''}
        ${onClick ? `cursor-pointer ${HOVER_ACCENT_BORDER} ${CTA.focusRing}` : ''}`}
    >
      <div className={TYPE.sectionTitle}>{label}</div>
      <div className={`${TYPE.kpiValue} ${COLOR.textPrimary} mt-1`}>{value}</div>
      <div className="flex items-baseline gap-2 mt-0.5 min-h-4">
        {delta && <span className={`text-xs font-medium tabular-nums ${COLOR[deltaTone].text}`}>{delta}</span>}
        {note && <span className={TYPE.meta}>{note}</span>}
      </div>
    </Tag>
  );
}

// ─────────────────────────────────────────────── EmptyState (§1.4 — never bare text)

export function EmptyState({ icon, cause, action }: {
  icon: ReactNode;
  cause: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center gap-2 py-8 text-center">
      <div className={COLOR.textSecondary}>{icon}</div>
      <p className={`${TYPE.body} ${COLOR.textSecondary} max-w-sm`}>{cause}</p>
      {action}
    </div>
  );
}

// ─────────────────────────────────────────────── Skeleton (shape of the content)

export function SkeletonRows({ rows = 3, cols = 4 }: { rows?: number; cols?: number }) {
  return (
    <div className="space-y-2 py-1" aria-hidden>
      {Array.from({ length: rows }).map((_, r) => (
        <div key={r} className="flex gap-3">
          {Array.from({ length: cols }).map((__, c) => (
            <div key={c} className="h-5 flex-1 rounded bg-slate-800/60 motion-safe:animate-pulse" />
          ))}
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────── ProgressBar (freshness → threshold)

export function ProgressBar({ ratio, tone }: { ratio: number; tone: SemanticTone }) {
  const pct = Math.min(Math.max(ratio, 0), 1) * 100;
  return (
    <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden" role="progressbar"
         aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
      <div className={`h-full rounded-full ${COLOR[tone].dot}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

// ─────────────────────────────────────────────── live relative time

export function useNow(intervalMs = 30_000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(t);
  }, [intervalMs]);
  return now;
}

/** Relative primary, absolute on hover (§3.8) — pair with title={iso}. */
export function fmtRelative(iso: string | null | undefined, now: number): string {
  if (!iso) return '—';
  const s = Math.max(0, Math.round((now - new Date(iso).getTime()) / 1000));
  if (s < 60) return `hace ${s} s`;
  if (s < 3600) return `hace ${Math.floor(s / 60)} min`;
  if (s < 172800) return `hace ${Math.round(s / 3600)} h`;
  return `hace ${Math.round(s / 86400)} d`;
}

export function fmtHours(hours: number | null): string {
  if (hours == null) return '—';
  if (hours < 1) return `${Math.round(hours * 60)} min`;
  if (hours < 48) return `${Math.round(hours * 10) / 10} h`;
  return `${Math.round(hours / 24)} d`;
}

export function fmtDateTime(iso: string | null | undefined): string {
  return iso ? iso.replace('T', ' ').slice(0, 16) : '—';
}

export function fmtDate(iso: string | null | undefined): string {
  return iso ? iso.slice(0, 10) : '—';
}

// Locale-aware formatters (checklist 10.1/10.2): null → "—" (missing), never a fake 0.
const _cop = new Intl.NumberFormat('es-CO', {
  style: 'currency', currency: 'COP', maximumFractionDigits: 0,
});
const _int = new Intl.NumberFormat('es-CO', { maximumFractionDigits: 0 });

/** COP currency; null ⇒ "—" (not measured), 0 ⇒ "$0" (a real zero). */
export function fmtCop(v: number | null | undefined): string {
  return v == null ? '—' : _cop.format(v);
}

/** Plain integer with locale grouping; null ⇒ "—". */
export function fmtInt(v: number | null | undefined): string {
  return v == null ? '—' : _int.format(v);
}

/** Percentage with one decimal; null ⇒ "—". */
export function fmtPct(v: number | null | undefined): string {
  return v == null ? '—' : `${_int.format(Math.round(v * 10) / 10)}%`;
}

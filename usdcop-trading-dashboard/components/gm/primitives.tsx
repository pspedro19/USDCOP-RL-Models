'use client';

/**
 * GM primitives (CTR-GM-UI-001) — panel, KPI tile, badges, section headers.
 * All palette via `lib/ui/gm-tokens.ts`; structure-only Tailwind here.
 */
import type { ReactNode } from 'react';

import { GM, GMT, GM_TONE_BADGE, GM_TONE_TEXT, type GmTone } from '@/lib/ui/gm-tokens';

export function GmPanel({ title, meta, actions, children, className = '', inner = false }: {
  title?: ReactNode;
  meta?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  inner?: boolean;
}) {
  return (
    <section className={`${inner ? GM.panelInner : GM.panel} gm-contain p-[18px] ${className}`}>
      {(title || actions || meta) && (
        <header className="flex items-center gap-3 mb-3.5">
          {title && <div className={`${GMT.panelTitle} ${GM.textStrong}`}>{title}</div>}
          <div className="flex-1" />
          {meta && <span className={`${GMT.micro} ${GM.textMuted} font-mono`}>{meta}</span>}
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

export function GmKpi({ label, value, tone = 'neutral', sub, big = false }: {
  label: string;
  value: string;
  tone?: GmTone;
  sub?: ReactNode;
  big?: boolean;
}) {
  return (
    <div className={`${GM.panel} gm-contain px-4 py-3.5`}>
      <div className={`${GMT.label} ${GM.textMuted} mb-1.5`}>{label}</div>
      {/* aria-live=polite: los lectores de pantalla anuncian SOLO cuando el valor
          cambia (KPI que se refresca), silencioso para KPIs estáticos (checklist 14.5). */}
      <div
        aria-live="polite"
        aria-atomic="true"
        aria-label={`${label}: ${value}`}
        className={`${big ? GMT.kpiBig : GMT.kpi} ${tone === 'neutral' ? GM.text : GM_TONE_TEXT[tone]}`}
      >
        {value}
      </div>
      {sub && <div className={`${GMT.micro} ${GM.textMuted} mt-1`}>{sub}</div>}
    </div>
  );
}

export function GmBadge({ tone = 'neutral', children, className = '' }: {
  tone?: GmTone; children: ReactNode; className?: string;
}) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-[.4px] ${GM_TONE_BADGE[tone]} ${className}`}>
      {children}
    </span>
  );
}

export function GmPageHeader({ kicker, title, subtitle, actions }: {
  kicker?: string; title: string; subtitle?: string; actions?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-5 mb-6">
      <div>
        {kicker && <div className={`${GMT.kicker} ${GM.accent} mb-2`}>{kicker}</div>}
        <h1 className={`m-0 ${GMT.h1} ${GM.headline}`}>{title}</h1>
        {subtitle && <p className={`mt-2 mb-0 text-[14.5px] ${GM.textSec}`}>{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2.5">{actions}</div>}
    </div>
  );
}

/** Signed pct/pnl text with mono font and semantic color. */
export function GmDelta({ value, suffix = '%', digits = 2 }: { value: number | null | undefined; suffix?: string; digits?: number }) {
  if (value == null) return <span className={`${GMT.mono} ${GM.textMuted}`}>—</span>;
  const tone: GmTone = value > 0 ? 'pos' : value < 0 ? 'neg' : 'neutral';
  return (
    <span className={`${GMT.mono} font-bold ${GM_TONE_TEXT[tone]}`}>
      {value > 0 ? '+' : ''}{value.toFixed(digits)}{suffix}
    </span>
  );
}

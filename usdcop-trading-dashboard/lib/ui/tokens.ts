/**
 * @deprecated ADAPTER (CTR-GM-UI-001 unificación de tokens, 2026-07).
 *
 * Los tokens legacy de /admin ahora COMPONEN sus exports desde `lib/ui/gm-tokens.ts`
 * y las CSS vars `--gm-*` (primitivos en `app/globals.css`). CERO hex propio en este
 * archivo: si un valor necesita hex, se añade (append) a gm-tokens y se referencia.
 *
 * Código nuevo: importa `lib/ui/gm-tokens.ts` directamente (GM/GMT/ROLE_BADGE_GM…).
 * Este adapter existe solo para que los components/admin existentes compilen sin
 * migrarlos uno a uno; se retira cuando el último consumidor migre.
 */
import { GM, ROLE_BADGE_GM } from './gm-tokens';

// ─────────────────────────────────────────────────────────────── typography (§1.2)

export const TYPE = {
  pageTitle: 'text-xl leading-7 font-semibold',                       // 20/28
  sectionTitle: 'text-[13px] leading-4 font-semibold uppercase tracking-[.06em] text-[var(--gm-text-sec)]',
  kpiValue: 'text-3xl leading-9 font-semibold tabular-nums',          // 30/36
  body: 'text-sm leading-5',                                          // 14/20
  meta: 'text-xs leading-4 text-[var(--gm-text-sec)]',                // 12/16
  mono: 'tabular-nums',
} as const;

// ─────────────────────────────────────────────────────────────── color (§1.3)

export const COLOR = {
  textPrimary: GM.text,
  textSecondary: GM.textSec,
  ok: { text: GM.pos, dot: 'bg-[var(--gm-pos)]', badge: GM.posBadge },
  warn: { text: GM.warn, dot: 'bg-[var(--gm-warn)]', badge: GM.warnBadge },
  error: { text: GM.neg, dot: 'bg-[var(--gm-neg)]', badge: GM.negBadge },
  info: { text: GM.info, dot: 'bg-[var(--gm-info)]', badge: GM.infoBadge },
  accent: { text: GM.accent, dot: 'bg-[var(--gm-accent)]', badge: GM.accentBadge },
  neutral: { text: GM.textSec, dot: 'bg-[rgba(148,163,184,.55)]', badge: GM.neutralBadge },
} as const;

/** Role hues (§1.3): hierarchy is hue, never red — SSOT en gm-tokens (ROLE_BADGE_GM). */
export const ROLE_BADGE: Record<'admin' | 'developer' | 'subscriber' | 'free', string> = ROLE_BADGE_GM;

// ─────────────────────────────────────────────────────────────── surfaces (§1.1)

export const SURFACE = {
  page: GM.page,
  card: 'bg-[var(--gm-panel)] border border-[var(--gm-border)] rounded-xl',
  cardStale: 'opacity-60',
  container: 'max-w-[1440px] mx-auto px-6',
  tableRowHover: GM.rowHover,
  input: 'rounded-lg bg-[var(--gm-panel-inner)] border border-[rgba(148,163,184,.16)] px-3 py-1.5 text-xs text-[var(--gm-text)] placeholder:text-[var(--gm-text-faint)]',
  overlay: 'bg-black/50 backdrop-blur-[1px]',
  drawer: 'bg-[var(--gm-page)] border-l border-[var(--gm-border)]',
} as const;

// ─────────────────────────────────────────────────────────────── interactive (§1.3/§4)

export const CTA = {
  /** Accent = brand: primary CTAs and active tab ONLY. */
  primary: 'bg-[rgba(34,211,238,.15)] hover:bg-[rgba(34,211,238,.25)] text-[var(--gm-accent)] border border-[rgba(34,211,238,.40)] rounded-lg font-semibold',
  ghost: 'bg-transparent hover:bg-[rgba(148,163,184,.08)] text-[var(--gm-text-sec)] hover:text-[var(--gm-text)] border border-[rgba(148,163,184,.16)] rounded-lg',
  /** Destructive red ONLY inside confirmation surfaces (§1.3). */
  destructive: 'bg-[rgba(251,113,133,.12)] hover:bg-[rgba(251,113,133,.20)] text-[var(--gm-neg)] border border-[rgba(251,113,133,.40)] rounded-lg font-semibold',
  focusRing: GM.focus,
  /** Active tab underline + fill (accent = brand). */
  tabActive: 'border-[var(--gm-accent)] text-[var(--gm-accent)] bg-[rgba(14,21,38,.60)]',
  /** Selected option in a segmented control. */
  segmentActive: 'bg-[rgba(34,211,238,.15)] text-[var(--gm-accent)]',
} as const;

/** Accent ring for state-dependent emphasis (e.g. Vote 2 pendiente §2.2). */
export const RING_ACCENT = 'rounded-xl ring-1 ring-[rgba(34,211,238,.50)]';

/** Search-match highlight (<mark>). */
export const HIGHLIGHT_MARK = 'bg-[rgba(34,211,238,.25)] text-inherit rounded-sm';

/** KpiTile hover affordance. */
export const HOVER_ACCENT_BORDER = 'hover:border-[rgba(34,211,238,.40)] hover:bg-[rgba(14,21,38,.85)]';

/** Subtle row wash for critical-severity audit rows. */
export const SEVERITY_ROW_BG = 'bg-[rgba(251,113,133,.05)]';

export type SemanticTone = 'ok' | 'warn' | 'error' | 'info' | 'accent' | 'neutral';

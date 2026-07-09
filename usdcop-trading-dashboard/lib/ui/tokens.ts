/**
 * Admin design tokens (CTR-ADMIN-UI-001 — spec `.claude/specs/platform/admin-ui-polish.md` §1).
 *
 * THE only place colors/typography live for /admin: components consume these class
 * strings, never ad-hoc palette classes (gate: zero hardcoded colors outside this file).
 * Semantic rule — each color means ONE thing: ok=state, warn=degraded/TEST,
 * error=failures+destructive, info=links, accent=brand CTAs. Roles get their own
 * hue family (violet/blue/teal/gray): red is danger, never hierarchy.
 */

// ─────────────────────────────────────────────────────────────── typography (§1.2)

export const TYPE = {
  pageTitle: 'text-xl leading-7 font-semibold',                       // 20/28
  sectionTitle: 'text-[13px] leading-4 font-semibold uppercase tracking-[.06em] text-[#94A3B8]',
  kpiValue: 'text-3xl leading-9 font-semibold tabular-nums',          // 30/36
  body: 'text-sm leading-5',                                          // 14/20
  meta: 'text-xs leading-4 text-[#94A3B8]',                           // 12/16
  mono: 'tabular-nums',
} as const;

// ─────────────────────────────────────────────────────────────── color (§1.3)

export const COLOR = {
  textPrimary: 'text-[#E2E8F0]',
  textSecondary: 'text-[#94A3B8]',
  ok: { text: 'text-[#10B981]', dot: 'bg-[#10B981]', badge: 'bg-[#10B981]/10 text-[#10B981] border-[#10B981]/40' },
  warn: { text: 'text-[#F59E0B]', dot: 'bg-[#F59E0B]', badge: 'bg-[#F59E0B]/10 text-[#F59E0B] border-[#F59E0B]/40' },
  error: { text: 'text-[#F43F5E]', dot: 'bg-[#F43F5E]', badge: 'bg-[#F43F5E]/10 text-[#F43F5E] border-[#F43F5E]/40' },
  info: { text: 'text-[#38BDF8]', dot: 'bg-[#38BDF8]', badge: 'bg-[#38BDF8]/10 text-[#38BDF8] border-[#38BDF8]/40' },
  accent: { text: 'text-[#22D3EE]', dot: 'bg-[#22D3EE]', badge: 'bg-[#22D3EE]/10 text-[#22D3EE] border-[#22D3EE]/40' },
  neutral: { text: 'text-[#94A3B8]', dot: 'bg-slate-500', badge: 'bg-slate-700/30 text-[#94A3B8] border-slate-600/40' },
} as const;

/** Role hues (§1.3): hierarchy is hue, never red. */
export const ROLE_BADGE: Record<'admin' | 'developer' | 'subscriber' | 'free', string> = {
  admin: 'bg-[#A78BFA]/10 text-[#A78BFA] border-[#A78BFA]/40',
  developer: 'bg-[#60A5FA]/10 text-[#60A5FA] border-[#60A5FA]/40',
  subscriber: 'bg-[#2DD4BF]/10 text-[#2DD4BF] border-[#2DD4BF]/40',
  free: 'bg-slate-700/30 text-[#94A3B8] border-slate-600/40',
};

// ─────────────────────────────────────────────────────────────── surfaces (§1.1)

export const SURFACE = {
  page: 'bg-slate-950',
  card: 'bg-slate-900/50 border border-[rgba(148,163,184,.12)] rounded-xl',
  cardStale: 'opacity-60',
  container: 'max-w-[1440px] mx-auto px-6',
  tableRowHover: 'hover:bg-slate-800/40',
  input: 'rounded-lg bg-slate-900 border border-slate-700 px-3 py-1.5 text-xs text-[#E2E8F0] placeholder:text-[#64748B]',
  overlay: 'bg-black/50 backdrop-blur-[1px]',
  drawer: 'bg-slate-900 border-l border-[rgba(148,163,184,.12)]',
} as const;

// ─────────────────────────────────────────────────────────────── interactive (§1.3/§4)

export const CTA = {
  /** Accent = brand: primary CTAs and active tab ONLY. */
  primary: 'bg-[#22D3EE]/15 hover:bg-[#22D3EE]/25 text-[#22D3EE] border border-[#22D3EE]/40 rounded-lg font-semibold',
  ghost: 'bg-transparent hover:bg-slate-800/60 text-[#94A3B8] hover:text-[#E2E8F0] border border-slate-700/60 rounded-lg',
  /** Destructive red ONLY inside confirmation surfaces (§1.3). */
  destructive: 'bg-[#F43F5E]/15 hover:bg-[#F43F5E]/25 text-[#F43F5E] border border-[#F43F5E]/40 rounded-lg font-semibold',
  focusRing: 'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#22D3EE] focus-visible:ring-offset-1 focus-visible:ring-offset-slate-950',
  /** Active tab underline + fill (accent = brand). */
  tabActive: 'border-[#22D3EE] text-[#22D3EE] bg-slate-900/60',
  /** Selected option in a segmented control. */
  segmentActive: 'bg-[#22D3EE]/15 text-[#22D3EE]',
} as const;

/** Accent ring for state-dependent emphasis (e.g. Vote 2 pendiente §2.2). */
export const RING_ACCENT = 'rounded-xl ring-1 ring-[#22D3EE]/50';

/** Search-match highlight (<mark>). */
export const HIGHLIGHT_MARK = 'bg-[#22D3EE]/25 text-inherit rounded-sm';

/** KpiTile hover affordance. */
export const HOVER_ACCENT_BORDER = 'hover:border-[#22D3EE]/40 hover:bg-slate-900/80';

/** Subtle row wash for critical-severity audit rows. */
export const SEVERITY_ROW_BG = 'bg-[#F43F5E]/5';

export type SemanticTone = 'ok' | 'warn' | 'error' | 'info' | 'accent' | 'neutral';

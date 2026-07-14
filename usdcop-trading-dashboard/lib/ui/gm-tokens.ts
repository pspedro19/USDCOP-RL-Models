/**
 * GlobalMarkets Terminal design tokens v2 (CTR-GM-UI-001 + Checklist Front-End Pro).
 *
 * Arquitectura (Checklist §9, 3 niveles): PRIMITIVOS = CSS vars en `app/globals.css`
 * (`--gm-*`, `--z-*`, `--gm-dur-*`) → SEMÁNTICOS = este archivo (GM/GMT/MOTION/Z) →
 * los componentes consumen solo estos nombres. Cero hex en componentes; cero z-index
 * fuera de la escala; motion solo por token.
 *
 * Tipografía en REM (accesible a zoom/font-size del usuario; WCAG 1.4.4) con `clamp()`
 * fluido en headings/KPI grandes — punto medio = tamaño del prototipo a 1440px.
 * Contraste: `textMuted` ahora #7C8AA3 (≥4.5:1 sobre #0A0D15); #64748B queda como
 * `textFaint` SOLO para elementos decorativos/no esenciales.
 */

// ─────────────────────────────────────────────────────────────── color (semántico)

export const GM = {
  /** Page + surfaces */
  page: 'bg-[var(--gm-page)]',
  panel: 'bg-[var(--gm-panel)] border border-[var(--gm-border)] rounded-2xl',
  panelInner: 'bg-[var(--gm-panel-inner)] border border-[rgba(148,163,184,.10)] rounded-xl',
  panelSoft: 'bg-[rgba(148,163,184,.05)] border border-[var(--gm-border)] rounded-xl',
  popover: 'bg-[#0E1526] border border-[rgba(148,163,184,.16)] rounded-xl shadow-[0_16px_40px_rgba(0,0,0,.5)]',
  headerBar: 'bg-[rgba(10,13,21,.9)] backdrop-blur-xl border-b border-[rgba(148,163,184,.10)]',

  /** Text */
  headline: 'text-[var(--gm-headline)]',
  text: 'text-[var(--gm-text)]',
  textStrong: 'text-[var(--gm-text-strong)]',
  textSec: 'text-[var(--gm-text-sec)]',
  textMuted: 'text-[var(--gm-text-muted)]',
  /** Decorativo/no esencial únicamente — NO cumple AA para texto informativo. */
  textFaint: 'text-[var(--gm-text-faint)]',

  /** Semantic */
  pos: 'text-[var(--gm-pos)]',
  neg: 'text-[var(--gm-neg)]',
  warn: 'text-[var(--gm-warn)]',
  info: 'text-[var(--gm-info)]',
  accent: 'text-[var(--gm-accent)]',

  posBadge: 'bg-[rgba(52,211,153,.12)] border border-[rgba(52,211,153,.3)] text-[var(--gm-pos)]',
  negBadge: 'bg-[rgba(251,113,133,.10)] border border-[rgba(251,113,133,.24)] text-[var(--gm-neg)]',
  warnBadge: 'bg-[rgba(245,158,11,.12)] border border-[rgba(245,158,11,.3)] text-[var(--gm-warn)]',
  infoBadge: 'bg-[rgba(59,130,246,.12)] border border-[rgba(59,130,246,.3)] text-[var(--gm-info)]',
  accentBadge: 'bg-[rgba(34,211,238,.10)] border border-[rgba(34,211,238,.28)] text-[var(--gm-accent)]',
  neutralBadge: 'bg-[rgba(148,163,184,.07)] border border-[rgba(148,163,184,.10)] text-[var(--gm-text-sec)]',

  /** Brand */
  brandGradient: 'bg-gradient-to-br from-[#22D3EE] via-[#3B82F6] to-[#8B5CF6]',
  ctaGradient: 'bg-gradient-to-br from-[#22D3EE] to-[#3B82F6] text-[#06121C]',
  onAccent: 'text-[#06121C]',

  /** Interactive (motion por token; targets ≥44px en variantes táctiles) */
  ctaPrimary: 'bg-gradient-to-br from-[#22D3EE] to-[#3B82F6] text-[#06121C] font-bold rounded-[11px] hover:opacity-90 transition-opacity duration-[var(--gm-dur-fast)]',
  ctaSoft: 'bg-[rgba(34,211,238,.10)] border border-[rgba(34,211,238,.28)] text-[var(--gm-accent)] font-bold rounded-[10px] hover:bg-[rgba(34,211,238,.16)] transition-colors duration-[var(--gm-dur-fast)]',
  ctaGhost: 'bg-[rgba(148,163,184,.06)] border border-[rgba(148,163,184,.14)] text-[var(--gm-text-strong)] rounded-[10px] hover:bg-[rgba(148,163,184,.10)] transition-colors duration-[var(--gm-dur-fast)]',
  ctaDanger: 'bg-[rgba(251,113,133,.08)] border border-[rgba(251,113,133,.2)] text-[var(--gm-neg)] rounded-[10px] hover:bg-[rgba(251,113,133,.14)] transition-colors duration-[var(--gm-dur-fast)]',
  focus: 'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--gm-accent)] focus-visible:ring-offset-1 focus-visible:ring-offset-[var(--gm-page)]',
  input: 'bg-[var(--gm-panel-inner)] border border-[rgba(148,163,184,.16)] rounded-[10px] text-[var(--gm-text)] placeholder:text-[var(--gm-text-faint)] px-3 py-2 text-[0.8125rem]',
  navActive: 'bg-[rgba(34,211,238,.10)] text-[var(--gm-accent)]',
  navIdle: 'text-[#9AA7BD] hover:bg-[rgba(148,163,184,.06)] hover:text-[var(--gm-text)] transition-colors duration-[var(--gm-dur-fast)]',
  rowHover: 'hover:bg-[rgba(148,163,184,.05)]',
} as const;

// ─────────────────────────────────────────────────────────────── type scale (rem)

/**
 * Escala en rem (prototipo Var B @1440px como punto medio). Headings/KPI grandes con
 * clamp() fluido; el resto rem fijo (densidad terminal estable, escalable por zoom).
 */
export const GMT = {
  h1: 'text-[clamp(1.5rem,1.2rem+1vw,1.75rem)] leading-tight font-extrabold tracking-[-.5px]',
  h2: 'text-[1.0625rem] font-bold',                      // 17px
  panelTitle: 'text-[0.8125rem] font-bold',              // 13px
  kicker: 'text-[0.75rem] font-semibold uppercase tracking-[1.2px]',
  label: 'text-[0.65625rem] font-semibold uppercase tracking-[.5px]',   // 10.5px
  body: 'text-[0.84375rem]',                             // 13.5px (denso; prosa usa .gm-prose ≥1rem)
  prose: 'text-[1rem] leading-relaxed',                  // lectura larga (Checklist §1)
  meta: 'text-[0.75rem]',                                // 12px
  micro: 'text-[0.65625rem]',                            // 10.5px
  /** Prices/KPIs: JetBrains Mono, tabular (Checklist §1 🔴). */
  mono: 'font-mono tabular-nums',
  kpi: 'text-[1.3125rem] font-extrabold font-mono tabular-nums',        // 21px
  kpiBig: 'text-[clamp(1.625rem,1.4rem+0.8vw,1.875rem)] font-extrabold font-mono tabular-nums',
} as const;

// ─────────────────────────────────────────────────────────────── motion & z (tokens)

/** Motion tokens (Checklist §8): SIEMPRE via clase, nunca duraciones sueltas. */
export const MOTION = {
  fast: 'duration-[var(--gm-dur-fast)]',
  base: 'duration-[var(--gm-dur-base)]',
  slow: 'duration-[var(--gm-dur-slow)]',
  easeOut: 'ease-[var(--gm-ease-out)]',
  easeInOut: 'ease-[var(--gm-ease-in-out)]',
} as const;

/** Z-index nombrado (Checklist §9): dropdown < sticky < backdrop < drawer < modal < toast. */
export const Z = {
  sticky: 'z-[var(--z-sticky)]',
  backdrop: 'z-[var(--z-backdrop)]',
  drawer: 'z-[var(--z-drawer)]',
  modal: 'z-[var(--z-modal)]',
  toast: 'z-[var(--z-toast)]',
} as const;

// ─────────────────────────────────────────────────────────────── tonos

export type GmTone = 'pos' | 'neg' | 'warn' | 'info' | 'accent' | 'neutral';

export const GM_TONE_TEXT: Record<GmTone, string> = {
  pos: GM.pos, neg: GM.neg, warn: GM.warn, info: GM.info, accent: GM.accent, neutral: GM.textSec,
};

export const GM_TONE_BADGE: Record<GmTone, string> = {
  pos: GM.posBadge, neg: GM.negBadge, warn: GM.warnBadge, info: GM.infoBadge,
  accent: GM.accentBadge, neutral: GM.neutralBadge,
};

/** Sign-aware tone for pct/pnl values. */
export function toneOf(n: number | null | undefined): GmTone {
  if (n == null || n === 0) return 'neutral';
  return n > 0 ? 'pos' : 'neg';
}

// ─────────────────────────────────────────────────────────────── admin console (append-only)
// Tokens que la consola /admin necesita y que requieren hex (este archivo es el ÚNICO
// lugar permitido para hex — CTR-ADMIN-CONSOLE-001 + CTR-GM-UI-001).

/**
 * Role hues (rol ≠ rojo): la jerarquía es matiz, nunca peligro.
 * admin=violeta · developer=azul · subscriber=teal · free=gris.
 */
export const ROLE_BADGE_GM: Record<'admin' | 'developer' | 'subscriber' | 'free', string> = {
  admin: 'bg-[rgba(167,139,250,.10)] text-[#A78BFA] border-[rgba(167,139,250,.40)]',
  developer: 'bg-[rgba(96,165,250,.10)] text-[#60A5FA] border-[rgba(96,165,250,.40)]',
  subscriber: 'bg-[rgba(45,212,191,.10)] text-[#2DD4BF] border-[rgba(45,212,191,.40)]',
  free: 'bg-[rgba(148,163,184,.08)] text-[var(--gm-text-sec)] border-[rgba(148,163,184,.20)]',
};

/** Violeta "experimental" (registro de modelos, actor de auditoría). */
export const GM_VIOLET = {
  text: 'text-[#A78BFA]',
  badge: 'bg-[rgba(167,139,250,.10)] border border-[rgba(167,139,250,.40)] text-[#A78BFA]',
} as const;

/**
 * Valores hex CRUDOS (único lugar permitido, CTR-GM-UI-001). Los consumidores que NO
 * pueden usar una clase Tailwind ni una CSS var — Recharts (`fill`/`stroke` como atributo
 * SVG) y props de color de iconos (`color=`/`iconColor=`) — importan de aquí en vez de
 * escribir el hex inline. Espejo 1:1 de los primitivos `--gm-*` de globals.css.
 */
export const GM_HEX = {
  accent: '#22D3EE',
  pos: '#34D399',
  neg: '#FB7185',
  violet: '#A78BFA',
  textSec: '#8494AC',
  tick: '#64748B',        // = --gm-text-faint (ejes/ticks decorativos)
  ref: '#4A5A72',         // = --gm-chart-ref
  onAccent: '#06121C',    // = --gm-on-accent
  disabled: '#475569',    // = --gm-text-disabled
  tooltipBg: '#0E1526',   // = --gm-chart-tooltip
  gridStroke: 'rgba(148,163,184,.16)',
  warn: '#F59E0B',        // = --gm-warn
  info: '#3B82F6',        // = --gm-info
  textIdle: '#9AA7BD',    // texto de eje/leyenda en gráficos (navIdle)
} as const;

/**
 * Paleta EXTENDIDA de gráficos analíticos (Recharts en /analysis, `gm-analysis.tsx`).
 * Series categóricas que exceden los 6 tonos semánticos base — viven aquí (único lugar
 * con hex) para no dispersarse por los componentes.
 */
export const GM_CHART = {
  tick: GM_HEX.tick,
  tooltipBg: GM_HEX.tooltipBg,
  text: GM_HEX.textIdle,
  pos: GM_HEX.pos,
  neg: GM_HEX.neg,
  warn: GM_HEX.warn,
  info: GM_HEX.info,
  accent: GM_HEX.accent,
  violet: '#8B5CF6',
  pink: '#EC4899',
  gold: '#D4AF37',
  indigo: '#6366F1',
  line: '#E6EDF7',        // = --gm-text (línea principal sobre fondo oscuro)
  maskFill: GM_HEX.tooltipBg,
} as const;

/**
 * Paleta del gráfico de velas (lightweight-charts, `TradingChartWithSignals`). Colores
 * de dominio (velas verde/rojo distintas de los semánticos GM; tinte por modelo;
 * semáforo de confianza; cromo del panel). Centralizada aquí — cero hex en el chart.
 */
export const GM_TV = {
  candleUp: '#22c55e',
  candleDown: '#ef4444',
  modelBlue: '#3B82F6',
  modelBlueSoft: '#60A5FA',
  modelViolet: '#8B5CF6',
  modelVioletSoft: '#A78BFA',
  bg: '#0f172a',
  axisText: '#94a3b8',
  grid: '#1e293b',
  crosshair: '#475569',
  border: '#334155',
  conf: { high: '#00C853', good: '#4CAF50', mid: '#FFC107', low: '#FF5722' },
} as const;

/** Kill-switch global (RiesgoSection): tarjeta + CTA según estado. */
export const GM_KILL = {
  cardActive: 'rounded-2xl border border-[rgba(251,113,133,.40)] bg-[rgba(251,113,133,.07)]',
  cardCalm: 'rounded-2xl border border-[rgba(52,211,153,.28)] bg-[rgba(52,211,153,.05)]',
  cardUnknown: 'rounded-2xl border border-[rgba(148,163,184,.20)] bg-[rgba(148,163,184,.05)]',
  /** Detener: el rojo pleno vive SOLO dentro de la superficie de confirmación. */
  haltBtn: 'bg-gradient-to-br from-[#FB7185] to-[#F43F5E] text-white font-extrabold rounded-[11px] hover:opacity-90 transition-opacity duration-[var(--gm-dur-fast)]',
  resumeBtn: 'bg-[rgba(52,211,153,.16)] text-[var(--gm-pos)] font-extrabold rounded-[11px] hover:bg-[rgba(52,211,153,.24)] transition-colors duration-[var(--gm-dur-fast)]',
} as const;

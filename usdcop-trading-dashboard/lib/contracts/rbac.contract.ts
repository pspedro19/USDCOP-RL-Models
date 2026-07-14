/**
 * RBAC + Monetization contract (CTR-RBAC-001) — THE single source of truth.
 *
 * Principle (spec `.claude/specs/platform/rbac-monetization.md`): the ROLE says who you
 * are (admin/developer/subscriber/free); the PLAN says what you paid for (entitlements).
 * Both are validated SERVER-SIDE on every sensitive request — the UI only reflects.
 * Deny-by-default: a page/API route with no entry in the matrices below is DENIED.
 *
 * Consumed by:
 *   - `middleware.ts`            → route gating (pages + APIs)
 *   - `<Nav/>` / hub cards       → what each role sees (render-only; never the defense)
 *   - route handlers             → `requirePermission()` fine-grained checks
 *   - `tests/rbac-coverage`      → CI fails if a real route is missing from the matrix
 */

// ─────────────────────────────────────────────────────────────── roles & permissions

export const ROLES = ['admin', 'developer', 'subscriber', 'free'] as const;
export type Role = (typeof ROLES)[number];

export const PERMISSIONS = [
  'research:read',      // backtest replay, versions, registry, experiments (read)
  'research:propose',   // propose experiments (Vote 1 stays automated)
  'approval:vote',      // Vote 2 + promote + deploy — admin only, always audited
  'signals:read',       // production signals ("Señales" view) per entitled assets
  'forecast:read',      // forecasting (server applies entitlement delay)
  'analysis:read',      // weekly/daily analysis (server applies entitlement delay)
  'execution:self',     // own SignalBridge: own keys, own orders, own kill switch
  'execution:global',   // system SignalBridge + global kill switch — admin only
  'market:read',        // realtime price/candles (rate-limited per user)
  'admin:all',          // user/plan management, audit log, flags
] as const;
export type Permission = (typeof PERMISSIONS)[number];

/** What each role can do. Deny-by-default: absence = denied. */
export const ROLE_PERMISSIONS: Record<Role, readonly Permission[]> = {
  admin: [
    'research:read', 'research:propose', 'approval:vote', 'signals:read',
    'forecast:read', 'analysis:read', 'execution:self', 'execution:global',
    'market:read', 'admin:all',
  ],
  developer: [
    'research:read', 'research:propose', 'forecast:read', 'analysis:read',
    'signals:read', 'market:read',
  ],
  subscriber: ['signals:read', 'forecast:read', 'analysis:read', 'execution:self', 'market:read'],
  free: ['forecast:read', 'analysis:read'], // server serves DELAYED content per entitlements
};

// ─────────────────────────────────────────────────────────────── entitlements (plan)

export type PlanId = 'free' | 'signals' | 'auto';

export interface ExecutionEntitlement {
  enabled: boolean;
  mode: 'paper' | 'live';
  paper_weeks_required: number;
  max_notional_usd: number;
  max_daily_loss_pct: number;
  max_open_positions: number;
}

export interface Entitlements {
  plan: PlanId;
  /** Asset ids the plan covers (maps 1:1 to the multi-asset registry). */
  assets: string[];
  forecast_delay_hours: number; // free serves week T−1
  analysis_delay_days: number;  // free serves summary T+7
  signals_realtime: boolean;
  execution: ExecutionEntitlement;
  expires_at: string | null;    // ISO; expired ⇒ auto-degrade to free
}

/** System ceilings per tier — users may LOWER execution limits, never raise them. */
export const PLAN_DEFAULTS: Record<PlanId, Entitlements> = {
  free: {
    plan: 'free', assets: ['usdcop'], forecast_delay_hours: 168, analysis_delay_days: 7,
    signals_realtime: false,
    execution: { enabled: false, mode: 'paper', paper_weeks_required: 4,
                 max_notional_usd: 0, max_daily_loss_pct: 0, max_open_positions: 0 },
    expires_at: null,
  },
  signals: {
    plan: 'signals', assets: ['usdcop'], forecast_delay_hours: 0, analysis_delay_days: 0,
    signals_realtime: true,
    execution: { enabled: false, mode: 'paper', paper_weeks_required: 4,
                 max_notional_usd: 0, max_daily_loss_pct: 0, max_open_positions: 0 },
    expires_at: null,
  },
  auto: {
    plan: 'auto', assets: ['usdcop'], forecast_delay_hours: 0, analysis_delay_days: 0,
    signals_realtime: true,
    execution: { enabled: true, mode: 'paper', paper_weeks_required: 4,
                 max_notional_usd: 5000, max_daily_loss_pct: 3.0, max_open_positions: 2 },
    expires_at: null,
  },
};

export function isExpired(e: Entitlements | null | undefined): boolean {
  return !!e?.expires_at && new Date(e.expires_at).getTime() < Date.now();
}

/** Effective entitlements: expired/missing ⇒ free; PARTIAL rows (e.g. raw `{"plan":"free"}`
 * in DB) are MERGED over the plan's defaults so consumers never see undefined fields
 * (QA iter-2 F7: `.assets.includes` crashed on a partial row — fixed at the SSOT). */
export function effectiveEntitlements(e: Entitlements | null | undefined): Entitlements {
  if (!e || isExpired(e)) return PLAN_DEFAULTS.free;
  const base = PLAN_DEFAULTS[e.plan] ?? PLAN_DEFAULTS.free;
  return { ...base, ...e, execution: { ...base.execution, ...(e.execution ?? {}) } };
}

// ─────────────────────────────────────────────────────────────── route matrices

export interface RouteRule {
  /** Path prefix (matched with startsWith after exact-match pass). */
  prefix: string;
  permission: Permission | 'public' | 'authenticated';
}

/**
 * PAGE routes × required permission. First match wins (longest prefix listed first
 * where ambiguity exists). Anything not matched ⇒ 'authenticated' minimum.
 */
export const PAGE_ROUTES: readonly RouteRule[] = [
  { prefix: '/login', permission: 'public' },
  { prefix: '/register', permission: 'public' },   // self-serve signup → admin approval
  { prefix: '/reset-password', permission: 'public' }, // forced temp-password consumption
  { prefix: '/pricing', permission: 'public' },
  { prefix: '/metodologia', permission: 'public' }, // transparency page — the sales weapon
  { prefix: '/legal', permission: 'public' },       // terminos / riesgo / privacidad
  { prefix: '/hub', permission: 'authenticated' },
  { prefix: '/dashboard', permission: 'research:read' },   // Backtest (replay/versions/gates)
  { prefix: '/forecasting', permission: 'forecast:read' },
  { prefix: '/analysis', permission: 'analysis:read' },
  { prefix: '/production', permission: 'signals:read' },   // "Señales" for clients
  { prefix: '/execution', permission: 'execution:self' },
  { prefix: '/admin', permission: 'admin:all' },
  // Archived pre-GlobalMarkets UI (CTR-GM-UI-001 migration) — admin-only reference copies.
  { prefix: '/legacy', permission: 'admin:all' },
  { prefix: '/account', permission: 'authenticated' },
  { prefix: '/catalog', permission: 'authenticated' },     // catálogo/watchlist/carrito (§4.3)
  { prefix: '/cart', permission: 'authenticated' },        // página completa del carrito (§4.3)
  { prefix: '/', permission: 'public' },                   // landing (exact match only)
] as const;

/** API routes × required permission. Deny-by-default enforced by coverage test. */
export const API_ROUTES: readonly RouteRule[] = [
  // public plumbing
  { prefix: '/api/auth', permission: 'public' },
  { prefix: '/api/health', permission: 'public' },
  { prefix: '/api/billing/webhook', permission: 'public' }, // signature-verified inside handler
  { prefix: '/api/billing/prices', permission: 'public' },  // plan prices shown on the public pricing page
  { prefix: '/api/public', permission: 'public' },          // marketing aggregates only (live-stats)
  { prefix: '/api/captcha', permission: 'public' },         // signed challenge for auth forms (answer never in response)
  // SignalBridge auth endpoints ARE the login (they mint the session) — must be public.
  // Longest-prefix beats the '/api/execution' → execution:self rule below.
  { prefix: '/api/execution/auth', permission: 'public' },
  // research surface (admin/developer)
  { prefix: '/api/registry', permission: 'research:read' },
  { prefix: '/api/backtest', permission: 'research:read' },
  { prefix: '/api/replay', permission: 'research:read' },
  { prefix: '/api/experiments', permission: 'research:read' },
  { prefix: '/api/models', permission: 'research:read' },
  { prefix: '/api/strategies', permission: 'research:read' },
  // approval — admin only (audited)
  { prefix: '/api/production/approve', permission: 'approval:vote' },
  { prefix: '/api/production/deploy', permission: 'approval:vote' },
  { prefix: '/api/registry/promote', permission: 'approval:vote' },
  // signals / production data
  { prefix: '/api/production', permission: 'signals:read' },
  { prefix: '/api/trading', permission: 'signals:read' },
  // gated content (server applies entitlement delay inside the handler)
  { prefix: '/api/data/analysis', permission: 'analysis:read' },
  { prefix: '/api/data', permission: 'authenticated' },
  { prefix: '/api/analysis', permission: 'analysis:read' },
  { prefix: '/api/forecasting', permission: 'forecast:read' },
  // market
  { prefix: '/api/market', permission: 'market:read' },
  { prefix: '/api/pipeline', permission: 'authenticated' },
  { prefix: '/api/proxy', permission: 'authenticated' },
  // execution
  { prefix: '/api/signalbridge/system', permission: 'execution:global' },
  { prefix: '/api/signalbridge/me', permission: 'execution:self' },
  { prefix: '/api/execution', permission: 'execution:self' },
  // admin
  { prefix: '/api/admin', permission: 'admin:all' },
  { prefix: '/api/users', permission: 'admin:all' },
  { prefix: '/api/billing', permission: 'authenticated' },
  // catálogo · watchlist · carrito (CTR-FE-BE-001 §4.3) — per-user rows, any role
  { prefix: '/api/catalog', permission: 'authenticated' },
  { prefix: '/api/watchlist', permission: 'authenticated' },
  { prefix: '/api/cart', permission: 'authenticated' },
  { prefix: '/api/agent', permission: 'research:read' },
  { prefix: '/api/signals', permission: 'signals:read' },
] as const;

// ─────────────────────────────────────────────────────────────── nav (render-only)

export interface NavEntry {
  href: string;
  /** Label per role — subscriber sees "Señales", internals see "Producción". */
  label: string;
  subscriberLabel?: string;
  permission: Permission | 'authenticated';
}

/** Hub/nav entries derive from this — never hardcode role checks in components. */
export const NAV_ENTRIES: readonly NavEntry[] = [
  { href: '/dashboard', label: 'Backtest', permission: 'research:read' },
  { href: '/production', label: 'Producción', subscriberLabel: 'Señales', permission: 'signals:read' },
  { href: '/forecasting', label: 'Forecasting', permission: 'forecast:read' },
  { href: '/analysis', label: 'Análisis', permission: 'analysis:read' },
  { href: '/execution', label: 'SignalBridge', permission: 'execution:self' },
  { href: '/admin', label: 'Admin', permission: 'admin:all' },
] as const;

// ─────────────────────────────────────────────────────────────── helpers

export function roleHasPermission(role: Role | string | undefined, perm: Permission): boolean {
  if (!role || !(ROLES as readonly string[]).includes(role)) return false;
  return ROLE_PERMISSIONS[role as Role].includes(perm);
}

/** Narrow an arbitrary string to a known Role. */
export function isRole(x: string | undefined | null): x is Role {
  return !!x && (ROLES as readonly string[]).includes(x);
}

/**
 * Membership check against an EFFECTIVE permission set (dynamic RBAC, migration
 * 056). Used by middleware/relay off the JWT-baked `permissions` claim; when the
 * claim is absent (legacy tokens) callers fall back to `roleHasPermission`.
 */
export function permsHave(perms: readonly string[] | undefined | null, perm: Permission): boolean {
  return !!perms && perms.includes(perm);
}

/**
 * Downgrade-only intersection for role PREVIEW ("Ver como"): the effective set an
 * admin sees while previewing role X = their real permissions ∩ X's permissions.
 * Because it only ever RESTRICTS, a forged view-as cookie can never escalate.
 */
export function intersectPerms(real: readonly string[], view: readonly string[]): string[] {
  const v = new Set(view);
  return real.filter((p) => v.has(p));
}

export function requiredPermissionFor(pathname: string, rules: readonly RouteRule[]):
    Permission | 'public' | 'authenticated' | null {
  // exact landing match first so '/' doesn't swallow everything
  if (pathname === '/') return 'public';
  const match = rules
    .filter((r) => r.prefix !== '/' && pathname.startsWith(r.prefix))
    .sort((a, b) => b.prefix.length - a.prefix.length)[0];
  return match ? match.permission : null;
}

export function navFor(role: Role): { href: string; label: string }[] {
  return NAV_ENTRIES.filter((e) =>
    e.permission === 'authenticated' ? true : roleHasPermission(role, e.permission),
  ).map((e) => ({
    href: e.href,
    label: role === 'subscriber' && e.subscriberLabel ? e.subscriberLabel : e.label,
  }));
}

'use client';

/**
 * TerminalShell — GlobalMarkets chrome (prototype Var B lines 42–128):
 * collapsible sidebar (brand · role-filtered nav · lang toggle · user · logout) +
 * sticky topbar (hamburger · "Mis activos" ticker · cart · Planes) + 1320px container.
 *
 * Nav derives from the CTR-RBAC-001 matrix (render-only; middleware is the defense),
 * order per VISUAL-SPEC-CHECKLIST: Inicio · Activos · Backtest · Producción/Señales ·
 * Forecasting · Análisis · SignalBridge · Admin (subscriber relabel + plan gate).
 *
 * Ticker sources (checklist "Chrome por rol"): research:read roles follow
 * /api/registry; subscriber/free consume /api/catalog. It renders symbol always,
 * price/change only when the API returns them (≠ null) and a Spark only when a
 * real series exists — data is NEVER invented here.
 *
 * i18n: every visible string comes from GM_DICT (lib/i18n/gm.ts); the ES/EN
 * toggle is live via setGmLang/useGmLang.
 * Wrap any migrated page: <TerminalShell active="production">…</TerminalShell>.
 */
import { useEffect, useState, type ReactNode } from 'react';
import { useRouter } from 'next/navigation';
import { signOut, useSession } from 'next-auth/react';
import { JetBrains_Mono } from 'next/font/google';
import {
  BarChart3, Calendar, Cpu, Eye, FileText, Home, Languages, LogOut, Menu,
  Settings, ShoppingCart, Sparkles, Store, Zap,
} from 'lucide-react';

import { VIEW_AS_ROLE_COOKIE } from '@/lib/contracts/admin-console.contract';
import { CART_CHANGED_EVENT, type CartResponse, type CatalogAsset, type CatalogResponse } from '@/lib/contracts/catalog.contract';
import { ROLES, roleHasPermission, type Permission, type Role } from '@/lib/contracts/rbac.contract';
import type { RegistryAssetEntry } from '@/lib/contracts/strategy-manifest.contract';
import { GM_DICT } from '@/lib/i18n/gm';
import { setGmLang, useGmLang, useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT, GM_TONE_TEXT } from '@/lib/ui/gm-tokens';
import { CartDrawer } from './CartDrawer';
import { NewsBell } from './NewsBell';
import { Spark } from './Spark';
import { useGmQuery } from './useGmQuery';

const jbMono = JetBrains_Mono({ subsets: ['latin'], variable: '--font-gm-mono', weight: ['400', '500', '600', '700'] });

export type GmSection =
  | 'hub' | 'catalog' | 'dashboard' | 'production' | 'forecasting' | 'analysis' | 'signalbridge' | 'admin';

interface NavItem {
  id: GmSection;
  label: string;
  icon: typeof Home;
  href: string;
  permission: Permission | null;
}

function useRole(): { role: Role; loading: boolean; name: string; authed: boolean; permissions: string[] | null } {
  const { data: session, status } = useSession();
  const user = session?.user as { role?: string; name?: string; email?: string; permissions?: string[] } | undefined;
  return {
    role: (status === 'loading' ? '__loading__' : (user?.role ?? 'free')) as Role,
    loading: status === 'loading',
    name: user?.name || user?.email || '',
    authed: status === 'authenticated',
    // Effective permission set baked into the JWT at login (dynamic RBAC, migration
    // 056). null ⇒ legacy token without the claim → fall back to the static matrix.
    permissions: Array.isArray(user?.permissions) ? user!.permissions! : null,
  };
}

/** Optional live fields the ticker renders ONLY when the API publishes them. */
interface TickerLiveFields {
  price?: number | null;
  change_pct?: number | null;
  series?: number[] | null;
  sparkline?: number[] | null;
}

interface TickerItem {
  id: string;
  symbol: string;
  price: number | null;
  changePct: number | null;
  series: number[] | null;
}

/** Lee una cookie NO-httpOnly del documento (gm-view-as-role es legible por diseño). */
function readCookie(name: string): string | null {
  if (typeof document === 'undefined') return null;
  const m = document.cookie.match(new RegExp(`(?:^|; )${name}=([^;]*)`));
  return m ? decodeURIComponent(m[1]) : null;
}

function toTickerItem(id: string, symbol: string, live: TickerLiveFields): TickerItem {
  const rawSeries = live.series ?? live.sparkline ?? null;
  const series = rawSeries?.filter((n) => Number.isFinite(n)) ?? null;
  return {
    id,
    symbol,
    price: typeof live.price === 'number' && Number.isFinite(live.price) ? live.price : null,
    changePct: typeof live.change_pct === 'number' && Number.isFinite(live.change_pct) ? live.change_pct : null,
    series: series && series.length >= 2 ? series : null,
  };
}

/** Content column width per page type. Dense grids/tables (análisis, backtest) use
 *  'wide'; the default suits most dashboards; 'full' is edge-to-edge for terminals. */
export type GmContentWidth = 'default' | 'wide' | 'full';
const GM_CONTENT_MAXW: Record<GmContentWidth, string> = {
  default: 'max-w-[1600px]',
  wide: 'max-w-[1860px]',
  full: 'max-w-none',
};

export function TerminalShell({ active, children, width = 'default' }: { active: GmSection; children: ReactNode; width?: GmContentWidth }) {
  const router = useRouter();
  const { role: sessionRole, name, authed, loading, permissions: sessionPerms } = useRole();
  const lang = useGmLang();
  const t = useGmT(GM_DICT);

  // "Ver como" (impersonación read-only): el espejo legible `gm-view-as-role` hace que la
  // nav rendericen como el rol SIMULADO — SIN conceder permisos (el servidor sigue
  // exigiendo el rol real para toda mutación; la cookie firmada httpOnly `gm-view-as` es
  // la fuente de verdad server-side). Al ser una restricción de navegación, es seguro.
  const [viewAs, setViewAs] = useState<Role | null>(null);
  useEffect(() => {
    const raw = readCookie(VIEW_AS_ROLE_COOKIE);
    setViewAs(raw && (ROLES as readonly string[]).includes(raw) ? (raw as Role) : null);
  }, []);
  const role = viewAs ?? sessionRole;

  const exitViewAs = async () => {
    try { await fetch('/api/admin/impersonate', { method: 'DELETE' }); } catch { /* noop */ }
    window.location.reload();
  };

  // Cambiar de vista "Ver como" desde el shell (solo admin real). Elegir el propio
  // rol admin = salir; cualquier otro rol = POST /impersonate (read-only, audita).
  const [viewAsBusy, setViewAsBusy] = useState(false);
  const switchViewAs = async (target: Role) => {
    if (target === sessionRole) { await exitViewAs(); return; }
    setViewAsBusy(true);
    try {
      const r = await fetch('/api/admin/impersonate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role: target, motivo: `Ver como ${target} (switch del shell)` }),
      });
      if (r.ok) window.location.href = '/hub';
      else setViewAsBusy(false);
    } catch { setViewAsBusy(false); }
  };

  const isSubscriber = role === 'subscriber';
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  // Carrito (CTR-FE-BE-001 §4.3): badge con count + drawer lateral. Oculto sin
  // sesión; el count se refresca con CART_CHANGED_EVENT (mutaciones en catálogo
  // o en el propio drawer). onUnauthenticated=noop: el badge nunca redirige.
  const [cartOpen, setCartOpen] = useState(false);
  const cart = useGmQuery<CartResponse>(authed ? '/api/cart' : null, { onUnauthenticated: () => {} });
  const cartCount = cart.data?.items.length ?? 0;
  useEffect(() => {
    const onChange = () => cart.reload();
    window.addEventListener(CART_CHANGED_EVENT, onChange);
    return () => window.removeEventListener(CART_CHANGED_EVENT, onChange);
  }, [cart.reload]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    setCollapsed(typeof window !== 'undefined' && window.localStorage.getItem('gm-sidebar') === '0');
  }, []);
  const toggleSidebar = () => {
    setCollapsed((c) => {
      window.localStorage.setItem('gm-sidebar', c ? '1' : '0');
      return !c;
    });
    setMobileOpen((o) => !o);
  };

  // Plan gate for SignalBridge in nav (same rule as GlobalNavbar / finding #28).
  const [execEnabled, setExecEnabled] = useState<boolean | null>(null);
  useEffect(() => {
    if (role !== 'subscriber') return;
    fetch('/api/billing/me').then((r) => (r.ok ? r.json() : null))
      .then((e) => setExecEnabled(!!e?.execution?.enabled)).catch(() => setExecEnabled(false));
  }, [role]);

  // Visibility uses the EFFECTIVE permission set (dynamic RBAC, migration 056):
  //  - preview ("Ver como"): the previewed role's static permissions (matches the
  //    server's downgrade-only preview; the shell only ever restricts here);
  //  - otherwise: the JWT-baked effective set (reflects role/override edits after
  //    re-login), falling back to the static role matrix for legacy tokens.
  const canSee = (perm: Permission): boolean => {
    if (viewAs) return roleHasPermission(viewAs, perm);
    if (sessionPerms) return sessionPerms.includes(perm);
    return roleHasPermission(sessionRole, perm);
  };

  // Prototype nav order (checklist "Chrome por rol"): Inicio · Activos · Backtest ·
  // Producción(admin)/Señales(sub) · Forecasting · Análisis · SignalBridge · Admin.
  const nav: NavItem[] = ([
    { id: 'hub', label: t('navHub'), icon: Home, href: '/hub', permission: null },
    { id: 'catalog', label: t('navCatalog'), icon: Store, href: '/catalog', permission: null },
    { id: 'dashboard', label: t('navBacktest'), icon: BarChart3, href: '/dashboard', permission: 'research:read' },
    { id: 'production', label: isSubscriber ? t('navSignals') : t('navProduction'), icon: Cpu, href: '/production', permission: 'signals:read' },
    { id: 'forecasting', label: t('navForecasting'), icon: Calendar, href: '/forecasting', permission: 'forecast:read' },
    { id: 'analysis', label: t('navAnalysis'), icon: FileText, href: '/analysis', permission: 'analysis:read' },
    { id: 'signalbridge', label: t('navExecution'), icon: Zap, href: '/execution/dashboard', permission: 'execution:self' },
    { id: 'admin', label: t('navAdmin'), icon: Settings, href: '/admin', permission: 'admin:all' },
  ] as NavItem[])
    .filter((i) => i.permission === null || roleHasPermission(role, i.permission))
    .filter((i) => i.id !== 'signalbridge' || !isSubscriber || execEnabled === true);

  // Ticker (§5.2 — asset lists never duplicated in the front): research roles follow
  // the registry SSOT; client roles (subscriber/free) consume /api/catalog
  // (authenticated) instead of collecting a 403 on the research-surface registry
  // (rbac.md §8). Both fetches wait for the session to resolve.
  const canReadRegistry = !loading && roleHasPermission(role, 'research:read');
  const registry = useGmQuery<{ assets?: Array<RegistryAssetEntry & TickerLiveFields> }>(
    canReadRegistry ? '/api/registry' : null);
  const catalog = useGmQuery<CatalogResponse>(
    !loading && authed && !canReadRegistry ? '/api/catalog' : null, { onUnauthenticated: () => {} });

  let tickerItems: TickerItem[];
  if (canReadRegistry) {
    tickerItems = (registry.data?.assets ?? []).map((a) => toTickerItem(a.asset_id, a.symbol, a));
  } else {
    const assets: Array<CatalogAsset & TickerLiveFields> = catalog.data?.assets ?? [];
    const watched = assets.filter((a) => a.in_watchlist);
    const source = watched.length ? watched : assets.filter((a) => a.status === 'available');
    tickerItems = source.map((a) => toTickerItem(a.asset_id, a.symbol, a));
  }

  const priceFmt = new Intl.NumberFormat(lang === 'es' ? 'es-CO' : 'en-US', { maximumFractionDigits: 2 });

  const logout = async () => {
    ['isAuthenticated', 'username', 'auth-token', 'refresh-token'].forEach((k) => {
      window.localStorage.removeItem(k); window.sessionStorage.removeItem(k);
    });
    await signOut({ callbackUrl: '/login' });
  };

  const toggleLang = () => setGmLang(lang === 'es' ? 'en' : 'es');
  const displayName = name || t('user');
  const sidebarVisible = !collapsed;

  return (
    <div className={`gm-root min-h-screen ${GM.page} ${jbMono.variable}`}>
      {/* mobile backdrop */}
      {mobileOpen && (
        <button
          aria-label={t('closeMenu')}
          onClick={() => setMobileOpen(false)}
          className="fixed inset-0 z-[var(--z-backdrop)] bg-[rgba(3,6,14,.55)] backdrop-blur-[2px] lg:hidden"
        />
      )}

      {/* sidebar — layout via explicit .gm-sidebar CSS (globals.css): the offset is
          load-bearing and must not depend on utility-class scanning */}
      <aside
        className="gm-sidebar"
        data-mobile-open={mobileOpen}
        data-visible={sidebarVisible}
      >
        <button onClick={() => router.push('/hub')} className={`flex items-center gap-3 px-1.5 mb-5 ${GM.focus}`}>
          <span className={`w-9 h-9 rounded-[11px] ${GM.brandGradient} flex items-center justify-center shadow-[0_4px_18px_rgba(34,211,238,.35)]`}>
            <Sparkles className="w-4.5 h-4.5 text-white" aria-hidden />
          </span>
          <span className="flex flex-col items-start leading-none">
            <span className={`text-[15px] font-extrabold ${GM.headline} tracking-[-.2px]`}>{t('brand')}</span>
            <span className={`text-[9.5px] font-semibold ${GM.textMuted} tracking-[1.4px] uppercase mt-[3px]`}>{t('terminal')}</span>
          </span>
        </button>

        <div className={`${GMT.micro} font-bold ${GM.textFaint} uppercase tracking-[1px] px-3 mb-2`}>{t('quickAccess')}</div>
        <nav className="flex flex-col gap-[3px]" aria-label={t('sections')}>
          {nav.map((item) => {
            const Icon = item.icon;
            const isActive = active === item.id;
            return (
              <button
                key={item.id}
                onClick={() => { setMobileOpen(false); router.push(item.href); }}
                aria-current={isActive ? 'page' : undefined}
                className={`flex items-center gap-3 h-11 px-3 rounded-[11px] text-left ${GM.focus}
                  ${isActive ? GM.navActive : GM.navIdle}`}
              >
                <Icon className="w-[18px] h-[18px] shrink-0" aria-hidden />
                <span className="text-[13.5px] font-semibold">{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="flex-1" />

        {/* "Ver como" — solo el admin REAL puede cambiar de vista (read-only, audita).
            Coincide con el prototipo Var B (switch "Ver como" en el menú lateral). */}
        {sessionRole === 'admin' && (
          <div className="mb-2">
            <label htmlFor="gm-view-as" className={`flex items-center gap-1.5 ${GMT.micro} font-bold ${GM.textFaint} uppercase tracking-[.8px] mb-1.5 px-1`}>
              <Eye className="w-3 h-3" aria-hidden /> {lang === 'es' ? 'Ver como' : 'View as'}
            </label>
            <select
              id="gm-view-as"
              data-testid="view-as-switch"
              value={viewAs ?? 'admin'}
              disabled={viewAsBusy}
              onChange={(e) => switchViewAs(e.target.value as Role)}
              aria-label={lang === 'es' ? 'Ver la plataforma como otro rol (solo lectura)' : 'View the platform as another role (read-only)'}
              className={`${GM.input} ${GM.focus} w-full h-10 text-[13px] font-semibold cursor-pointer disabled:opacity-50`}
            >
              <option value="admin">{lang === 'es' ? 'Admin (tú)' : 'Admin (you)'}</option>
              <option value="developer">Developer</option>
              <option value="subscriber">{lang === 'es' ? 'Suscriptor' : 'Subscriber'}</option>
              <option value="free">{lang === 'es' ? 'Gratis / Invitado' : 'Free / Guest'}</option>
            </select>
          </div>
        )}

        <div className="flex items-center gap-2">
          <button
            onClick={toggleLang}
            title={t('language')}
            aria-label={t('language')}
            className={`${GM.ctaGhost} ${GM.focus} flex items-center gap-1.5 h-11 px-3 text-[0.75rem] font-semibold font-mono`}
          >
            <Languages className="w-4 h-4" aria-hidden /> {lang.toUpperCase()}
          </button>
          <div className={`flex-1 min-w-0 flex items-center gap-2 h-10 px-2.5 ${GM.panelSoft}`}>
            <span className={`w-[26px] h-[26px] rounded-[7px] ${GM.brandGradient} flex items-center justify-center text-[11px] font-extrabold ${GM.onAccent} shrink-0`}>
              {displayName.slice(0, 2).toUpperCase()}
            </span>
            <span className={`text-[12px] font-semibold ${GM.textStrong} truncate`}>{displayName}</span>
          </div>
          <button onClick={logout} title={t('logout')} aria-label={t('logout')}
            className={`${GM.ctaDanger} ${GM.focus} w-11 h-11 shrink-0 flex items-center justify-center`}>
            <LogOut className="w-4 h-4" aria-hidden />
          </button>
        </div>
      </aside>

      {/* content column — offset via explicit .gm-content CSS */}
      <div className="gm-content" data-shifted={sidebarVisible}>
        {/* banner persistente de impersonación (read-only) — visible en toda página */}
        {viewAs && (
          <div
            role="status"
            className={`${GM.warnBadge} ${GMT.meta} sticky top-0 z-[var(--z-sticky)] flex items-center gap-2 px-5 py-2`}
          >
            <Eye className="w-4 h-4 shrink-0" aria-hidden />
            <span className="font-semibold">
              {lang === 'es' ? `Viendo como ${viewAs} — solo lectura` : `Viewing as ${viewAs} — read-only`}
            </span>
            <button
              onClick={exitViewAs}
              className={`${GM.ctaGhost} ${GM.focus} ml-auto h-8 px-3 text-[0.75rem] font-semibold shrink-0`}
            >
              {lang === 'es' ? 'Salir' : 'Exit'}
            </button>
          </div>
        )}
        {/* topbar */}
        <header className={`sticky top-0 z-[var(--z-sticky)] h-[54px] flex items-center gap-3.5 px-5 ${GM.headerBar}`}>
          <button onClick={toggleSidebar} title={t('menu')} aria-label={t('toggleMenu')}
            className={`${GM.ctaGhost} ${GM.focus} w-11 h-11 flex items-center justify-center shrink-0`}>
            <Menu className="w-4.5 h-4.5" aria-hidden />
          </button>
          <span className={`hidden sm:flex items-center gap-1.5 ${GMT.micro} font-bold ${GM.textMuted} uppercase tracking-[.8px] shrink-0`}>
            {t('myAssets')}
          </span>
          <div className="flex items-center gap-5 overflow-hidden flex-1">
            {tickerItems.slice(0, 5).map((a) => {
              const tone = a.changePct == null ? 'accent' : a.changePct >= 0 ? 'pos' : 'neg';
              const dot = tone === 'pos' ? 'bg-[var(--gm-pos)]'
                : tone === 'neg' ? 'bg-[var(--gm-neg)]' : 'bg-[var(--gm-accent)]';
              return (
                <button
                  key={a.id}
                  onClick={() => router.push('/catalog')}
                  aria-label={`${a.symbol} — ${t('goCatalog')}`}
                  className={`flex items-center gap-2 h-11 shrink-0 ${GM.focus}`}
                >
                  <span className={`w-1.5 h-1.5 rounded-full ${dot}`} aria-hidden />
                  <span className={`text-[12px] font-bold ${GM.textStrong} ${GMT.mono}`}>
                    {a.symbol}
                  </span>
                  {a.price != null && (
                    <span className={`text-[12px] font-medium ${GM.textSec} ${GMT.mono}`}>
                      {priceFmt.format(a.price)}
                    </span>
                  )}
                  {a.changePct != null && (
                    <span className={`text-[11.5px] font-bold ${GMT.mono} ${GM_TONE_TEXT[tone]}`}>
                      {a.changePct >= 0 ? '+' : ''}{a.changePct.toFixed(2)}%
                    </span>
                  )}
                  {a.series && (
                    <Spark seed={a.id} data={a.series} tone={tone} size="sm" />
                  )}
                </button>
              );
            })}
          </div>
          {authed && <NewsBell />}
          {authed && (
            <button
              onClick={() => setCartOpen(true)}
              title={t('cart')}
              aria-label={`${t('openCart')} (${cartCount})`}
              className={`${GM.ctaGhost} ${GM.focus} relative w-11 h-11 flex items-center justify-center shrink-0`}
              data-testid="topbar-cart"
            >
              <ShoppingCart className="w-4.5 h-4.5" aria-hidden />
              {cartCount > 0 && (
                <span className={`absolute -top-1.5 -right-1.5 min-w-[18px] h-[18px] px-1 rounded-full
                  ${GM.ctaGradient} flex items-center justify-center text-[10px] font-extrabold font-mono`}>
                  {cartCount}
                </span>
              )}
            </button>
          )}
          <button onClick={() => router.push('/pricing')}
            className={`${GM.ctaSoft} ${GM.focus} h-11 px-3.5 text-[0.78125rem] shrink-0`}>
            {t('plans')}
          </button>
        </header>

        {/* main fills the content column and centers its inner block via
            justify-content (bulletproof). NOTE: `w-full` + `mx-auto` does NOT
            center — width:100% resolves the auto margins to 0 before max-width
            clamps the box, leaving content pinned left on wide screens. */}
        <main className="flex-1 w-full min-w-0 flex justify-center">
          <div className={`w-full ${GM_CONTENT_MAXW[width]} px-6 pt-[26px] pb-[90px]`}>
            {children}
          </div>
        </main>
      </div>

      <CartDrawer open={cartOpen} onClose={() => setCartOpen(false)} />
    </div>
  );
}

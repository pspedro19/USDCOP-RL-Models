'use client';

/**
 * HubView — vista HUB del GlobalMarkets Terminal (CTR-GM-UI-001; prototipo
 * "Var B" líneas 160–245): saludo por rol + liveStats REALES a la derecha,
 * grid 3-col de module cards derivadas de la matriz RBAC (módulos denegados a
 * roles cliente = candado con teaser → "Ver planes" /pricing) y franja
 * "Mis activos" alimentada por el registry SSOT.
 *
 * Datos (cero números inventados):
 *   - /api/public/live-stats  → KPIs del bundle publicado (ratios ocultos si N<20)
 *   - /api/registry           → activos (sin precios: el registry no los publica)
 *   - /api/billing/me         → plan-gate SignalBridge para suscriptores
 * Toda la paleta viene de `lib/ui/gm-tokens.ts` — cero hex sueltos aquí.
 */
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useSession } from 'next-auth/react';
import {
  BarChart3, Bitcoin, Calendar, ChevronRight, Coins, Cpu, Eye, FileText,
  Gem, LineChart, Lock, Settings, Store, Zap,
} from 'lucide-react';

import { defineGmDict, useGmT } from '@/lib/i18n/gm-core';
import { ROLES, roleHasPermission, type Permission, type Role } from '@/lib/contracts/rbac.contract';
import type { RegistryIndex } from '@/lib/contracts/strategy-manifest.contract';
import { canShowRatios } from '@/lib/contracts/ui.contract';
import { GM, GMT, GM_TONE_BADGE, GM_TONE_TEXT, toneOf, type GmTone } from '@/lib/ui/gm-tokens';
import { AsyncBoundary, GmBadge, GmPageHeader, GmSkeleton, useGmQuery } from '@/components/gm';

// ─────────────────────────────────────────────── i18n (strings de esta vista)

/** Prototipo Var B: cardMeta.catalog (l. 2050), lockMsg exactos (l. 2080-2082),
 *  L.common.seePlans, identidad "Invitado" (login('free','Invitado')). */
const HUB_DICT = defineGmDict({
  es: {
    catalogTitle: 'Activos',
    catalogSubtitle: 'Watchlist y carrito',
    catalogDesc: 'Explora forex, cripto, acciones y materias primas. Sigue activos o desbloquéalos para operar.',
    catalogFeatures: '4 categorías|Watchlist|Carrito de activos',
    lockSignals: 'Activa señales en vivo con un plan.',
    lockExec: 'Ejecución automática — plan Auto.',
    seePlans: 'Ver planes',
    guest: 'Invitado',
  },
  en: {
    catalogTitle: 'Assets',
    catalogSubtitle: 'Watchlist & cart',
    catalogDesc: 'Browse forex, crypto, stocks and commodities. Follow assets or unlock them to trade.',
    catalogFeatures: '4 categories|Watchlist|Asset cart',
    lockSignals: 'Unlock live signals with a plan.',
    lockExec: 'Auto-execution — Auto plan.',
    seePlans: 'See plans',
    guest: 'Guest',
  },
});

type HubT = (key: keyof (typeof HUB_DICT)['es']) => string;

// ─────────────────────────────────────────────────────── módulos (RBAC-driven)

interface HubModule {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  icon: typeof BarChart3;
  tone: GmTone;
  href: string;
  features: string[];
  /** 'authenticated' ⇒ visible para TODOS los roles (p.ej. catálogo de activos). */
  permission: Permission | 'authenticated';
  /** Teaser mostrado cuando la card está bloqueada (conversión → /pricing). */
  lockMsg?: string;
}

/** Orden del prototipo (modOrder l. 2059): catalog · backtest · signals ·
 *  forecasting · analysis · execution · admin. */
function buildModules(isSubscriber: boolean, t: HubT): HubModule[] {
  return [
    {
      id: 'catalog',
      title: t('catalogTitle'),
      subtitle: t('catalogSubtitle'),
      description: t('catalogDesc'),
      icon: Store, tone: 'accent', href: '/catalog',
      features: t('catalogFeatures').split('|'),
      permission: 'authenticated',
    },
    {
      id: 'dashboard',
      title: 'Backtest',
      subtitle: 'Backtest y aprobación',
      description: 'Revisa el backtest OOS por estrategia, con curva de equity, gates y replay por versión.',
      icon: BarChart3, tone: 'info', href: '/dashboard',
      features: ['Replay por versión', 'Gates', 'Historial de trades'],
      permission: 'research:read', // solo admin/developer (matriz RBAC)
    },
    {
      id: 'production',
      title: isSubscriber ? 'Señales' : 'Producción',
      subtitle: 'Estrategia en tiempo real',
      description: 'Estrategia activa en producción: señal semanal, equity, posición y P&L en vivo.',
      icon: Cpu, tone: 'pos', href: '/production',
      features: ['Señal semanal', 'Equity al día', 'P&L en vivo'],
      permission: 'signals:read',
      lockMsg: t('lockSignals'),
    },
    {
      id: 'forecasting',
      title: 'Forecasting',
      subtitle: 'Predicciones a medio plazo',
      description: 'Proyecciones semanales multi-activo (USD/COP, Oro, BTC) con intervalos de confianza.',
      icon: Calendar, tone: 'warn', href: '/forecasting',
      features: ['9 modelos ML', 'Intervalos', 'Multi-activo'],
      permission: 'forecast:read',
    },
    {
      id: 'analysis',
      title: 'Análisis',
      subtitle: 'Inteligencia de mercado',
      description: 'Análisis AI multi-activo con indicadores macro, señales, noticias y calendario económico.',
      icon: FileText, tone: 'info', href: '/analysis',
      features: ['Análisis AI', 'Gráficos macro', 'Chat asistente'],
      permission: 'analysis:read',
    },
    {
      id: 'execution',
      title: 'SignalBridge',
      subtitle: 'Ejecución automatizada',
      description: 'Conecta tus exchanges y ejecuta señales automáticamente con gestión de riesgo y kill switch.',
      icon: Zap, tone: 'neg', href: '/execution/dashboard',
      features: ['Paper-first', 'Gestión de riesgo', 'Kill switch'],
      permission: 'execution:self',
      lockMsg: t('lockExec'),
    },
    {
      id: 'admin',
      title: 'Admin',
      subtitle: 'Consola de administración',
      description: 'Cola de aprobación de usuarios, gestión de planes, salud del sistema y auditoría.',
      icon: Settings, tone: 'accent', href: '/admin',
      features: ['Usuarios', 'Sistema', 'Auditoría'],
      permission: 'admin:all',
    },
  ];
}

const ROLE_LABEL: Record<Role, string> = {
  admin: 'Admin',
  developer: 'Developer',
  subscriber: 'Suscriptor',
  free: 'Free',
};

/** Cuenta demo compartida (bootstrap_guest en SignalBridge). El email por defecto
 *  se puede sobreescribir en build con NEXT_PUBLIC_GUEST_EMAIL. */
const GUEST_EMAIL = process.env.NEXT_PUBLIC_GUEST_EMAIL || 'guest@demo.local';

// ────────────────────────────────────────────────────────── liveStats (bundle)

interface LiveStats {
  strategy_name?: string;
  year?: number;
  unavailable?: boolean;
  return_ytd_pct?: number | null;
  max_dd_pct?: number | null;
  n_trades?: number | null;
  sharpe?: number | null;
  bundle_date?: string | null;
}

/** KPIs del bundle publicado — skeleton→empty→error vía AsyncBoundary. */
function HubLiveStats() {
  const stats = useGmQuery<LiveStats>('/api/public/live-stats');
  return (
    <AsyncBoundary
      state={stats}
      empty={(s) => !!s.unavailable || s.return_ytd_pct == null}
      emptyProps={{
        title: 'Sin datos en vivo',
        body: 'El bundle publicado aún no está disponible — corre el pipeline de producción.',
      }}
      skeleton={
        <div className="flex gap-2.5" aria-busy>
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className={`h-[66px] w-[118px] ${GM.panelSoft} motion-safe:animate-pulse`} />
          ))}
        </div>
      }
    >
      {(s) => {
        const items: Array<{ label: string; value: string; tone: GmTone }> = [
          { label: 'Estrategia', value: s.strategy_name ?? '—', tone: 'neutral' },
          {
            label: `Retorno ${s.year ?? ''} YTD`,
            value: `${(s.return_ytd_pct ?? 0) >= 0 ? '+' : ''}${(s.return_ytd_pct ?? 0).toFixed(2)}%`,
            tone: toneOf(s.return_ytd_pct),
          },
          {
            label: 'Max DD',
            value: s.max_dd_pct != null ? `−${Math.abs(s.max_dd_pct).toFixed(1)}%` : '—',
            tone: 'neg',
          },
          canShowRatios(s.n_trades) && s.sharpe != null
            ? { label: 'Sharpe', value: s.sharpe.toFixed(2), tone: 'accent' }
            : { label: 'Operaciones', value: String(s.n_trades ?? '—'), tone: 'accent' },
        ];
        return (
          <div className="flex flex-col items-end gap-1.5" data-testid="hub-live-stats">
            <div className="flex flex-wrap gap-2.5">
              {items.map((it) => (
                <div key={it.label} className={`min-w-[118px] px-[15px] py-3 ${GM.panel} rounded-[13px]`}>
                  <div className={`${GMT.label} ${GM.textMuted} mb-1.5`}>{it.label}</div>
                  <div className={`text-[19px] font-extrabold font-mono tabular-nums ${it.tone === 'neutral' ? GM.text : GM_TONE_TEXT[it.tone]}`}>
                    {it.value}
                  </div>
                </div>
              ))}
            </div>
            <span className={`${GMT.micro} ${GM.textFaint}`}>
              cifras del bundle publicado{s.bundle_date ? ` · ${s.bundle_date}` : ''} — no recomputadas
            </span>
          </div>
        );
      }}
    </AsyncBoundary>
  );
}

// ─────────────────────────────────────────────────────── franja "Mis activos"

const ASSET_CLASS_META: Record<string, { tone: GmTone; icon: typeof Coins; label: string }> = {
  fx: { tone: 'accent', icon: Coins, label: 'FX' },
  crypto: { tone: 'warn', icon: Bitcoin, label: 'Cripto' },
  commodity: { tone: 'pos', icon: Gem, label: 'Mat. primas' },
};

/** Activos del registry SSOT — símbolo + clase; sin precios inventados.
 *  El registry crudo es superficie research (rbac.md §8): los roles cliente no lo
 *  fetchean (evita el 403) y ven la franja bloqueada con CTA a planes. */
function HubMyAssets({ canRead }: { canRead: boolean }) {
  const registry = useGmQuery<RegistryIndex>(canRead ? '/api/registry' : null);
  const count = registry.data?.assets?.length ?? 0;
  if (!canRead) {
    return (
      <section className={`${GM.panel} p-[22px] mt-5`} data-testid="hub-my-assets">
        <header className="flex items-center gap-2.5 mb-2">
          <Lock className={`w-[18px] h-[18px] ${GM.textMuted}`} aria-hidden />
          <h3 className={`m-0 text-[15px] font-bold ${GM.text}`}>Mis activos</h3>
        </header>
        <p className={`m-0 text-[12.5px] ${GM.textSec}`}>
          El detalle de activos y estrategias es parte de los planes pagos.{' '}
          <a href="/pricing" className={`font-bold ${GM.accent}`}>Ver planes →</a>
        </p>
      </section>
    );
  }
  return (
    <section className={`${GM.panel} p-[22px] mt-5`} data-testid="hub-my-assets">
      <header className="flex items-center gap-2.5 mb-4">
        <Eye className={`w-[18px] h-[18px] ${GM.accent}`} aria-hidden />
        <h3 className={`m-0 text-[15px] font-bold ${GM.text}`}>Mis activos</h3>
        {count > 0 && (
          <span className={`text-[11px] font-semibold ${GM.textMuted}`}>· {count} activos</span>
        )}
        <div className="flex-1" />
        {/* Catálogo · watchlist · carrito (CTR-FE-BE-001 §4.3) */}
        <a href="/catalog" className={`${GM.ctaSoft} ${GM.focus} h-8 px-3 inline-flex items-center gap-1.5 text-[12px]`}
          data-testid="hub-manage-assets">
          Gestionar <ChevronRight className="w-3.5 h-3.5" aria-hidden />
        </a>
      </header>
      <AsyncBoundary
        state={registry}
        empty={(r) => (r.assets ?? []).length === 0}
        emptyProps={{
          title: 'Sin activos publicados',
          body: 'El registry aún no tiene activos — publica un bundle para verlos aquí.',
        }}
        skeleton={
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3" aria-busy>
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className={`h-[68px] ${GM.panelSoft} motion-safe:animate-pulse`} />
            ))}
          </div>
        }
      >
        {(r) => (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {r.assets.map((a) => {
              const meta = ASSET_CLASS_META[a.asset_class ?? ''] ?? {
                tone: 'neutral' as GmTone, icon: LineChart, label: a.asset_class ?? '',
              };
              const Icon = meta.icon;
              return (
                <div key={a.asset_id} className={`${GM.panelInner} p-3.5 flex items-center gap-2.5`}
                  data-testid={`hub-asset-${a.asset_id}`}>
                  <span className={`w-9 h-9 rounded-[9px] flex items-center justify-center shrink-0 border ${GM_TONE_BADGE[meta.tone]}`}>
                    <Icon className="w-[18px] h-[18px]" aria-hidden />
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className={`text-[13px] font-bold ${GM.text} font-mono truncate`}>{a.symbol}</div>
                    <div className={`${GMT.micro} ${GM.textMuted} truncate`}>{a.display_name}</div>
                  </div>
                  {meta.label && <GmBadge tone={meta.tone}>{meta.label}</GmBadge>}
                </div>
              );
            })}
          </div>
        )}
      </AsyncBoundary>
    </section>
  );
}

// ──────────────────────────────────────────────────────────────── module card

function ModuleCard({ m, locked, seePlans, onGo }: {
  m: HubModule; locked: boolean; seePlans: string; onGo: () => void;
}) {
  const Icon = m.icon;
  return (
    <button
      data-testid={`hub-card-${m.id}`}
      onClick={onGo}
      className={`${GM.panel} ${GM.focus} ${GM.rowHover} group relative p-[22px] text-left transition-all
        ${locked ? 'opacity-85' : ''}`}
    >
      <div className="flex items-start gap-3.5 mb-3.5">
        <span className={`w-[46px] h-[46px] rounded-xl flex items-center justify-center shrink-0
          ${locked ? GM_TONE_BADGE.neutral : GM_TONE_BADGE[m.tone]}`}>
          <Icon className="w-[22px] h-[22px]" aria-hidden />
        </span>
        <div className="flex-1 min-w-0">
          <div className={`text-[17px] font-bold ${GM.headline}`}>{m.title}</div>
          <div className={`text-[12.5px] ${GM.textSec} mt-[3px]`}>{m.subtitle}</div>
        </div>
        {locked
          ? <Lock className={`w-[18px] h-[18px] ${GM.textMuted} shrink-0`} aria-label="módulo bloqueado" />
          : <ChevronRight className={`w-[18px] h-[18px] ${GM.textFaint} shrink-0 transition-transform group-hover:translate-x-0.5`} aria-hidden />}
      </div>
      <p className={`m-0 mb-4 text-[13px] leading-[1.55] ${GM.textSec}`}>{m.description}</p>
      <div className="flex flex-wrap gap-1.5">
        {m.features.map((f) => (
          <span key={f} className={`${GM.neutralBadge} rounded-[7px] px-[9px] py-1 text-[11px] font-medium`}>
            {f}
          </span>
        ))}
      </div>
      {locked && (
        <div className={`mt-4 flex items-center justify-between gap-3 px-3 py-2.5 ${GM.panelInner}`}>
          <span className={`text-[11.5px] ${GM.textSec}`}>{m.lockMsg}</span>
          <span className={`text-[12px] font-bold ${GM.accent} shrink-0`}>{seePlans} →</span>
        </div>
      )}
    </button>
  );
}

// ──────────────────────────────────────────────────────────────────────── view

export default function HubView() {
  const router = useRouter();
  const t = useGmT(HUB_DICT);
  // Cards render desde el contrato RBAC (CTR-RBAC-001): la matriz decide qué ve cada
  // rol; el middleware es la defensa real — esto es solo reflexión de UI.
  const { data: session, status } = useSession();
  const sessionLoading = status === 'loading';
  const role = ((session?.user as { role?: string } | undefined)?.role ?? 'free') as Role;
  // Preview "Ver como": si hay cookie de rol simulado y el rol REAL es admin, el hub
  // renderiza como el rol previsualizado (downgrade-only, coherente con la nav del shell
  // y el gate real del servidor). Sin esto, un admin previsualizando veía las cards admin.
  const [viewAs, setViewAs] = useState<Role | null>(null);
  useEffect(() => {
    const raw = typeof document !== 'undefined'
      ? document.cookie.split('; ').find((c) => c.startsWith('gm-view-as-role='))?.split('=')[1]
      : undefined;
    setViewAs(raw && (ROLES as readonly string[]).includes(raw) ? (raw as Role) : null);
  }, []);
  const effectiveRole = viewAs ?? role;
  const isSubscriber = effectiveRole === 'subscriber';
  // Effective permission set baked into the JWT (dynamic RBAC, migración 056): refleja
  // ediciones de rol/override tras re-login. null ⇒ token legacy → matriz estática.
  const sessionPerms = Array.isArray((session?.user as { permissions?: string[] } | undefined)?.permissions)
    ? (session!.user as { permissions?: string[] }).permissions!
    : null;
  // En preview usa los permisos (estáticos) del rol simulado; si no, el set horneado.
  const canSee = (perm: Permission): boolean =>
    viewAs ? roleHasPermission(viewAs, perm)
      : sessionPerms ? sessionPerms.includes(perm)
        : roleHasPermission(role, perm);
  // Identidad "Invitado" cuando la sesión es la cuenta demo compartida (guest).
  const email = (session?.user as { email?: string } | undefined)?.email ?? '';
  const isGuest = email.toLowerCase() === GUEST_EMAIL.toLowerCase();

  // Plan-gate SignalBridge: un suscriptor cuyo PLAN no incluye ejecución ve la card
  // bloqueada (upsell a Auto) aunque su ROL tenga execution:self (misma regla que el shell).
  const [execEnabled, setExecEnabled] = useState<boolean | null>(null);
  useEffect(() => {
    if (effectiveRole !== 'subscriber') return;
    fetch('/api/billing/me').then((r) => (r.ok ? r.json() : null))
      .then((e) => setExecEnabled(!!e?.execution?.enabled)).catch(() => setExecEnabled(false));
  }, [effectiveRole]);

  if (sessionLoading) return <GmSkeleton label="Cargando tu terminal…" />;

  const modules = buildModules(isSubscriber, t);
  // §3.1 ux-navigation: roles cliente ven los módulos DENEGADOS como teaser bloqueado
  // (conversión), nunca como ausencia. Roles internos no ven lo que no es suyo.
  // Orden fiel al prototipo: una sola pasada; la card bloqueada queda EN SU SITIO.
  const isClientRole = effectiveRole === 'free' || effectiveRole === 'subscriber';
  const planLockedIds = effectiveRole === 'subscriber' && execEnabled === false ? ['execution'] : [];
  const cards = modules
    .map((m) => {
      const visible = m.permission === 'authenticated' || canSee(m.permission);
      const locked = (isClientRole && !visible && (m.id === 'production' || m.id === 'execution'))
        || planLockedIds.includes(m.id);
      if (!visible && !locked) return null;
      return { m, locked };
    })
    .filter((c): c is { m: HubModule; locked: boolean } => c !== null);

  return (
    <div className="motion-safe:animate-in motion-safe:fade-in">
      <GmPageHeader
        kicker={`Bienvenido · ${isGuest ? t('guest') : ROLE_LABEL[effectiveRole] ?? 'Free'}`}
        title="Tu terminal"
        subtitle="Elige un módulo o gestiona tus activos y suscripciones."
        actions={<HubLiveStats />}
      />

      {/* Module grid (prototipo: 3 columnas, gap 16px, bloqueadas en su posición) */}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {cards.map(({ m, locked }) => (
          <ModuleCard
            key={m.id}
            m={m}
            locked={locked}
            seePlans={t('seePlans')}
            onGo={() => router.push(locked ? '/pricing' : m.href)}
          />
        ))}
      </div>

      <HubMyAssets canRead={canSee('research:read')} />
    </div>
  );
}

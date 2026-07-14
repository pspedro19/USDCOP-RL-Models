'use client';

/**
 * PricingView — public /pricing on the GlobalMarkets design system
 * (CTR-GM-UI-001, prototype Var B lines 1384-1421; CTR-RBAC-001 §B.5).
 *
 * Tiers mirror PLAN_DEFAULTS (the SSOT) — feature flags here NEVER grant access; the
 * middleware + entitlements do. Prices: la página anterior no publicaba montos COP
 * ("Suscripción mensual") y los montos del backend (lib/billing/wompi.ts) son
 * placeholders explícitos (§B.5 "business decision") — se conserva el copy sin monto.
 * Marketing metric policy (audit I-8/B.2): headline = forward production, never backtest.
 * Plan actual: useSession (opcional) + GET /api/billing/me; la página sigue siendo pública.
 */
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useSession } from 'next-auth/react';
import {
  Bitcoin, Blocks, Bolt, Check, Coins, Gem, LineChart, Plus, Radio, ShieldCheck, X,
} from 'lucide-react';

import { GM, GMT, GM_TONE_BADGE, GM_HEX, type GmTone } from '@/lib/ui/gm-tokens';
import { PLAN_DEFAULTS, type Entitlements, type PlanId } from '@/lib/contracts/rbac.contract';
import type { CatalogCategoryId, CatalogResponse } from '@/lib/contracts/catalog.contract';
import { defineGmDict, useGmT } from '@/lib/i18n/gm-core';
import { AsyncBoundary } from '../AsyncBoundary';
import { formatCop, useBillingPrices, useCart } from '../useCart';
import { useGmQuery } from '../useGmQuery';
import { PublicFooter, PublicHeader } from './PublicChrome';

const TIERS: ReadonlyArray<{
  id: PlanId;
  name: string;
  tagline: string;
  cta: string;
  highlight: boolean;
  icon: typeof ShieldCheck;
  iconColor: string;
}> = [
  { id: 'free', name: 'Free', tagline: 'Conoce el sistema', cta: 'Crear cuenta',
    highlight: false, icon: ShieldCheck, iconColor: GM_HEX.textSec },
  { id: 'signals', name: 'Señales Pro', tagline: 'Señales y análisis al día', cta: 'Suscribirme',
    highlight: true, icon: Radio, iconColor: GM_HEX.accent },
  { id: 'auto', name: 'Auto Premium', tagline: 'Ejecución en tu propio exchange', cta: 'Suscribirme',
    highlight: false, icon: Bolt, iconColor: GM_HEX.neg },
];

/**
 * Feature matrix — the SAME ordered rows for every tier, with the included/excluded
 * boolean DERIVED from that plan's entitlements (PLAN_DEFAULTS, the RBAC SSOT) so the
 * marketing table can never drift from what the plan actually grants. Marketing rows
 * that are not a single entitlement field (add-ons, notifications) are expressed as a
 * predicate over the plan's real entitlements (both gated on paid = signals_realtime).
 */
const PLAN_FEATURES: ReadonlyArray<{ label: string; includes: (e: Entitlements) => boolean }> = [
  { label: 'Análisis y forecasting (contenido base)', includes: () => true },
  { label: 'Análisis y forecasting al día (sin retraso)',
    includes: (e) => e.analysis_delay_days === 0 && e.forecast_delay_hours === 0 },
  { label: 'Señales en vivo (USD/COP)', includes: (e) => e.signals_realtime },
  { label: 'Add-ons por activo (Oro / BTC)', includes: (e) => e.signals_realtime },
  { label: 'Notificaciones email / telegram', includes: (e) => e.signals_realtime },
  { label: 'Ejecución automática (SignalBridge, paper-first)', includes: (e) => e.execution.enabled },
];

// ── Add-ons por activo (prototipo Var B l.1405-1421) ─────────────────────────

/** Icono/tono por clase de activo — mismo mapping visual que CatalogView. */
const ADDON_CLASS_META: Record<CatalogCategoryId, { icon: typeof Coins; tone: GmTone }> = {
  fx: { icon: Coins, tone: 'accent' },
  crypto: { icon: Bitcoin, tone: 'warn' },
  equity_index: { icon: LineChart, tone: 'info' },
  commodity: { icon: Gem, tone: 'pos' },
};

const ADDON_DICT = defineGmDict({
  es: {
    title: 'Add-ons por activo',
    sub: 'Añade activos individuales a tu plan Señales o Auto. El catálogo sale del registry publicado.',
    empty: 'Sin activos disponibles',
    emptyBody: 'El registry aún no publica activos disponibles como add-on.',
    add: 'añadir al carrito',
    inCart: 'En carrito',
    unlocked: 'Desbloqueado',
    guestCta: 'Crear cuenta',
    guestBody: 'Crea tu cuenta para ver el catálogo de activos y añadir add-ons a tu plan.',
    priceSoon: 'precio en la pasarela',
  },
  en: {
    title: 'Per-asset add-ons',
    sub: 'Add individual assets to your Signals or Auto plan. The catalog derives from the published registry.',
    empty: 'No assets available',
    emptyBody: 'The registry does not publish add-on assets yet.',
    add: 'add to cart',
    inCart: 'In cart',
    unlocked: 'Unlocked',
    guestCta: 'Create account',
    guestBody: 'Create your account to browse the asset catalog and add add-ons to your plan.',
    priceSoon: 'price at checkout',
  },
});

/**
 * Sección de add-ons: con sesión, cards desde GET /api/catalog (SSOT — cero
 * listas duplicadas en el front) con precio honesto ('—' si el backend no
 * publica monto) y botón "añadir al carrito" (useCart, compartido con el
 * drawer). Sin sesión NO se inventa la lista (el endpoint es authenticated,
 * deny-by-default): CTA de registro.
 */
function PricingAddons({ authed, onRegister }: { authed: boolean; onRegister: () => void }) {
  const t = useGmT(ADDON_DICT);
  const catalog = useGmQuery<CatalogResponse>(authed ? '/api/catalog' : null, {
    onUnauthenticated: () => {}, // página pública: jamás redirige a /login
  });
  const { inCart, isPending, addItem } = useCart(authed);

  return (
    <section className={`mt-[22px] ${GM.panel} p-[22px]`} data-testid="pricing-addons">
      <div className="flex items-center gap-2.5 mb-1">
        <Blocks className={`w-[18px] h-[18px] ${GM.accent}`} aria-hidden />
        <h2 className={`m-0 text-[15px] font-bold ${GM.text}`}>{t('title')}</h2>
      </div>
      <p className={`m-0 mb-4 ${GMT.meta} ${GM.textSec}`}>{t('sub')}</p>

      {!authed ? (
        <div className={`${GM.panelInner} flex flex-wrap items-center gap-4 px-4 py-3.5`}>
          <p className={`m-0 flex-1 min-w-[220px] ${GMT.body} ${GM.textSec}`}>{t('guestBody')}</p>
          <button
            onClick={onRegister}
            data-testid="pricing-addons-register"
            className={`${GM.ctaPrimary} ${GM.focus} h-11 px-[18px] text-[13.5px] shrink-0`}
          >
            {t('guestCta')}
          </button>
        </div>
      ) : (
        <AsyncBoundary
          state={catalog}
          skeleton={
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3" aria-busy>
              {Array.from({ length: 3 }).map((_, i) => (
                <div key={i} className="h-[62px] rounded-xl bg-[rgba(148,163,184,.07)] border border-[rgba(148,163,184,.10)] motion-safe:animate-pulse" />
              ))}
            </div>
          }
          empty={(d) => d.assets.filter((a) => a.status === 'available').length === 0}
          emptyProps={{ title: t('empty'), body: t('emptyBody') }}
        >
          {(d) => (
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {d.assets.filter((a) => a.status === 'available').map((a) => {
                const meta = ADDON_CLASS_META[a.asset_class] ?? ADDON_CLASS_META.fx;
                const Icon = meta.icon;
                const added = inCart(a.asset_id);
                return (
                  <div
                    key={a.asset_id}
                    className={`${GM.panelInner} flex items-center gap-3 px-3.5 py-3`}
                    data-testid={`pricing-addon-${a.asset_id}`}
                  >
                    <span className={`w-9 h-9 rounded-[10px] flex items-center justify-center shrink-0 border ${GM_TONE_BADGE[meta.tone]}`}>
                      <Icon className="w-4 h-4" aria-hidden />
                    </span>
                    <div className="flex-1 min-w-0">
                      <div className={`text-[13px] font-bold ${GM.text} ${GMT.mono} truncate`}>{a.symbol}</div>
                      <div className={`${GMT.micro} ${GM.textSec} truncate`}>
                        {a.name} · <span className={GMT.mono}>
                          {/* precio honesto: solo montos publicados por el backend, jamás inventados */}
                          {a.addon_price_month != null
                            ? `$${a.addon_price_month.toLocaleString('es-CO')}/mes`
                            : `— · ${t('priceSoon')}`}
                        </span>
                      </div>
                    </div>
                    {a.entitled ? (
                      <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-[.4px] shrink-0 ${GM.posBadge}`}>
                        {t('unlocked')}
                      </span>
                    ) : (
                      <button
                        onClick={() => addItem(a.asset_id)}
                        disabled={added || isPending(a.asset_id)}
                        aria-label={`${t('add')}: ${a.symbol}`}
                        title={added ? t('inCart') : undefined}
                        className={`${GM.ctaSoft} ${GM.focus} w-11 h-11 flex items-center justify-center shrink-0 disabled:opacity-60`}
                      >
                        {added ? <Check className="w-4 h-4" aria-hidden /> : <Plus className="w-4 h-4" aria-hidden />}
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </AsyncBoundary>
      )}
    </section>
  );
}

export function PricingView() {
  const router = useRouter();
  const { status } = useSession();
  const prices = useBillingPrices(); // precios reales COP desde la SSOT (/api/billing/prices)
  const [busy, setBusy] = useState<string | null>(null);
  const [currentPlan, setCurrentPlan] = useState<PlanId | null>(null);

  // Plan actual (solo con sesión) — server-resolved entitlements, read-only.
  useEffect(() => {
    if (status !== 'authenticated') return;
    let cancelled = false;
    fetch('/api/billing/me')
      .then((r) => (r.ok ? r.json() : null))
      .then((e: { plan?: PlanId } | null) => {
        if (!cancelled && e?.plan) setCurrentPlan(e.plan);
      })
      .catch(() => { /* pública: sin plan visible */ });
    return () => { cancelled = true; };
  }, [status]);

  const subscribe = async (plan: PlanId) => {
    if (plan === 'free') return router.push('/register');
    setBusy(plan);
    try {
      const res = await fetch('/api/billing/checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plan }),
      });
      if (res.status === 401) return router.push('/login?callbackUrl=/pricing');
      const data = await res.json();
      if (res.ok && data.checkoutUrl) window.location.href = data.checkoutUrl;
      else alert(data.error ?? 'La pasarela de pagos aún no está configurada.');
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className={`min-h-screen flex flex-col ${GM.page}`} data-testid="gm-pricing">
      <PublicHeader />

      <main className="flex-1 w-full flex justify-center px-6 pt-8">
       <div className="w-full max-w-[1080px]">
        <header className="text-center mb-7">
          <h1 className={`m-0 text-[30px] font-extrabold ${GM.headline} tracking-[-.6px]`}>
            Planes y precios
          </h1>
          <p className={`mt-2.5 mx-auto mb-0 text-[14px] ${GM.textSec} max-w-[620px] leading-[1.6]`}>
            Señales cuantitativas multi-activo con gestión de riesgo institucional. Las métricas
            publicadas corresponden al desempeño <strong className={GM.textStrong}>forward de producción</strong>.
          </p>
        </header>

        <div className="grid md:grid-cols-3 gap-4">
          {TIERS.map((tier) => {
            const Icon = tier.icon;
            const isCurrent = currentPlan === tier.id;
            const ent = PLAN_DEFAULTS[tier.id];
            const priceCop = prices.planCop(tier.id);
            return (
              <section
                key={tier.id}
                data-testid={`pricing-tier-${tier.id}`}
                className={`relative rounded-[18px] p-6 flex flex-col border ${
                  tier.highlight
                    ? 'border-[rgba(34,211,238,.45)] bg-[rgba(34,211,238,.05)] shadow-[0_12px_40px_rgba(34,211,238,.12)]'
                    : 'border-[rgba(148,163,184,.14)] bg-[rgba(14,21,38,.6)]'
                }`}
              >
                {tier.highlight && !isCurrent && (
                  <span className="absolute top-3.5 right-4 text-[9.5px] font-extrabold tracking-[.6px] text-[var(--gm-on-accent)] bg-[var(--gm-accent)] px-2.5 py-[3px] rounded-full">
                    MÁS POPULAR
                  </span>
                )}
                {isCurrent && (
                  <span
                    data-testid="pricing-current-plan"
                    className={`absolute top-3.5 right-4 rounded-full px-2.5 py-[3px] text-[9.5px] font-extrabold tracking-[.6px] ${GM.posBadge}`}
                  >
                    TU PLAN ACTUAL
                  </span>
                )}

                <div className="flex items-center gap-2.5 mb-1">
                  <Icon className="w-[22px] h-[22px]" style={{ color: tier.iconColor }} aria-hidden />
                  <h2 className={`m-0 text-[18px] font-extrabold ${GM.headline}`}>{tier.name}</h2>
                </div>
                <p className={`m-0 ${GMT.meta} ${GM.textSec} mb-4`}>{tier.tagline}</p>
                <p className={`m-0 text-[26px] font-extrabold ${GM.text} font-mono tabular-nums`}>
                  {tier.id === 'free' ? 'Gratis' : priceCop != null ? formatCop(priceCop) : '—'}
                  {tier.id !== 'free' && (
                    <span className={`text-[12px] font-medium ${GM.textMuted}`}>/mes</span>
                  )}
                </p>

                <ul className="mt-5 space-y-2.5 flex-1 list-none p-0 m-0">
                  {PLAN_FEATURES.map((f) => {
                    const included = f.includes(ent);
                    return (
                      <li key={f.label} className="flex items-start gap-2">
                        {included
                          ? <Check className="w-4 h-4 mt-0.5 text-[var(--gm-pos)] shrink-0" aria-hidden />
                          : <X className="w-4 h-4 mt-0.5 text-[var(--gm-text-disabled)] shrink-0" aria-hidden />}
                        <span className={`text-[12.5px] leading-[1.4] ${included ? GM.textStrong : GM.textFaint}`}>
                          {f.label}
                        </span>
                      </li>
                    );
                  })}
                </ul>

                <button
                  onClick={() => subscribe(tier.id)}
                  disabled={busy === tier.id || isCurrent}
                  data-testid={`pricing-cta-${tier.id}`}
                  className={`mt-5 h-11 text-[13.5px] font-bold disabled:opacity-50 ${GM.focus} ${
                    tier.highlight ? GM.ctaPrimary : GM.ctaGhost
                  }`}
                >
                  {isCurrent ? 'Plan activo' : busy === tier.id ? 'Redirigiendo…' : tier.cta}
                </button>
              </section>
            );
          })}
        </div>

        {/* Add-ons por activo (prototipo l.1405-1421) — catálogo SSOT + carrito compartido */}
        <PricingAddons
          authed={status === 'authenticated'}
          onRegister={() => router.push('/register')}
        />

        <div className={`mt-6 text-center ${GMT.micro} ${GM.textMuted} max-w-[620px] mx-auto leading-[1.6] space-y-1.5`}>
          <p className="m-0">
            El tier Auto opera <strong className={GM.textStrong}>paper-first</strong>: 4 semanas
            simuladas antes de habilitar dinero real, con kill switch propio.
          </p>
          <p className="m-0">
            Contenido informativo y educativo; no constituye asesoría financiera. Rendimientos
            pasados no garantizan resultados futuros.
          </p>
        </div>
       </div>
      </main>

      <PublicFooter />
    </div>
  );
}

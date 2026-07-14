'use client';

/**
 * CartView — página /cart del GlobalMarkets Terminal (CTR-FE-BE-001 §4.3;
 * prototipo "Var B" líneas 306–362): grid 1.5fr/1fr con selector de plan
 * (radio), lista de add-ons con quitar, y panel de total sticky (top-74) con
 * CTA gradiente "Ir a pagar" + disclaimer.
 *
 * Estado/acciones 100% en `useCart` (compartido con CartDrawer — DRY). Montos:
 * los precios COP del backend son placeholders (spec §B.5) ⇒ '—' honesto; el
 * monto real se confirma en la pasarela. El carrito NO se limpia al checkout
 * (webhook, rbac.md regla 7). Paleta gm-tokens; strings CART_DICT (ES/EN).
 */
import { useRouter } from 'next/navigation';
import { Bitcoin, Coins, Gem, LineChart, Lock, Plus, Trash2 } from 'lucide-react';

import type { CatalogCategoryId } from '@/lib/contracts/catalog.contract';
import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT, GM_TONE_BADGE, type GmTone } from '@/lib/ui/gm-tokens';
import { AsyncBoundary } from '../AsyncBoundary';
import { CART_DICT, formatCop, useCart, useCartPlanOptions } from '../useCart';

/** Icono por clase de activo (mismo mapping visual que CatalogView). */
const CLASS_META: Record<CatalogCategoryId, { icon: typeof Coins; tone: GmTone }> = {
  fx: { icon: Coins, tone: 'accent' },
  crypto: { icon: Bitcoin, tone: 'warn' },
  equity_index: { icon: LineChart, tone: 'info' },
  commodity: { icon: Gem, tone: 'pos' },
};

export default function CartView() {
  const router = useRouter();
  const t = useGmT(CART_DICT);
  const {
    cart, count, removeItem,
    plan, setPlan, checkingOut, checkoutError, checkout,
  } = useCart();
  const planOptions = useCartPlanOptions();

  return (
    <div className="max-w-[860px] mx-auto motion-safe:animate-in motion-safe:fade-in" data-testid="gm-cart-view">
      {/* header: título + count mono (prototipo l.309-312) */}
      <div className="flex items-center gap-3 mb-[22px]">
        <h1 className={`m-0 ${GMT.h1} ${GM.headline}`}>{t('title')}</h1>
        <span className={`${GMT.meta} font-bold ${GM.textMuted} ${GMT.mono}`}>{count}</span>
      </div>

      <AsyncBoundary
        state={cart}
        empty={(d) => d.items.length === 0}
        emptyProps={{
          title: t('empty'),
          body: t('emptyHint'),
          action: (
            <button
              onClick={() => router.push('/catalog')}
              className={`${GM.ctaSoft} ${GM.focus} inline-flex items-center gap-1.5 h-11 px-[18px] text-[13.5px]`}
            >
              <Plus className="w-4 h-4" aria-hidden /> {t('goCatalog')}
            </button>
          ),
        }}
      >
        {(d) => {
          // Total real: precio del plan (SSOT) + Σ add-ons del carrito. Si algún
          // add-on no tiene precio publicado, el total es indeterminado ⇒ '—' honesto.
          const planCop = plan === 'free' ? 0 : planOptions.find((p) => p.id === plan)?.priceCop ?? null;
          const missingAddonPrice = d.items.some((ci) => ci.addon_price_month == null);
          const addonsTotal = d.items.reduce((s, ci) => s + (ci.addon_price_month ?? 0), 0);
          const grandTotal = planCop == null || missingAddonPrice ? null : planCop + addonsTotal;
          return (
          <div className="grid md:grid-cols-[1.5fr_1fr] gap-4 items-start">
            {/* columna izquierda: plan + add-ons */}
            <div className="flex flex-col gap-4">
              {/* selector de plan (radio) */}
              <section className={`${GM.panel} p-[18px]`}>
                <div className={`${GMT.panelTitle} ${GM.textStrong} mb-3`}>{t('planSection')}</div>
                <div className="flex flex-col gap-2" role="radiogroup" aria-label="plan">
                  {planOptions.map((p) => {
                    const active = plan === p.id;
                    return (
                      <button
                        key={p.id}
                        role="radio"
                        aria-checked={active}
                        onClick={() => setPlan(p.id)}
                        data-testid={`cart-plan-${p.id}`}
                        className={`${active ? GM.ctaSoft : GM.ctaGhost} ${GM.focus}
                          flex items-center gap-3 px-3.5 py-3 min-h-[44px] text-left`}
                      >
                        <span className={`w-4 h-4 rounded-full border flex items-center justify-center shrink-0
                          ${active ? 'border-[var(--gm-accent)]' : 'border-[rgba(148,163,184,.4)]'}`} aria-hidden>
                          {active && <span className="w-2 h-2 rounded-full bg-[var(--gm-accent)]" />}
                        </span>
                        <span className="flex-1 min-w-0">
                          <span className={`block text-[13.5px] font-bold ${GM.text}`}>{p.name}</span>
                          <span className={`block ${GMT.micro} ${GM.textSec}`}>{p.tagline}</span>
                        </span>
                        <span className={`${GMT.meta} font-bold ${GM.textStrong} ${GMT.mono} shrink-0`}>{p.price}</span>
                      </button>
                    );
                  })}
                </div>
              </section>

              {/* add-ons con quitar */}
              <section className={`${GM.panel} p-[18px]`}>
                <div className={`${GMT.panelTitle} ${GM.textStrong} mb-3`}>{t('addonsSection')}</div>
                <div className="flex flex-col gap-2">
                  {d.items.map((ci) => {
                    const meta = CLASS_META[ci.asset_class ?? 'fx'] ?? CLASS_META.fx;
                    const Icon = meta.icon;
                    return (
                      <div key={ci.asset_id}
                        className={`${GM.panelInner} flex items-center gap-3 px-[13px] py-3`}
                        data-testid={`cart-page-item-${ci.asset_id}`}>
                        <span className={`w-9 h-9 rounded-[10px] flex items-center justify-center shrink-0 border ${GM_TONE_BADGE[meta.tone]}`}>
                          <Icon className="w-4 h-4" aria-hidden />
                        </span>
                        <div className="flex-1 min-w-0">
                          <div className={`text-[13px] font-bold ${GM.text} ${GMT.mono} truncate`}>{ci.symbol}</div>
                          <div className={`${GMT.micro} ${GM.textSec} truncate`}>{ci.name}</div>
                        </div>
                        <span className={`${GMT.meta} font-bold ${GM.textStrong} ${GMT.mono}`}>
                          {ci.addon_price_month != null ? `$${ci.addon_price_month.toLocaleString('es-CO')}` : '—'}
                        </span>
                        <button
                          onClick={() => removeItem(ci.asset_id)}
                          aria-label={`${t('remove')} ${ci.symbol}`}
                          className={`${GM.ctaDanger} ${GM.focus} w-11 h-11 flex items-center justify-center shrink-0`}
                        >
                          <Trash2 className="w-4 h-4" aria-hidden />
                        </button>
                      </div>
                    );
                  })}
                </div>
              </section>
            </div>

            {/* panel total sticky (prototipo: sticky top-74) */}
            <aside className={`${GM.panel} p-5 md:sticky md:top-[74px]`} data-testid="cart-total-panel">
              <div className={`${GMT.panelTitle} ${GM.textStrong} mb-3.5`}>{t('total')}</div>
              <div className="flex items-center justify-between mb-2">
                <span className={`${GMT.meta} ${GM.textSec}`}>
                  {t('planLine')} ({planOptions.find((p) => p.id === plan)?.name})
                </span>
                <span className={`${GMT.meta} ${GM.textStrong} ${GMT.mono}`}>
                  {plan === 'free' ? t('freePrice') : formatCop(planCop)}
                </span>
              </div>
              <div className="flex items-center justify-between mb-3">
                <span className={`${GMT.meta} ${GM.textSec}`}>{t('addonLine')} ({count})</span>
                <span className={`${GMT.meta} ${GM.textStrong} ${GMT.mono}`}>
                  {missingAddonPrice ? '—' : formatCop(addonsTotal)}
                </span>
              </div>
              <div className="flex items-baseline justify-between pt-3 border-t border-[rgba(148,163,184,.10)] mb-4">
                <span className={`text-[14px] font-bold ${GM.headline}`}>{t('total')}</span>
                <span className={`text-[24px] font-extrabold ${GM.accent} ${GMT.mono}`}>
                  {grandTotal == null ? '—' : formatCop(grandTotal)}
                  <span className={`${GMT.meta} ${GM.textMuted} font-medium`}>{t('month')}</span>
                </span>
              </div>
              {checkoutError && (
                <p className={`${GMT.micro} ${GM.neg} mb-3`} role="alert">{checkoutError}</p>
              )}
              <button
                onClick={checkout}
                disabled={plan === 'free' || checkingOut}
                data-testid="cart-checkout"
                className={`${GM.ctaPrimary} ${GM.focus} w-full h-[46px] flex items-center justify-center gap-2 text-[14px]
                  disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                <Lock className="w-4 h-4" aria-hidden />
                {checkingOut
                  ? t('redirecting')
                  : plan === 'free'
                    ? t('choosePaid')
                    : grandTotal != null
                      ? `${t('checkout')} · ${formatCop(grandTotal)}`
                      : t('checkout')}
              </button>
              <p className={`mt-3 mb-0 ${GMT.micro} ${GM.textMuted} text-center leading-relaxed`}>
                {t('disclaimer')}
              </p>
            </aside>
          </div>
          );
        }}
      </AsyncBoundary>
    </div>
  );
}

'use client';

/**
 * CartDrawer — drawer lateral del carrito (CTR-FE-BE-001 §4.3; prototipo
 * "Var B" líneas ~1560–1610): overlay + aside derecho con lista de add-ons
 * (quitar), selector de plan (free/signals/auto), total sticky y CTA
 * "Ir a pagar" → POST /api/cart/checkout → redirect a la URL del proveedor.
 *
 * Estado/acciones: TODO vive en `useCart` (compartido con /cart CartView —
 * DRY); este componente es solo la piel del drawer + link "Ver carrito
 * completo" → /cart. Montos: los precios COP del backend son placeholders
 * explícitos (spec §B.5) — se muestra '—' y el monto real se confirma en la
 * pasarela; cero números inventados. Paleta 100% gm-tokens; strings vía
 * CART_DICT (ES/EN).
 */
import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowRight, Lock, Plus, ShoppingCart, Trash2, X } from 'lucide-react';

import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT } from '@/lib/ui/gm-tokens';
import { AsyncBoundary } from './AsyncBoundary';
import { CART_DICT, useCart, useCartPlanOptions } from './useCart';

export function CartDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const router = useRouter();
  const t = useGmT(CART_DICT);
  // enabled=open ⇒ sin fetch cerrado y (re)carga en cada apertura.
  const {
    cart, items, count, removeItem,
    plan, setPlan, checkingOut, checkoutError, checkout,
  } = useCart(open);
  const planOptions = useCartPlanOptions();

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  const goCatalog = () => { onClose(); router.push('/catalog'); };
  const goFullCart = () => { onClose(); router.push('/cart'); };

  return (
    <>
      {/* overlay */}
      <button
        aria-label="cerrar carrito"
        onClick={onClose}
        className="fixed inset-0 z-[var(--z-backdrop)] bg-[rgba(3,6,14,.6)] backdrop-blur-[3px] motion-safe:animate-in motion-safe:fade-in"
      />
      {/* drawer */}
      <aside
        role="dialog"
        aria-modal="true"
        aria-label={t('title')}
        className={`fixed top-0 right-0 bottom-0 z-[var(--z-drawer)] w-[420px] max-w-[92vw] flex flex-col
          bg-[var(--gm-surface-drawer)] border-l border-[rgba(148,163,184,.14)] shadow-[-20px_0_60px_rgba(0,0,0,.5)]
          motion-safe:animate-in motion-safe:slide-in-from-right motion-safe:duration-200`}
        data-testid="cart-drawer"
      >
        {/* header */}
        <div className="px-[22px] py-5 border-b border-[rgba(148,163,184,.10)] flex items-center justify-between gap-2">
          <div className="flex items-center gap-2.5 min-w-0">
            <ShoppingCart className={`w-[18px] h-[18px] ${GM.accent}`} aria-hidden />
            <h2 className={`m-0 ${GMT.h2} ${GM.headline}`}>{t('title')}</h2>
            <span className={`${GMT.meta} font-bold ${GM.textMuted} ${GMT.mono}`}>{count}</span>
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            <button
              onClick={goFullCart}
              data-testid="cart-drawer-full-link"
              className={`${GM.ctaGhost} ${GM.focus} h-11 px-3 flex items-center gap-1.5 ${GMT.micro} font-bold`}
            >
              {t('viewFull')} <ArrowRight className="w-3.5 h-3.5" aria-hidden />
            </button>
            <button onClick={onClose} aria-label="cerrar"
              className={`${GM.ctaGhost} ${GM.focus} w-8 h-8 flex items-center justify-center`}>
              <X className="w-4 h-4" aria-hidden />
            </button>
          </div>
        </div>

        {/* body */}
        <div className="flex-1 overflow-y-auto px-[22px] py-[18px]">
          <AsyncBoundary
            state={cart}
            empty={(d) => d.items.length === 0}
            emptyProps={{
              title: t('empty'),
              body: t('emptyHint'),
              action: (
                <button onClick={goCatalog}
                  className={`${GM.ctaSoft} ${GM.focus} inline-flex items-center gap-1.5 h-10 px-4 text-[13px]`}>
                  <Plus className="w-4 h-4" aria-hidden /> {t('goCatalog')}
                </button>
              ),
            }}
          >
            {(d) => (
              <>
                {/* plan */}
                <div className={`${GMT.label} ${GM.textMuted} mb-2.5`}>{t('planSection')}</div>
                <div className="flex flex-col gap-2 mb-5" role="radiogroup" aria-label="plan">
                  {planOptions.map((p) => {
                    const active = plan === p.id;
                    return (
                      <button
                        key={p.id}
                        role="radio"
                        aria-checked={active}
                        onClick={() => setPlan(p.id)}
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

                {/* add-ons */}
                <div className={`${GMT.label} ${GM.textMuted} mb-2.5`}>{t('addonsSection')}</div>
                <div className="flex flex-col gap-2">
                  {d.items.map((ci) => (
                    <div key={ci.asset_id}
                      className={`${GM.panelInner} flex items-center gap-3 px-3.5 py-3`}
                      data-testid={`cart-item-${ci.asset_id}`}>
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
                  ))}
                </div>
              </>
            )}
          </AsyncBoundary>
        </div>

        {/* footer / total — solo con items */}
        {count > 0 && (
          <div className="px-[22px] py-[18px] border-t border-[rgba(148,163,184,.12)] bg-[rgba(6,10,20,.5)]">
            <div className="flex items-center justify-between mb-1">
              <span className={`${GMT.meta} ${GM.textSec}`}>
                {t('planLine')} ({planOptions.find((p) => p.id === plan)?.name})
              </span>
              <span className={`${GMT.meta} ${GM.textStrong} ${GMT.mono}`}>
                {plan === 'free' ? t('freePrice') : '—'}
              </span>
            </div>
            <div className="flex items-center justify-between mb-3">
              <span className={`${GMT.meta} ${GM.textSec}`}>{t('addonLine')} ({count})</span>
              <span className={`${GMT.meta} ${GM.textStrong} ${GMT.mono}`}>—</span>
            </div>
            <div className="flex items-baseline justify-between pt-3 border-t border-[rgba(148,163,184,.10)] mb-4">
              <span className={`text-[14px] font-bold ${GM.headline}`}>{t('total')}</span>
              <span className={`text-[22px] font-extrabold ${GM.accent} ${GMT.mono}`}>
                —<span className={`${GMT.meta} ${GM.textMuted} font-medium`}>{t('month')}</span>
              </span>
            </div>
            {checkoutError && (
              <p className={`${GMT.micro} ${GM.neg} mb-3`} role="alert">{checkoutError}</p>
            )}
            <button
              onClick={checkout}
              disabled={plan === 'free' || checkingOut}
              className={`${GM.ctaPrimary} ${GM.focus} w-full h-[46px] flex items-center justify-center gap-2 text-[14px]
                disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              <Lock className="w-4 h-4" aria-hidden />
              {checkingOut ? t('redirecting') : plan === 'free' ? t('choosePaid') : t('checkout')}
            </button>
            <p className={`mt-3 mb-0 ${GMT.micro} ${GM.textMuted} text-center leading-relaxed`}>
              {t('disclaimer')}
            </p>
          </div>
        )}
      </aside>
    </>
  );
}

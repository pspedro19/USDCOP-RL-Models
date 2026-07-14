'use client';

/**
 * useCart — estado/acciones compartidas del carrito (CTR-FE-BE-001 §4.3).
 *
 * Fuente única para TODAS las superficies del carrito (CartDrawer, /cart
 * CartView, sección add-ons de PricingView): GET /api/cart, add/remove con
 * CART_CHANGED_EVENT (las demás superficies se refrescan solas), plan
 * seleccionado y checkout POST /api/cart/checkout → redirect a la pasarela.
 *
 * Regla 7 (rbac.md): el carrito NO se limpia en el checkout — la verdad es el
 * webhook del proveedor; aquí solo se redirige a la pasarela.
 *
 * i18n: `CART_DICT` (defineGmDict ES+EN) también es compartido para que el
 * drawer y la página usen exactamente las mismas cadenas (DRY).
 */
import { useCallback, useEffect, useMemo, useState } from 'react';

import { apiFetch, ClientApiError } from '@/lib/api/gm-client';
import {
  CART_CHANGED_EVENT,
  type BillingPricesResponse,
  type CartCheckoutResponse,
  type CartItem,
  type CartResponse,
} from '@/lib/contracts/catalog.contract';
import type { PlanId } from '@/lib/contracts/rbac.contract';
import { defineGmDict, useGmT } from '@/lib/i18n/gm-core';
import type { AsyncState } from './AsyncBoundary';
import { useGmQuery } from './useGmQuery';

// ─────────────────────────────────────────────────────────── i18n (compartido)

export const CART_DICT = defineGmDict({
  es: {
    title: 'Carrito',
    empty: 'Tu carrito está vacío',
    emptyHint: 'Añade activos desde el catálogo para ampliar tu plan.',
    goCatalog: 'Ir al catálogo',
    planSection: 'Tu plan',
    addonsSection: 'Add-ons de activos',
    planLine: 'Plan',
    addonLine: 'Add-ons',
    total: 'Total',
    month: '/mes',
    checkout: 'Ir a pagar',
    redirecting: 'Redirigiendo…',
    choosePaid: 'Elige un plan de pago',
    freePrice: 'Gratis',
    viewFull: 'Ver carrito completo',
    remove: 'quitar',
    checkoutFailed: 'No se pudo iniciar el pago.',
    disclaimer:
      'El monto final se confirma en la pasarela de pago. Cancela cuando quieras. ' +
      'La activación llega tras la confirmación del proveedor.',
    planFreeTagline: 'Contenido con retraso · sin señales en vivo',
    planSignalsTagline: 'Señales en tiempo real por activo',
    planAutoTagline: 'Ejecución automatizada paper-first',
  },
  en: {
    title: 'Cart',
    empty: 'Your cart is empty',
    emptyHint: 'Add assets from the catalog to extend your plan.',
    goCatalog: 'Go to catalog',
    planSection: 'Your plan',
    addonsSection: 'Asset add-ons',
    planLine: 'Plan',
    addonLine: 'Add-ons',
    total: 'Total',
    month: '/mo',
    checkout: 'Checkout',
    redirecting: 'Redirecting…',
    choosePaid: 'Pick a paid plan',
    freePrice: 'Free',
    viewFull: 'View full cart',
    remove: 'remove',
    checkoutFailed: 'Could not start the payment.',
    disclaimer:
      'The final amount is confirmed at the payment gateway. Cancel anytime. ' +
      'Activation arrives after the provider confirms.',
    planFreeTagline: 'Delayed content · no live signals',
    planSignalsTagline: 'Real-time signals per asset',
    planAutoTagline: 'Paper-first automated execution',
  },
});

// ─────────────────────────────────────────────────────────── prices (SSOT)

const COP_FMT = new Intl.NumberFormat('es-CO', {
  style: 'currency', currency: 'COP', maximumFractionDigits: 0,
});

/** Format a whole-COP amount (es-CO). `null`/`undefined` ⇒ '—' (never invent a price). */
export function formatCop(cop: number | null | undefined): string {
  return typeof cop === 'number' ? COP_FMT.format(cop) : '—';
}

export interface BillingPrices {
  /** Monthly plan price in whole COP; 0 for free, `null` while unknown/unloaded. */
  planCop: (plan: PlanId) => number | null;
  /** asset_id → add-on monthly price (whole COP); only published assets present. */
  addons: Record<string, number>;
  loading: boolean;
}

/**
 * Shared read of GET /api/billing/prices (the public billing SSOT). One hook for
 * every price-showing surface (Pricing tiers, Cart plan rows + total) so the
 * front never hardcodes an amount. Public endpoint ⇒ never redirects to /login.
 */
export function useBillingPrices(): BillingPrices {
  const q = useGmQuery<BillingPricesResponse>('/api/billing/prices', { onUnauthenticated: () => {} });
  return useMemo(() => {
    const map = new Map((q.data?.plans ?? []).map((p) => [p.plan, p.price_month_cop] as const));
    return {
      planCop: (plan: PlanId) => (map.has(plan) ? map.get(plan)! : plan === 'free' ? 0 : null),
      addons: q.data?.addons ?? {},
      loading: q.loading,
    };
  }, [q.data, q.loading]);
}

// ─────────────────────────────────────────────────────────── plan options

export interface CartPlanOption {
  id: PlanId;
  name: string;
  tagline: string;
  /** Formatted price for display ('Gratis' for free, '—' until loaded). */
  price: string;
  /** Numeric monthly price in whole COP (0 for free, `null` if unknown). */
  priceCop: number | null;
}

/** Opciones de plan localizadas con precio real desde la SSOT (/api/billing/prices). */
export function useCartPlanOptions(): CartPlanOption[] {
  const t = useGmT(CART_DICT);
  const prices = useBillingPrices();
  return useMemo(() => {
    const opt = (id: PlanId, name: string, tagline: string): CartPlanOption => {
      const priceCop = prices.planCop(id);
      return { id, name, tagline, priceCop, price: id === 'free' ? t('freePrice') : formatCop(priceCop) };
    };
    return [
      opt('free', 'Free', t('planFreeTagline')),
      opt('signals', 'Signals', t('planSignalsTagline')),
      opt('auto', 'Auto', t('planAutoTagline')),
    ];
  }, [t, prices]);
}

// ─────────────────────────────────────────────────────────── hook

export interface CartController {
  /** Estado async crudo — se enchufa 1:1 en <AsyncBoundary state={…}>. */
  cart: AsyncState<CartResponse>;
  items: CartItem[];
  count: number;
  /** true mientras hay una mutación add/remove en vuelo para ese asset. */
  isPending: (assetId: string) => boolean;
  inCart: (assetId: string) => boolean;
  addItem: (assetId: string) => Promise<void>;
  removeItem: (assetId: string) => Promise<void>;
  /** Plan elegido para el checkout (default: signals, el destacado). */
  plan: PlanId;
  setPlan: (p: PlanId) => void;
  checkingOut: boolean;
  checkoutError: string | null;
  /** POST /api/cart/checkout → redirect. El carrito NO se limpia (webhook, regla 7). */
  checkout: () => Promise<void>;
}

/**
 * @param enabled — con `false` no se hace fetch (drawer cerrado / sin sesión);
 *   al pasar a `true` se (re)carga, conservando el comportamiento previo del
 *   drawer de recargar en cada apertura.
 */
export function useCart(enabled: boolean = true): CartController {
  const t = useGmT(CART_DICT);
  const cart = useGmQuery<CartResponse>(enabled ? '/api/cart' : null, {
    // El carrito nunca fuerza redirect a /login (mismo criterio que el badge del shell).
    onUnauthenticated: () => {},
  });
  const [plan, setPlan] = useState<PlanId>('signals');
  const [pending, setPending] = useState<Set<string>>(new Set());
  const [checkingOut, setCheckingOut] = useState(false);
  const [checkoutError, setCheckoutError] = useState<string | null>(null);

  // Cualquier superficie que mute el carrito emite CART_CHANGED_EVENT: todas
  // las instancias del hook se re-sincronizan sin prop-drilling.
  useEffect(() => {
    if (!enabled) return;
    const onChange = () => cart.reload();
    window.addEventListener(CART_CHANGED_EVENT, onChange);
    return () => window.removeEventListener(CART_CHANGED_EVENT, onChange);
  }, [enabled, cart.reload]); // eslint-disable-line react-hooks/exhaustive-deps

  const items = cart.data?.items ?? [];
  const ids = useMemo(() => new Set(items.map((i) => i.asset_id)), [cart.data]); // eslint-disable-line react-hooks/exhaustive-deps

  const mutate = useCallback(async (assetId: string, fn: () => Promise<unknown>) => {
    setPending((p) => new Set(p).add(assetId));
    try {
      await fn();
      window.dispatchEvent(new CustomEvent(CART_CHANGED_EVENT)); // ⇒ reload propio + resto
    } catch {
      /* la recarga posterior refleja el estado real del servidor */
    } finally {
      setPending((p) => { const n = new Set(p); n.delete(assetId); return n; });
    }
  }, []);

  const addItem = useCallback(
    (assetId: string) =>
      mutate(assetId, () =>
        apiFetch('/api/cart', { method: 'POST', body: JSON.stringify({ asset_id: assetId }) })),
    [mutate],
  );

  const removeItem = useCallback(
    (assetId: string) =>
      mutate(assetId, () =>
        apiFetch(`/api/cart/${encodeURIComponent(assetId)}`, { method: 'DELETE' })),
    [mutate],
  );

  const checkout = useCallback(async () => {
    if (plan === 'free' || checkingOut) return;
    setCheckingOut(true);
    setCheckoutError(null);
    try {
      const { data } = await apiFetch<CartCheckoutResponse>('/api/cart/checkout', {
        method: 'POST',
        body: JSON.stringify({ plan }),
      });
      // Redirect a la pasarela. NO limpiamos el carrito: la activación (y la
      // limpieza) la decide el webhook firma-verificado del proveedor (regla 7).
      window.location.href = data.checkout_url;
    } catch (e) {
      setCheckoutError(e instanceof ClientApiError ? e.message : t('checkoutFailed'));
      setCheckingOut(false);
    }
  }, [plan, checkingOut, t]);

  return {
    cart,
    items,
    count: items.length,
    isPending: (assetId) => pending.has(assetId),
    inCart: (assetId) => ids.has(assetId),
    addItem,
    removeItem,
    plan,
    setPlan,
    checkingOut,
    checkoutError,
    checkout,
  };
}

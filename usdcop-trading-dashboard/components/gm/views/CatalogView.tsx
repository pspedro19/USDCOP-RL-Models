'use client';

/**
 * CatalogView — vista CATALOG del GlobalMarkets Terminal (CTR-FE-BE-001 §4.3;
 * prototipo "Var B" líneas 246–306): tabs por categoría con counts + grid de
 * asset cards con badge (PRÓXIMAMENTE / DESBLOQUEADO), toggle de watchlist
 * (estrella) y botón "Añadir" al carrito para activos disponibles no incluidos
 * en el plan.
 *
 * Datos: TODO sale de GET /api/catalog (registry SSOT + entitlements +
 * watchlist compuestos server-side) — cero listas de activos duplicadas en el
 * front. El estado "en carrito" sale de GET /api/cart. Mutaciones →
 * POST/DELETE /api/{watchlist,cart} y se emite CART_CHANGED_EVENT para que el
 * badge del shell se refresque. Paleta 100% vía gm-tokens (cero hex aquí).
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Bitcoin, Check, Coins, Gem, LineChart, Lock, ShoppingCart, Star,
} from 'lucide-react';

import { apiFetch } from '@/lib/api/gm-client';
import {
  CART_CHANGED_EVENT,
  type CatalogAsset,
  type CatalogCategoryId,
  type CatalogResponse,
  type CartResponse,
} from '@/lib/contracts/catalog.contract';
import { GM, GMT, GM_TONE_BADGE, type GmTone } from '@/lib/ui/gm-tokens';
import { AsyncBoundary, GmBadge, GmPageHeader, useGmQuery } from '@/components/gm';

const CLASS_META: Record<CatalogCategoryId, { icon: typeof Coins; tone: GmTone }> = {
  fx: { icon: Coins, tone: 'accent' },
  crypto: { icon: Bitcoin, tone: 'warn' },
  equity_index: { icon: LineChart, tone: 'info' },
  commodity: { icon: Gem, tone: 'pos' },
};

type Tab = 'all' | CatalogCategoryId;

export default function CatalogView() {
  const router = useRouter();
  const catalog = useGmQuery<CatalogResponse>('/api/catalog');
  const cart = useGmQuery<CartResponse>('/api/cart');
  const [tab, setTab] = useState<Tab>('all');
  const [pending, setPending] = useState<Set<string>>(new Set());

  // El drawer también muta el carrito — mantener las cards sincronizadas.
  useEffect(() => {
    const onChange = () => cart.reload();
    window.addEventListener(CART_CHANGED_EVENT, onChange);
    return () => window.removeEventListener(CART_CHANGED_EVENT, onChange);
  }, [cart.reload]); // eslint-disable-line react-hooks/exhaustive-deps

  const inCart = useMemo(
    () => new Set((cart.data?.items ?? []).map((i) => i.asset_id)),
    [cart.data],
  );

  const withPending = useCallback(async (key: string, fn: () => Promise<unknown>) => {
    setPending((p) => new Set(p).add(key));
    try {
      await fn();
    } catch {
      /* la recarga posterior refleja el estado real; sin toast por ahora */
    } finally {
      setPending((p) => { const n = new Set(p); n.delete(key); return n; });
    }
  }, []);

  const toggleWatch = (a: CatalogAsset) =>
    withPending(`watch:${a.asset_id}`, async () => {
      if (a.in_watchlist) {
        await apiFetch(`/api/watchlist/${encodeURIComponent(a.asset_id)}`, { method: 'DELETE' });
      } else {
        await apiFetch('/api/watchlist', { method: 'POST', body: JSON.stringify({ asset_id: a.asset_id }) });
      }
      catalog.reload();
    });

  const toggleCart = (a: CatalogAsset) =>
    withPending(`cart:${a.asset_id}`, async () => {
      const wasInCart = inCart.has(a.asset_id);
      if (wasInCart) {
        await apiFetch(`/api/cart/${encodeURIComponent(a.asset_id)}`, { method: 'DELETE' });
      } else {
        await apiFetch('/api/cart', { method: 'POST', body: JSON.stringify({ asset_id: a.asset_id }) });
      }
      cart.reload();
      window.dispatchEvent(new CustomEvent(CART_CHANGED_EVENT));
      // Comprar = añadir y llevar al carrito para completar el pago; quitar no navega.
      if (!wasInCart) router.push('/cart');
    });

  return (
    <div className="motion-safe:animate-in motion-safe:fade-in">
      <GmPageHeader
        kicker="Catálogo"
        title="Activos y estrategias"
        subtitle="Sigue activos gratis con la watchlist o añádelos a tu plan como add-ons. Todo el catálogo deriva del registry publicado."
      />

      <AsyncBoundary
        state={catalog}
        empty={(d) => d.assets.length === 0}
        emptyProps={{
          title: 'Catálogo vacío',
          body: 'El registry aún no publica activos — corre el pipeline de publicación.',
        }}
      >
        {(d) => {
          const tabs: Array<{ id: Tab; label: string; count: number }> = [
            { id: 'all', label: 'Todos', count: d.assets.length },
            ...d.categories.map((c) => ({ id: c.id as Tab, label: c.label, count: c.count })),
          ];
          const visible = d.assets.filter((a) => tab === 'all' || a.asset_class === tab);
          return (
            <>
              {/* Tabs de categoría (prototipo: pills con count mono) */}
              <div className="flex flex-wrap gap-2 mb-5" role="tablist" aria-label="categorías">
                {tabs.map((t) => (
                  <button
                    key={t.id}
                    role="tab"
                    aria-selected={tab === t.id}
                    onClick={() => setTab(t.id)}
                    className={`${tab === t.id ? GM.ctaSoft : GM.ctaGhost} ${GM.focus}
                      flex items-center gap-2 h-9 px-3.5 text-[13px] font-semibold`}
                  >
                    {t.label}
                    <span className={`${GMT.micro} font-bold font-mono px-[7px] py-px rounded-full ${GM.neutralBadge}`}>
                      {t.count}
                    </span>
                  </button>
                ))}
              </div>

              {visible.length === 0 ? (
                <p className={`${GMT.body} ${GM.textMuted}`}>Sin activos en esta categoría.</p>
              ) : (
                <div className="grid gap-3.5 md:grid-cols-2 xl:grid-cols-3">
                  {visible.map((a) => (
                    <AssetCard
                      key={a.asset_id}
                      asset={a}
                      inCart={inCart.has(a.asset_id)}
                      busy={pending.has(`watch:${a.asset_id}`) || pending.has(`cart:${a.asset_id}`)}
                      onWatch={() => toggleWatch(a)}
                      onCart={() => toggleCart(a)}
                    />
                  ))}
                </div>
              )}
            </>
          );
        }}
      </AsyncBoundary>
    </div>
  );
}

function AssetCard({ asset: a, inCart, busy, onWatch, onCart }: {
  asset: CatalogAsset;
  inCart: boolean;
  busy: boolean;
  onWatch: () => void;
  onCart: () => void;
}) {
  const meta = CLASS_META[a.asset_class] ?? CLASS_META.fx;
  const Icon = meta.icon;
  const soon = a.status === 'coming_soon';
  return (
    <div className={`${GM.panel} relative p-[18px] ${soon ? 'opacity-85' : ''}`}
      data-testid={`catalog-asset-${a.asset_id}`}>
      {/* Badge top-right: PRÓXIMAMENTE gana sobre DESBLOQUEADO (soon nunca es entitled) */}
      {soon && <GmBadge tone="warn" className="absolute top-3.5 right-3.5">Próximamente</GmBadge>}
      {a.entitled && <GmBadge tone="pos" className="absolute top-3.5 right-3.5">Desbloqueado</GmBadge>}

      <div className="flex items-center gap-3 mb-4">
        <span className={`w-[42px] h-[42px] rounded-[11px] flex items-center justify-center shrink-0 border ${GM_TONE_BADGE[meta.tone]}`}>
          <Icon className="w-5 h-5" aria-hidden />
        </span>
        <div className="min-w-0">
          <div className={`text-[16px] font-extrabold ${GM.headline} font-mono truncate`}>{a.symbol}</div>
          <div className={`${GMT.meta} ${GM.textSec} truncate`}>{a.name}</div>
        </div>
      </div>

      {/* Precio: sin fuente en vivo aún ⇒ '—' honesto (cero números inventados) */}
      <div className="flex items-end justify-between mb-4">
        <div>
          <div className={`${GMT.kpi} ${a.price != null ? GM.text : GM.textFaint} leading-none`}>
            {a.price != null ? a.price.toLocaleString('es-CO') : '—'}
          </div>
          <div className={`${GMT.micro} ${GM.textMuted} mt-1.5`}>
            {a.change_pct != null
              ? `${a.change_pct >= 0 ? '+' : ''}${a.change_pct.toFixed(2)}%`
              : 'precio en vivo próximamente'}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={onWatch}
          disabled={busy}
          aria-pressed={a.in_watchlist}
          className={`${a.in_watchlist ? GM.ctaSoft : GM.ctaGhost} ${GM.focus}
            flex-1 h-[38px] flex items-center justify-center gap-1.5 text-[12.5px] font-semibold disabled:opacity-60`}
        >
          <Star className="w-4 h-4" fill={a.in_watchlist ? 'currentColor' : 'none'} aria-hidden />
          {a.in_watchlist ? 'Siguiendo' : 'Seguir'}
        </button>

        {soon ? (
          <button disabled
            className={`${GM.ctaGhost} flex-1 h-[38px] flex items-center justify-center gap-1.5 text-[12.5px] font-semibold opacity-60 cursor-not-allowed`}>
            <Lock className="w-4 h-4" aria-hidden /> Próximamente
          </button>
        ) : !a.entitled ? (
          <button
            onClick={onCart}
            disabled={busy}
            className={`${inCart ? GM.ctaSoft : GM.ctaPrimary} ${GM.focus}
              flex-1 h-[38px] flex items-center justify-center gap-1.5 text-[12.5px] disabled:opacity-60`}
          >
            {inCart
              ? <><Check className="w-4 h-4" aria-hidden /> En carrito</>
              : <><ShoppingCart className="w-4 h-4" aria-hidden /> Añadir</>}
          </button>
        ) : null}
      </div>
    </div>
  );
}

'use client';

/**
 * Catálogo (CTR-ADMIN-CONSOLE-001 · pestaña Catálogo). Lista los activos del registry
 * SSOT (RegistryBuilder) — solo lectura desde la consola: el registry lo genera el
 * pipeline, así que el toggle por fila va DESHABILITADO con tooltip = `note`. Precio
 * add-on desconocido → "—" (nunca inventado).
 */
import { Store } from 'lucide-react';

import type { AdminCatalogResponse } from '@/lib/contracts/admin-console.contract';
import { COLOR, CTA, TYPE } from '@/lib/ui/tokens';

import { REFRESH, useAdminWidget } from './useAdminWidget';
import { Badge, Card, EmptyState, SkeletonRows, fmtRelative, useNow } from './ui';

const DASH = '—';

export function CatalogSection() {
  const cat = useAdminWidget<AdminCatalogResponse>('/api/admin/catalog', { refreshMs: REFRESH.catalog });
  const now = useNow(30_000);
  const d = cat.data;

  const meta = cat.updatedAt
    ? <span title={new Date(cat.updatedAt).toISOString()}>{fmtRelative(new Date(cat.updatedAt).toISOString(), now)}</span>
    : null;

  return (
    <div className="space-y-4" data-testid="admin-section-catalogo-admin">
      <Card
        title="Catálogo de activos"
        icon={<Store className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
        info={d?.note ?? 'Registry generado por el pipeline (RegistryBuilder) — solo lectura desde la consola.'}
        badge={d ? <Badge tone="neutral">{d.assets.length}</Badge> : null}
        meta={meta} stale={cat.stale}
      >
        {cat.error && !d && (
          <EmptyState
            icon={<Store className="w-8 h-8" aria-hidden />}
            cause={<>No se pudo cargar el catálogo: {cat.error}</>}
            action={<button onClick={cat.reload} className={`${CTA.primary} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Reintentar</button>}
          />
        )}
        {cat.loading && !d && <SkeletonRows rows={4} cols={5} />}
        {d && d.assets.length === 0 && !cat.error && (
          <EmptyState icon={<Store className="w-8 h-8" aria-hidden />} cause="Sin activos en el catálogo." />
        )}
        {d && d.assets.length > 0 && (
          <>
            <p className={`${TYPE.meta} mb-3`}>{d.note}</p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className={`text-left ${COLOR.textSecondary} border-b border-slate-800`}>
                    <th className="py-2 pr-3">Símbolo</th><th className="pr-3">Nombre</th>
                    <th className="pr-3">Clase</th><th className="pr-3">Estado</th>
                    <th className="pr-3 text-right">Add-on / mes</th><th className="pr-3 text-right">Publicado</th>
                  </tr>
                </thead>
                <tbody>
                  {d.assets.map((a) => (
                    <tr key={a.asset_id} className="h-10 border-b border-slate-800/50">
                      <td className={`pr-3 font-medium ${TYPE.mono} ${COLOR.textPrimary}`}>{a.symbol}</td>
                      <td className={`pr-3 ${COLOR.textSecondary}`}>{a.name}</td>
                      <td className={`pr-3 ${COLOR.textSecondary}`}>{a.asset_class}</td>
                      <td className="pr-3">
                        <Badge tone={a.status === 'available' ? 'ok' : 'warn'}>
                          {a.status === 'available' ? 'Disponible' : 'Próximamente'}
                        </Badge>
                      </td>
                      <td className={`pr-3 text-right ${TYPE.mono} ${COLOR.textSecondary}`}>
                        {a.addon_price_month == null ? DASH : `$${a.addon_price_month.toLocaleString()}`}
                      </td>
                      <td className="pr-3 text-right">
                        <label className="inline-flex items-center gap-1.5 cursor-not-allowed" title={d.note}>
                          <input
                            type="checkbox"
                            disabled
                            checked={a.status === 'available'}
                            aria-label={`publicación de ${a.symbol} (solo lectura — ${d.note})`}
                            className="cursor-not-allowed opacity-60"
                          />
                        </label>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </Card>
    </div>
  );
}

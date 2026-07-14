/**
 * GET /api/admin/catalog — pestaña Catálogo de la consola (admin:all).
 *
 * Reusa lib/services/catalog-registry.ts (la ÚNICA fuente que convierte el registry
 * SSOT en entradas de catálogo — jamás duplicar la lista). Sin PATCH: el registry lo
 * genera el pipeline (RegistryBuilder); el toggle en la UI va deshabilitado con
 * tooltip "registry generado por pipeline".
 */
import { ok } from '@/lib/api/envelope';
import { requirePermission } from '@/lib/api/relay';
import type { AdminCatalogResponse } from '@/lib/contracts/admin-console.contract';
import { allCatalogAssets } from '@/lib/services/catalog-registry';

export async function GET(req: Request) {
  const gate = requirePermission(req, 'admin:all');
  if (gate instanceof Response) return gate;

  const assets = await allCatalogAssets();
  const body: AdminCatalogResponse = {
    assets: assets.map((a) => ({
      asset_id: a.asset_id,
      symbol: a.symbol,
      name: a.name,
      asset_class: a.asset_class,
      status: a.status,
      addon_price_month: a.addon_price_month,
    })),
    mutable: false,
    note: 'Registry generado por el pipeline (RegistryBuilder) — solo lectura desde la consola.',
  };
  return ok(body, { meta: { asOf: new Date().toISOString() } });
}

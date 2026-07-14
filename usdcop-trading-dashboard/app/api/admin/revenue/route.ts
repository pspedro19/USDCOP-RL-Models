/**
 * GET /api/admin/revenue — pestaña Ingresos (admin:all).
 *
 * Métricas REALES agregadas en vivo sobre sb_users.entitlements + audit_log
 * (lib/admin/revenue.ts). Sin mocks: con 0 suscriptores de pago las cifras son
 * ceros genuinos; lo que no se mide (reembolsos, contracargos, split de
 * expansión/contracción) va null → la UI pinta "—". Los precios salen del SSOT
 * compartido con el checkout de Wompi (lib/billing/prices.ts).
 */
import { computeRevenue } from '@/lib/admin/revenue';
import { ok, fail } from '@/lib/api/envelope';
import { requirePermission } from '@/lib/api/relay';

export async function GET(req: Request) {
  const gate = requirePermission(req, 'admin:all');
  if (gate instanceof Response) return gate;

  try {
    const { revenue } = await computeRevenue();
    return ok(revenue, { meta: { asOf: new Date().toISOString() } });
  } catch (e) {
    return fail('UPSTREAM_ERROR', `revenue aggregation failed: ${String(e)}`, 503);
  }
}

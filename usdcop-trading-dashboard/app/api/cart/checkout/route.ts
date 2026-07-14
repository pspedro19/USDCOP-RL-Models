/**
 * POST /api/cart/checkout (CTR-FE-BE-001 §4.3) — turn the cart into a hosted
 * checkout session. Body: { plan: 'signals' | 'auto' }.
 *
 * DESIGN DECISION (spec asked to choose + document): instead of an internal
 * HTTP forward to POST /api/billing/checkout (which would need base-URL
 * resolution, cookie propagation and a middleware re-entry just to reach a
 * 30-line handler in the SAME process), this route calls the SAME dependency
 * that route uses — `getBillingProvider().createCheckout()` from `lib/billing`
 * (DIP boundary, CTR-RBAC-001 rule 7). The only duplicated logic is the 3-line
 * email lookup; the provider abstraction stays the single billing entry point,
 * and this route stays enveloped (`ok/fail`) unlike the legacy billing route.
 *
 * Flow: validate plan → read cart → drop already-entitled add-ons (defensive:
 * never charge for owned assets) → provider checkout (reference encodes
 * user+plan+addOns for the webhook) → return checkout_url. The cart is NOT
 * cleared here — only the payment webhook grants entitlements, and clearing is
 * deferred until the webhook confirms (re-running checkout is safe/idempotent
 * in effect: same cart ⇒ equivalent session).
 */
import { fail, ok } from '@/lib/api/envelope';
import { requireSession } from '@/lib/api/relay';
import { getEntitlements } from '@/lib/auth/entitlements';
import { getBillingProvider } from '@/lib/billing';
import { query } from '@/lib/db/postgres-client';
import type { CartCheckoutResponse } from '@/lib/contracts/catalog.contract';

export const dynamic = 'force-dynamic';

export async function POST(req: Request) {
  const gate = requireSession(req);
  if (gate instanceof Response) return gate;

  let body: { plan?: unknown };
  try {
    body = await req.json();
  } catch {
    return fail('BAD_REQUEST', 'JSON inválido.', 400);
  }
  if (body.plan !== 'signals' && body.plan !== 'auto') {
    return fail('BAD_REQUEST', "plan debe ser 'signals' o 'auto'.", 400);
  }

  let email: string | undefined;
  let cartAssets: string[];
  try {
    const [userRes, cartRes] = await Promise.all([
      query<{ email: string }>('SELECT email FROM sb_users WHERE id = $1', [gate.userId]),
      query<{ asset_id: string }>(
        'SELECT asset_id FROM user_cart WHERE user_id = $1 ORDER BY created_at ASC',
        [gate.userId],
      ),
    ]);
    email = userRes.rows[0]?.email;
    cartAssets = cartRes.rows.map((r) => r.asset_id);
  } catch {
    return fail('UPSTREAM_UNAVAILABLE', 'Base de datos no disponible.', 502);
  }
  if (!email) return fail('NOT_FOUND', 'Usuario no encontrado.', 404);

  // Defensive: never send already-owned assets to the provider.
  const entitlements = await getEntitlements(gate.userId);
  const addOnAssets = cartAssets.filter((a) => !entitlements.assets.includes(a));

  try {
    const session = await getBillingProvider().createCheckout({
      userId: gate.userId, email, plan: body.plan, addOnAssets,
    });
    const data: CartCheckoutResponse = {
      provider: session.provider,
      checkout_url: session.checkoutUrl,
      reference: session.reference,
      addon_assets: addOnAssets,
    };
    return ok(data);
  } catch (e) {
    // Provider not configured (no keys): actionable 503, never a raw 500 stack.
    return fail('BILLING_NOT_CONFIGURED', 'Pasarela de pago no configurada.', 503,
      { detail: String(e) });
  }
}

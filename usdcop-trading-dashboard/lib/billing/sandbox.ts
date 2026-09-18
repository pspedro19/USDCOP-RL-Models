/**
 * Sandbox billing provider — exercises the REAL payment path without a merchant account.
 *
 * WHY THIS EXISTS: the Wompi integration is complete but inert, because live keys require a
 * merchant account nobody can conjure. Without a second provider the purchase flow could
 * not be run end to end at all: checkout returned 503 and the webhook, the sealed-quote
 * comparison, the entitlement grant and the cart clear were reachable only from unit tests.
 *
 * WHAT IT IS NOT: a bypass. It deliberately goes through the same gauntlet as Wompi —
 * an HMAC-signed webhook whose signature is verified, a server-to-server confirmation the
 * webhook cross-checks before touching a row, and the same `unauthenticatedFields`
 * declaration. A flow that passes here passes because the checks ran, not because they
 * were skipped.
 *
 * HOW IT IS KEPT OUT OF A REAL DEPLOYMENT. Three independent switches, all required:
 * `BILLING_PROVIDER=sandbox`, an explicit `BILLING_SANDBOX_ENABLED=true`, and a
 * `BILLING_SANDBOX_SECRET`. Construction throws if any is missing, so no single stray env
 * var can quietly make purchases free.
 *
 * The enable flag is deliberately NOT `NODE_ENV`: `next start` sets `NODE_ENV=production`
 * for every production BUILD, including the one running on a staging box, so keying the
 * refusal to it would have meant "the sandbox only works under `next dev`" — which is
 * exactly where an end-to-end payment test is least useful. `APP_ENV` is the app's own
 * deployment marker and is still honoured: a box that calls itself production refuses
 * regardless of the other two switches.
 */
import { createHmac, timingSafeEqual } from 'crypto';

import type {
  AuthoritativeTransaction,
  BillingProvider,
  CheckoutRequest,
  CheckoutSession,
  TransactionLookup,
  WebhookVerification,
} from './provider';
import { encodeReference } from './provider';

/** Status vocabulary, mirroring Wompi's so the normalization logic is the same shape. */
const STATUS_MAP = {
  APPROVED: 'payment.approved',
  DECLINED: 'payment.declined',
  VOIDED: 'payment.declined',
} as const;

type SandboxStatus = keyof typeof STATUS_MAP;

/** In-memory record of what the sandbox "charged", so `fetchTransaction` has an authority. */
const issued = new Map<string, AuthoritativeTransaction>();

export function sandboxSecret(): string {
  const s = process.env.BILLING_SANDBOX_SECRET;
  if (!s) throw new Error('BILLING_SANDBOX_SECRET not configured');
  return s;
}

/** Signature over the fields a real provider signs — deliberately NOT over `reference`. */
export function signSandboxEvent(id: string, status: string, amountInCents: number): string {
  return createHmac('sha256', sandboxSecret())
    .update(`${id}|${status}|${amountInCents}`)
    .digest('hex');
}

/**
 * Record a transaction so the webhook's server-to-server confirmation can find it. Called
 * by the sandbox payment page at the moment it "charges".
 */
export function recordSandboxTransaction(t: AuthoritativeTransaction): void {
  issued.set(t.id, t);
}

export class SandboxProvider implements BillingProvider {
  readonly name = 'sandbox';

  constructor() {
    if ((process.env.APP_ENV ?? '').toLowerCase() === 'production') {
      throw new Error('the sandbox billing provider is refused on a production deployment');
    }
    if (process.env.BILLING_SANDBOX_ENABLED !== 'true') {
      throw new Error('sandbox billing requires BILLING_SANDBOX_ENABLED=true');
    }
    sandboxSecret(); // fail closed at construction, not at the first charge
  }

  async createCheckout(req: CheckoutRequest): Promise<CheckoutSession> {
    const reference = encodeReference(req.userId, req.plan, req.addOnAssets ?? []);
    return {
      provider: this.name,
      // A local page that stands in for the provider's hosted checkout.
      checkoutUrl: `/api/billing/sandbox?reference=${encodeURIComponent(reference)}`,
      reference,
    };
  }

  async verifyWebhook(rawBody: string, _headers: Headers): Promise<WebhookVerification> {
    let body: {
      id?: string; reference?: string; status?: string;
      amount_in_cents?: number; currency?: string; signature?: string;
    };
    try {
      body = JSON.parse(rawBody);
    } catch {
      return { valid: false, error: 'unparseable sandbox webhook body' };
    }

    const { id, reference, status, amount_in_cents: amount, signature } = body;
    if (!id || !reference || !status || typeof amount !== 'number' || !signature) {
      return { valid: false, error: 'sandbox webhook missing required fields' };
    }

    const expected = signSandboxEvent(id, status, amount);
    const a = Buffer.from(signature, 'utf8');
    const b = Buffer.from(expected, 'utf8');
    if (a.length !== b.length || !timingSafeEqual(a, b)) {
      return { valid: false, error: 'sandbox signature mismatch' };
    }

    const type = STATUS_MAP[status as SandboxStatus];
    if (!type) {
      // Signature good, but the status carries no transition we model. Acknowledge and
      // drop — never invent a transition the provider did not express.
      return {
        valid: true,
        ignored: { reason: `unmapped sandbox status ${status}`, reference, providerEventId: id, providerStatus: status },
      };
    }

    return {
      valid: true,
      event: {
        type,
        reference,
        amountInCents: amount,
        currency: body.currency ?? 'COP',
        providerEventId: `sandbox:${id}:${status}`,
        providerTransactionId: id,
        providerStatus: status,
        // Same honesty as the Wompi adapter: the signature covers id|status|amount, so
        // `reference` arrives unauthenticated and the webhook must confirm it against
        // the sealed quote before crediting anybody.
        unauthenticatedFields: ['reference'],
        raw: body,
      },
    };
  }

  async fetchTransaction(id: string): Promise<TransactionLookup> {
    const t = issued.get(id);
    if (!t) return { kind: 'not_found', reason: `sandbox has no transaction ${id}` };
    return { kind: 'found', transaction: t };
  }
}

/**
 * Wompi provider (Colombia) — https://docs.wompi.co
 *
 * Checkout: Wompi's hosted Web Checkout via URL parameters (public key + reference +
 * amount + integrity signature). Webhook: Wompi POSTs an event whose
 * `signature.checksum` = SHA256(<properties values concatenated> + timestamp + events_secret).
 *
 * Env: WOMPI_PUBLIC_KEY, WOMPI_EVENTS_SECRET, WOMPI_INTEGRITY_SECRET,
 *      BILLING_PRICES_COP (JSON {"signals":..cents.., "auto":..cents..}), NEXTAUTH_URL.
 */
import { createHash } from 'node:crypto';

import { planPricesCents } from './prices';
import type {
  BillingProvider, CheckoutRequest, CheckoutSession, WebhookVerification,
} from './provider';
import { encodeReference } from './provider';

// Prices are shared SSOT (lib/billing/prices.ts) so admin revenue and checkout
// never diverge. Env override: BILLING_PRICES_COP.
const pricesCopCents = planPricesCents;

export class WompiProvider implements BillingProvider {
  readonly name = 'wompi';

  async createCheckout(req: CheckoutRequest): Promise<CheckoutSession> {
    const publicKey = process.env.WOMPI_PUBLIC_KEY;
    if (!publicKey) throw new Error('WOMPI_PUBLIC_KEY not configured');

    const reference = encodeReference(req.userId, req.plan, req.addOnAssets ?? []);
    const amountInCents = Math.round(pricesCopCents()[req.plan] ?? 0);
    const currency = 'COP';

    // Integrity signature: SHA256(reference + amount + currency + integrity_secret)
    const integritySecret = process.env.WOMPI_INTEGRITY_SECRET ?? '';
    const signature = createHash('sha256')
      .update(`${reference}${amountInCents}${currency}${integritySecret}`)
      .digest('hex');

    const redirect = `${process.env.NEXTAUTH_URL ?? ''}/account/billing`;
    const params = new URLSearchParams({
      'public-key': publicKey,
      currency,
      'amount-in-cents': String(amountInCents),
      reference,
      'signature:integrity': signature,
      'redirect-url': redirect,
      'customer-data:email': req.email,
    });

    return {
      provider: this.name,
      checkoutUrl: `https://checkout.wompi.co/p/?${params.toString()}`,
      reference,
    };
  }

  async verifyWebhook(rawBody: string, _headers: Headers): Promise<WebhookVerification> {
    const secret = process.env.WOMPI_EVENTS_SECRET;
    if (!secret) return { valid: false, error: 'WOMPI_EVENTS_SECRET not configured' };

    let body: WompiEvent;
    try {
      body = JSON.parse(rawBody);
    } catch {
      return { valid: false, error: 'invalid JSON' };
    }

    const { signature, timestamp, data } = body;
    if (!signature?.checksum || !signature.properties || timestamp == null) {
      return { valid: false, error: 'missing signature fields' };
    }

    // checksum = SHA256(concat(value of each signature.properties path) + timestamp + secret)
    const concatenated = signature.properties
      .map((path) => String(getPath(body.data, path.replace(/^transaction\./, 'transaction.')) ?? ''))
      .join('');
    const expected = createHash('sha256')
      .update(`${concatenated}${timestamp}${secret}`)
      .digest('hex');
    if (expected !== signature.checksum) return { valid: false, error: 'bad checksum' };

    const tx = data?.transaction;
    const approved = body.event === 'transaction.updated' && tx?.status === 'APPROVED';
    const declined = body.event === 'transaction.updated' &&
      ['DECLINED', 'ERROR', 'VOIDED'].includes(tx?.status ?? '');

    return {
      valid: true,
      event: {
        type: approved ? 'payment.approved'
          : declined ? 'payment.declined'
          : 'subscription.cancelled',
        reference: tx?.reference ?? '',
        amountInCents: tx?.amount_in_cents,
        raw: body,
      },
    };
  }
}

interface WompiEvent {
  event: string;
  data?: { transaction?: { id?: string; status?: string; reference?: string;
                           amount_in_cents?: number } };
  signature?: { checksum: string; properties: string[] };
  timestamp?: number;
}

function getPath(obj: unknown, path: string): unknown {
  return path.split('.').reduce<unknown>(
    (acc, k) => (acc && typeof acc === 'object' ? (acc as Record<string, unknown>)[k] : undefined),
    obj,
  );
}

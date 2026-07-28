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

import { planPricesCents, addonPricesCop } from './prices';
import type {
  BillingEventType, BillingProvider, CheckoutRequest, CheckoutSession, TransactionLookup,
  WebhookVerification,
} from './provider';
import { encodeReference } from './provider';

// Prices are shared SSOT (lib/billing/prices.ts) so admin revenue and checkout
// never diverge. Env override: BILLING_PRICES_COP.
const pricesCopCents = planPricesCents;

/** Currency of every Wompi checkout this integration issues. */
const WOMPI_CURRENCY = 'COP';

/**
 * EXHAUSTIVE map `transaction.status` → normalized transition (CODEX P0-1).
 * `null` = the status is real but carries no transition we apply (the payment is
 * still in flight). A status NOT present here is unknown ⇒ ignored, never guessed.
 * Wompi statuses: https://docs.wompi.co/docs/colombia/estados-y-eventos/
 */
const WOMPI_STATUS_MAP: Record<string, BillingEventType | null> = {
  APPROVED: 'payment.approved',
  DECLINED: 'payment.declined',
  ERROR: 'payment.declined',
  VOIDED: 'payment.declined',
  PENDING: null,
};

/** Event names this integration understands. Anything else is ignored. */
const WOMPI_HANDLED_EVENTS = new Set(['transaction.updated']);

/**
 * Which `data.transaction` path backs each field of the normalized event. Wompi's
 * `signature.properties` is the EXHAUSTIVE list of paths its checksum covers; every
 * field whose path is absent from that list arrives UNAUTHENTICATED and is mutable by
 * anyone who can replay the event (CODEX CXD-056).
 *
 * In practice Wompi signs `transaction.id|status|amount_in_cents` and does NOT sign
 * `transaction.reference` — the field that decides which order, and therefore which
 * user, gets credited. No code here can fix that: two events differing only in the
 * reference produce the same, genuinely valid checksum. So we do not pretend to
 * authenticate it — we REPORT it, and the route closes the replay at the ledger
 * (`lib/billing/event-ledger.ts`, `billing_events.provider_event_id`).
 */
const EVENT_FIELD_SOURCE: Readonly<Record<string, string>> = {
  providerEventId: 'transaction.id',
  type: 'transaction.status',
  amountInCents: 'transaction.amount_in_cents',
  currency: 'transaction.currency',
  reference: 'transaction.reference',
};

function unauthenticatedFieldsOf(properties: readonly string[]): readonly string[] {
  const signed = new Set(properties);
  return Object.entries(EVENT_FIELD_SOURCE)
    .filter(([, path]) => !signed.has(path))
    .map(([field]) => field);
}

/**
 * The outbound HTTP port (CXD-059). Injected so the confirmation logic is exercised
 * end-to-end in tests with zero network, and so the transport can be swapped
 * (instrumented client, proxy) without touching the decision.
 */
export type HttpFetch = (url: string, init?: RequestInit) => Promise<Response>;

/**
 * Base URL of the query API. Wompi splits sandbox/production, and the key prefix is
 * what tells them apart (`pub_prod_…` vs `pub_test_…`). Deriving it from the key that
 * is ALREADY configured avoids a second, silently-divergent setting: pointing at
 * sandbox with production keys would make every real payment "not found" and reject
 * legitimate money. `WOMPI_API_URL` overrides for self-hosted/proxied setups.
 */
function apiBaseUrl(): string {
  const override = process.env.WOMPI_API_URL?.trim();
  if (override) return override.replace(/\/+$/, '');
  return process.env.WOMPI_PUBLIC_KEY?.trim().startsWith('pub_prod_')
    ? 'https://production.wompi.co/v1'
    : 'https://sandbox.wompi.co/v1';
}

/** Milliseconds before an unanswered confirmation counts as `unavailable` (⇒ 503, retry). */
function confirmTimeoutMs(): number {
  const raw = Number(process.env.BILLING_CONFIRM_TIMEOUT_MS);
  return Number.isFinite(raw) && raw > 0 ? raw : 5_000;
}

export class WompiProvider implements BillingProvider {
  readonly name = 'wompi';

  /**
   * `globalThis.fetch` is resolved LAZILY (not captured as a default parameter value)
   * so a test that stubs the global still routes through this provider.
   */
  constructor(private readonly http: HttpFetch = (url, init) => globalThis.fetch(url, init)) {}

  /**
   * `GET /v1/transactions/:id` — the authoritative account of the payment.
   *
   * NEVER throws and NEVER returns `found` on doubt: a timeout, a 5xx, a transport
   * error or a body we cannot parse all yield `unavailable`, which the route turns
   * into a retryable 503 with zero mutations. Only a 200 whose `data` we fully parsed
   * can confirm anything.
   */
  async fetchTransaction(providerTransactionId: string): Promise<TransactionLookup> {
    const id = providerTransactionId?.trim();
    if (!id) return { kind: 'not_found', reason: 'empty transaction id' };

    // The query endpoint is read-only and keyed by an unguessable id; the public key
    // is sent when configured because some Wompi environments expect it. It is PUBLIC
    // by definition (it also ships in the checkout URL), so this is not a secret
    // crossing a new boundary — and a 401/403 is treated as "cannot tell" below
    // rather than as a verdict, so a wrong key can never reject a real payment.
    const publicKey = process.env.WOMPI_PUBLIC_KEY?.trim();
    let res: Response;
    try {
      res = await this.http(`${apiBaseUrl()}/transactions/${encodeURIComponent(id)}`, {
        method: 'GET',
        headers: {
          accept: 'application/json',
          ...(publicKey ? { authorization: `Bearer ${publicKey}` } : {}),
        },
        signal: AbortSignal.timeout(confirmTimeoutMs()),
        cache: 'no-store',
      });
    } catch (e) {
      // Transport error / abort. We do NOT know the truth ⇒ retryable, never a verdict.
      return { kind: 'unavailable', reason: `transport: ${(e as Error)?.name ?? 'error'}` };
    }

    // ONLY an explicit 404 is the provider DENYING the transaction. Every other
    // non-2xx (401/403 misconfigured key, 429 throttling, 5xx, a proxy's 502) means we
    // could not ask — which is `unavailable` ⇒ 503 ⇒ the provider retries. Classifying
    // those as `not_found` would let a bad key or a rate limit permanently REJECT
    // payments that really happened; fail-closed must never become fail-destructive.
    if (res.status === 404) return { kind: 'not_found', reason: 'provider has no such transaction' };
    if (!res.ok) return { kind: 'unavailable', reason: `provider http ${res.status}` };

    let body: WompiTransactionResponse;
    try {
      body = JSON.parse(await res.text());
    } catch {
      return { kind: 'unavailable', reason: 'unparseable provider response' };
    }
    const tx = body?.data;
    if (!tx || typeof tx !== 'object' || typeof tx.id !== 'string') {
      return { kind: 'unavailable', reason: 'provider response missing transaction' };
    }

    const rawAmount = tx.amount_in_cents;
    const amountInCents = typeof rawAmount === 'string' ? Number(rawAmount) : rawAmount;
    return {
      kind: 'found',
      transaction: {
        id: tx.id,
        reference: typeof tx.reference === 'string' ? tx.reference : '',
        status: typeof tx.status === 'string' ? tx.status : '',
        amountInCents: Number.isFinite(amountInCents as number) ? (amountInCents as number) : undefined,
        currency: typeof tx.currency === 'string' ? tx.currency : undefined,
      },
    };
  }

  async createCheckout(req: CheckoutRequest): Promise<CheckoutSession> {
    // Fail CLOSED on incomplete configuration: an incomplete config must never
    // produce a payable URL (CODEX P0-2 — a URL signed with an empty integrity
    // secret is a URL whose amount anyone can rewrite).
    const publicKey = process.env.WOMPI_PUBLIC_KEY?.trim();
    if (!publicKey) throw new Error('WOMPI_PUBLIC_KEY not configured');
    const integritySecret = process.env.WOMPI_INTEGRITY_SECRET?.trim();
    if (!integritySecret) throw new Error('WOMPI_INTEGRITY_SECRET not configured');
    const redirect = process.env.NEXTAUTH_URL?.trim();
    if (!redirect) throw new Error('NEXTAUTH_URL not configured');

    const reference = encodeReference(req.userId, req.plan, req.addOnAssets ?? []);
    const base = Math.round(pricesCopCents()[req.plan] ?? 0);
    if (!Number.isFinite(base) || base <= 0) {
      // No published price for a PAID plan ⇒ we do not invent one (prices are an
      // operator decision, spec §B.5). Charging 0 would grant access for free.
      throw new Error(`no published price for plan '${req.plan}'`);
    }
    let addons = 0;
    for (const id of req.addOnAssets ?? []) {
      const price = addonPricesCop()[id];
      if (typeof price !== 'number' || !Number.isFinite(price) || price <= 0) {
        throw new Error(`no published price for add-on '${id}'`);
      }
      addons += Math.round(price * 100);
    }
    const amountInCents = base + addons;
    const currency = WOMPI_CURRENCY;

    // Integrity signature: SHA256(reference + amount + currency + integrity_secret)
    const signature = createHash('sha256')
      .update(`${reference}${amountInCents}${currency}${integritySecret}`)
      .digest('hex');

    const params = new URLSearchParams({
      'public-key': publicKey,
      currency,
      'amount-in-cents': String(amountInCents),
      reference,
      'signature:integrity': signature,
      'redirect-url': `${redirect}/account/billing`,
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
    const status = String(tx?.status ?? '');
    const reference = tx?.reference ?? '';
    const txId = tx?.id ?? (reference || 'unknown');
    const providerEventId = `wompi:${txId}:${status || body.event}`;

    // ── EXHAUSTIVE mapping. Anything not explicitly listed is ACKNOWLEDGED and
    // DROPPED; it is never translated into a transition the provider did not send.
    if (!WOMPI_HANDLED_EVENTS.has(body.event)) {
      return { valid: true, ignored: { reason: `unhandled event '${body.event}'`, reference, providerEventId } };
    }
    if (!(status in WOMPI_STATUS_MAP)) {
      return {
        valid: true,
        ignored: { reason: 'unknown transaction status', reference, providerEventId, providerStatus: status },
      };
    }
    const type = WOMPI_STATUS_MAP[status];
    if (type === null) {
      return {
        valid: true,
        ignored: { reason: 'no state transition (payment still in flight)', reference, providerEventId, providerStatus: status },
      };
    }

    const rawAmount = tx?.amount_in_cents;
    const amountInCents = typeof rawAmount === 'string' ? Number(rawAmount) : rawAmount;

    return {
      valid: true,
      event: {
        type,
        reference,
        amountInCents: Number.isFinite(amountInCents as number) ? (amountInCents as number) : undefined,
        currency: typeof tx?.currency === 'string' ? tx.currency : undefined,
        providerEventId,
        // The provider's own id + raw status, so the route can confirm this event
        // against `GET /transactions/:id` before mutating anything (CXD-059).
        providerTransactionId: typeof tx?.id === 'string' ? tx.id : undefined,
        providerStatus: status || undefined,
        unauthenticatedFields: unauthenticatedFieldsOf(signature.properties),
        raw: body,
      },
    };
  }
}

interface WompiEvent {
  event: string;
  data?: { transaction?: { id?: string; status?: string; reference?: string;
                           amount_in_cents?: number | string; currency?: string } };
  signature?: { checksum: string; properties: string[] };
  timestamp?: number;
}

/** `GET /v1/transactions/:id` response (only the fields we cross-check). */
interface WompiTransactionResponse {
  data?: { id?: string; status?: string; reference?: string;
           amount_in_cents?: number | string; currency?: string };
}

function getPath(obj: unknown, path: string): unknown {
  return path.split('.').reduce<unknown>(
    (acc, k) => (acc && typeof acc === 'object' ? (acc as Record<string, unknown>)[k] : undefined),
    obj,
  );
}

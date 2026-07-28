/**
 * Billing provider abstraction (CTR-RBAC-001 R6, rule 9: the payment provider is the
 * source of truth; webhooks update entitlements; NEVER trust client claims).
 *
 * Dependency-inversion: routes depend on THIS interface; concrete providers (Wompi,
 * PayU, MercadoPago, Stripe) implement it. Swap via env `BILLING_PROVIDER`.
 */
import type { PlanId } from '@/lib/contracts/rbac.contract';

export interface CheckoutRequest {
  userId: string;
  email: string;
  plan: Exclude<PlanId, 'free'>;
  /** Optional per-asset add-ons (maps 1:1 to the multi-asset registry). */
  addOnAssets?: string[];
}

export interface CheckoutSession {
  provider: string;
  /** URL the browser is redirected to for payment. */
  checkoutUrl: string;
  /** Opaque reference that will come back in the webhook (encodes user+plan). */
  reference: string;
}

/** Transitions the platform knows how to apply. NOTHING else may be synthesized. */
export type BillingEventType =
  | 'payment.approved'
  | 'payment.declined'
  | 'subscription.cancelled'
  | 'payment.refunded'
  | 'payment.charged_back';

export interface NormalizedBillingEvent {
  type: BillingEventType;
  reference: string;
  amountInCents?: number;
  /** ISO-4217 code as reported by the provider; checked against the sealed quote. */
  currency?: string;
  /** Stable provider-side id (idempotency key of the append-only event ledger). */
  providerEventId: string;
  /**
   * The provider's OWN transaction id — the key of `fetchTransaction`. Distinct from
   * `providerEventId`, which is our composite ledger key. Absent ⇒ the event cannot be
   * confirmed server-to-server and therefore must NOT mutate anything (CXD-059).
   */
  providerTransactionId?: string;
  /** Raw provider status string, compared against the authoritative record as-is. */
  providerStatus?: string;
  /**
   * Fields of THIS event that the provider's signature does NOT cover, and which are
   * therefore attacker-mutable on a replay of an otherwise valid event (CODEX CXD-056).
   * Wompi signs only `transaction.id|status|amount_in_cents`, so `reference` — the
   * field that decides who gets credited — arrives UNAUTHENTICATED. Consumers must
   * confirm every listed field against something they control (the sealed quote in
   * `checkout_orders`, or the `provider_event_id` binding in `billing_events`) before
   * acting on it. See `lib/billing/event-ledger.ts`.
   */
  unauthenticatedFields: readonly string[];
  raw: unknown;
}

export interface WebhookVerification {
  valid: boolean;
  /** Normalized event after signature verification. Absent ⇒ nothing to apply. */
  event?: NormalizedBillingEvent;
  /**
   * Signature verified but the event carries NO state transition we understand
   * (e.g. Wompi `PENDING`, or an event type added by the provider after this code
   * was written). It is acknowledged and dropped — a provider event is NEVER
   * translated into a transition it did not express (CODEX P0-1).
   */
  ignored?: { reason: string; reference?: string; providerEventId?: string; providerStatus?: string };
  error?: string;
}

/**
 * The transaction AS THE PROVIDER REPORTS IT over its own authenticated API — the only
 * account of a payment that an attacker cannot author (CXD-059). Every field here is
 * cross-checked against the webhook body before a single row is mutated.
 */
export interface AuthoritativeTransaction {
  id: string;
  /** The field the webhook checksum does NOT cover. This copy is the authority. */
  reference: string;
  /** Raw provider status, NOT normalized: compared verbatim with the event's. */
  status: string;
  amountInCents?: number;
  currency?: string;
}

/**
 * Outcome of a server-to-server lookup. Three outcomes, never two: "cannot tell"
 * (`unavailable`) MUST be distinguishable from "the provider denies it" (`not_found`),
 * because they get opposite HTTP answers — 503 so the provider retries, versus a 4xx
 * that ends the attempt. Collapsing them either loses a real payment or accepts a
 * forgery during an outage.
 */
export type TransactionLookup =
  | { kind: 'found'; transaction: AuthoritativeTransaction }
  /** The provider answered and has no such transaction (404), or the id is unusable. */
  | { kind: 'not_found'; reason: string }
  /** Timeout, 5xx, transport error, unparseable body — we simply do not know. */
  | { kind: 'unavailable'; reason: string };

export interface BillingProvider {
  readonly name: string;
  createCheckout(req: CheckoutRequest): Promise<CheckoutSession>;
  /** MUST verify the provider's signature; invalid ⇒ {valid:false}. */
  verifyWebhook(rawBody: string, headers: Headers): Promise<WebhookVerification>;
  /**
   * Confirm a transaction against the provider's API (the injectable port of
   * CXD-059). MUST NOT throw: transport failures are returned as `unavailable`, so a
   * provider outage can never be mistaken for a rejection — or for an approval.
   */
  fetchTransaction(providerTransactionId: string): Promise<TransactionLookup>;
}

/**
 * reference = billing correlation id — encodes who paid for what.
 *
 * FAIL-CLOSED ROUND-TRIP (P0-5): the parts are joined with `_` but `decodeReference`
 * parses the add-on slot as `[^_]*`, so an asset id containing `_` (asset ids are
 * operator data — `BILLING_ADDON_PRICES_COP` is an env override) yields a reference
 * NOTHING can decode. The checkout would still be payable, and every webhook for that
 * payment would then die on the reference/quote cross-check: the customer is charged
 * and can never be credited, with no retry able to fix it. So we verify the round-trip
 * HERE, before a payable URL exists, instead of restating the grammar in a second
 * regex that could drift from the parser.
 */
export function encodeReference(userId: string, plan: PlanId, addOns: string[] = []): string {
  const reference = `sub_${plan}_${userId}_${addOns.join('+') || 'base'}_${Date.now()}`;
  const decoded = decodeReference(reference);
  const sameAddOns = decoded
    && decoded.addOns.length === addOns.length
    && decoded.addOns.every((a, i) => a === addOns[i]);
  if (!decoded || decoded.userId !== userId || decoded.plan !== plan || !sameAddOns) {
    throw new Error('billing reference does not round-trip; refusing to issue a payable checkout');
  }
  return reference;
}

export function decodeReference(reference: string):
    { plan: PlanId; userId: string; addOns: string[] } | null {
  const m = reference.match(/^sub_(free|signals|auto)_([0-9a-f-]{36})_([^_]*)_\d+$/i);
  if (!m) return null;
  return {
    plan: m[1] as PlanId,
    userId: m[2],
    addOns: m[3] === 'base' ? [] : m[3].split('+'),
  };
}

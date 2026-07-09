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

export interface WebhookVerification {
  valid: boolean;
  /** Normalized event after signature verification. */
  event?: {
    type: 'payment.approved' | 'payment.declined' | 'subscription.cancelled';
    reference: string;
    amountInCents?: number;
    raw: unknown;
  };
  error?: string;
}

export interface BillingProvider {
  readonly name: string;
  createCheckout(req: CheckoutRequest): Promise<CheckoutSession>;
  /** MUST verify the provider's signature; invalid ⇒ {valid:false}. */
  verifyWebhook(rawBody: string, headers: Headers): Promise<WebhookVerification>;
}

/** reference = billing correlation id — encodes who paid for what. */
export function encodeReference(userId: string, plan: PlanId, addOns: string[] = []): string {
  return `sub_${plan}_${userId}_${addOns.join('+') || 'base'}_${Date.now()}`;
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

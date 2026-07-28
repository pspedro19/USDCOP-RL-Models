/**
 * Server-to-server confirmation of a webhook event — CODEX finding **CXD-059**.
 *
 * ── WHAT THE LEDGER DOES NOT DO ─────────────────────────────────────────────────
 * `event-ledger.ts` (CXD-056) makes the SECOND use of a `provider_event_id`
 * impossible. It cannot make the FIRST use safe. Wompi's checksum covers
 * `transaction.id|status|amount_in_cents` and NOT `transaction.reference`, so anyone
 * holding a validly-signed APPROVED body can open his own checkout for the same plan
 * — same amount, same currency — and POST that body with HIS reference. Signature:
 * valid. Ledger: first sighting. Sealed quote: matches, because it is his own quote.
 * Every local check agrees and he is credited without paying. No amount of code
 * reading the request body can detect this: the forged and the genuine event are
 * byte-identical except in the one field nobody signed.
 *
 * ── WHY THIS IS NOT AN OPERATOR DECISION ────────────────────────────────────────
 * A previous round recorded fail-closed confirmation as an operator decision because
 * a provider outage would block legitimate payments. That framing was wrong and CODEX
 * rejected it: **no provider outage can justify crediting unauthenticated money.**
 * Failing closed does not LOSE a legitimate payment — it postpones it; the provider
 * retries (and, if it does not, the operator replays) and the retry credits. Failing
 * open gives money away irreversibly. The two error costs are not comparable, so
 * there is no trade-off to delegate.
 *
 * ── THE RULE ────────────────────────────────────────────────────────────────────
 * Before ANY mutation of an order or of entitlements, the event must be confirmed
 * against `provider.fetchTransaction()`, and `id`, `reference`, `status`, `amount`
 * and `currency` must ALL agree. Three outcomes, deliberately distinct:
 *
 *   confirmed   → proceed to the (unchanged) ledger + sealed-quote checks
 *   unavailable → 503, mutate NOTHING, not even the ledger — the ledger is what makes
 *                 a retry a no-op, so poisoning it during an outage would deny the
 *                 paid user forever
 *   mismatch    → 4xx + security incident, mutate NOTHING
 *
 * This module is PURE (no I/O) so every branch is testable without a network, and so
 * the decision cannot drift into the transport code. The transport is
 * `BillingProvider.fetchTransaction` — an interface, per the DIP already used for
 * `createCheckout`/`verifyWebhook`.
 */
import type { NormalizedBillingEvent, TransactionLookup } from './provider';

export type ConfirmationVerdict =
  | { kind: 'confirmed' }
  /** We could not reach a verdict. Retryable: the caller answers 503 and mutates nothing. */
  | { kind: 'unavailable'; reason: string }
  /** The provider's record contradicts the event (or denies it). Never retryable. */
  | { kind: 'mismatch'; field: ConfirmedField; authoritative: AuthoritativeSnapshot };

export type ConfirmedField =
  | 'transaction_id' | 'existence' | 'reference' | 'status' | 'amount' | 'currency';

/**
 * What the provider said, for the AUDIT ROW ONLY. It names another customer's
 * reference, so it must never reach an HTTP response body.
 */
export interface AuthoritativeSnapshot {
  id?: string;
  reference?: string;
  status?: string;
  amountInCents?: number;
  currency?: string;
}

/**
 * Compare the authoritative record with the event. Fail-closed by construction: the
 * ONLY path to `confirmed` is one where every comparable field is present AND equal.
 * A field the provider omits is NOT "compatible" — it is a mismatch, because an
 * absent value cannot authenticate anything.
 */
export function confirmAgainstProvider(
  event: NormalizedBillingEvent,
  lookup: TransactionLookup,
): ConfirmationVerdict {
  if (lookup.kind === 'unavailable') return { kind: 'unavailable', reason: lookup.reason };

  // The event carries no provider transaction id ⇒ there is nothing to confirm it
  // with. We refuse rather than fall back to trusting the body.
  if (!event.providerTransactionId) {
    return { kind: 'mismatch', field: 'transaction_id', authoritative: {} };
  }
  if (lookup.kind === 'not_found') {
    return { kind: 'mismatch', field: 'existence', authoritative: {} };
  }

  const tx = lookup.transaction;
  const snapshot: AuthoritativeSnapshot = {
    id: tx.id, reference: tx.reference, status: tx.status,
    amountInCents: tx.amountInCents, currency: tx.currency,
  };
  const reject = (field: ConfirmedField): ConfirmationVerdict =>
    ({ kind: 'mismatch', field, authoritative: snapshot });

  if (!tx.id || tx.id !== event.providerTransactionId) return reject('transaction_id');
  // ── THE FIELD THE SIGNATURE NEVER COVERED. This single comparison is the fix.
  if (!tx.reference || tx.reference !== event.reference) return reject('reference');
  // Raw status, compared verbatim: normalizing first would let two different provider
  // statuses that happen to map to the same transition confirm each other.
  if (!tx.status || !event.providerStatus || tx.status !== event.providerStatus) return reject('status');
  if (typeof tx.amountInCents !== 'number' || !Number.isFinite(tx.amountInCents)
      || tx.amountInCents !== event.amountInCents) return reject('amount');
  if (!tx.currency || tx.currency !== event.currency) return reject('currency');

  return { kind: 'confirmed' };
}

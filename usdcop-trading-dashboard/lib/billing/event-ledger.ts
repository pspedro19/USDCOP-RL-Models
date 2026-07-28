/**
 * Provider-event ledger semantics — the ONLY place the replay of a signed webhook is
 * actually stopped (CODEX finding CXD-056).
 *
 * ── WHY THIS EXISTS: the reference is NOT authenticated ──────────────────────────
 * Wompi signs an event as `sha256(<values of signature.properties> + timestamp +
 * events_secret)`, and its `signature.properties` are `transaction.id`,
 * `transaction.status` and `transaction.amount_in_cents`. **`transaction.reference` —
 * the field that decides WHICH ORDER, and therefore WHICH USER, gets credited — is
 * not covered by the checksum.** Two events with the same transaction id/status/amount
 * and different references carry the byte-identical, genuinely valid checksum: a
 * signature verifier cannot tell the forged one from the real one, and no amount of
 * code in `wompi.ts` can change that. We state this instead of implying the signature
 * protects it (`.claude/rules/rbac.md` §7 + K-031: never declare a guarantee that is
 * not enforced).
 *
 * ── WHAT DOES ENFORCE IT ────────────────────────────────────────────────────────
 * `billing_events.provider_event_id` is UNIQUE (migration 059) and the reference the
 * event was FIRST seen with is stored next to it in `order_reference`. That row is the
 * authoritative, durable binding `provider event → reference`:
 *
 *   INSERT ... ON CONFLICT (provider_event_id) DO NOTHING RETURNING provider_event_id
 *     rowCount = 1  → first sighting; the binding is now written and committed.
 *     rowCount = 0  → the id already exists ⇒ classify it here:
 *                     identical  ⇒ provider retry, no-op, credit NOTHING again
 *                     different  ⇒ SECURITY INCIDENT, reject; NEVER continue.
 *
 * ── WHAT THIS DOES *NOT* ENFORCE (CXD-059) ──────────────────────────────────────
 * This ledger stops the SECOND use of a provider event. It cannot stop the FIRST: an
 * attacker who opens his own checkout for the same plan (same amount, same currency)
 * and replays a signed body under HIS reference collides with nothing — the id has
 * never been seen. Every local check agrees and he is credited.
 *
 * That gap is closed one layer earlier, by `lib/billing/confirmation.ts`: the route
 * confirms the transaction server-to-server (`BillingProvider.fetchTransaction` →
 * `GET /v1/transactions/:id`) BEFORE any mutation, and requires the authoritative
 * reference to match. An earlier round recorded that as an OPERATOR DECISION because
 * failing closed would let a provider outage block payments; CODEX rejected the
 * framing and was right — an outage postpones a legitimate credit (the retry applies
 * it), while failing open gives money away. The two layers are independent and each
 * is tested alone.
 */

export interface LedgerRow {
  order_reference: string;
  event_type: string;
  payload: unknown;
}

export interface IncomingEvent {
  reference: string;
  type: string;
  raw: unknown;
}

export type LedgerVerdict =
  /** Same id, same reference, same type, same payload ⇒ a provider retry. */
  | { kind: 'duplicate' }
  /** Same id, something else changed ⇒ the reference/payload is being replayed. */
  | { kind: 'conflict'; boundReference: string; changed: 'reference' | 'event_type' | 'payload' };

/**
 * Order-insensitive digest of a JSON value. `jsonb` does not preserve key order, so a
 * raw string comparison against the stored payload would report false conflicts;
 * sorting keys makes the round-trip stable while still detecting ANY value change.
 */
export function canonicalJsonDigest(value: unknown): string {
  return hash(canonicalize(value));
}

function canonicalize(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value ?? null);
  if (Array.isArray(value)) return `[${value.map(canonicalize).join(',')}]`;
  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([, v]) => v !== undefined)
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${canonicalize(v)}`).join(',')}}`;
}

/** FNV-1a 64-bit (hex). Not a security primitive — an equality fingerprint. */
function hash(s: string): string {
  let h = 0xcbf29ce484222325n;
  for (let i = 0; i < s.length; i += 1) {
    h ^= BigInt(s.charCodeAt(i));
    h = (h * 0x100000001b3n) & 0xffffffffffffffffn;
  }
  return h.toString(16).padStart(16, '0');
}

/**
 * Decide what a `provider_event_id` collision means. Fail-closed by construction: the
 * ONLY input that yields `duplicate` is one that matches the stored row in every field
 * we can compare. Anything else is a conflict, including a payload we cannot parse.
 */
export function classifyLedgerConflict(prior: LedgerRow, incoming: IncomingEvent): LedgerVerdict {
  if (prior.order_reference !== incoming.reference) {
    return { kind: 'conflict', boundReference: prior.order_reference, changed: 'reference' };
  }
  if (prior.event_type !== incoming.type) {
    return { kind: 'conflict', boundReference: prior.order_reference, changed: 'event_type' };
  }
  if (canonicalJsonDigest(prior.payload) !== canonicalJsonDigest(incoming.raw ?? {})) {
    return { kind: 'conflict', boundReference: prior.order_reference, changed: 'payload' };
  }
  return { kind: 'duplicate' };
}

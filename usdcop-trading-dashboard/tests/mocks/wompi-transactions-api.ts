/**
 * Scripted fake of the provider's authoritative query API — `GET /v1/transactions/:id`
 * (CODEX finding CXD-059, implemented in `lib/billing/wompi.ts::fetchTransaction`).
 *
 * The webhook now refuses to mutate anything until the provider CONFIRMS the payment
 * server-to-server, so every billing test runs in a world where that API exists. This
 * fake is the single model of it: one copy, so the suites cannot disagree about what
 * the provider does.
 *
 * It reproduces exactly the response shapes the implementation branches on:
 *   200 + {data:{...}}   the transaction, verbatim and IMMUTABLE once recorded
 *   404                  the provider denies it          ⇒ `not_found` ⇒ 4xx
 *   5xx / thrown         we cannot tell                  ⇒ `unavailable` ⇒ 503
 *   200 + non-JSON       we cannot tell (gateway page)   ⇒ `unavailable` ⇒ 503
 *
 * Install with `vi.stubGlobal('fetch', api.fetch)`: the provider resolves
 * `globalThis.fetch` lazily for exactly this reason, so the real `WompiProvider` —
 * URL building, status classification, parsing — is what runs, with only the transport
 * replaced.
 */

export interface ProviderTx {
  id: string;
  status: string;
  reference: string;
  amount_in_cents: number;
  currency: string;
}

export type ProviderApiMode = 'up' | 'down' | 'http5xx' | 'garbage';

/**
 * A transport error carrying exactly the things that must never reach a client: an
 * absolute path and a credential. Tests assert the response body is free of both.
 */
export const leakyTransportError = () => new Error(
  'connect ETIMEDOUT at C:\\srv\\usdcop\\lib\\billing\\wompi.ts:203 (authorization=Bearer pub_prod_hunter2)',
);

export function createWompiTransactionsApi() {
  const txs = new Map<string, ProviderTx>();
  const calls: string[] = [];
  let mode: ProviderApiMode = 'up';

  return {
    txs,
    calls,
    setMode(next: ProviderApiMode) { mode = next; },
    reset() { txs.clear(); calls.length = 0; mode = 'up'; },

    /**
     * Record what the provider REALLY has. First write wins: a provider's record of a
     * transaction is immutable, and a test that could silently rewrite it would be
     * able to make a forgery "true".
     */
    record(tx: ProviderTx) {
      if (!txs.has(tx.id)) txs.set(tx.id, tx);
    },

    /** Overwrite unconditionally — for suites whose subject is a LATER defence layer. */
    forceRecord(tx: ProviderTx) { txs.set(tx.id, tx); },

    fetch: async (input: unknown): Promise<Response> => {
      const url = String(typeof input === 'string' ? input : (input as { url?: string })?.url ?? input);
      calls.push(url);
      if (mode === 'down') throw leakyTransportError();
      if (mode === 'http5xx') {
        return new Response('upstream timeout at C:\\srv\\usdcop\\gateway (secret=hunter2)', { status: 504 });
      }
      if (mode === 'garbage') return new Response('<html>maintenance</html>', { status: 200 });

      const id = decodeURIComponent(url.split('/').pop()!.split('?')[0]);
      const tx = txs.get(id);
      if (!tx) return new Response(JSON.stringify({ error: { reason: 'not found' } }), { status: 404 });
      return new Response(JSON.stringify({ data: tx }), {
        status: 200, headers: { 'content-type': 'application/json' },
      });
    },
  };
}

/**
 * Billing provider factory — swap providers via env `BILLING_PROVIDER` (default wompi).
 * Routes import ONLY from here (dependency inversion; see provider.ts).
 */
import type { BillingProvider } from './provider';
import { SandboxProvider } from './sandbox';
import { WompiProvider } from './wompi';

export * from './provider';

const providers: Record<string, () => BillingProvider> = {
  wompi: () => new WompiProvider(),
  // Exercises the real payment path (signed webhook + server-to-server confirmation)
  // without a merchant account, so the purchase flow is runnable end to end before live
  // keys exist. It refuses to construct in production and without its own secret — see
  // sandbox.ts. NOT a bypass: the same checks run, they just run against a local issuer.
  sandbox: () => new SandboxProvider(),
  // payu / mercadopago / stripe: implement BillingProvider and register here.
};

export function getBillingProvider(): BillingProvider {
  const name = (process.env.BILLING_PROVIDER ?? 'wompi').toLowerCase();
  const factory = providers[name];
  if (!factory) throw new Error(`unknown BILLING_PROVIDER '${name}'`);
  return factory();
}

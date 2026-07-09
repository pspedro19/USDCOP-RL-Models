/**
 * Billing provider factory — swap providers via env `BILLING_PROVIDER` (default wompi).
 * Routes import ONLY from here (dependency inversion; see provider.ts).
 */
import type { BillingProvider } from './provider';
import { WompiProvider } from './wompi';

export * from './provider';

const providers: Record<string, () => BillingProvider> = {
  wompi: () => new WompiProvider(),
  // payu / mercadopago / stripe: implement BillingProvider and register here.
};

export function getBillingProvider(): BillingProvider {
  const name = (process.env.BILLING_PROVIDER ?? 'wompi').toLowerCase();
  const factory = providers[name];
  if (!factory) throw new Error(`unknown BILLING_PROVIDER '${name}'`);
  return factory();
}

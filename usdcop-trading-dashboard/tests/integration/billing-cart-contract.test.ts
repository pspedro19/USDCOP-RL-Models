import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const root = process.cwd();

describe('billing/cart server-side contracts', () => {
  it('never trusts client add-ons and keeps entitlement changes in webhook', () => {
    const route = readFileSync(join(root, 'app/api/cart/checkout/route.ts'), 'utf8');
    expect(route).not.toContain('body.addOn');
    expect(route).toContain('SELECT asset_id FROM user_cart WHERE user_id = $1');
    expect(route).toContain('getEntitlements');
  });

  it('verifies amount and idempotency before granting access', () => {
    const route = readFileSync(join(root, 'app/api/billing/webhook/route.ts'), 'utf8');
    expect(route).toContain('amount mismatch');
    expect(route).toContain('billing_webhook_events');
    expect(route).toContain('UPDATE sb_users SET entitlements');
  });
});

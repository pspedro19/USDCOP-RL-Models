import { createHash } from 'node:crypto';
import { beforeEach, expect, it } from 'vitest';

import { WompiProvider } from '../../../../usdcop-trading-dashboard/lib/billing/wompi';

const SECRET = 'events-secret';
const USER_A = '11111111-1111-4111-8111-111111111111';
const USER_B = '22222222-2222-4222-8222-222222222222';

function signedApproval(reference: string) {
  const transaction = {
    id: 'tx_replay_1',
    status: 'APPROVED',
    reference,
    amount_in_cents: 9_900_000,
    currency: 'COP',
  };
  const timestamp = 1_700_000_000;
  const properties = [
    'transaction.id',
    'transaction.status',
    'transaction.amount_in_cents',
  ];
  const checksum = createHash('sha256')
    .update(`${transaction.id}${transaction.status}${transaction.amount_in_cents}${timestamp}${SECRET}`)
    .digest('hex');
  return {
    event: 'transaction.updated',
    data: { transaction },
    signature: { properties, checksum },
    timestamp,
  };
}

beforeEach(() => {
  process.env.WOMPI_EVENTS_SECRET = SECRET;
});

it('exposes reference as unauthenticated for the stateful route to confirm', async () => {
  const provider = new WompiProvider();
  const original = signedApproval(`sub_signals_${USER_A}_base_1700000000000`);
  const tampered = structuredClone(original);
  tampered.data.transaction.reference =
    `sub_signals_${USER_B}_base_1700000000000`;

  const first = await provider.verifyWebhook(
    JSON.stringify(original),
    new Headers(),
  );
  const replay = await provider.verifyWebhook(
    JSON.stringify(tampered),
    new Headers(),
  );

  expect(first.valid).toBe(true);
  expect(replay.valid).toBe(true);
  expect(first.event?.providerEventId).toBe(replay.event?.providerEventId);
  expect(first.event?.unauthenticatedFields).toContain('reference');
  expect(replay.event?.unauthenticatedFields).toContain('reference');
  expect(first.event?.reference).not.toBe(replay.event?.reference);
});

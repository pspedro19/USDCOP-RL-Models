import { validIsoTimestamp } from '../../../../usdcop-trading-dashboard/lib/contracts/policy.contract';

const cases = [
  ['year_0000', '0000-01-01T00:00:00Z', false],
  ['year_0001', '0001-01-01T00:00:00Z', true],
  ['leap_1900', '1900-02-29T00:00:00Z', false],
  ['leap_2000', '2000-02-29T00:00:00Z', true],
] as const;

let failures = 0;
for (const [id, value, expected] of cases) {
  const actual = validIsoTimestamp(value);
  console.log(JSON.stringify({ id, value, expected, actual }));
  if (actual !== expected) failures += 1;
}

if (failures > 0) {
  console.error(`POLICY_TIMESTAMP_PARITY_RED failures=${failures}`);
  process.exitCode = 1;
}

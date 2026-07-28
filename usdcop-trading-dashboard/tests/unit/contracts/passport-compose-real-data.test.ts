/**
 * S-04 — the §6 guard on the REAL published surface, not on a fixture.
 * =====================================================================
 *
 * The contract tests (`passport-contract.test.ts` + `tests/unit/test_passport_contract.py`)
 * proved the guard works on hand-built envs. The auto-red-team then ran the composer
 * against the actual `public/data/**` and found it publishing
 *
 *     REAL btc_hodl_b1 backtest: n_trades={"value":null} | sharpe={"value":0.793}
 *                              | p_value={"value":0.0242} | insufficient_trades=false
 *     validateStrategyPassport(real btc_hodl_b1) = []
 *
 * for a strategy whose `trades_2025.json` has ONE element. The guard was byte-identical
 * in both languages and completely inert, because it keyed off `headline.trades` — a
 * field only the three `smart_simple_*` manifests publish. Every Gold, BTC and SPX500
 * bundle (and `spx500_*`, which spells it `n_trades`) sailed through.
 *
 * This test composes the REAL artifacts. A fixture cannot catch a defect whose whole
 * nature is "the real data does not look like the fixture".
 */
import { describe, it, expect } from 'vitest';

import {
  composeControlTower,
  composeStrategyPassport,
  listPassportStrategies,
} from '@/lib/passport/compose';
import {
  MIN_TRADES_FOR_RATIOS,
  SMALL_SAMPLE_SUPPRESSED_FIELDS,
  isAvailable,
  validateControlTower,
  validateStrategyPassport,
  type EnvPerformance,
  type Sourced,
} from '@/lib/contracts/passport.contract';

describe('composeStrategyPassport over the real public/data (S-04)', () => {
  it('publishes no ratio whose N is not published and >= 20 — for EVERY strategy', async () => {
    const strategies = await listPassportStrategies();
    expect(strategies.length, 'registry.json publishes no strategies').toBeGreaterThan(0);

    const offenders: string[] = [];
    for (const { strategy_id } of strategies) {
      const passport = await composeStrategyPassport(strategy_id);
      expect(passport, `${strategy_id}: composer returned null`).toBeTruthy();
      const errors = validateStrategyPassport(passport);
      if (errors.length) offenders.push(`${strategy_id}: ${errors.join(' | ')}`);

      for (const [env, block] of Object.entries(passport!.performance)) {
        const perf = block as EnvPerformance;
        const n = perf.n_trades?.value;
        const nOk = typeof n === 'number' && n >= MIN_TRADES_FOR_RATIOS;
        if (nOk) continue;
        for (const key of SMALL_SAMPLE_SUPPRESSED_FIELDS) {
          const field = (perf as unknown as Record<string, Sourced<unknown>>)[key];
          if (isAvailable(field)) {
            offenders.push(
              `${strategy_id}.performance.${env}.${key} = ${JSON.stringify(field.value)} `
              + `with n_trades=${JSON.stringify(n)}`,
            );
          }
        }
      }
    }
    expect(offenders, 'quant-constitution §6 violated on the real published surface').toEqual([]);
  });

  it('the Control Tower rows obey the same rule', async () => {
    const tower = await composeControlTower(new Date('2026-07-28T00:00:00Z'));
    expect(validateControlTower(tower)).toEqual([]);
    const offenders = tower.sleeves
      .filter((s) => {
        const n = s.n_trades?.value;
        const nOk = typeof n === 'number' && n >= MIN_TRADES_FOR_RATIOS;
        return !nOk && (isAvailable(s.sharpe) || isAvailable(s.dsr_family));
      })
      .map((s) => `${s.strategy_id}: sharpe=${JSON.stringify(s.sharpe.value)} `
        + `dsr=${JSON.stringify(s.dsr_family.value)} n=${JSON.stringify(s.n_trades.value)}`);
    expect(offenders).toEqual([]);
  });
});

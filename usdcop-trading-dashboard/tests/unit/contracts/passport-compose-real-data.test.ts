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
  DSR_BAR,
  MIN_TRADES_FOR_RATIOS,
  SMALL_SAMPLE_SUPPRESSED_FIELDS,
  isAvailable,
  validateControlTower,
  validateSourced,
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

/**
 * BL-32 — the OTHER half of the same defect: the guards check that a block EXISTS, never
 * that it says anything.
 *
 * `validateStrategyPassport` only asserts `'governance' in payload`, so replacing the
 * composed governance block with `governance: {} as never` in `lib/passport/compose.ts`
 * shipped a Passport with **cero trials, cero DSR, cero N y cero `policy_hash`** and left
 * the suite at 30 passed. Deleting the KEY went red; emptying it did not.
 *
 * Honest nuance (and the reason these tests do NOT demand values): a strategy with no
 * trials yet, no signed withdrawal protocol or no `policy_hash` is a LEGITIMATE state —
 * `policy_hash` has no producer at all until BL-45. So what is required is the contract's
 * own alternative to a value: the `Sourced` shape with `status: 'unavailable'` and a
 * `pending` naming the backlog item that owes it. A `{}` cannot express that; `null` +
 * `pending: 'BL-45 …'` can. The rule enforced is "no mute holes", not "no holes".
 */
describe('composed Passport blocks are populated, never a mute {} (BL-32)', () => {
  /** Field names declared by each block in `passport.contract.ts`. Interfaces are erased
   *  at runtime, so the expectation is written out here; the Python twin
   *  (`tests/unit/test_passport_contract.py::test_python_fixture_mirrors_the_ts_blocks`)
   *  parses the same interfaces and fails if these lists and the contract drift apart. */
  const BLOCK_FIELDS: Record<string, string[]> = {
    identity: ['strategy_id', 'asset_id', 'display_name', 'surface', 'engine_type',
      'status', 'active_version', 'timeframe'],
    governance: ['n_trials_total', 'n_trials_forecast', 'n_trials_action', 'n_family',
      'n_cluster', 'n_global', 'dsr_family', 'dsr_bar', 'approval_status', 'gates',
      'withdrawal_protocol', 'retirement_signal', 'retirement_reason'],
    lineage: ['model_versions', 'spec_fingerprint', 'feature_set_hash', 'policy_hash',
      'lineage_graph'],
    live: ['open_orders', 'last_fill_at', 'quarantined', 'reconciled',
      'kill_switch_engaged', 'deploy_status', 'last_signal_at'],
    risk: ['current_exposure', 'vol_target_pct', 'vol_forecast_pct', 'm_forward', 'm_dd',
      'rho_max', 'turnover'],
  };

  // Rojo con: `governance: {} as never` en lib/passport/compose.ts (idem lineage/live/risk).
  it('declares every field of every block for EVERY real strategy', async () => {
    const strategies = await listPassportStrategies();
    expect(strategies.length, 'registry.json publishes no strategies').toBeGreaterThan(0);

    const offenders: string[] = [];
    for (const { strategy_id } of strategies) {
      const passport = await composeStrategyPassport(strategy_id);
      expect(passport, `${strategy_id}: composer returned null`).toBeTruthy();
      for (const [block, fields] of Object.entries(BLOCK_FIELDS)) {
        const rec = (passport as unknown as Record<string, unknown>)[block];
        if (!rec || typeof rec !== 'object') {
          offenders.push(`${strategy_id}.${block}: ${JSON.stringify(rec)} (no es un objeto)`);
          continue;
        }
        const missing = fields.filter((f) => !(f in (rec as Record<string, unknown>)));
        if (missing.length) offenders.push(`${strategy_id}.${block}: faltan ${missing.join(',')}`);
      }
    }
    expect(offenders, 'un bloque del Passport se compuso vacío').toEqual([]);
  });

  // Rojo con: `governance: {} as never` en lib/passport/compose.ts.
  it('governance carries the trial counts, the DSR family and its bar — value or named owner', async () => {
    const strategies = await listPassportStrategies();
    const offenders: string[] = [];

    for (const { strategy_id } of strategies) {
      const passport = await composeStrategyPassport(strategy_id);
      const gov = passport!.governance as unknown as Record<string, Sourced<unknown>>;

      // §2 — the bar travels with the block; without it a DSR is an unjudgeable number.
      if (passport!.governance?.dsr_bar !== DSR_BAR) {
        offenders.push(`${strategy_id}.governance.dsr_bar = ${JSON.stringify(passport!.governance?.dsr_bar)} (esperado ${DSR_BAR})`);
      }
      for (const field of ['n_trials_total', 'n_trials_forecast', 'n_trials_action',
        'n_family', 'n_cluster', 'n_global', 'dsr_family'] as const) {
        const sourcedField = gov?.[field];
        const errors = validateSourced(sourcedField, `${strategy_id}.governance.${field}`);
        if (errors.length) { offenders.push(...errors); continue; }
        // Either a published number with its artifact, or an explicit `pending` naming
        // the backlog item that owes it. Never `{}`, never a bare null.
        if (!isAvailable(sourcedField) && !/\bBL-\d+/.test(String(sourcedField.source.pending))) {
          offenders.push(
            `${strategy_id}.governance.${field}: sin valor y sin dueño — pending=`
            + `${JSON.stringify(sourcedField.source.pending)} no nombra ningún BL`,
          );
        }
      }
    }
    expect(offenders, 'governance compuesto sin trials/DSR ni dueño declarado').toEqual([]);
  });

  // Rojo con: `policy_hash: {} as never` (o borrar la línea) en el bloque `lineage` de compose.ts.
  it('lineage carries policy_hash / feature_set_hash / spec_fingerprint with a named owner', async () => {
    const strategies = await listPassportStrategies();
    const offenders: string[] = [];

    for (const { strategy_id } of strategies) {
      const passport = await composeStrategyPassport(strategy_id);
      const lineage = passport!.lineage as unknown as Record<string, Sourced<unknown>>;
      for (const field of ['policy_hash', 'feature_set_hash', 'spec_fingerprint'] as const) {
        const sourcedField = lineage?.[field];
        const errors = validateSourced(sourcedField, `${strategy_id}.lineage.${field}`);
        if (errors.length) { offenders.push(...errors); continue; }
        if (!isAvailable(sourcedField) && !/\bBL-\d+/.test(String(sourcedField.source.pending))) {
          offenders.push(
            `${strategy_id}.lineage.${field}: sin hash y sin dueño — pending=`
            + `${JSON.stringify(sourcedField.source.pending)} no nombra ningún BL`,
          );
        }
      }
    }
    expect(offenders, 'lineage compuesto sin fingerprints ni dueño declarado').toEqual([]);
  });
});

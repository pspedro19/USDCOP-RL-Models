/**
 * CTR-PASSPORT-001 contract tests (BL-32) — TS side of the Py↔TS mirror.
 * Python twin: `tests/unit/test_passport_contract.py` (same vocabularies, same
 * verdicts). Both runners assert the SAME invariants so a drift on either side
 * goes red.
 *
 * What is actually being locked here is not "does the object have fields" but the
 * three things that make the Passport honest:
 *   1. an unavailable field can never carry a value or hide who owes it;
 *   2. a published field can never be anonymous;
 *   3. N<20 can never publish a Sharpe/p-value/DSR.
 */
import { describe, it, expect } from 'vitest';

import {
  BOOK_STATES,
  DSR_BAR,
  FORBIDDEN_PASSPORT_ACTIONS,
  HEALTH_CLOCKS,
  MIN_TRADES_FOR_RATIOS,
  N_MAX_TRIALS,
  PASSPORT_CONTRACT_ID,
  PASSPORT_ENVS,
  RETIREMENT_SIGNALS,
  SOURCE_STATUSES,
  isAvailable,
  sanitizeNumber,
  sourced,
  suppressSmallSample,
  unavailable,
  validateControlTower,
  validateSourced,
  validateStrategyPassport,
  type EnvPerformance,
} from '@/lib/contracts/passport.contract';

// ────────────────────────────────────────────────────────────── vocabularies

describe('vocabularies (mirror of src/contracts/passport.py)', () => {
  it('declares the five FABRIC §24.4 environments in order', () => {
    expect([...PASSPORT_ENVS]).toEqual(['backtest', 'held_out', 'paper', 'canary', 'live']);
  });
  it('declares the five §24.5 book states', () => {
    expect([...BOOK_STATES]).toEqual(['CHAMPION', 'CANARY', 'PAPER', 'REDUCED', 'QUARANTINED']);
  });
  it('has exactly two source statuses — "estimated" would be a modelling call', () => {
    expect([...SOURCE_STATUSES]).toEqual(['published', 'unavailable']);
  });
  it('treats "unknown" as a first-class retirement signal', () => {
    expect(RETIREMENT_SIGNALS).toContain('unknown');
  });
  it('declares the three clocks including the one with no producer', () => {
    expect([...HEALTH_CLOCKS]).toEqual(['data', 'model', 'exec']);
  });
  it('pins the constitutional constants', () => {
    expect(MIN_TRADES_FOR_RATIOS).toBe(20);
    expect(N_MAX_TRIALS).toBe(989);
    expect(DSR_BAR).toBe(0.95);
  });
});

// ────────────────────────────────────────────────────────── Sourced primitive

describe('Sourced', () => {
  it('published fields must name their artifact', () => {
    expect(validateSourced(sourced(1.23, 'public/data/x.json'), 'f')).toEqual([]);
    const anonymous = { value: 1, source: { path: null, status: 'published', pending: null } };
    expect(validateSourced(anonymous, 'f').join()).toMatch(/MUST name their artifact/);
  });

  it('unavailable fields must be null AND declare who owes them', () => {
    expect(validateSourced(unavailable('BL-22 fact_pnl'), 'f')).toEqual([]);
    const withValue = { value: 0, source: { path: null, status: 'unavailable', pending: 'BL-22' } };
    expect(validateSourced(withValue, 'f').join()).toMatch(/MUST have value=null/);
    const noOwner = { value: null, source: { path: null, status: 'unavailable', pending: null } };
    expect(validateSourced(noOwner, 'f').join()).toMatch(/pending on/);
  });

  it('never lets a non-finite number through (strategy-contract §2)', () => {
    expect(sanitizeNumber(Infinity)).toBeNull();
    expect(sanitizeNumber(NaN)).toBeNull();
    expect(sourced(Infinity, 'p').value).toBeNull();
    expect(JSON.stringify(sourced(Infinity, 'p'))).not.toContain('Infinity');
  });

  it('isAvailable is false for unavailable AND for published-but-null', () => {
    expect(isAvailable(sourced(1, 'p'))).toBe(true);
    expect(isAvailable(unavailable('BL-x'))).toBe(false);
    expect(isAvailable(sourced(null, 'p'))).toBe(false);
  });
});

// ─────────────────────────────────────────────────────── small-sample guard §6

function envWith(nTrades: number | null): EnvPerformance {
  return {
    env: 'live',
    period_label: sourced<string>('2026', 'p'),
    return_pct: sourced(3.36, 'p'),
    n_trades: nTrades == null ? unavailable('n/d') : sourced(nTrades, 'p'),
    max_dd_pct: sourced(1.5, 'p'),
    win_rate_pct: sourced(72.7, 'p'),
    profit_factor: sourced(2.408, 'p'),
    sharpe: sourced(1.9, 'p'),
    calmar: sourced(2.24, 'p'),
    p_value: sourced(0.03, 'p'),
    dsr_family: sourced(0.42, 'p'),
    timing_ratio: sourced(0.02, 'p'),
    insufficient_trades: false,
  };
}

describe('suppressSmallSample (quant-constitution §6)', () => {
  it('with N<20 strips ratios/p/DSR but keeps count, return and drawdown', () => {
    const out = suppressSmallSample(envWith(11));
    expect(out.insufficient_trades).toBe(true);
    for (const key of ['sharpe', 'calmar', 'p_value', 'dsr_family'] as const) {
      expect(out[key].value).toBeNull();
      expect(out[key].source.status).toBe('unavailable');
      expect(out[key].source.pending).toMatch(/N=11/);
    }
    // Descriptive quantities describe what happened — they survive.
    expect(out.return_pct.value).toBe(3.36);
    expect(out.n_trades.value).toBe(11);
    expect(out.max_dd_pct.value).toBe(1.5);
    expect(out.win_rate_pct.value).toBe(72.7);
  });

  it('with N>=20 leaves everything intact', () => {
    const out = suppressSmallSample(envWith(20));
    expect(out.insufficient_trades).toBe(false);
    expect(out.sharpe.value).toBe(1.9);
  });

  it('with an UNDETERMINABLE N fails closed (S-04)', () => {
    // The first version returned untouched when N was unknown ("absence of N is
    // not evidence of N<20"). Backwards for a publication guard: the manifests
    // that omit the count are exactly the ones with 1-3 trades (btc_hodl_b1
    // published Sharpe 0.793 / p=0.0242 off ONE trade). No N ⇒ no ratio.
    const out = suppressSmallSample(envWith(null));
    for (const key of ['sharpe', 'calmar', 'p_value', 'dsr_family'] as const) {
      expect(out[key].value).toBeNull();
      expect(out[key].source.status).toBe('unavailable');
      expect(out[key].source.pending).toMatch(/no determinable/);
    }
    expect(out.insufficient_trades).toBe(true);
    expect(out.return_pct.value).toBe(3.36);   // descriptive survives
  });

  it('with a published-but-null N also fails closed (the real artifact shape)', () => {
    const env = envWith(11);
    env.n_trades = sourced<number>(null, 'p');
    const out = suppressSmallSample(env);
    expect(out.sharpe.value).toBeNull();
    expect(out.insufficient_trades).toBe(true);
  });
});

// ────────────────────────────────────────────────────────── payload validators

function minimalPassport(overrides: Record<string, unknown> = {}) {
  const performance = Object.fromEntries(
    PASSPORT_ENVS.map((e) => [e, { ...envWith(25), env: e }]),
  );
  return {
    contract: PASSPORT_CONTRACT_ID,
    contract_version: '1.0.0',
    strategy_id: 'smart_simple_v11',
    generated_at: '2026-07-28T00:00:00Z',
    identity: {}, governance: {}, lineage: {},
    performance, live: {}, risk: {},
    ...overrides,
  };
}

describe('validateStrategyPassport', () => {
  it('accepts a well-formed passport', () => {
    expect(validateStrategyPassport(minimalPassport())).toEqual([]);
  });

  it('requires all five environments to be declared, even when empty', () => {
    const p = minimalPassport();
    delete (p.performance as Record<string, unknown>).canary;
    expect(validateStrategyPassport(p).join()).toMatch(/missing canary/);
  });

  it('rejects a Sharpe published next to N<20 (the §6 violation this exists to stop)', () => {
    const p = minimalPassport();
    (p.performance as Record<string, EnvPerformance>).live = envWith(3); // NOT suppressed
    const errors = validateStrategyPassport(p);
    expect(errors.join()).toMatch(/performance\.live\.sharpe: published with N=3/);
    expect(errors.join()).toMatch(/performance\.live\.p_value/);
  });

  it('rejects a Sharpe published with an UNDETERMINABLE N (S-04)', () => {
    const p = minimalPassport();
    const env = envWith(3);
    env.n_trades = unavailable<number>('el manifiesto no publica el conteo de trades');
    (p.performance as Record<string, EnvPerformance>).live = env;
    const errors = validateStrategyPassport(p).join();
    expect(errors).toMatch(/performance\.live\.sharpe/);
    expect(errors).toMatch(/N no determinable/);
    expect(errors).toMatch(/performance\.live\.p_value/);
  });

  it('rejects a Sharpe published with a published-but-null N (S-04)', () => {
    const p = minimalPassport();
    const env = envWith(3);
    env.n_trades = sourced<number>(null, 'public/data/strategies/x/manifest.json');
    (p.performance as Record<string, EnvPerformance>).live = env;
    const errors = validateStrategyPassport(p).join();
    expect(errors).toMatch(/performance\.live\.sharpe/);
    expect(errors).toMatch(/N no determinable/);
  });

  it('rejects any action affordance — the surface is DIAGNOSTIC', () => {
    for (const action of FORBIDDEN_PASSPORT_ACTIONS) {
      const errors = validateStrategyPassport(minimalPassport({ [action]: true }));
      expect(errors.join()).toMatch(new RegExp(`must not expose action .${action}`));
    }
  });
});

function minimalTower(overrides: Record<string, unknown> = {}) {
  return {
    contract: PASSPORT_CONTRACT_ID,
    contract_version: '1.0.0',
    generated_at: '2026-07-28T00:00:00Z',
    book: { state_counts: { CHAMPION: 1, PAPER: 2, CANARY: null, REDUCED: null, QUARANTINED: null } },
    sleeves: [{ strategy_id: 'smart_simple_v11', retirement_signal: 'unknown' }],
    data: { n_max_trials: sourced(N_MAX_TRIALS, 'src/contracts/passport.py') },
    paired_tests: [],
    pending_interfaces: [],
    ...overrides,
  };
}

describe('validateControlTower', () => {
  it('accepts a well-formed snapshot', () => {
    expect(validateControlTower(minimalTower())).toEqual([]);
  });

  it('pins N_MAX to the declared spend cap (it must never drift into a metric)', () => {
    const t = minimalTower({ data: { n_max_trials: sourced(500, 'x') } });
    expect(validateControlTower(t).join()).toMatch(/n_max_trials must be 989/);
  });

  it('rejects unknown book states', () => {
    const t = minimalTower({ book: { state_counts: { CHAMPION: 1, WINNER: 3 } } });
    expect(validateControlTower(t).join()).toMatch(/unknown states WINNER/);
  });

  it('rejects a sleeve with an invented retirement signal', () => {
    const t = minimalTower({ sleeves: [{ strategy_id: 'x', retirement_signal: 'probably_fine' }] });
    expect(validateControlTower(t).join()).toMatch(/retirement_signal: bad value/);
  });

  it('rejects a sleeve Sharpe with an undeterminable N (S-04, tower half)', () => {
    const t = minimalTower({
      sleeves: [{
        strategy_id: 'btc_hodl_b1',
        retirement_signal: 'unknown',
        n_trades: unavailable<number>('el manifiesto no publica el conteo de trades'),
        sharpe: sourced(0.793, 'public/data/registry.json'),
        dsr_family: sourced(0.8357, 'public/data/production/approval_state.json'),
      }],
    });
    const errors = validateControlTower(t).join();
    expect(errors).toMatch(/sleeves\[0\]\.sharpe/);
    expect(errors).toMatch(/N no determinable/);
    expect(errors).toMatch(/sleeves\[0\]\.dsr_family/);
  });

  it('rejects a sleeve Sharpe with N<20', () => {
    const t = minimalTower({
      sleeves: [{
        strategy_id: 'x', retirement_signal: 'unknown',
        n_trades: sourced(1, 'p'), sharpe: sourced(0.793, 'p'),
      }],
    });
    expect(validateControlTower(t).join()).toMatch(/published with N=1/);
  });

  it('leaves a sleeve with enough trades intact', () => {
    const t = minimalTower({
      sleeves: [{
        strategy_id: 'x', retirement_signal: 'unknown',
        n_trades: sourced(34, 'p'), sharpe: sourced(3.35, 'p'),
      }],
    });
    expect(validateControlTower(t)).toEqual([]);
  });

  it('rejects action affordances on the tower too', () => {
    expect(validateControlTower(minimalTower({ promote: {} })).join())
      .toMatch(/must not expose action .promote/);
  });
});

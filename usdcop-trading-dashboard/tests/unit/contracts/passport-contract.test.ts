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
  ENV_PERFORMANCE_FIELDS,
  ENV_PERFORMANCE_PLAIN_FIELDS,
  FORBIDDEN_PASSPORT_ACTIONS,
  HEALTH_CLOCKS,
  MIN_TRADES_FOR_RATIOS,
  N_MAX_TRIALS,
  PASSPORT_BLOCK_FIELDS,
  PASSPORT_CONTRACT_ID,
  PASSPORT_ENVS,
  PASSPORT_PLAIN_FIELDS,
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
  // F-07: el tercer reloj del §23 se llama `pnl` (datos/modelo/PnL), NO `exec`.
  // El productor real (`src/monitoring/system_health_contract.py::Clock`) emite
  // `pnl`; mientras el consumidor dijo `exec`, el reloj publicado se perdía.
  it('declares the three §23 clocks with the producer\'s vocabulary', () => {
    expect([...HEALTH_CLOCKS]).toEqual(['data', 'model', 'pnl']);
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

// Los bloques de abajo están DELIBERADAMENTE poblados. La primera versión de este
// fixture era `identity: {}, governance: {}, lineage: {}, live: {}, risk: {}` — es
// decir, el propio hueco que el validador no veía: un Passport con cero trials, cero
// DSR, cero linaje y cero riesgo pasaba con 0 errores. Un fixture más vacío que el
// payload real no puede detectar un payload vacío.
// Los nombres son espejo 1:1 de `PASSPORT_BLOCK_FIELDS` (passport.contract.ts), que a
// su vez el test Python ata a las `export interface`.

function identityBlock(): Record<string, unknown> {
  return {
    strategy_id: 'smart_simple_v11',
    asset_id: 'usdcop',
    display_name: 'Smart Simple v11',
    surface: 'action',
    engine_type: 'composite',
    status: 'production',
    active_version: sourced<string>('v11', 'public/data/registry.json'),
    timeframe: 'H5',
  };
}

function governanceBlock(): Record<string, unknown> {
  const gov = 'data/control-tower/governance.json';
  return {
    n_trials_total: sourced(63, gov),
    n_trials_forecast: sourced(21, gov),
    n_trials_action: sourced(42, gov),
    n_family: sourced(12, gov),
    n_cluster: sourced(30, gov),
    n_global: sourced(63, gov),
    dsr_family: sourced(0.83, 'data/approvals/smart_simple_v11.json'),
    dsr_bar: DSR_BAR,
    approval_status: sourced<string>('APPROVED', 'data/approvals/smart_simple_v11.json'),
    gates: sourced<unknown[]>([], 'data/approvals/smart_simple_v11.json'),
    withdrawal_protocol: sourced<string>(
      '.claude/specs/assets/usdcop/WITHDRAWAL-PROTOCOL.md', gov),
    retirement_signal: 'unknown',
    retirement_reason: 'sin evaluación de retiro POR ESTRATEGIA publicada (BL-25)',
  };
}

function lineageBlock(): Record<string, unknown> {
  return {
    model_versions: sourced<unknown[]>([], 'public/data/strategies/smart_simple_v11/manifest.json'),
    spec_fingerprint: unavailable<string>('BL-17 — fingerprints canónicos'),
    feature_set_hash: unavailable<string>('BL-39 — feature contracts por estrategia-versión'),
    policy_hash: unavailable<string>('BL-45 — motor de políticas (policy_hash/params_hash)'),
    lineage_graph: unavailable<unknown>('BL-24 — nodes/edges de linaje'),
  };
}

function liveBlock(): Record<string, unknown> {
  return {
    open_orders: unavailable('BL-21 — event sourcing exec.*'),
    last_fill_at: unavailable<string>('BL-21 — event sourcing exec.*'),
    quarantined: unavailable<boolean>('BL-21 — cuarentena por reconciliación'),
    reconciled: unavailable<boolean>('BL-21/BL-22 — reconciliación contra fills'),
    kill_switch_engaged: unavailable<boolean>('BL-30 — kill switch independiente de Airflow'),
    deploy_status: unavailable<string>('sin deploy_status.json publicado'),
    last_signal_at: sourced<string>('2026-07-27T13:00:00Z',
      'public/data/strategies/smart_simple_v11/manifest.json'),
  };
}

function riskBlock(): Record<string, unknown> {
  return {
    current_exposure: unavailable('BL-26 — portfolio_snapshot'),
    vol_target_pct: unavailable('BL-27 — allocator v1'),
    vol_forecast_pct: unavailable('BL-27 — allocator v1'),
    m_forward: unavailable('BL-27 — multiplicadores m_forward/m_dd'),
    m_dd: unavailable('BL-27 — multiplicadores m_forward/m_dd'),
    rho_max: unavailable('BL-26 — matriz de correlación entre sleeves'),
    turnover: unavailable('BL-22 — fact_position'),
  };
}

/** Los cinco bloques cuyo CONTENIDO exige el contrato, con el builder que produce una
 *  instancia realista de cada uno. Los tests negativos se conducen desde este mapa: un
 *  único test agregado se pondría verde en cuanto UNO de los bloques sobreviviese. */
const BLOCK_BUILDERS: Record<string, () => Record<string, unknown>> = {
  identity: identityBlock,
  governance: governanceBlock,
  lineage: lineageBlock,
  live: liveBlock,
  risk: riskBlock,
};

function minimalPassport(overrides: Record<string, unknown> = {}) {
  const performance = Object.fromEntries(
    PASSPORT_ENVS.map((e) => [e, { ...envWith(25), env: e }]),
  );
  return {
    contract: PASSPORT_CONTRACT_ID,
    contract_version: '1.0.0',
    strategy_id: 'smart_simple_v11',
    generated_at: '2026-07-28T00:00:00Z',
    identity: identityBlock(),
    governance: governanceBlock(),
    lineage: lineageBlock(),
    performance,
    live: liveBlock(),
    risk: riskBlock(),
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

// ───────────────────────────────────── contenido por bloque (BL-32, el hueco real)

/**
 * El validador comprobaba la PRESENCIA de las ocho claves de nivel superior y nada
 * más: `governance: {}`, `lineage: {}`, `risk: {}` validaban con 0 errores, o sea que
 * se podía publicar un Passport sin gobernanza, sin linaje y sin riesgo.
 *
 * Matiz honesto y motivo por el que estos tests NO exigen valores: `policy_hash` no
 * tiene productor hasta BL-45, `dsr_family` está `unavailable` para casi todas las
 * estrategias y un activo sin trials es un estado LEGÍTIMO. Lo que se exige es la
 * forma alternativa que el contrato ya define — `value: null` + `status:'unavailable'`
 * + `pending` no vacío. La regla es "sin agujeros MUDOS", no "sin agujeros".
 */
describe('per-block content (BL-32)', () => {
  // Rojo con: `identity: {}` (idem governance/lineage/live/risk) en minimalPassport,
  // que es EXACTAMENTE como estaba este fixture antes de este cambio.
  for (const [block, build] of Object.entries(BLOCK_BUILDERS)) {
    it(`rejects an empty ${block} block`, () => {
      const errors = validateStrategyPassport(minimalPassport({ [block]: {} }));
      for (const field of PASSPORT_BLOCK_FIELDS[block]) {
        expect(errors).toContain(`passport.${block}: missing '${field}'`);
      }
      // …y con el bloque poblado no sobra ningún error.
      expect(validateStrategyPassport(minimalPassport({ [block]: build() }))).toEqual([]);
    });
  }

  it('rejects governance without n_trials_total (cero trials, DSR sin deflactar)', () => {
    const gov = governanceBlock();
    delete gov.n_trials_total;
    expect(validateStrategyPassport(minimalPassport({ governance: gov })))
      .toContain("passport.governance: missing 'n_trials_total'");
  });

  it('rejects identity without strategy_id', () => {
    const identity = identityBlock();
    delete identity.strategy_id;
    expect(validateStrategyPassport(minimalPassport({ identity })))
      .toContain("passport.identity: missing 'strategy_id'");
  });

  it('rejects an env-performance block that declares the env and nothing else', () => {
    const performance = Object.fromEntries(PASSPORT_ENVS.map((e) => [e, { ...envWith(25), env: e }]));
    performance.paper = { env: 'paper' } as unknown as EnvPerformance;
    const errors = validateStrategyPassport(minimalPassport({ performance }));
    expect(errors).toContain("passport.performance.paper: missing 'n_trades'");
    expect(errors).toContain("passport.performance.paper: missing 'return_pct'");
  });

  it('rejects a MUTE hole: value null with an empty pending', () => {
    const gov = governanceBlock();
    gov.n_trials_total = { value: null, source: { path: null, status: 'unavailable', pending: '' } };
    expect(validateStrategyPassport(minimalPassport({ governance: gov })).join())
      .toMatch(/governance\.n_trials_total: unavailable fields MUST declare what they are pending on/);
  });

  it('rejects a field that is not Sourced at all (a bare `{}` says nothing)', () => {
    const gov = governanceBlock();
    gov.dsr_family = {};
    const errors = validateStrategyPassport(minimalPassport({ governance: gov })).join();
    expect(errors).toMatch(/governance\.dsr_family: missing 'value'/);
    expect(errors).toMatch(/governance\.dsr_family: missing\/invalid 'source'/);
  });

  // El test que impide que la regla degenere en "sin agujeros" y rompa el estado real:
  // `policy_hash` NO tiene productor hasta BL-45 y aun así el Passport debe validar.
  it('ACCEPTS a declared hole: value null + pending naming a BL', () => {
    const gov = governanceBlock();
    gov.n_trials_total = unavailable('BL-09/BL-10 — el activo no tiene conteo de trials proyectado');
    gov.dsr_family = unavailable('BL-18 — sin gate DSR publicado para esta estrategia');
    const lineage = lineageBlock();
    lineage.policy_hash = unavailable<string>('BL-45 — motor de políticas (policy_hash/params_hash)');
    expect(validateStrategyPassport(minimalPassport({ governance: gov, lineage }))).toEqual([]);
  });

  it('pins the constitutional bar and the retirement vocabulary inside the block', () => {
    const loose = governanceBlock();
    loose.dsr_bar = 0.5;
    expect(validateStrategyPassport(minimalPassport({ governance: loose })).join())
      .toMatch(/governance\.dsr_bar must be 0\.95/);
    const invented = governanceBlock();
    invented.retirement_signal = 'probably_fine';
    expect(validateStrategyPassport(minimalPassport({ governance: invented })).join())
      .toMatch(/governance\.retirement_signal: bad value/);
  });

  it('declares the plain fields as a SUBSET of the block fields (mirror sanity)', () => {
    for (const [block, plain] of Object.entries(PASSPORT_PLAIN_FIELDS)) {
      for (const field of plain) {
        expect(PASSPORT_BLOCK_FIELDS[block], `${block}.${field}`).toContain(field);
      }
    }
    for (const field of ENV_PERFORMANCE_PLAIN_FIELDS) {
      expect(ENV_PERFORMANCE_FIELDS).toContain(field);
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

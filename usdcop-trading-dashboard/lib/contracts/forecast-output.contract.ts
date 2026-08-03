/**
 * Forecast Output Contract (DIAGNOSTIC surface)
 * =============================================
 * Typed prediction record for the forecasting zoo (FABRIC §15.2 / BL-15).
 *
 * A ForecastOutput is a DIAGNOSTIC artifact: it estimates a price/return for a
 * horizon and NOTHING else — no decision, no PnL, no orders. A ForecastOutput is
 * rejected by TYPE before any logic runs:
 *
 * QUE SE APLICA HOY, Y QUE NO (BL-15, corregido 2026-08-03) — espejo exacto del
 * docstring de src/contracts/forecast_output.py:
 *   Hasta 2026-08-03 este bloque nombraba strategy_output como el unico registro
 *   admitido por el libro. Ese tipo NO EXISTE en el codigo: era un nombre sin
 *   referente afirmado en presente. Lo cierto es que el rechazo
 *   de ForecastOutput por tipo si es real; que el canal accionable existente es
 *   UniversalSignalRecord (src/contracts/signal_contract.py); que la barrera que
 *   de verdad se ejecuta es de esquemas de URI en src/orchestration/dataset_uri.py;
 *   y que ningun allocator valida aun su entrada por contrato — brecha viva de BL-15.
 *   - `diagnostic_only` is the literal type `true` — a record claiming to be
 *     actionable does not typecheck.
 *   - The shape shares no fields with StrategyTrade (strategy.contract.ts).
 *
 * INGEST WALL (BL-15 remedio)
 * ---------------------------
 * `parseForecastOutput()` / `ingestForecastOutputs()` are the only supported way
 * an untrusted payload (file, API, CSV) becomes a record, and they are
 * fail-closed: unknown keys are REJECTED (never dropped), `diagnostic_only` must
 * arrive as the literal `true` (never coerced), and every record is validated.
 *
 * TIMESTAMPS (strict, bilateral)
 * ------------------------------
 * Grammar accepted by BOTH runtimes (this file and src/contracts/forecast_output.py):
 *
 *     YYYY-MM-DDTHH:MM:SS[.ffffff][Z|±HH:MM]
 *
 * with a REAL calendar check (leap years), HH<=23, MM<=59, SS<=59 and offsets in
 * ±00:00..±23:59. `Date.parse` is NOT used: it silently ROLLS impossible dates
 * (`2026-02-30` -> `2026-03-02`), accepts a space separator and interprets naive
 * strings in the machine's local zone — three ways for TypeScript to accept what
 * Python rejects. The parser below is deterministic and timezone-independent.
 *
 * The three timestamps must share tz-awareness: all aware or all naive. Mixed is
 * a contract rejection on both sides (`.claude/rules/data-governance.md`: COP is
 * America/Bogota, XAU/BTC are instant-based TIMESTAMPTZ — mixing conventions has
 * no defined ordering).
 *
 * Spec: .claude/specs/planes/04-CTR-QLAB-FABRIC-004.md §15.2 +
 *       .claude/specs/planes/backlog/BL-15-contrato-forecast-output.md
 * Python mirror: src/contracts/forecast_output.py
 * Shared case table: tests/fixtures/forecast_output_cases.v1.json (both runners)
 * Contract: CTR-FORECAST-OUTPUT-001
 */

// -----------------------------------------------------------------------------
// Constants (mirrored in forecast_output.py — change BOTH sides)
// -----------------------------------------------------------------------------

export const CONTRACT_ID = 'CTR-FORECAST-OUTPUT-001';

/**
 * Allowed prediction types. A "direction" belief is NOT a prediction type —
 * it goes in `direction_probability` (and a raw score is not a probability;
 * see quant-constitution).
 */
export const PREDICTION_TYPES = ['return', 'log_return', 'price'] as const;

export type PredictionType = (typeof PREDICTION_TYPES)[number];

/** Closed field sets — anything else is an ingest-wall rejection. */
export const FORECAST_OUTPUT_FIELDS = [
  'forecast_id', 'forecast_spec_id', 'asset', 'model_id', 'horizon',
  'as_of', 'available_at', 'target_time',
  'prediction', 'model_fingerprint', 'data_snapshot_id',
  'direction_probability', 'diagnostic_only',
] as const;

export const PREDICTION_FIELDS = ['type', 'point', 'lower', 'upper'] as const;

export const DIRECTION_PROBABILITY_FIELDS = ['up'] as const;

/** Required non-empty string identifiers. */
export const REQUIRED_STRING_FIELDS = [
  'forecast_id', 'forecast_spec_id', 'asset', 'model_id', 'horizon',
  'model_fingerprint', 'data_snapshot_id',
] as const;

export const TIMESTAMP_FIELDS = ['as_of', 'available_at', 'target_time'] as const;

/** Strict ISO8601 grammar — same characters as the Python mirror. */
const ISO8601_RE =
  /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(\.\d{1,6})?(Z|[+-]\d{2}:\d{2})?$/;

/** Thrown by the ingest wall (mirror of Python's ForecastOutputError). */
export class ForecastOutputError extends Error {
  readonly errors: string[];
  constructor(errors: string[]) {
    super(errors.join('; '));
    this.name = 'ForecastOutputError';
    this.errors = errors;
  }
}

// -----------------------------------------------------------------------------
// Nested records
// -----------------------------------------------------------------------------

/** Point prediction with optional interval bounds. */
export interface ForecastPrediction {
  type: PredictionType;
  point: number;
  lower?: number | null;   // interval lower bound (same unit as point)
  upper?: number | null;   // interval upper bound
}

/** Calibrated probability that the target moves up over the horizon. */
export interface DirectionProbability {
  up: number;              // in [0, 1]
}

// -----------------------------------------------------------------------------
// Core record
// -----------------------------------------------------------------------------

/**
 * One model x horizon prediction, point-in-time honest.
 *
 * `as_of` is the information cutoff, `available_at` is when the forecast
 * physically existed (>= as_of, anti-look-ahead), `target_time` is the
 * timestamp the prediction refers to (> as_of).
 */
export interface ForecastOutput {
  // --- Identity ---
  forecast_id: string;             // unique id, e.g. uuid
  forecast_spec_id: string;        // e.g. "usdcop_forecast_zoo_v3"
  asset: string;                   // e.g. "usdcop"
  model_id: string;                // e.g. "ridge_v2"
  horizon: string;                 // e.g. "5d"

  // --- Point-in-time ---
  as_of: string;                   // ISO8601 — information cutoff
  available_at: string;            // ISO8601 — when the forecast existed (>= as_of)
  target_time: string;             // ISO8601 — what the prediction refers to (> as_of)

  // --- Prediction ---
  prediction: ForecastPrediction;

  // --- Lineage ---
  model_fingerprint: string;       // e.g. "sha256:..."
  data_snapshot_id: string;

  // --- Optional belief ---
  direction_probability?: DirectionProbability | null;

  // --- Wall: literal `true` — an "actionable forecast" does not typecheck ---
  diagnostic_only: true;
}

// -----------------------------------------------------------------------------
// Timestamp parsing (strict — Date.parse is deliberately NOT used)
// -----------------------------------------------------------------------------

interface ParsedTimestamp {
  /** Whole seconds since the epoch for a validated instant. Naive strings are
   *  compared against each other on the same wall-clock reference (never the
   *  machine's local zone). Seconds and microseconds are kept APART so the
   *  comparison stays exact: a single microsecond counter overflows the 2**53
   *  exact-integer range past ~year 2255, which would silently make two
   *  far-future timestamps compare equal in TS while Python distinguishes them. */
  seconds: number;
  micros: number;
  aware: boolean;
}

/** -1 | 0 | 1 — exact ordering of two parsed timestamps. */
function cmpTimestamp(a: ParsedTimestamp, b: ParsedTimestamp): number {
  if (a.seconds !== b.seconds) return a.seconds < b.seconds ? -1 : 1;
  if (a.micros !== b.micros) return a.micros < b.micros ? -1 : 1;
  return 0;
}

/**
 * Howard Hinnant's days-from-civil — the same integer arithmetic Python's
 * `datetime` performs internally, and the same helper the policy_version
 * mirror uses.
 *
 * `Date.UTC` is deliberately NOT used here: it remaps years 0-99 onto
 * 1900-1999 (a legacy `Date` quirk), so `0099-12-31` became 1999-12-31 while
 * `0100-01-01` stayed in year 100 — inverting an ordering that Python gets
 * right and making the two runtimes disagree on a record neither grammar
 * rejects.
 */
function daysFromCivil(year: number, month: number, day: number): number {
  const y = year - (month <= 2 ? 1 : 0);
  const era = Math.floor((y >= 0 ? y : y - 399) / 400);
  const yoe = y - era * 400;
  const doy = Math.floor((153 * (month + (month > 2 ? -3 : 9)) + 2) / 5) + day - 1;
  const doe = yoe * 365 + Math.floor(yoe / 4) - Math.floor(yoe / 100) + doy;
  return era * 146097 + doe - 719468;
}

function daysInMonth(year: number, month: number): number {
  if (month === 2) {
    const leap = (year % 4 === 0 && year % 100 !== 0) || year % 400 === 0;
    return leap ? 29 : 28;
  }
  return [4, 6, 9, 11].includes(month) ? 30 : 31;
}

function parseContractTimestamp(
  name: string,
  value: unknown,
): { parsed: ParsedTimestamp | null; error: string | null } {
  if (typeof value !== 'string' || value === '') {
    return { parsed: null, error: `${name} must be a non-empty ISO8601 string` };
  }
  const m = ISO8601_RE.exec(value);
  if (!m) {
    return {
      parsed: null,
      error: `${name} is not strict ISO8601 (YYYY-MM-DDTHH:MM:SS[.ffffff][Z|±HH:MM]): ${JSON.stringify(value)}`,
    };
  }

  const year = Number(m[1]);
  const month = Number(m[2]);
  const day = Number(m[3]);
  const hour = Number(m[4]);
  const minute = Number(m[5]);
  const second = Number(m[6]);
  const frac = m[7];
  const offsetRaw = m[8];

  // Year 0000 is outside `datetime`'s domain (Python raises for year 0), so the
  // mirror rejects it explicitly instead of one side accepting what the other
  // cannot even construct. Supported domain on BOTH sides: 0001..9999.
  if (year < 1) {
    return { parsed: null, error: `${name} has an impossible year: ${JSON.stringify(value)}` };
  }
  if (month < 1 || month > 12) {
    return { parsed: null, error: `${name} has an impossible month: ${JSON.stringify(value)}` };
  }
  if (day < 1 || day > daysInMonth(year, month)) {
    return { parsed: null, error: `${name} has an impossible calendar date: ${JSON.stringify(value)}` };
  }
  if (hour > 23 || minute > 59 || second > 59) {
    return { parsed: null, error: `${name} has an impossible time: ${JSON.stringify(value)}` };
  }

  const micros = frac ? Math.round(Number(frac) * 1_000_000) : 0;

  let offsetMinutes = 0;
  let aware = false;
  if (offsetRaw !== undefined) {
    aware = true;
    if (offsetRaw !== 'Z') {
      const sign = offsetRaw[0] === '+' ? 1 : -1;
      const oh = Number(offsetRaw.slice(1, 3));
      const om = Number(offsetRaw.slice(4, 6));
      if (oh > 23 || om > 59) {
        return { parsed: null, error: `${name} has an impossible UTC offset: ${JSON.stringify(value)}` };
      }
      offsetMinutes = sign * (oh * 60 + om);
    }
  }

  // Pure integer arithmetic (all components already validated), so the result is
  // independent of the machine's timezone AND of `Date`'s legacy year remapping.
  const epochSeconds =
    daysFromCivil(year, month, day) * 86400 +
    hour * 3600 +
    minute * 60 +
    second -
    offsetMinutes * 60;

  return { parsed: { seconds: epochSeconds, micros, aware }, error: null };
}

// -----------------------------------------------------------------------------
// Validation (exact mirror of validate_forecast_payload() in Python)
// -----------------------------------------------------------------------------

/**
 * Finite AND bilaterally representable.
 *
 * JS has a single float64 numeric type, so an INTEGER-VALUED magnitude above
 * 2**53-1 loses precision the moment `JSON.parse` reads it (9007199254740993
 * silently became ...992) while Python keeps the literal exact — the two
 * runtimes then disagree about whether `lower <= point`. Such values are
 * rejected on BOTH sides.
 *
 * The rule is stated as "integer-valued and outside the safe range" precisely
 * because that is decidable from TypeScript, where the int/float distinction is
 * already gone by the time validation runs. It is deliberately over-strict for
 * huge float64 integers such as 1e300 (rejected on both sides): a prediction of
 * that magnitude is meaningless in every unit this contract carries, and
 * symmetry is worth more than reach.
 */
function finite(v: unknown): v is number {
  if (typeof v !== 'number' || !Number.isFinite(v)) return false;
  if (Number.isInteger(v) && !Number.isSafeInteger(v)) return false;
  return true;
}

/** Python's `repr()` for the values that reach these messages. */
function pyRepr(v: unknown): string {
  if (typeof v === 'string') return `'${v}'`;
  if (v === null || v === undefined) return 'None';
  if (v === true) return 'True';
  if (v === false) return 'False';
  return String(v);
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/**
 * Structural + semantic validation of an untrusted record.
 * Returns the list of contract violations (empty array = valid).
 * No NaN/Infinity anywhere; unknown fields rejected; diagnostic_only exactly true.
 */
export function validateForecastOutput(raw: unknown): string[] {
  if (!isPlainObject(raw)) {
    return ['forecast_output must be an object'];
  }
  const o = raw;
  const errors: string[] = [];

  const unknown = Object.keys(o)
    .filter((k) => !(FORECAST_OUTPUT_FIELDS as readonly string[]).includes(k))
    .sort();
  if (unknown.length > 0) {
    errors.push(`unknown field(s) not in ${CONTRACT_ID}: [${unknown.map((k) => `'${k}'`).join(', ')}]`);
  }

  for (const name of REQUIRED_STRING_FIELDS) {
    const v = o[name];
    if (typeof v !== 'string' || !v.trim()) {
      errors.push(`${name} must be a non-empty string`);
    }
  }

  // --- timestamps: strict grammar, then awareness homogeneity, then order ---
  const parsed: Record<string, ParsedTimestamp | null> = {};
  for (const name of TIMESTAMP_FIELDS) {
    const { parsed: p, error } = parseContractTimestamp(name, o[name]);
    parsed[name] = p;
    if (error) errors.push(error);
  }

  const allParsed = TIMESTAMP_FIELDS.every((n) => parsed[n] !== null);
  if (allParsed) {
    const awareFlags = TIMESTAMP_FIELDS.map((n) => parsed[n]!.aware);
    if (new Set(awareFlags).size > 1) {
      const detail = TIMESTAMP_FIELDS
        .map((n) => `${n}=${parsed[n]!.aware ? 'aware' : 'naive'}`)
        .join(', ');
      errors.push(
        'timestamps must be all timezone-aware or all naive — mixed naive/aware ' +
          `has no defined ordering (${detail})`,
      );
    } else {
      const asOf = parsed.as_of!;
      const availableAt = parsed.available_at!;
      const targetTime = parsed.target_time!;
      if (cmpTimestamp(availableAt, asOf) < 0) {
        errors.push('available_at must be >= as_of (anti-look-ahead)');
      }
      if (cmpTimestamp(targetTime, asOf) <= 0) {
        errors.push('target_time must be > as_of');
      }
    }
  }

  // --- prediction ----------------------------------------------------------
  const p = o.prediction;
  if (!isPlainObject(p)) {
    errors.push('prediction must be an object');
  } else {
    const unknownP = Object.keys(p)
      .filter((k) => !(PREDICTION_FIELDS as readonly string[]).includes(k))
      .sort();
    if (unknownP.length > 0) {
      errors.push(
        `prediction has unknown field(s) not in ${CONTRACT_ID}: [${unknownP.map((k) => `'${k}'`).join(', ')}]`,
      );
    }
    if (!(PREDICTION_TYPES as readonly unknown[]).includes(p.type)) {
      errors.push(`prediction.type ${pyRepr(p.type)} not in ${'(' + PREDICTION_TYPES.map((t) => `'${t}'`).join(', ') + ')'}`);
    }
    const point = p.point;
    const lower = p.lower;
    const upper = p.upper;
    checkFinite('prediction.point', point, errors, false);
    checkFinite('prediction.lower', lower, errors, true);
    checkFinite('prediction.upper', upper, errors, true);
    if (finite(lower) && finite(upper) && finite(point)) {
      if (lower > upper) {
        errors.push('prediction.lower must be <= prediction.upper');
      } else if (!(lower <= point && point <= upper)) {
        errors.push('prediction.point must lie within [lower, upper]');
      }
    }
  }

  // --- direction_probability (optional) ------------------------------------
  const dp = o.direction_probability;
  if (dp !== null && dp !== undefined) {
    if (!isPlainObject(dp)) {
      errors.push('direction_probability must be an object');
    } else {
      const unknownDp = Object.keys(dp)
        .filter((k) => !(DIRECTION_PROBABILITY_FIELDS as readonly string[]).includes(k))
        .sort();
      if (unknownDp.length > 0) {
        errors.push(
          `direction_probability has unknown field(s) not in ${CONTRACT_ID}: [${unknownDp.map((k) => `'${k}'`).join(', ')}]`,
        );
      }
      const up = dp.up;
      if (!finite(up) || up < 0 || up > 1) {
        errors.push('direction_probability.up must be a finite number in [0, 1]');
      }
    }
  }

  // --- the wall ------------------------------------------------------------
  if (o.diagnostic_only !== true) {
    errors.push('diagnostic_only must be true — forecasts are DIAGNOSTIC');
  }

  return errors;
}

function checkFinite(name: string, value: unknown, errors: string[], allowNull: boolean): void {
  if (value === null || value === undefined) {
    if (!allowNull) errors.push(`${name} is required`);
    return;
  }
  if (!finite(value)) {
    errors.push(`${name} must be a finite number (no NaN/Inf), got ${pyRepr(value)}`);
  }
}

/** Type guard for untrusted JSON (files, API responses). */
export function isForecastOutput(raw: unknown): raw is ForecastOutput {
  return validateForecastOutput(raw).length === 0;
}

// -----------------------------------------------------------------------------
// Ingest wall
// -----------------------------------------------------------------------------

/** THE ingest wall: an untrusted payload becomes a record, or nothing. */
export function parseForecastOutput(raw: unknown): ForecastOutput {
  const errors = validateForecastOutput(raw);
  if (errors.length > 0) throw new ForecastOutputError(errors);
  return raw as ForecastOutput;
}

/** All-or-nothing ingest of a batch (one bad row = nothing enters). */
export function ingestForecastOutputs(rows: readonly unknown[]): ForecastOutput[] {
  const problems: string[] = [];
  const out: ForecastOutput[] = [];
  rows.forEach((row, i) => {
    const errors = validateForecastOutput(row);
    if (errors.length > 0) problems.push(`[${i}] ${errors.join('; ')}`);
    else out.push(row as ForecastOutput);
  });
  if (problems.length > 0) {
    throw new ForecastOutputError([
      `${problems.length} row(s) rejected by ${CONTRACT_ID}: ${problems.join(' | ')}`,
    ]);
  }
  return out;
}

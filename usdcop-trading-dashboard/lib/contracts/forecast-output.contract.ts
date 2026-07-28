/**
 * Forecast Output Contract (DIAGNOSTIC surface)
 * =============================================
 * Typed prediction record for the forecasting zoo (FABRIC §15.2 / BL-15).
 *
 * A ForecastOutput is a DIAGNOSTIC artifact: it estimates a price/return for a
 * horizon and NOTHING else — no decision, no PnL, no orders. The allocator/book
 * accepts exclusively strategy_output records; a ForecastOutput is rejected by
 * TYPE before any logic runs:
 *   - `diagnostic_only` is the literal type `true` — a record claiming to be
 *     actionable does not typecheck.
 *   - The shape shares no fields with StrategyTrade (strategy.contract.ts).
 *
 * Spec: .claude/specs/planes/04-CTR-QLAB-FABRIC-004.md §15.2 +
 *       .claude/specs/planes/backlog/BL-15-contrato-forecast-output.md
 * Python mirror: src/contracts/forecast_output.py
 * Contract: CTR-FORECAST-OUTPUT-001
 */

// -----------------------------------------------------------------------------
// Constants (mirrored in forecast_output.py — change BOTH sides)
// -----------------------------------------------------------------------------

/**
 * Allowed prediction types. A "direction" belief is NOT a prediction type —
 * it goes in `direction_probability` (and a raw score is not a probability;
 * see quant-constitution).
 */
export const PREDICTION_TYPES = ['return', 'log_return', 'price'] as const;

export type PredictionType = (typeof PREDICTION_TYPES)[number];

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
// Validation (mirrors ForecastOutput.validate())
// -----------------------------------------------------------------------------

function finite(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

function parseTs(v: unknown): number | null {
  if (typeof v !== 'string' || !v.trim()) return null;
  const t = Date.parse(v);
  return Number.isNaN(t) ? null : t;
}

/**
 * Structural + numeric validation of an untrusted record.
 * Returns the list of contract violations (empty array = valid).
 * No NaN/Infinity anywhere; diagnostic_only must be exactly true.
 */
export function validateForecastOutput(raw: unknown): string[] {
  const errors: string[] = [];
  if (typeof raw !== 'object' || raw === null) {
    return ['forecast_output must be an object'];
  }
  const o = raw as Record<string, unknown>;

  for (const name of [
    'forecast_id', 'forecast_spec_id', 'asset', 'model_id',
    'horizon', 'model_fingerprint', 'data_snapshot_id',
  ]) {
    const v = o[name];
    if (typeof v !== 'string' || !v.trim()) {
      errors.push(`${name} must be a non-empty string`);
    }
  }

  const asOf = parseTs(o.as_of);
  const availableAt = parseTs(o.available_at);
  const targetTime = parseTs(o.target_time);
  if (asOf === null) errors.push('as_of is not valid ISO8601');
  if (availableAt === null) errors.push('available_at is not valid ISO8601');
  if (targetTime === null) errors.push('target_time is not valid ISO8601');
  if (asOf !== null && availableAt !== null && availableAt < asOf) {
    errors.push('available_at must be >= as_of (anti-look-ahead)');
  }
  if (asOf !== null && targetTime !== null && targetTime <= asOf) {
    errors.push('target_time must be > as_of');
  }

  const p = o.prediction as Record<string, unknown> | undefined;
  if (typeof p !== 'object' || p === null) {
    errors.push('prediction must be an object');
  } else {
    if (!PREDICTION_TYPES.includes(p.type as PredictionType)) {
      errors.push(`prediction.type ${String(p.type)} not in [${PREDICTION_TYPES.join(', ')}]`);
    }
    if (!finite(p.point)) errors.push('prediction.point must be a finite number (no NaN/Inf)');
    if (p.lower != null && !finite(p.lower)) errors.push('prediction.lower must be a finite number (no NaN/Inf)');
    if (p.upper != null && !finite(p.upper)) errors.push('prediction.upper must be a finite number (no NaN/Inf)');
    if (finite(p.lower) && finite(p.upper) && finite(p.point)) {
      if (p.lower > p.upper) {
        errors.push('prediction.lower must be <= prediction.upper');
      } else if (!(p.lower <= p.point && p.point <= p.upper)) {
        errors.push('prediction.point must lie within [lower, upper]');
      }
    }
  }

  const dp = o.direction_probability as Record<string, unknown> | null | undefined;
  if (dp != null) {
    if (typeof dp !== 'object') {
      errors.push('direction_probability must be an object');
    } else if (!finite(dp.up) || (dp.up as number) < 0 || (dp.up as number) > 1) {
      errors.push('direction_probability.up must be a finite number in [0, 1]');
    }
  }

  if (o.diagnostic_only !== true) {
    errors.push('diagnostic_only must be true — forecasts are DIAGNOSTIC');
  }

  return errors;
}

/** Type guard for untrusted JSON (files, API responses). */
export function isForecastOutput(raw: unknown): raw is ForecastOutput {
  return validateForecastOutput(raw).length === 0;
}

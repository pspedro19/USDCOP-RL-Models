/**
 * Contrato de REPLAY DIARIO — C039 (CTR-REPLAY-001).
 *
 * Espejo de `src/contracts/replay_schema.py`. Los dos lados se cambian en el MISMO commit:
 * un espejo que se actualiza solo por un lado deja de ser un espejo y se convierte en dos
 * contratos que se contradicen en silencio.
 *
 * QUE PROBLEMA CIERRA
 * -------------------
 * Los bundles publicaban su serie diaria reducida a `{d, eq}` mientras el productor calculaba
 * el stream completo: exposicion ejecutada, bruto, coste y neto. Reconstruir desde
 * `precio x leverage` difiere de la equity publicada **hasta 6.60 pp por trade** en SPX,
 * porque el PnL del motor viene de `open_to_open_return` mientras `entry_price`/`exit_price`
 * son niveles de CIERRE usados solo como referencia.
 *
 * POR QUE `returnConvention` ES OBLIGATORIA
 * -----------------------------------------
 * El tipo es generico: BTC (24/7), Gold (metals) y SPX (exchange hours) NO tienen por que
 * compartir convencion. Un consumidor que compare contra la serie equivocada obtiene un numero
 * plausible y no se entera.
 *
 * POR QUE `exposureExec` Y NO `targetExposure`
 * --------------------------------------------
 * La fuente es `weights_exec`: exposicion EJECUTADA, no un objetivo.
 */

export const REPLAY_CONTRACT_ID = 'CTR-REPLAY-001' as const;

/** Tolerancia de la invariante 1 (net === gross - cost). Espeja TOLERANCIA_NETO. */
export const REPLAY_NET_TOLERANCE = 1e-9;

/** Tolerancia de la invariante 2 (recurrencia de equity). Espeja TOLERANCIA_EQUITY_REL. */
export const REPLAY_EQUITY_REL_TOLERANCE = 1e-4;

export const RETURN_CONVENTIONS = ['open_to_open', 'close_to_close'] as const;
export type ReturnConvention = (typeof RETURN_CONVENTIONS)[number];

/**
 * Una fila = un dia de la serie reconstruible.
 * Campos DECIMALES (0.01 = 1%), nunca porcentajes.
 */
export interface DailyReplayRow {
  d: string;                    // "YYYY-MM-DD"
  eq: number;                   // equity al cierre del dia, en moneda
  exposure_exec: number;        // exposicion EJECUTADA (weights_exec)
  gross_return_decimal: number; // retorno del dia ANTES de costes
  cost_return_decimal: number;  // coste del dia, POSITIVO (se resta del bruto)
  net_return_decimal: number;   // gross - cost
}

export interface DailyReplayDocument {
  kind: 'daily_replay';
  strategy_id: string;
  year: number;
  initial_capital: number;
  /** OBLIGATORIA. Sin ella el consumidor no sabe contra que serie del activo comparar. */
  return_convention: ReturnConvention;
  rows: DailyReplayRow[];
}

/**
 * Documento legacy `{kind: 'daily_equity', rows: [{d, eq}]}`. Se tipa para poder LEER bundles
 * antiguos, pero el consumidor cuantitativo debe fallar cerrado ante el: no tiene exposicion ni
 * descomposicion de costes, y adivinarlas es justo el error que C039 cierra.
 */
export interface LegacyDailyEquityDocument {
  kind: 'daily_equity';
  initial_capital: number;
  rows: Array<{ d: string; eq: number }>;
}

export function isDailyReplayDocument(
  doc: DailyReplayDocument | LegacyDailyEquityDocument,
): doc is DailyReplayDocument {
  return doc.kind === 'daily_replay';
}

/** Invariantes 1 y 4 sobre una fila. Devuelve los fallos; vacio = conforme. */
export function validateReplayRow(row: DailyReplayRow): string[] {
  const fallos: string[] = [];
  const numericos: Array<[string, number]> = [
    ['eq', row.eq],
    ['exposure_exec', row.exposure_exec],
    ['gross_return_decimal', row.gross_return_decimal],
    ['cost_return_decimal', row.cost_return_decimal],
    ['net_return_decimal', row.net_return_decimal],
  ];
  for (const [nombre, v] of numericos) {
    if (v === null || v === undefined || !Number.isFinite(v)) {
      fallos.push(`${row.d}: ${nombre} no es finito (${String(v)})`);
    }
  }
  if (fallos.length > 0) return fallos;

  const esperado = row.gross_return_decimal - row.cost_return_decimal;
  if (Math.abs(esperado - row.net_return_decimal) > REPLAY_NET_TOLERANCE) {
    fallos.push(`${row.d}: net=${row.net_return_decimal} pero gross-cost=${esperado}`);
  }
  // El coste es POSITIVO por convencion. Si el espejo lo invirtiera, la invariante 1 seguiria
  // cuadrando y el error pasaria desapercibido: por eso se comprueba aparte.
  if (row.cost_return_decimal < 0) {
    fallos.push(
      `${row.d}: cost_return_decimal=${row.cost_return_decimal} es NEGATIVO; ` +
        'la convencion es positivo-se-resta',
    );
  }
  return fallos;
}

/** Invariantes 1-4 sobre el documento entero. */
export function validateReplayDocument(doc: DailyReplayDocument): string[] {
  const fallos: string[] = [];
  if (!RETURN_CONVENTIONS.includes(doc.return_convention)) {
    fallos.push(
      `return_convention=${String(doc.return_convention)} no esta declarada; ` +
        `validas: ${RETURN_CONVENTIONS.join(', ')}`,
    );
  }
  for (const row of doc.rows) fallos.push(...validateReplayRow(row));
  if (fallos.length > 0) return fallos;

  let equity = doc.initial_capital;
  for (const row of doc.rows) {
    equity *= 1 + row.net_return_decimal;
    if (Math.abs(equity - row.eq) / Math.max(Math.abs(row.eq), 1e-9) > REPLAY_EQUITY_REL_TOLERANCE) {
      fallos.push(`${row.d}: eq publicada ${row.eq} pero la recurrencia da ${equity}`);
      break; // a partir del primer desajuste el resto es consecuencia
    }
  }
  return fallos;
}

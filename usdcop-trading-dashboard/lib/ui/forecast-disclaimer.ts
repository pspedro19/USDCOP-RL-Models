/**
 * Forecast disclaimer — SSOT del caveat de honestidad de /forecasting (BL-02/BL-03/BL-04).
 *
 * CTR-QUANT-CONSTITUTION-001: la superficie de forecasting es DIAGNÓSTICA, no de señales.
 * Este archivo es el ÚNICO origen del texto del banner; lo consumen:
 *   - components/gm/views/ForecastingView.tsx        (vista GM activa)
 *   - components/forecasting/ForecastingDashboard.tsx (legacy /legacy/forecasting)
 * Regla 6 FABRIC: toda métrica/mensaje de gobierno con definición única — no dupliques
 * estos strings; impórtalos.
 */

/** data-testid compartido del banner (los tests de BL-01 lo exigen en ambas vistas). */
export const FORECAST_DISCLAIMER_TESTID = 'da-caveat';

/** Titular fuerte del banner — visible en TODA superficie de forecasting (BL-02). */
export const FORECAST_DISCLAIMER_HEADLINE = 'DIAGNÓSTICO — NO ES UNA SEÑAL DE INVERSIÓN';

/** Rama direccional (replay causal USD/COP) — encabezado en negrita. */
export const FORECAST_DISCLAIMER_DIRECTIONAL_TITLE = 'Replay shadow, no señal ejecutable.';

/** Rama direccional — cuerpo. */
export const FORECAST_DISCLAIMER_DIRECTIONAL_BODY =
  'Los horizontes se seleccionan solo con resultados maduros y pueden abstenerse. El '
  + 'contrato permanece bloqueado para trading hasta superar estabilidad, costos y forward paper.';

/** Rama zoo/genérica (model zoo y weekly inference) — encabezado en negrita. */
export const FORECAST_DISCLAIMER_ZOO_TITLE = 'Superficie de diagnóstico, no de señales.';

/** Rama zoo/genérica — cuerpo. */
export const FORECAST_DISCLAIMER_ZOO_BODY =
  'La precisión direccional media de estos modelos es ≈52% — estadísticamente '
  + 'indistinguible de una moneda al aire tras ajustar por los 9 modelos probados.';

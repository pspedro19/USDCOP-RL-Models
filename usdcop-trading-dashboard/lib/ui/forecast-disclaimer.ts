/**
 * Forecast disclaimer — SSOT del wording honesto de /forecasting (BL-02/BL-03/BL-04).
 *
 * CTR-QUANT-CONSTITUTION-001: la superficie de forecasting es DIAGNÓSTICA, no de señales.
 * Este archivo es el ÚNICO origen del texto del banner y de las etiquetas neutras que lo
 * acompañan (dirección, columna de acierto); lo consumen:
 *   - components/gm/views/ForecastingView.tsx         (vista GM activa)
 *   - components/forecasting/ForecastingDashboard.tsx (legacy /legacy/forecasting)
 *   - components/forecasting/WeeklyInferenceView.tsx  (legacy weekly Gold/BTC)
 * Regla 6 FABRIC: toda métrica/mensaje de gobierno con definición única — no dupliques
 * estos strings; impórtalos.
 *
 * REGLA DURA (CXD-032 re-remediación): **ningún constante de este módulo contiene una
 * cifra**. Una afirmación numérica de desempeño solo es honesta si se deriva de los datos
 * de LA superficie donde se muestra; un número congelado en el copy migra a superficies
 * donde es falso (así el "≈52% / 9 modelos" del model zoo USD/COP acabó afirmándose sobre
 * la inferencia semanal de Oro, que es una política de REGLAS y no corre ningún zoo).
 * El candado estático `test_disclaimer_copy_carries_no_hardcoded_numbers` lo impide.
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

/** Rama model zoo (USD/COP y BTC: las 9 familias ML) — encabezado en negrita. */
export const FORECAST_DISCLAIMER_ZOO_TITLE = 'Superficie de diagnóstico, no de señales.';

/**
 * Rama model zoo — cuerpo. SIN cifras: la precisión direccional real se publica modelo a
 * modelo en la propia pantalla y cada superficie deriva la suya (el legacy añade la media
 * de la carga como línea extra). Un ≈52% congelado aquí sería falso en BTC (DA ≈0.46) y
 * doblemente falso en las superficies de reglas.
 */
export const FORECAST_DISCLAIMER_ZOO_BODY =
  'La precisión direccional medida de estos modelos —publicada modelo a modelo en esta '
  + 'misma pantalla— es indistinguible de una moneda al aire tras ajustar por el número '
  + 'de modelos y horizontes probados.';

/** Rama weekly inference basada en REGLAS (Oro/BTC semanal) — encabezado en negrita. */
export const FORECAST_DISCLAIMER_WEEKLY_TITLE =
  'Inferencia por reglas: superficie de diagnóstico, no de señales.';

/**
 * Rama weekly inference — cuerpo. Esta superficie NO corre el model zoo: describir aquí
 * "9 modelos" o una precisión media del zoo sería, literalmente, una afirmación falsa
 * sobre la naturaleza del producto (rechazo Codex a b86083e).
 */
export const FORECAST_DISCLAIMER_WEEKLY_BODY =
  'Esta inferencia semanal la produce una política basada en REGLAS congelada: no hay '
  + 'conjunto de modelos ML ni probabilidad calibrada detrás de cada dirección. Lo que se '
  + 'muestra es el sesgo histórico de esa regla frente al resultado ya ocurrido, y su '
  + 'acierto se publica sin ajustar por los intentos probados.';

/**
 * Etiquetas NEUTRAS de dirección (BL-03 / FABRIC §24.3). LONG/SHORT son etiquetas
 * IMPERATIVAS de orden: describen lo que un ejecutor haría, no lo que la superficie
 * diagnóstica sabe. Se sustituyen por la descripción del sesgo, sin color ni verbo.
 */
export const FORECAST_DIRECTION_LABEL_UP = 'Sesgo al alza';
export const FORECAST_DIRECTION_LABEL_DOWN = 'Sesgo a la baja';
export const FORECAST_DIRECTION_LABEL_FLAT = 'Sin exposición';
export const FORECAST_DIRECTION_LABEL_UNKNOWN = 'Sin dato direccional';

/**
 * Columna de acierto de las tablas semanales (a11y, rechazo Codex a b86083e): el acierto
 * NO puede comunicarse solo con un glifo ✓/· y un color. La columna necesita nombre
 * accesible y cada celda un `Sí`/`No` leído por lector de pantalla.
 */
export const FORECAST_HIT_COLUMN_LABEL = 'Acierto direccional';
export const FORECAST_HIT_YES_LABEL = 'Sí';
export const FORECAST_HIT_NO_LABEL = 'No';

/**
 * BL-20 (fase UI) — render test de la sección admin de interpretabilidad.
 *
 * Constitución (A.7 + quant-constitution §6): la superficie presenta el hallazgo
 * como DIAGNÓSTICO del predictor débil (features con signo inestable entre años),
 * jamás como "importancia de features para operar":
 *  1. Header fijo del artefacto: "SHAP explica el modelo, no el mercado".
 *  2. Kill-flags renderizados DEL ARTEFACTO (frontend no recomputa condiciones).
 *  3. Cero verde/rojo direccional en los valores SHAP (signos en tinta neutra).
 *  4. Rule-based etiquetado "atribución, no SHAP"; sin Sharpe/p-value solemnes.
 *  5. Read-only: cero botones en los paneles.
 */
import { describe, it, expect, afterEach } from 'vitest';
import { cleanup, render, screen, within } from '@testing-library/react';

// El pool singleThread comparte el registro de módulos: el auto-cleanup de RTL
// solo queda registrado en el primer archivo de la corrida. Cleanup explícito
// aquí ⇒ este archivo es verde en cualquier orden y no contamina a los vecinos.
afterEach(() => cleanup());

import {
  LinearShapPanel,
  RuleAttributionPanel,
  buildSignMatrix,
} from '@/components/admin/InterpretabilitySection';

import { REAL_LINEAR, REAL_RULE, trimLinear, trimRule } from '../../support/interp-fixtures';

/**
 * Fixtures = subconjunto FIEL de los artefactos REALES trackeados en
 * `<repo>/data/interpretability/**` (tests/support/interp-fixtures.ts), no una copia a mano.
 * La copia a mano que vivía aquí ya había divergido del generador (sin `artifact_id`/
 * `provenance`, sin `fit.n_fits`/`n_train_scheme` ⇒ 2 errores de `tsc`, y con cifras de PnL
 * de una corrida anterior). Un fixture inventado prueba el render de un artefacto que no existe.
 */
const FEATURES = ['return_10d', 'high', 'day_of_week'];
const YEARS = ['2020', '2026'];
const LINEAR = trimLinear(REAL_LINEAR, FEATURES, YEARS);
const RULE = trimRule(REAL_RULE, 2);

/**
 * Formateo % del panel, reimplementado a propósito (no se importa `fmtSignedPct`): así la
 * aserción compara contra el VALOR del artefacto, no contra la función bajo test.
 */
const pct = (v: number, digits = 1) => `${v < 0 ? '−' : ''}${(Math.abs(v) * 100).toFixed(digits)}%`;

// Clases direccionales prohibidas sobre valores SHAP (verde/rojo semántico GM).
const DIRECTIONAL = '[class*="gm-pos"], [class*="gm-neg"], [class*="emerald"], [class*="red-"]';

describe('buildSignMatrix (helper puro — solo formatea el by_year del artefacto)', () => {
  it('deriva el signo por año y marca los kill-flags que trae el artefacto', () => {
    const m = buildSignMatrix(LINEAR);
    expect(m.years).toEqual(['2020', '2026']);

    const r10 = m.rows.find((r) => r.feature === 'return_10d')!;
    expect(r10.flagged).toBe(true);          // viene de kill_flags_sign_change_by_year
    expect(r10.signs).toEqual([1, -1]);      // + en 2020, − en 2026 (signo inestable)

    // Los dos controles de que el flag NO se recomputa en el frontend, ambos con los
    // valores REALES del artefacto (el generador mira los 7 años; aquí se muestran 2):
    const high = m.rows.find((r) => r.feature === 'high')!;
    expect(high.signs).toEqual([1, 1]);      // estable en los años mostrados…
    expect(high.flagged).toBe(true);         // …y aun así flagged por el generador.

    const dow = m.rows.find((r) => r.feature === 'day_of_week')!;
    expect(dow.signs).toEqual([1, -1]);      // cambia de signo en los años mostrados…
    expect(dow.flagged).toBe(false);         // …y el generador NO lo flagea (magnitud inmaterial).
  });

  it('feature ausente en un año ⇒ 0 (nunca NaN)', () => {
    const m = buildSignMatrix({
      ...LINEAR,
      by_year: { '2020': LINEAR.by_year['2020'], '2026': [] },
    });
    expect(m.rows.find((r) => r.feature === 'return_10d')!.signs).toEqual([1, 0]);
  });
});

describe('LinearShapPanel (zoo ridge — diagnóstico, no importancia para operar)', () => {
  it('muestra el header constitucional del artefacto', () => {
    render(<LinearShapPanel summary={LINEAR} />);
    expect(
      screen.getAllByText(/SHAP explica el modelo, no el mercado/i).length,
    ).toBeGreaterThan(0);
  });

  it('presenta los kill-flags como diagnóstico de inestabilidad (conteo X/N del artefacto)', () => {
    render(<LinearShapPanel summary={LINEAR} />);
    // 2 de las 3 features retenidas están en `kill_flags_sign_change_by_year` del artefacto real.
    expect(screen.getByText('2/3')).toBeInTheDocument();
    expect(screen.getAllByText(/diagnóstico/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/no es importancia para operar/i)).toBeInTheDocument();
  });

  it('cero verde/rojo direccional sobre los valores SHAP', () => {
    const { container } = render(<LinearShapPanel summary={LINEAR} />);
    expect(container.querySelectorAll(DIRECTIONAL)).toHaveLength(0);
  });

  it('tabla accesible con la matriz de signos por año (+/− en tinta neutra)', () => {
    render(<LinearShapPanel summary={LINEAR} />);
    const table = screen.getByRole('table', { name: /signo.*por año/i });
    expect(within(table).getByRole('columnheader', { name: '2020' })).toBeInTheDocument();
    expect(within(table).getByRole('columnheader', { name: '2026' })).toBeInTheDocument();
  });

  it('read-only: cero botones', () => {
    render(<LinearShapPanel summary={LINEAR} />);
    expect(screen.queryAllByRole('button')).toHaveLength(0);
  });
});

describe('RuleAttributionPanel (spx500 — atribución de reglas, NO SHAP)', () => {
  it('etiqueta la superficie como atribución, no SHAP', () => {
    render(<RuleAttributionPanel summary={RULE} />);
    expect(screen.getAllByText(/atribución de reglas/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/no SHAP/i).length).toBeGreaterThan(0);
  });

  it('muestra la descomposición beta/timing del artefacto sin métricas solemnes', () => {
    render(<RuleAttributionPanel summary={RULE} />);
    // beta y timing DEL artefacto publicado (×100, 1 decimal) — no cifras copiadas a mano
    expect(screen.getByText(pct(RULE.pnl_decomposition.pnl_beta))).toBeInTheDocument();
    expect(screen.getByText(pct(RULE.pnl_decomposition.pnl_timing_cov_pos_ret))).toBeInTheDocument();
    // sin Sharpe / p-value: esto es diagnóstico, no claim de edge
    expect(screen.queryByText(/sharpe/i)).toBeNull();
    expect(screen.queryByText(/p[-‑]?value|p\s*=/i)).toBeNull();
  });

  it('cero verde/rojo direccional también en el PnL por año', () => {
    const { container } = render(<RuleAttributionPanel summary={RULE} />);
    expect(container.querySelectorAll(DIRECTIONAL)).toHaveLength(0);
  });

  it('read-only: cero botones', () => {
    render(<RuleAttributionPanel summary={RULE} />);
    expect(screen.queryAllByRole('button')).toHaveLength(0);
  });
});

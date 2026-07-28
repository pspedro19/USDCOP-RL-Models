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
  type InterpLinearSummary,
  type InterpRuleSummary,
} from '@/components/admin/InterpretabilitySection';

/** Subconjunto fiel de public/data/interpretability/zoo/usdcop/ridge/2026-07-27/summary.json. */
const LINEAR: InterpLinearSummary = {
  nota: 'SHAP explica el modelo, no el mercado; solo test-folds; diagnostico 0 trials',
  surface: 'zoo',
  asset: 'usdcop',
  model_id: 'ridge',
  model_type: 'linear',
  method: 'linear_shap_closed_form',
  attribution_not_shap: false,
  version: '2026-07-27',
  generated_at: '2026-07-28T02:49:42.472712+00:00',
  fit: {
    scheme: 'ultimo fit walk-forward (origen = ultima fila, purga 5d)',
    origin: '2026-07-27',
    n_train: 1649,
    horizon: 5,
    purge_days: 5,
    scaler: 'StandardScaler train-only',
    params: { alpha: 1.0, fit_intercept: true },
  },
  scope:
    'phi_j = coef_j*(x_j-mu_j)/sigma_j del ULTIMO fit aplicado a todo el historico de features — diagnostico del modelo congelado, no evidencia OOS ni claim de edge',
  base_value: -4.037946557593295e-5,
  n_rows: 1654,
  n_features: 3,
  top_features: [
    { rank: 1, feature: 'return_10d', coef: 0.0076015815605016916, mean_abs_shap: 0.0056194033765918, mean_shap: -1.8402207004886957e-5 },
    { rank: 2, feature: 'high', coef: -0.0066361391759335525, mean_abs_shap: 0.005195130453732321, mean_shap: 4.586156554639145e-5 },
    { rank: 3, feature: 'day_of_week', coef: -0.00011993827699173584, mean_abs_shap: 0.00010177717550288016, mean_shap: 1.0908035456776284e-9 },
  ],
  by_year: {
    '2020': [
      { feature: 'return_10d', mean_abs_shap: 0.00784172534234886, mean_shap: 0.0003202165670858094 },
      { feature: 'high', mean_abs_shap: 0.005111235278331515, mean_shap: 0.0048435335961069855 },
      { feature: 'day_of_week', mean_abs_shap: 0.00010176732610898587, mean_shap: 1.1371216400314353e-6 },
    ],
    '2026': [
      { feature: 'high', mean_abs_shap: 0.007710007938919618, mean_shap: 0.007710007938919618 },
      { feature: 'return_10d', mean_abs_shap: 0.00582594308954068, mean_shap: -0.0028993645490742064 },
      { feature: 'day_of_week', mean_abs_shap: 9.913323564481966e-5, mean_shap: -4.5314257306754676e-6 },
    ],
  },
  // El flag viene DEL ARTEFACTO (generador Python); la UI no lo recomputa.
  kill_flags_sign_change_by_year: ['return_10d'],
};

/** Subconjunto fiel de public/data/interpretability/rule_based/spx500/.../summary.json. */
const RULE: InterpRuleSummary = {
  nota: 'SHAP explica el modelo, no el mercado; solo test-folds; diagnostico 0 trials',
  surface: 'rule_based',
  asset: 'spx500',
  model_id: 'spx500_regime_gated_v1',
  model_type: 'rule_based',
  method: 'rule_attribution',
  attribution_not_shap: true,
  version: '2026-07-27',
  generated_at: '2026-07-28T02:49:43.384982+00:00',
  scope:
    'atribucion de reglas sobre la misma serie del adapter publicado (daily/252); descomposicion pnl_gross = beta + timing, timing = n*cov(pos,ret) — diagnostico, no claim de edge',
  rules: {
    gate: 'trend_on = close > MA200 (dumb baseline ma200_always_on del sleeve)',
    pct_days_trend_on: 0.7310839733098325,
    pct_days_position_active: 0.5815183180158631,
    avg_exposure: 0.4728929893158234,
    n_trades: 4479,
  },
  pnl_decomposition: {
    n_days: 7943,
    pnl_gross: 1.0915988226461462,
    pnl_beta: 1.5529968368518372,
    pnl_timing_cov_pos_ret: -0.46139801420569104,
    costs: 0.08911379302987395,
    pnl_net: 1.0024850296162722,
  },
  by_year: {
    '2018': {
      n_days: 251,
      pnl_gross: 0.08506739933442643,
      pnl_beta: -0.04192729838755579,
      pnl_timing_cov_pos_ret: 0.1269946977219822,
      costs: 0.002588324710288278,
      pnl_net: 0.08247907462413816,
      pct_days_trend_on: 0.8087649402390438,
      pct_days_position_active: 0.549800796812749,
      avg_exposure: 0.630156719192474,
    },
    '2026': {
      n_days: 141,
      pnl_gross: -0.00630529578886381,
      pnl_beta: 0.05147576018170504,
      pnl_timing_cov_pos_ret: -0.05778105597056885,
      costs: 0.001978589831034038,
      pnl_net: -0.008283885619897848,
      pct_days_trend_on: 0.9078014184397163,
      pct_days_position_active: 0.7872340425531915,
      avg_exposure: 0.5875270391784111,
    },
  },
};

// Clases direccionales prohibidas sobre valores SHAP (verde/rojo semántico GM).
const DIRECTIONAL = '[class*="gm-pos"], [class*="gm-neg"], [class*="emerald"], [class*="red-"]';

describe('buildSignMatrix (helper puro — solo formatea el by_year del artefacto)', () => {
  it('deriva el signo por año y marca los kill-flags que trae el artefacto', () => {
    const m = buildSignMatrix(LINEAR);
    expect(m.years).toEqual(['2020', '2026']);

    const r10 = m.rows.find((r) => r.feature === 'return_10d')!;
    expect(r10.flagged).toBe(true);          // viene de kill_flags_sign_change_by_year
    expect(r10.signs).toEqual([1, -1]);      // + en 2020, − en 2026 (signo inestable)

    const high = m.rows.find((r) => r.feature === 'high')!;
    expect(high.flagged).toBe(false);        // el flag NO se recomputa en el frontend
    expect(high.signs).toEqual([1, 1]);
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
    expect(screen.getByText('1/3')).toBeInTheDocument();
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
    // beta 155.3% y timing −46.1% del bundle (×100, 1 decimal)
    expect(screen.getByText('155.3%')).toBeInTheDocument();
    expect(screen.getByText('−46.1%')).toBeInTheDocument();
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

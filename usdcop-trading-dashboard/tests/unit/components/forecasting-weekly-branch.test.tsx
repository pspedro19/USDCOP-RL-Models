/**
 * BL-02 (hueco documentado) — LA RAMA `weekly_inference` DE /forecasting, VIVA.
 *
 * El candado estático (tests/regression/test_forecasting_caveat_present.py) exige que la
 * expresión de variante de ForecastingView.tsx contenga `'weekly'` y ramifique por
 * `isModelZoo`, y muerde ante la mutación `{isModelZoo && (<ForecastDisclaimer …/>)}`
 * por profundidad de llaves. Pero HOY esa rama es un NO-OP en runtime: los cuatro
 * activos de `lib/contracts/analysis-assets.ts` declaran `forecast_mode: 'model_zoo'`,
 * así que `isModelZoo` es siempre true, `AssetWeeklyBody` es inalcanzable desde la vista
 * GM y NINGÚN test de render podía distinguir la rama `weekly` de su ausencia. El candado
 * congelaba una rama que nadie podía ver, y el banner de la superficie `weekly` no estaba
 * verificado por nada ejecutable.
 *
 * Este fichero cierra ese hueco: INYECTA la SSOT de activos (único punto de entrada del
 * `forecast_mode` a la vista) con un activo en `weekly_inference` y renderiza la vista GM
 * REAL —barrel `@/components/gm` sin stubbear, `useGmQuery` → `apiFetch` → `fetch`
 * mockeado por ruta— para ejercitar la rama de punta a punta.
 *
 * La coherencia entre esta rama, el candado Python y los datos reales (hoy: CERO
 * consumidores) la fija `test_weekly_branch_and_its_lock_stay_coherent_with_the_data`
 * en el mismo fichero de regresión. Los dos tests son un par: éste prueba que la rama
 * FUNCIONA, aquél declara que hoy nadie la alcanza.
 */
import React from 'react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, cleanup, within } from '@testing-library/react';

import { installConsoleGate } from '../../support/console-gate';
import { stubChartLayout } from '../../support/chart-layout';

import {
  FORECAST_DISCLAIMER_TESTID,
  FORECAST_DISCLAIMER_HEADLINE,
  FORECAST_DISCLAIMER_WEEKLY_TITLE,
  FORECAST_DISCLAIMER_WEEKLY_BODY,
  FORECAST_DISCLAIMER_ZOO_TITLE,
  FORECAST_DISCLAIMER_ZOO_BODY,
  FORECAST_DIRECTION_LABEL_UP,
  FORECAST_DIRECTION_LABEL_DOWN,
} from '@/lib/ui/forecast-disclaimer';

// ────────────────────────────────────────────────────────────── mocks de entorno

// next/font solo existe dentro del runtime de Next: sin esto el barrel `@/components/gm`
// (→ TerminalShell → JetBrains_Mono) revienta al importarse. Con él la vista se monta con
// sus componentes REALES (AsyncBoundary, primitives, useGmQuery) — nada del árbol bajo
// prueba se stubbea, que es lo que hace que el test vea la rama de verdad.
vi.mock('next/font/google', () => ({
  JetBrains_Mono: () => ({ variable: 'font-jb-mono', className: 'font-jb-mono', style: { fontFamily: 'monospace' } }),
  Inter: () => ({ variable: 'font-inter', className: 'font-inter', style: { fontFamily: 'sans-serif' } }),
}));

let currentRole: 'admin' | 'subscriber' | 'free' | null = 'free';
vi.mock('next-auth/react', () => ({
  useSession: () => (currentRole === null
    ? { data: null, status: 'unauthenticated' }
    : { data: { user: { role: currentRole } }, status: 'authenticated' }),
}));

let currentQs = '';
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), prefetch: vi.fn() }),
  useSearchParams: () => new URLSearchParams(currentQs),
}));

/**
 * LA INYECCIÓN. `forecast_mode` entra a ForecastingView por UN sitio —
 * `ANALYSIS_ASSETS` de la SSOT de activos— así que sustituir ese módulo es la forma
 * mínima y honesta de darle un consumidor a la rama muerta sin tocar producción.
 * Se conserva un activo `model_zoo` (el defecto real, usdcop) para que la sustitución
 * no altere ninguna otra decisión de la vista.
 */
const WEEKLY_ASSET_ID = 'rulesfixture';
const WEEKLY_ASSET_NAME = 'Activo por reglas (fixture)';
vi.mock('@/lib/contracts/analysis-assets', () => {
  const ASSETS = [
    {
      asset_id: 'usdcop', symbol: 'USD/COP', chart_symbol: 'USDCOP', display_name: 'USD/COP',
      asset_class: 'fx', forecast_mode: 'model_zoo',
    },
    {
      asset_id: 'rulesfixture', symbol: 'RUL/USD', chart_symbol: 'RULUSD',
      display_name: 'Activo por reglas (fixture)',
      asset_class: 'commodity', forecast_mode: 'weekly_inference',
    },
  ];
  const ids = ASSETS.map((a) => a.asset_id);
  return {
    ANALYSIS_ASSETS: ASSETS,
    ANALYSIS_ASSET_IDS: ids,
    DEFAULT_ANALYSIS_ASSET: 'usdcop',
    isValidAnalysisAsset: (id: string | null | undefined) => !!id && ids.includes(id),
    resolveAnalysisAsset: (id: string | null | undefined) =>
      (id && ids.includes(id) ? id : 'usdcop'),
    getAnalysisAsset: (id: string | null | undefined) =>
      ASSETS.find((a) => a.asset_id === id) ?? ASSETS[0],
  };
});

// recharts necesita layout (jsdom mide 0×0): sin esto la curva del cuerpo weekly no se
// monta y el test diría cubrir una superficie con gráfico cubriendo una sin él.
stubChartLayout();

// Consola limpia = parte del contrato (tolerancia cero, igual que la suite hermana).
installConsoleGate();

// ────────────────────────────────────────────────────────────── fixtures
// Forma fiel de public/forecasting/<asset>/{index,weekly_inference_<year>,forward}.json.

const INDEX_DOC = {
  asset_id: WEEKLY_ASSET_ID,
  display_name: WEEKLY_ASSET_NAME,
  chart_symbol: 'RULUSD',
  years: [2025],
  primary_strategy_id: 'rules_trend_b2',
  strategies: [
    { strategy_id: 'rules_trend_b2', strategy_name: 'Reglas · Trend-follower Daily (B2)', strategy_type: 'rule_based' },
  ],
};

const WEEKS = [
  {
    iso_week: '2025-W01', week_start: '2024-12-30', week_end: '2025-01-03',
    direction: 'LONG', exposure: 0.505, exposure_raw: 0.758, regime: 'compression',
    confidence: 1.0, expected_return_pct: -0.3, realized_return_pct: 0.53,
    buyhold_return_pct: 1.25, entry_price: 2607.18, close_price: 2639.69, hit: true,
  },
  {
    iso_week: '2025-W02', week_start: '2025-01-06', week_end: '2025-01-10',
    direction: 'SHORT', exposure: 0.4, exposure_raw: 0.4, regime: 'stretched',
    confidence: 0.62, expected_return_pct: 0.2, realized_return_pct: -0.41,
    buyhold_return_pct: 0.9, entry_price: 2639.69, close_price: 2663.51, hit: false,
  },
];

const WEEKLY_DOC = {
  asset_id: WEEKLY_ASSET_ID, display_name: WEEKLY_ASSET_NAME, symbol: 'RUL/USD',
  chart_symbol: 'RULUSD', asset_class: 'commodity', year: 2025,
  generated_at: '2026-07-27', kind: 'weekly_inference',
  strategies: [{
    strategy_id: 'rules_trend_b2', strategy_name: 'Reglas · Trend-follower Daily (B2)',
    strategy_type: 'rule_based', is_primary: true,
    weeks: WEEKS,
    summary: {
      weeks_total: 52, weeks_in_market: 52, weeks_flat: 0, hit_rate_pct: 69.2,
      ytd_strategy_return_pct: 42.47, ytd_buyhold_return_pct: 73.5, avg_exposure: 0.605,
    },
  }],
};

/** fetch router por sufijo de URL (el resto: 404 legacy, que NO bloquea la vista). */
function mockFetch(routes: Array<[RegExp, { ok: boolean; status?: number; body?: unknown }]>) {
  (global as any).fetch = vi.fn(async (input: unknown) => {
    const url = String(input);
    for (const [re, r] of routes) {
      if (re.test(url)) {
        return {
          ok: r.ok, status: r.status ?? (r.ok ? 200 : 404),
          json: async () => r.body ?? { error: `HTTP ${r.status ?? 404}` },
          text: async () => JSON.stringify(r.body ?? null),
        };
      }
    }
    return { ok: false, status: 404, json: async () => ({ error: 'HTTP 404' }), text: async () => '' };
  });
}

afterEach(() => {
  cleanup();
  currentRole = 'free';
  currentQs = '';
});

// ═══════════════════════════════════════════════════════════════════════════════

describe('ForecastingView — rama weekly_inference VIVA (BL-02)', () => {
  beforeEach(() => {
    mockFetch([
      [new RegExp(`/api/forecasting/${WEEKLY_ASSET_ID}/index\\.json$`), { ok: true, body: INDEX_DOC }],
      [new RegExp(`/api/forecasting/${WEEKLY_ASSET_ID}/weekly_inference_2025\\.json$`), { ok: true, body: WEEKLY_DOC }],
      [/forward\.json$/, { ok: false, status: 404 }],
    ]);
    currentQs = `asset=${WEEKLY_ASSET_ID}`;
  });

  async function renderWeeklyAsset() {
    const { ForecastingView } = await import('@/components/gm/views/ForecastingView');
    const utils = render(<ForecastingView />);
    // El cuerpo weekly (AssetWeeklyBody) es la prueba de que la rama se alcanza de verdad.
    await screen.findByTestId('forecasting-weekly-inference');
    return utils;
  }

  // ROJO si se borra la rama weekly: `variant={directionalSelected ? 'directional' : 'zoo'}`
  it('monta el banner con la variante weekly (copy de REGLAS, jamás el del model zoo)', async () => {
    await renderWeeklyAsset();
    const banner = screen.getByTestId(FORECAST_DISCLAIMER_TESTID);
    expect(banner).toBeVisible();
    const text = banner.textContent ?? '';
    expect(text).toContain(FORECAST_DISCLAIMER_HEADLINE);
    expect(text, 'la superficie de reglas debe declarar su propia naturaleza')
      .toContain(FORECAST_DISCLAIMER_WEEKLY_TITLE);
    expect(text).toContain(FORECAST_DISCLAIMER_WEEKLY_BODY);
    // El doble aserto es el que hace red la mutación: sin rama weekly cae el zoo aquí.
    expect(text, 'una superficie de REGLAS no puede afirmar el titular del model zoo')
      .not.toContain(FORECAST_DISCLAIMER_ZOO_TITLE);
    expect(text, 'una superficie de REGLAS no corre 9 modelos ML: el cuerpo del zoo es falso aquí')
      .not.toContain(FORECAST_DISCLAIMER_ZOO_BODY);
  });

  // ROJO si se borra la rama de render weekly: `) : isModelZoo ? (` → `) : (` con el zoo.
  it('renderiza el cuerpo weekly real (AssetWeeklyBody), no el model zoo', async () => {
    const { container } = await renderWeeklyAsset();
    expect(screen.getByTestId('forecasting-weekly-inference')).toBeInTheDocument();
    // Etiquetas neutras de dirección desde los datos publicados (BL-03 sobre la rama viva).
    const table = container.querySelector('table') as HTMLTableElement;
    expect(table, 'la rama weekly debe pintar su tabla semanal').toBeTruthy();
    const w01 = within(container).getByText('2025-W01').closest('tr') as HTMLElement;
    expect(w01.textContent).toContain(FORECAST_DIRECTION_LABEL_UP);
    const w02 = within(container).getByText('2025-W02').closest('tr') as HTMLElement;
    expect(w02.textContent).toContain(FORECAST_DIRECTION_LABEL_DOWN);
    expect(table.textContent ?? '', 'etiquetas imperativas de orden en superficie diagnóstica')
      .not.toMatch(/\b(LONG|SHORT)\b/);
  });

  // ROJO si el banner se envuelve en `{isModelZoo && (…)}` (la mutación de BL-02):
  // con el activo en weekly_inference el banner desaparece para TODOS los roles.
  it.each([null, 'free', 'subscriber', 'admin'] as const)(
    'el banner weekly es incondicional respecto al rol (rol=%s)',
    async (role) => {
      currentRole = role as typeof currentRole;
      await renderWeeklyAsset();
      const banner = screen.getByTestId(FORECAST_DISCLAIMER_TESTID);
      expect(banner).toBeVisible();
      expect(banner.textContent ?? '').toContain(FORECAST_DISCLAIMER_WEEKLY_TITLE);
    },
  );
});

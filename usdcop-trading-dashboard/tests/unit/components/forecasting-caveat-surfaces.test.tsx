/**
 * BL-02/BL-03/BL-04 — re-remediación tras CXD-032 (segundo rechazo de Codex).
 *
 * Render tests (Vitest + testing-library, fixtures reales) de las TRES superficies
 * de forecasting; el candado estático (tests/regression/test_forecasting_caveat_present.py)
 * cubre la estructura del fuente, ESTOS tests cubren el render real:
 *
 *  1. BL-02 — el disclaimer (componente compartido ForecastDisclaimer, SSOT
 *     lib/ui/forecast-disclaimer.ts) se monta INCONDICIONAL y queda VISIBLE en:
 *     GM ForecastingView (todos los modos), legacy ForecastingDashboard y
 *     legacy WeeklyInferenceView (Gold/BTC — el gap exacto de CXD-032).
 *  2. BL-03 — WeeklyInferenceView sin LONG verde / SHORT rojo; columna proxy
 *     de convicción presente y etiquetada "Convicción de regla (proxy; no
 *     probabilidad)" — `confidence` NO es una probabilidad (no existe
 *     probability_up en el contrato weekly; no se inventa).
 *  3. BL-03 a11y — tablas weekly: caption sr-only, th scope=col/row, región de
 *     scroll focusable (patrón PaperCandidatesPanel, commit 624465c).
 *  4. BL-04 — el BODY del disclaimer legacy viene del SSOT (no solo el título) y
 *     la frase derivada de los datos se mantiene (nunca un número hardcodeado).
 *
 * MUTACIONES PROBADAS (cada una deja este archivo en rojo — ver reporte BL-02/03/04):
 *  M1  quitar <ForecastDisclaimer/> de una superficie            → getByTestId falla
 *  M2  envolverlo en `{false && ...}` / rama por asset           → getByTestId falla
 *  M3  atributo `hidden` o style display:none (elemento/ancestro)→ assertHardVisible falla
 *  M4  className "hidden"/"invisible"/"sr-only" (Tailwind, que jsdom no computa
 *      — por eso se asserta la CLASE además de toBeVisible)      → assertHardVisible falla
 *  M5  reponer LONG emerald / SHORT red en WeeklyInferenceView   → test de tono neutro falla
 *  M6  re-etiquetar la convicción como "Prob."/"probabilidad"    → tests de etiqueta fallan
 *  M7  quitar caption/scope/región focusable de una tabla weekly → tests a11y fallan
 *  M8  cuerpo legacy hardcodeado en vez de SSOT                  → test de body SSOT falla
 */
import React from 'react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, within, cleanup, waitFor } from '@testing-library/react';

import {
  FORECAST_DISCLAIMER_TESTID,
  FORECAST_DISCLAIMER_HEADLINE,
  FORECAST_DISCLAIMER_ZOO_TITLE,
  FORECAST_DISCLAIMER_ZOO_BODY,
  FORECAST_DISCLAIMER_DIRECTIONAL_TITLE,
} from '@/lib/ui/forecast-disclaimer';

// ────────────────────────────────────────────────────────────── mocks de entorno

// Sesión admin (las vistas solo la usan para hints internos / free-badge).
vi.mock('next-auth/react', () => ({
  useSession: () => ({ data: { user: { role: 'admin' } }, status: 'authenticated' }),
}));

// next/navigation para la vista GM (estado en la URL).
let currentQs = '';
const routerReplace = vi.fn();
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: routerReplace, push: vi.fn(), prefetch: vi.fn() }),
  useSearchParams: () => new URLSearchParams(currentQs),
}));

// Barrel GM: TerminalShell arrastra next/font (no carga en vitest/jsdom) — se stubbea
// TODO el barrel con primitives mínimos. El banner bajo prueba NO viene de aquí:
// es el componente compartido real de components/forecasting/ForecastDisclaimer.
vi.mock('@/components/gm', () => {
  const R = require('react');
  const AsyncBoundary = ({ state, children, empty }: any) =>
    state?.data && !(empty && empty(state.data))
      ? children(state.data)
      : R.createElement('div', { 'data-testid': 'gm-async-empty' });
  return {
    AsyncBoundary,
    GmBadge: ({ children, tone }: any) =>
      R.createElement('span', { 'data-tone': tone ?? 'neutral' }, children),
    GmDelta: ({ value }: any) => R.createElement('span', null, String(value ?? '—')),
    GmKpi: ({ label, value, sub }: any) =>
      R.createElement('div', null, `${label}: ${value}`, sub ? ` · ${sub}` : ''),
    GmPageHeader: ({ title, subtitle, actions }: any) =>
      R.createElement('header', null, title, subtitle, actions),
    GmPanel: ({ title, meta, actions, children }: any) =>
      R.createElement('section', null, title, meta, actions, children),
    GmEmpty: () => R.createElement('div'),
    GmErrorRetry: () => R.createElement('div'),
    GmSkeleton: () => R.createElement('div'),
    Spark: () => R.createElement('span'),
    useGmQuery: (_url: string | null) =>
      ({ data: null, error: null, loading: false, reload: () => {} }),
  };
});

// ────────────────────────────────────────────────────────────── helpers

/**
 * Visibilidad DURA por estructura del render (no solo jest-dom): jsdom no computa
 * las hojas de estilo de Tailwind, así que `toBeVisible()` NO ve `className="hidden"`.
 * Este helper recorre el elemento y TODOS sus ancestros y falla ante: atributo
 * `hidden`, `aria-hidden`, clases de ocultamiento Tailwind, o estilos inline
 * display/visibility/opacity — exactamente las mutaciones M3/M4.
 */
function assertHardVisible(el: HTMLElement) {
  for (let node: HTMLElement | null = el; node && node !== document.body; node = node.parentElement) {
    expect(node.hasAttribute('hidden'), `<${node.tagName.toLowerCase()}> tiene atributo hidden`).toBe(false);
    expect(node.getAttribute('aria-hidden'), `<${node.tagName.toLowerCase()}> aria-hidden`).not.toBe('true');
    const cls = typeof node.className === 'string' ? node.className : '';
    expect(
      /(^|\s)(hidden|invisible|sr-only)(\s|$)/.test(cls),
      `<${node.tagName.toLowerCase()}> lleva clase de ocultamiento: "${cls}"`,
    ).toBe(false);
    expect(node.style.display, `display inline en <${node.tagName.toLowerCase()}>`).not.toBe('none');
    expect(node.style.visibility).not.toBe('hidden');
    expect(node.style.opacity).not.toBe('0');
  }
  expect(el).toBeVisible();
}

/** El banner compartido, visible y con el copy SSOT (titular + rama zoo). */
function assertZooBannerVisible(scope: HTMLElement | typeof screen = screen as any) {
  const q = scope === screen ? screen : within(scope as HTMLElement);
  const banner = (q as any).getByTestId(FORECAST_DISCLAIMER_TESTID) as HTMLElement;
  assertHardVisible(banner);
  expect(banner.textContent).toContain(FORECAST_DISCLAIMER_HEADLINE);
  expect(banner.textContent).toContain(FORECAST_DISCLAIMER_ZOO_TITLE);
  expect(banner.textContent).toContain(FORECAST_DISCLAIMER_ZOO_BODY);
  return banner;
}

/** a11y de tabla weekly (patrón PaperCandidatesPanel 624465c). */
function assertWeeklyTableA11y(root: HTMLElement) {
  // Región de scroll horizontal focusable (axe scrollable-region-focusable).
  const regions = Array.from(root.querySelectorAll('[role="region"]'))
    .filter((r) => r.querySelector('table'));
  expect(regions.length, 'falta el contenedor role="region" con la tabla weekly').toBeGreaterThan(0);
  const region = regions[0] as HTMLElement;
  expect(region.getAttribute('tabindex'), 'la región de scroll debe ser focusable (tabIndex=0)').toBe('0');
  expect(region.getAttribute('aria-label'), 'la región necesita accessible name').toBeTruthy();
  const table = region.querySelector('table') as HTMLTableElement;
  // Caption sr-only (nombre accesible de la tabla, invisible para videntes).
  const caption = table.querySelector('caption');
  expect(caption, 'la tabla weekly necesita <caption>').toBeTruthy();
  expect(caption!.className).toMatch(/sr-only/);
  expect((caption!.textContent ?? '').length).toBeGreaterThan(10);
  // Encabezados de columna y de fila con scope.
  const colHeaders = table.querySelectorAll('thead th[scope="col"]');
  expect(colHeaders.length, 'todos los th de columna necesitan scope="col"').toBeGreaterThanOrEqual(8);
  const rowHeaders = table.querySelectorAll('tbody th[scope="row"]');
  expect(rowHeaders.length, 'la celda Semana debe ser th scope="row"').toBeGreaterThan(0);
}

/** fetch router por sufijo de URL. */
function mockFetch(routes: Array<[RegExp, { ok: boolean; status?: number; body?: unknown; text?: string }]>) {
  (global as any).fetch = vi.fn(async (input: unknown) => {
    const url = String(input);
    for (const [re, r] of routes) {
      if (re.test(url)) {
        return {
          ok: r.ok,
          status: r.status ?? (r.ok ? 200 : 404),
          json: async () => r.body ?? null,
          text: async () => r.text ?? JSON.stringify(r.body ?? null),
        };
      }
    }
    return { ok: false, status: 404, json: async () => null, text: async () => '' };
  });
}

// ────────────────────────────────────────────────────────────── fixtures reales
// Subconjunto fiel de public/forecasting/xauusd/{index,weekly_inference_2025}.json.

const GOLD_INDEX = {
  asset_id: 'xauusd',
  display_name: 'Oro (Gold)',
  chart_symbol: 'XAUUSD',
  years: [2026, 2025],
  primary_strategy_id: 'gold_trend_b2',
  strategies: [
    { strategy_id: 'gold_trend_b2', strategy_name: 'Gold · Trend-follower Daily (B2)', strategy_type: 'rule_based' },
  ],
};

const GOLD_WEEKLY_2025 = {
  asset_id: 'xauusd',
  display_name: 'Oro (Gold)',
  symbol: 'XAU/USD',
  chart_symbol: 'XAUUSD',
  asset_class: 'commodity',
  year: 2025,
  generated_at: '2026-07-27',
  kind: 'weekly_inference',
  strategies: [{
    strategy_id: 'gold_trend_b2',
    strategy_name: 'Gold · Trend-follower Daily (B2)',
    strategy_type: 'rule_based',
    is_primary: true,
    weeks: [
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
      {
        iso_week: '2025-W03', week_start: '2025-01-13', week_end: '2025-01-17',
        direction: 'FLAT', exposure: 0, exposure_raw: 0, regime: 'event',
        confidence: null, expected_return_pct: 0, realized_return_pct: 0,
        buyhold_return_pct: 0.4, entry_price: 2663.51, close_price: 2674.1, hit: false,
      },
    ],
    summary: {
      weeks_total: 52, weeks_in_market: 52, weeks_flat: 0, hit_rate_pct: 69.2,
      ytd_strategy_return_pct: 42.47, ytd_buyhold_return_pct: 73.5, avg_exposure: 0.605,
    },
  }],
};

const CSV_WITH_DA = [
  'view_type,inference_week,model_name,horizon_days,direction_accuracy,model_avg_direction_accuracy,model_avg_rmse,image_forecast,image_backtest',
  'forward_forecast,2026-W28,CONSENSUS,5,52.1,51.8,0.0123,forward_consensus_2026_W28.png,',
  'backtest,,ridge,5,53.2,,,,backtest_ridge_h5.png',
  'backtest,,bayesian_ridge,5,49.7,,,,backtest_bayesian_ridge_h5.png',
].join('\n');

// Sin ningún direction_accuracy numérico: el banner debe renderizar IGUAL (BL-02:
// "hidden por rama pasa" — el caveat no puede depender de que haya stats).
const CSV_NO_DA = [
  'view_type,inference_week,model_name,horizon_days,direction_accuracy,model_avg_direction_accuracy,model_avg_rmse,image_forecast,image_backtest',
  'forward_forecast,2026-W28,CONSENSUS,5,,,,forward_consensus_2026_W28.png,',
].join('\n');

const CONVICTION_LABEL = 'Convicción de regla (proxy; no probabilidad)';

afterEach(() => {
  cleanup();
  // OJO: NO vi.restoreAllMocks() — resetearía el mock global de ResizeObserver del
  // setup (recharts crashearía en el siguiente mount y React desmontaría el árbol).
  routerReplace.mockClear();
});

// ═══════════════════════════════════════════════ 1 · componente compartido

// Import no-analizable por Vite: si el componente compartido no existe, el rojo es
// del TEST que documenta el gap raíz de CXD-032, no un fallo de colección del archivo.
const DISCLAIMER_MODULE = '@/components/forecasting/' + 'ForecastDisclaimer';
const importDisclaimer = () =>
  import(/* @vite-ignore */ DISCLAIMER_MODULE).catch(() => null);

describe('ForecastDisclaimer (componente compartido, SSOT)', () => {
  it('existe en components/forecasting/ForecastDisclaimer y renderiza el copy SSOT', async () => {
    const mod = await importDisclaimer();
    expect(mod, 'components/forecasting/ForecastDisclaimer.tsx no existe (CXD-032 §1)').toBeTruthy();
    const ForecastDisclaimer = (mod as any).ForecastDisclaimer;
    expect(ForecastDisclaimer).toBeTypeOf('function');

    const { unmount } = render(<ForecastDisclaimer />);
    assertZooBannerVisible();
    unmount();

    // Variante direccional (replay causal USD/COP): también SSOT, también visible.
    render(<ForecastDisclaimer variant="directional" />);
    const banner = screen.getByTestId(FORECAST_DISCLAIMER_TESTID);
    assertHardVisible(banner);
    expect(banner.textContent).toContain(FORECAST_DISCLAIMER_HEADLINE);
    expect(banner.textContent).toContain(FORECAST_DISCLAIMER_DIRECTIONAL_TITLE);
  });

  it('acepta una línea derivada (children) sin perder el body SSOT', async () => {
    const mod = await importDisclaimer();
    expect(mod, 'components/forecasting/ForecastDisclaimer.tsx no existe (CXD-032 §1)').toBeTruthy();
    const ForecastDisclaimer = (mod as any).ForecastDisclaimer;
    render(
      <ForecastDisclaimer>
        <span>la precision direccional media es 51.7% sobre 3 mediciones</span>
      </ForecastDisclaimer>,
    );
    const banner = assertZooBannerVisible();
    expect(banner.textContent).toContain('51.7% sobre 3 mediciones');
  });
});

// ═══════════════════════════════════ 2 · legacy WeeklyInferenceView (Gold/BTC)

describe('WeeklyInferenceView (legacy Gold/BTC)', () => {
  beforeEach(() => {
    mockFetch([
      [/\/index\.json$/, { ok: true, body: GOLD_INDEX }],
      [/weekly_inference_2025\.json$/, { ok: true, body: GOLD_WEEKLY_2025 }],
      [/forward\.json$/, { ok: false }],
    ]);
  });

  async function renderWeekly() {
    const { WeeklyInferenceView } = await import('@/components/forecasting/WeeklyInferenceView');
    const utils = render(<WeeklyInferenceView assetId="xauusd" />);
    await screen.findByText('2025-W01');
    return utils;
  }

  it('BL-02: monta el disclaimer compartido, visible (gap CXD-032 §1)', async () => {
    await renderWeekly();
    assertZooBannerVisible();
  });

  it('BL-03: dirección LONG/SHORT en tono NEUTRO (sin verde/rojo)', async () => {
    await renderWeekly();
    const badges = screen.getAllByText(/\b(LONG|SHORT)\b/);
    expect(badges.length).toBeGreaterThanOrEqual(2);
    for (const b of badges) {
      expect(b.className, `badge de dirección "${b.textContent}" con color de compra/venta`)
        .not.toMatch(/emerald|text-red|green-|bg-red/);
    }
  });

  it('BL-03: columna de convicción proxy presente, honesta y sin wording probabilístico', async () => {
    await renderWeekly();
    // Etiqueta mandatoria: convicción, NO probabilidad (no existe probability_up weekly).
    expect(screen.getByText(CONVICTION_LABEL)).toBeTruthy();
    // Valores desde `confidence` del JSON publicado (1.0 → 100%, 0.62 → 62%, null → —).
    const w01 = screen.getByText('2025-W01').closest('tr') as HTMLElement;
    expect(within(w01).getByText('100%')).toBeTruthy();
    const w02 = screen.getByText('2025-W02').closest('tr') as HTMLElement;
    expect(within(w02).getByText('62%')).toBeTruthy();
    // Prohibido venderla como probabilidad (M6).
    expect(document.body.textContent).not.toMatch(/probabilidad estimada/i);
    expect(screen.queryByText(/^Prob\.?\s*\(proxy\)$/)).toBeNull();
  });

  it('BL-03: a11y de la tabla weekly (caption sr-only, scope col/row, región focusable)', async () => {
    const { container } = await renderWeekly();
    assertWeeklyTableA11y(container);
  });
});

// ═══════════════════════════════ 3 · legacy ForecastingDashboard (zoo USD/COP)

describe('ForecastingDashboard (legacy model zoo)', () => {
  async function renderDashboard(csv: string) {
    mockFetch([[/bi_dashboard_unified\.csv$/, { ok: true, text: csv }]]);
    const { ForecastingDashboard } = await import('@/components/forecasting/ForecastingDashboard');
    const utils = render(<ForecastingDashboard />);
    await waitFor(() => expect(screen.queryByText(/Cargando datos del Forecasting/)).toBeNull());
    return utils;
  }

  it('BL-04: el BODY del banner viene del SSOT y conserva la frase derivada de los datos', async () => {
    await renderDashboard(CSV_WITH_DA);
    const banner = assertZooBannerVisible();
    // Frase derivada (useMemo sobre los rows cargados) — la honestidad dinámica se conserva.
    expect(banner.textContent).toMatch(/precision direccional media es 51\.7% sobre 3 mediciones/);
  });

  it('BL-02: el banner renderiza aunque no haya stats de DA (sin rama que lo oculte)', async () => {
    await renderDashboard(CSV_NO_DA);
    assertZooBannerVisible();
  });
});

// ═══════════════════════════════════════ 4 · GM ForecastingView (todos los modos)

describe('ForecastingView (GM) — banner incondicional por modo', () => {
  beforeEach(() => {
    // El CSV del zoo falla: el banner NO puede depender de datos cargados.
    mockFetch([[/bi_dashboard_unified\.csv$/, { ok: false, status: 404 }]]);
  });

  async function renderGm(qs: string) {
    currentQs = qs;
    const { ForecastingView } = await import('@/components/gm/views/ForecastingView');
    return render(<ForecastingView />);
  }

  it('modo replay direccional (usdcop por defecto): banner visible con rama direccional', async () => {
    await renderGm('asset=usdcop');
    const banner = screen.getByTestId(FORECAST_DISCLAIMER_TESTID);
    assertHardVisible(banner);
    expect(banner.textContent).toContain(FORECAST_DISCLAIMER_HEADLINE);
    expect(banner.textContent).toContain(FORECAST_DISCLAIMER_DIRECTIONAL_TITLE);
  });

  it('modo model zoo (model=ALL): banner visible con rama zoo, aun sin datos', async () => {
    await renderGm('asset=usdcop&model=ALL');
    assertZooBannerVisible();
  });
});

// ═══════════════════ 5 · GM AssetWeeklyBody (weekly inference con piel GM)

describe('AssetWeeklyBody (GM weekly inference)', () => {
  async function getBody() {
    const mod: any = await import('@/components/gm/views/ForecastingView');
    return mod.AssetWeeklyBody;
  }

  it('se exporta para cobertura de render (CXD-032 §6: packs con Vitest render)', async () => {
    const AssetWeeklyBody = await getBody();
    expect(
      AssetWeeklyBody,
      'AssetWeeklyBody debe exportarse desde ForecastingView.tsx para poder renderizarse en tests',
    ).toBeTypeOf('function');
  });

  it('BL-03: etiqueta la convicción como proxy NO probabilístico y cumple a11y de tabla', async () => {
    const AssetWeeklyBody = await getBody();
    expect(AssetWeeklyBody).toBeTypeOf('function');
    const { container } = render(
      <AssetWeeklyBody data={GOLD_WEEKLY_2025} strategyId="gold_trend_b2" forward={null} />,
    );
    await screen.findByText('2025-W01');
    // Etiqueta mandatoria + prohibición de wording probabilístico sobre `confidence`.
    expect(screen.getByText(CONVICTION_LABEL)).toBeTruthy();
    expect(document.body.textContent).not.toMatch(/probabilidad estimada/i);
    expect(screen.queryByText(/^Prob\.?\s*\(proxy\)$/)).toBeNull();
    // Valores desde confidence.
    const w02 = screen.getByText('2025-W02').closest('tr') as HTMLElement;
    expect(within(w02).getByText('62%')).toBeTruthy();
    // a11y (mismo patrón que la tabla legacy).
    assertWeeklyTableA11y(container);
  });
});

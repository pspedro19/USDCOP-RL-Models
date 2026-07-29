/**
 * BL-34 — /replay renderiza CERO superficie de Voto 2 (render twin del candado estático).
 * =======================================================================================
 * Gemelo de `tests/regression/test_replay_is_read_only.py`. El candado estático cubre el
 * PERÍMETRO (qué ficheros alcanza /replay y qué pueden contener); este cubre el RENDER de la
 * implementación compartida `/dashboard`+`/replay`, que el perímetro exime a propósito.
 *
 * Ataca M-A directamente:
 *   `ForecastingBacktestSection.tsx:1619`
 *   `const canPromote = userRole === 'admin' && !readOnly`  →  `const canPromote = true`
 * Con esa mutación, la MISMA sección montada en variante `/replay` (readOnly) con sesión
 * ADMIN monta ApprovalPanel + DeployPanel: "Aprobar (Voto 2/2)", "Rechazar", "Desplegar a
 * producción". Eso viola `approval-gates.md` invariante 3 (los botones de aprobación viven
 * SOLO en /dashboard) y añade una segunda puerta al doble voto.
 *
 * POR QUÉ HAY UN CONTROL POSITIVO (léelo antes de borrarlo)
 * --------------------------------------------------------
 * Un test que sólo afirma "no hay botones" es verde también cuando los mocks no montan
 * NADA — la forma más común de candado falso. Por eso cada caso se ejecuta en pareja:
 *   1. variante /dashboard (readOnly=false, admin) → los tres controles DEBEN existir;
 *   2. variante /replay    (readOnly=true,  admin) → CERO controles.
 * Si el fixture deja de producir el panel, (1) se pone rojo y nadie puede confundir
 * "invisible" con "inexistente".
 *
 * La consulta es por ROL ACCESIBLE (`queryAllByRole('button', {name})`), no por testid ni
 * por clase: cualquier botón que un lector de pantalla anuncie como aprobar/rechazar/
 * desplegar/promover falla, lo haya escrito quien lo haya escrito y esté donde esté —
 * incluido un widget nuevo dentro del propio fichero compartido, que es justo el hueco
 * que el candado estático NO puede cerrar.
 *
 * MUTACIONES PROBADAS (cada una deja este fichero en rojo):
 *  M-A   `const canPromote = true`                                → replay muestra los 3 botones
 *  M-A'  `... && !readOnly || true`                               → idem
 *  M-B'  añadir un botón "Aprobar" fuera de `{canPromote && …}`   → idem
 *  M-C   borrar `{readOnly && …}` con la nota de solo-lectura     → falta replay-readonly-note
 */
import React from 'react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, cleanup, waitFor } from '@testing-library/react';

// ───────────────────────────────────────────────────────────────── sesión (rol)
//
// El peor caso para /replay es ADMIN: es el único rol que en /dashboard SÍ ve el Voto 2,
// así que un test con sesión `free` sería verde por accidente y no probaría nada.
let currentRole: string | null = 'admin';
vi.mock('next-auth/react', () => ({
  useSession: () => (currentRole === null
    ? { data: null, status: 'unauthenticated' }
    : { data: { user: { role: currentRole } }, status: 'authenticated' }),
}));

// Barrel GM: TerminalShell arrastra next/font (no carga en vitest/jsdom). Se stubbean SOLO
// primitivos de presentación — los botones bajo prueba los pinta ForecastingBacktestSection
// real, no este stub.
vi.mock('@/components/gm', () => {
  const R = require('react');
  return {
    GmBadge: ({ children, tone }: any) => R.createElement('span', { 'data-tone': tone ?? 'neutral' }, children),
    GmKpi: ({ label, value }: any) => R.createElement('div', null, `${label}: ${value}`),
    GmPanel: ({ title, actions, children }: any) =>
      R.createElement('section', null, title, actions, children),
    GmSkeleton: ({ label }: any) => R.createElement('div', { 'data-testid': 'gm-skeleton' }, label),
  };
});

// Recharts no mide en jsdom (ResponsiveContainer = 0×0) y ensucia la consola.
vi.mock('recharts', () => {
  const R = require('react');
  const P = ({ children }: any) => R.createElement('div', null, children);
  return {
    AreaChart: P, Area: P, XAxis: P, YAxis: P, ReferenceLine: P,
    Tooltip: P, ResponsiveContainer: P,
  };
});

// El chart pesado entra por `lazy(() => import(...))`; sin stub, Suspense nunca resuelve.
vi.mock('@/components/charts/TradingChartWithSignals', () => {
  const R = require('react');
  return { default: () => R.createElement('div', { 'data-testid': 'chart-stub' }) };
});

import { ForecastingBacktestSection } from '@/components/production/ForecastingBacktestSection';

// ────────────────────────────────────────────────────────────────── fixtures

const SID = 'smart_simple_v11';

const SUMMARY = {
  strategy_id: SID,
  strategy_name: 'Smart Simple v1.1',
  year: 2025,
  initial_capital: 10000,
  direction_accuracy_pct: 55.2,
  statistical_tests: { p_value: 0.006, significant: true },
  strategies: {
    [SID]: {
      final_equity: 12563, total_return_pct: 25.63, sharpe: 3.35, max_dd_pct: 6.1,
      win_rate_pct: 64.7, profit_factor: 2.9, trading_days: 250, n_long: 20, n_short: 14,
      exit_reasons: { take_profit: 20, hard_stop: 4, time_exit: 10 },
    },
  },
};

const TRADES = {
  strategy_id: SID,
  trades: Array.from({ length: 12 }, (_, i) => ({
    trade_id: `t${i}`,
    timestamp: `2025-0${(i % 9) + 1}-10`,
    exit_timestamp: `2025-0${(i % 9) + 1}-17`,
    direction: i % 2 === 0 ? 'LONG' : 'SHORT',
    entry_price: 4000 + i, exit_price: 4010 + i,
    pnl_pct: i % 3 === 0 ? -0.4 : 0.9, pnl_abs: i % 3 === 0 ? -40 : 90,
    exit_reason: 'take_profit', size: 1,
  })),
};

/**
 * Los dos paneles de Voto 2 son MUTUAMENTE EXCLUYENTES por estado:
 *   ApprovalPanel  → `if (approval.status !== 'PENDING_APPROVAL') return null`  (Aprobar/Rechazar)
 *   DeployPanel    → `if (approval.status !== 'APPROVED') return null`          (Desplegar)
 * Por eso CADA caso se ejecuta en los dos estados: con uno solo, la mitad de la superficie
 * de Voto 2 nunca se ejercería y el test sería verde por no haberla montado jamás.
 */
type ApprovalStatus = 'PENDING_APPROVAL' | 'APPROVED';
let approvalStatus: ApprovalStatus = 'PENDING_APPROVAL';

const approvalState = () => ({
  status: approvalStatus,
  strategy: SID,
  strategy_name: 'Smart Simple v1.1',
  backtest_recommendation: 'PROMOTE',
  backtest_confidence: 1,
  approved_by: approvalStatus === 'APPROVED' ? 'admin' : undefined,
  approved_at: approvalStatus === 'APPROVED' ? '2026-01-02T00:00:00Z' : undefined,
  gates: [
    { gate: 'min_return_pct', label: 'Retorno mínimo', passed: true, value: 25.6, threshold: -15 },
    { gate: 'min_trades', label: 'Trades mínimos', passed: true, value: 34, threshold: 10 },
    { gate: 'statistical_significance', label: 'p-value', passed: true, value: 0.006, threshold: 0.05 },
  ],
  created_at: '2026-01-01T00:00:00Z',
  last_updated: '2026-01-01T00:00:00Z',
});

const json = (body: unknown) => ({
  ok: true, status: 200, json: async () => body, text: async () => JSON.stringify(body),
});
const notFound = () => ({ ok: false, status: 404, json: async () => ({}), text: async () => '' });

function installFetch() {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.startsWith('/api/registry')) {
      return json({
        strategies: [{
          strategy_id: SID, display_name: 'Smart Simple v1.1', asset_id: 'usdcop',
          pipeline_type: 'ml_forecasting', status: 'experimental',
          backtest_years: [2025], return_pct: 25.63, sharpe: 3.35, p_value: 0.006,
        }],
        assets: [{ asset_id: 'usdcop', display_name: 'USD/COP' }],
        default: { strategy_id: SID },
      });
    }
    if (url.startsWith('/api/production/approval')) return json({ ok: true, data: approvalState() });
    if (url.startsWith('/api/production/deploy/status')) return json({ status: 'idle' });
    if (url.includes('/data/production/summary_2025.json')) return json(SUMMARY);
    if (url.includes(`/data/production/trades/${SID}_2025.json`)) return json(TRADES);
    return notFound();               // manifest, deploy/status, market price, …
  });
  vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch);
  return fetchMock;
}

/** Monta la sección y espera a que el bundle esté pintado (fin del estado `loading`). */
async function mount(readOnly: boolean) {
  const view = render(<ForecastingBacktestSection readOnly={readOnly} />);
  await waitFor(() => {
    expect(screen.queryByTestId('gm-skeleton')).toBeNull();
    // Ancla del bundle publicado: si esto no aparece, el fixture no cargó y cualquier
    // aserto de ausencia sería vacío.
    expect(screen.getAllByText('Smart Simple v1.1').length).toBeGreaterThan(0);
  }, { timeout: 5000 });
  return view;
}

/** Cualquier botón cuyo NOMBRE ACCESIBLE sea un verbo de Voto 2, venga de donde venga. */
const VOTE2_BUTTON = /aprobar|aprobaci[oó]n|rechazar|desplegar|promover|voto\s*2/i;
const vote2Buttons = () =>
  screen.queryAllByRole('button', { name: VOTE2_BUTTON });

describe('BL-34 — /replay es READ-ONLY en render (approval-gates.md invariante 3)', () => {
  beforeEach(() => {
    currentRole = 'admin';
    approvalStatus = 'PENDING_APPROVAL';
    installFetch();
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('CONTROL POSITIVO: la variante /dashboard (admin) SÍ monta la superficie de Voto 2', async () => {
    // PENDING_APPROVAL → ApprovalPanel (Aprobar / Rechazar)
    approvalStatus = 'PENDING_APPROVAL';
    await mount(false);
    expect(
      vote2Buttons().length,
      'el fixture dejó de montar ApprovalPanel: sin este control, el test de /replay sería '
      + 'verde por vacuidad (no porque los botones estén ausentes)',
    ).toBeGreaterThan(0);
    expect(screen.getByRole('button', { name: /Aprobar \(Voto 2\/2\)/i })).toBeTruthy();
    expect(screen.getByRole('button', { name: /^Rechazar$/i })).toBeTruthy();
    cleanup();

    // APPROVED → DeployPanel (Desplegar a producción)
    approvalStatus = 'APPROVED';
    await mount(false);
    expect(screen.getByRole('button', { name: /Desplegar a producción/i })).toBeTruthy();
  });

  it('M-A: la variante /replay con sesión ADMIN no monta NINGÚN control de Voto 2', async () => {
    for (const status of ['PENDING_APPROVAL', 'APPROVED'] as const) {
      approvalStatus = status;
      await mount(true);
      expect(
        vote2Buttons().map((b) => b.textContent?.trim()),
        `[${status}] Con readOnly=true (variante /replay) y rol admin no puede existir ningún `
        + 'botón de aprobación/rechazo/despliegue: el Voto 2 vive SOLO en /dashboard y decide '
        + 'sobre los números del bundle publicado (approval-gates.md inv. 3, '
        + 'quant-constitution §7). Mutación que esto detecta: `const canPromote = true`.',
      ).toEqual([]);
      // Redundante a propósito: si alguien renombra los botones, el regex de arriba podría
      // dejar de casar; el copy de los paneles tampoco debe existir.
      expect(screen.queryByText(/Aprobar \(Voto 2\/2\)/i)).toBeNull();
      expect(screen.queryByText(/Desplegar a producción/i)).toBeNull();
      cleanup();
    }
  });

  it('M-A en todos los roles: ni admin ni ningún otro rol ve Voto 2 en /replay', async () => {
    for (const role of ['admin', 'developer', 'subscriber', 'free', null] as const) {
      for (const status of ['PENDING_APPROVAL', 'APPROVED'] as const) {
        currentRole = role;
        approvalStatus = status;
        await mount(true);
        expect(
          vote2Buttons().map((b) => b.textContent?.trim()),
          `rol ${role ?? 'anon'} (${status}) vio controles de Voto 2 en /replay`,
        ).toEqual([]);
        cleanup();
      }
    }
  });

  it('M-C: /replay muestra la nota de solo-lectura y nombra /dashboard; /dashboard no la muestra', async () => {
    const { unmount } = await mount(true);
    const note = screen.getByTestId('replay-readonly-note');
    expect(note).toBeVisible();
    expect(note.textContent ?? '').toMatch(/\/dashboard/);
    unmount();
    cleanup();
    await mount(false);
    expect(
      screen.queryByTestId('replay-readonly-note'),
      'la nota "esto es solo lectura" es FALSA en /dashboard, donde sí se vota',
    ).toBeNull();
  });

  it('los gates del Voto 1 siguen visibles en /replay (read-only ≠ ciego)', async () => {
    await mount(true);
    // Investigar es el propósito de /replay: quitar los gates junto con los botones sería
    // "arreglar" el hallazgo rompiendo la superficie.
    expect(screen.getByText('Retorno mínimo')).toBeTruthy();
    expect(screen.getByText('p-value')).toBeTruthy();
  });
});

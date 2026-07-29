/**
 * BL-05 — TEST DE MONTAJE: el A/B de candidatas vive DENTRO de /production.
 *
 * Hueco que cierra (verificación por mutación, 2026-07-28): los 13 tests de
 * PaperCandidatesPanel.test.tsx renderizan el panel AISLADO con un fixture, así que
 * anular su montaje en ProductionView.tsx:982 —`{false && paperLedger && <PaperCandidatesPanel/>}`,
 * exactamente el estado que BL-05 existe para arreglar— dejaba la suite en 13 passed / 578
 * passed, idéntica. Un panel perfecto y desconectado de la página seguía siendo "verde".
 *
 * Aquí se monta la VISTA REAL (`ProductionView`) con la cadena de datos real
 * (useGmQuery → apiFetch → fetch mockeado por ruta). Nada del panel se stubbea:
 * si la línea 982 deja de montarlo, o si la URL del ledger cambia, o si el bundle
 * de producción deja de resolver, la tabla desaparece del documento y esto se pone rojo.
 *
 * MUTACIONES QUE LO PONEN ROJO (probadas):
 *  M1  `{false && paperLedger && <PaperCandidatesPanel …/>}`   → no hay tabla "candidatas"
 *  M2  borrar la línea 982 entera                              → ídem
 *  M3  cambiar la URL del ledger a una ruta inexistente        → paperLedger=null ⇒ sin tabla
 *  M4  montar el panel también en la vista CLIENTE (RBAC §8)   → el test de rol free falla
 */
import React from 'react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';

import { stubChartLayout } from '../../support/chart-layout';

// ────────────────────────────────────────────────────────────── mocks de entorno
//
// next/font solo existe dentro del runtime de Next: sin este mock el BARREL
// `@/components/gm` (→ TerminalShell → JetBrains_Mono) revienta al importarse y la
// vista entera es intesteable. Con él, la vista se monta con sus componentes REALES
// (AsyncBoundary, primitives, useGmQuery) — nada del árbol bajo prueba se stubbea.
vi.mock('next/font/google', () => ({
  JetBrains_Mono: () => ({ variable: 'font-jb-mono', className: 'font-jb-mono', style: { fontFamily: 'monospace' } }),
  Inter: () => ({ variable: 'font-inter', className: 'font-inter', style: { fontFamily: 'sans-serif' } }),
}));

// Sesión por ROL: el ledger es un internal del juez sellado (RBAC §8) — la vista
// cliente (free/subscriber) ni siquiera pide la URL.
let currentRole: 'admin' | 'developer' | 'subscriber' | 'free' = 'admin';
vi.mock('next-auth/react', () => ({
  useSession: () => ({ data: { user: { role: currentRole } }, status: 'authenticated' }),
}));

// El gráfico de velas (lightweight-charts) no aporta nada a esta garantía y arrastra
// canvas/WebGL: se stubbea SOLO él, vía su módulo, para que `next/dynamic` lo resuelva.
vi.mock('@/components/charts/TradingChartWithSignals', () => ({
  __esModule: true,
  default: () => React.createElement('div', { 'data-testid': 'chart-stub' }),
}));

// recharts necesita layout (jsdom mide 0×0) — la curva de equity se monta de verdad.
stubChartLayout();

// ────────────────────────────────────────────────────────────── fixtures reales
// Subconjuntos fieles de los ficheros publicados en public/data/production/**.

/** public/data/production/paper/candidates_ledger_2026.json */
const LEDGER = {
  contract: 'CTR-QUANT-CONSTITUTION-001',
  anchor: '2026-01-01 (directiva operador 2026-07-22)',
  labels: {
    smart_simple_v11: 'forward real todo 2026 (producción)',
    smart_simple_v12: 'replay descriptivo Ene->2026-07-21 (mirado, trial pagado) + forward post-freeze',
    v13: 'EXCLUIDA hasta freeze (abrir 2026 = +1 trial, decisión del operador)',
  },
  judge_note: 'el juez sellado de v12/v14 consume SOLO judge_window (post-freeze)',
  generated_at: '2026-07-27',
  strategies: {
    smart_simple_v11: {
      ret_2026_ytd_pct: 3.36, n_trades: 11,
      note_n: 'N<20 => solo conteo y PnL', judge_window: null,
    },
    smart_simple_v12: {
      ret_2026_ytd_pct: 3.12, n_trades: 11, note_n: 'N<20 => solo conteo y PnL',
      judge_window: {
        starts_after: '2026-07-21', n_trades: 0, pnl_pct_compound: 0.0,
        note: 'N<20 => solo conteo y PnL',
      },
    },
    smart_simple_v14: {
      ret_2026_ytd_pct: null, n_trades: null,
      judge_window: { starts_after: '2026-07-24', n_trades: null, pnl_pct_compound: null },
    },
  },
};

/** public/data/production/summary.json (forward 2026, N<20 ⇒ sharpe null) */
const SUMMARY = {
  generated_at: '2026-07-21T17:58:47.257408',
  strategy_name: 'Smart Simple v2.0.0',
  strategy_id: 'smart_simple_v11',
  year: 2026,
  initial_capital: 10000.0,
  n_trades: 11,
  insufficient_trades: true,
  strategies: {
    buy_and_hold: { final_equity: 8554.0, total_return_pct: -14.46 },
    smart_simple_v11: {
      final_equity: 10335.82, total_return_pct: 3.36, calmar: 2.24, sharpe: null,
      max_dd_pct: 1.5, win_rate_pct: 72.7, profit_factor: 2.408, trading_days: 55,
      exit_reasons: { week_end: 7, take_profit: 3, hard_stop: 1 },
      n_long: 10, n_short: 1, insufficient_trades: true,
    },
  },
  statistical_tests: { p_value: null, significant: false, insufficient_trades: true, min_trades_for_stats: 20 },
};

const STRATEGIES = {
  strategies: [{
    strategy_id: 'smart_simple_v11', strategy_name: 'Smart Simple v2.0.0',
    status: 'LIVE', year: 2026, return_pct: 3.36, mode: 'live', is_active_default: true,
  }],
  active_strategy_id: 'smart_simple_v11',
};

const APPROVAL = {
  status: 'LIVE', strategy: 'smart_simple_v11', strategy_name: 'Smart Simple v2.0.0',
  backtest_recommendation: 'PROMOTE', backtest_confidence: 1,
  gates: [], created_at: '2026-01-01', last_updated: '2026-07-21',
};

const TRADES = {
  strategy_id: 'smart_simple_v11',
  trades: [{
    trade_id: 1, timestamp: '2026-01-05', exit_timestamp: '2026-01-09', side: 'LONG',
    entry_price: 4300, exit_price: 4326, pnl_usd: 61, pnl_pct: 0.61,
    exit_reason: 'week_end', equity_at_entry: 10000, equity_at_exit: 10061, leverage: 1,
  }],
};

/** Router de fetch por ruta — mismas URLs que pide ProductionView. */
function mockFetch(overrides: Array<[RegExp, { ok: boolean; status?: number; body?: unknown }]> = []) {
  const routes: Array<[RegExp, { ok: boolean; status?: number; body?: unknown }]> = [
    ...overrides,
    [/\/api\/production\/strategies/, { ok: true, body: STRATEGIES }],
    [/summary_\d{4}\.json/, { ok: true, body: { ...SUMMARY, year: 2025 } }],
    [/\/data\/production\/summary\.json/, { ok: true, body: SUMMARY }],
    [/\/api\/production\/(approval|status)/, { ok: true, body: APPROVAL }],
    [/\/api\/production\/live/, { ok: false, status: 404 }],
    [/\/data\/production\/trades\//, { ok: true, body: TRADES }],
    [/\/api\/market\/realtime-price/, { ok: true, body: { price: 4300, change: 1, changePct: 0.02 } }],
    [/candidates_ledger_2026\.json/, { ok: true, body: LEDGER }],
    [/system_health\.json/, { ok: false, status: 404 }],
  ];
  const seen: string[] = [];
  (global as any).fetch = vi.fn(async (input: unknown) => {
    const url = String(input);
    seen.push(url);
    for (const [re, r] of routes) {
      if (re.test(url)) {
        return { ok: r.ok, status: r.status ?? (r.ok ? 200 : 404), json: async () => r.body ?? null };
      }
    }
    return { ok: false, status: 404, json: async () => null };
  });
  return seen;
}

async function renderProduction() {
  const { ProductionView } = await import('@/components/gm/views/ProductionView');
  return render(<ProductionView />);
}

beforeEach(() => {
  currentRole = 'admin';
  mockFetch();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('ProductionView — el panel A/B de candidatas está MONTADO en la página (BL-05)', () => {
  // ROJO con: `{false && paperLedger && <PaperCandidatesPanel …/>}` en ProductionView.tsx:982.
  it('rol admin: la tabla de candidatas se renderiza dentro de /production', async () => {
    await renderProduction();

    const table = await screen.findByRole('table', { name: /candidatas/i }, { timeout: 5000 });
    expect(table).toBeInTheDocument();
    // La tabla es la del ledger publicado, no cualquier tabla: sus filas son las
    // candidatas del A/B (v11 producción vs v12/v14 paper con juez sellado).
    for (const sid of Object.keys(LEDGER.strategies)) {
      expect(table.textContent, `falta la candidata ${sid} en la tabla montada`).toContain(sid);
    }
  });

  // ROJO con: montar el panel sin el gate de rol (`{paperLedger && …}` sirviendo también
  // a la vista cliente) — el juez sellado es un internal (RBAC §8, ux-navigation P3).
  it('rol free: la vista cliente NO monta el A/B (internals del juez sellado)', async () => {
    currentRole = 'free';
    const seen = mockFetch();
    await renderProduction();

    // Se espera a que la vista tenga datos (el bundle carga igual) antes de negar.
    await screen.findByTestId('prod-kpis', undefined, { timeout: 5000 });
    await waitFor(() => {
      expect(screen.queryByRole('table', { name: /candidatas/i })).toBeNull();
    });
    // Y ni siquiera se pide la URL del ledger: el gate está en el fetch, no solo en el render.
    expect(seen.some((u) => /candidates_ledger/.test(u))).toBe(false);
  });
});

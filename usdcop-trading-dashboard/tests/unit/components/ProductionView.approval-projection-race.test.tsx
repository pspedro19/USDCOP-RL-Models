/**
 * D1 — `TypeError: Cannot read properties of undefined (reading 'filter')` en /production.
 *
 * DEFECTO REAL (reproducido contra el contenedor, 5 de 6 cargas de /production con
 * sesión admin): el estado de aprobación tiene DOS proyecciones con DOS contratos
 * distintos y la vista elige por rol (CXD-057):
 *   · cliente  → `/api/production/status`   = allowlist sanitizada, **SIN `gates`**
 *   · research → `/api/production/approval` = documento íntegro, **CON `gates`**
 *
 * `useSession()` arranca en `status: 'loading'` con `data: undefined`, y
 * `ProductionView` colapsa eso a `role ?? 'free'` ⇒ **isClientView = true en el primer
 * render**, así que la consulta sale hacia la proyección de CLIENTE aunque el usuario
 * sea admin. Cuando la sesión resuelve, `isClientView` pasa a false: la URL cambia a
 * `/api/production/approval` y `<ApprovalPanel>` empieza a montarse… pero `useGmQuery`
 * conservaba el `data` de la URL ANTERIOR. El panel recibía entonces la proyección
 * sanitizada, donde `gates` no existe, y `approval.gates.filter(...)` reventaba.
 *
 * Es una CARRERA entre dos respuestas HTTP —`/api/auth/session` vs
 * `/api/production/status`— y por eso no era determinista: si la sesión gana, la URL ya
 * había cambiado antes de que llegase dato alguno y no hay error. Aquí se fija el orden
 * perdedor a mano, que es el único modo de convertir una carrera en una prueba.
 *
 * MUTACIONES QUE LO PONEN ROJO (probadas):
 *  M1  revertir el reset por `path` en useGmQuery.ts  → TypeError …reading 'filter'
 *  M2  revertir el gate `status === 'loading'` de ProductionView → se pide la
 *      proyección de cliente siendo admin (test "no consulta la proyección de cliente")
 */
import React from 'react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, act, waitFor } from '@testing-library/react';

import { stubChartLayout } from '../../support/chart-layout';

// next/font solo existe en el runtime de Next; sin esto el barrel @/components/gm no importa.
vi.mock('next/font/google', () => ({
  JetBrains_Mono: () => ({ variable: 'font-jb-mono', className: 'font-jb-mono', style: { fontFamily: 'monospace' } }),
  Inter: () => ({ variable: 'font-inter', className: 'font-inter', style: { fontFamily: 'sans-serif' } }),
}));

// ── sesión CONTROLABLE: reproduce el ciclo real loading → authenticated(admin).
// next-auth no entrega la sesión en el primer render (no hay `session` inicial en
// AuthSessionProvider), así que este mock es fiel al runtime, no una comodidad.
const sessionStore = vi.hoisted(() => {
  let state: { data: unknown; status: string } = { data: undefined, status: 'loading' };
  const listeners = new Set<() => void>();
  return {
    get: () => state,
    set(next: { data: unknown; status: string }) {
      state = next;
      listeners.forEach((l) => l());
    },
    subscribe(l: () => void) {
      listeners.add(l);
      return () => { listeners.delete(l); };
    },
    reset() {
      state = { data: undefined, status: 'loading' };
    },
  };
});

vi.mock('next-auth/react', () => ({
  useSession: () => {
    const [, force] = React.useState(0);
    React.useEffect(() => sessionStore.subscribe(() => force((n) => n + 1)), []);
    return sessionStore.get();
  },
}));

vi.mock('@/components/charts/TradingChartWithSignals', () => ({
  __esModule: true,
  default: () => React.createElement('div', { 'data-testid': 'chart-stub' }),
}));

stubChartLayout();

// ────────────────────────────────────────────────────────────── fixtures reales
// Recortes fieles de data/approvals/approval_state.json y de lo que
// `toPublicApproval()` (lib/approvals/store.ts) deja pasar.

/** GET /api/production/approval — documento ÍNTEGRO (research:read). */
const APPROVAL_FULL = {
  status: 'PENDING_APPROVAL',
  strategy: 'smart_simple_v11',
  strategy_name: 'Smart Simple v2.0.0',
  backtest_recommendation: 'HOLD',
  backtest_confidence: 0.8333,
  gates: [
    { gate: 'return', label: 'Retorno', value: '+3.36%', threshold: '> -15%', passed: true },
    { gate: 'sharpe', label: 'Sharpe', value: '1.20', threshold: '> 0', passed: true },
    { gate: 'max_dd', label: 'Max DD', value: '1.5%', threshold: '< 20%', passed: true },
    { gate: 'trades', label: 'Trades', value: '34', threshold: '>= 10', passed: true },
    { gate: 'p_value', label: 'p-value', value: '0.006', threshold: '< 0.05', passed: true },
    { gate: 'deflated_sharpe', label: 'DSR', value: '0.50', threshold: '> 0.95', passed: false },
  ],
  created_at: '2026-01-01',
  last_updated: '2026-07-21',
};

/** GET /api/production/status — allowlist de `toPublicApproval`: NO existe `gates`. */
const APPROVAL_PUBLIC = {
  status: APPROVAL_FULL.status,
  strategy: APPROVAL_FULL.strategy,
  strategy_name: APPROVAL_FULL.strategy_name,
  approved_at: null,
  created_at: APPROVAL_FULL.created_at,
  last_updated: APPROVAL_FULL.last_updated,
};

const SUMMARY = {
  generated_at: '2026-07-21T17:58:47.257408',
  strategy_name: 'Smart Simple v2.0.0',
  strategy_id: 'smart_simple_v11',
  year: 2026,
  initial_capital: 10000.0,
  n_trades: 11,
  insufficient_trades: true,
  strategies: {
    smart_simple_v11: {
      final_equity: 10335.82, total_return_pct: 3.36, calmar: 2.24, sharpe: null,
      max_dd_pct: 1.5, win_rate_pct: 72.7, profit_factor: 2.408, trading_days: 55,
      exit_reasons: { week_end: 7 }, n_long: 10, n_short: 1, insufficient_trades: true,
    },
  },
  statistical_tests: { p_value: null, significant: false, insufficient_trades: true },
};

const STRATEGIES = {
  strategies: [{
    strategy_id: 'smart_simple_v11', strategy_name: 'Smart Simple v2.0.0',
    status: 'LIVE', year: 2026, return_pct: 3.36, mode: 'live', is_active_default: true,
  }],
  active_strategy_id: 'smart_simple_v11',
};

const TRADES = {
  strategy_id: 'smart_simple_v11',
  trades: [{
    trade_id: 1, timestamp: '2026-01-05', exit_timestamp: '2026-01-09', side: 'LONG',
    entry_price: 4300, exit_price: 4326, pnl_usd: 61, pnl_pct: 0.61,
    exit_reason: 'week_end', equity_at_entry: 10000, equity_at_exit: 10061, leverage: 1,
  }],
};

/** URLs pedidas, en orden — permite afirmar QUÉ proyección se consultó. */
let seen: string[] = [];

function installFetch() {
  seen = [];
  (global as any).fetch = vi.fn(async (input: unknown) => {
    const url = String(input);
    seen.push(url);
    const reply = (body: unknown, status = 200) => ({
      ok: status < 400, status, json: async () => body,
    });
    // El orden importa: /approval antes que /status (el segundo es prefijo del primero
    // en ninguna dirección, pero se explicita para que el router no dependa del azar).
    if (/\/api\/production\/approval/.test(url)) return reply(APPROVAL_FULL);
    if (/\/api\/production\/status/.test(url)) return reply(APPROVAL_PUBLIC);
    if (/\/api\/production\/strategies/.test(url)) return reply(STRATEGIES);
    if (/summary_\d{4}\.json/.test(url)) return reply(null, 404);
    if (/\/data\/production\/summary\.json/.test(url)) return reply(SUMMARY);
    if (/\/api\/production\/live/.test(url)) return reply(null, 404);
    if (/\/data\/production\/trades\//.test(url)) return reply(TRADES);
    if (/realtime-price/.test(url)) return reply({ price: 4300, change: 1, changePct: 0.02 });
    return reply(null, 404);
  });
}

async function renderProduction() {
  const { ProductionView } = await import('@/components/gm/views/ProductionView');
  return render(<ProductionView />);
}

/** Deja correr microtareas hasta que todas las promesas de fetch encoladas resuelvan. */
async function settle(rounds = 6) {
  for (let i = 0; i < rounds; i++) {
    await act(async () => { await Promise.resolve(); });
  }
}

beforeEach(() => {
  sessionStore.reset();
  installFetch();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('D1 — /production no puede renderizar la proyección de CLIENTE en el panel de research', () => {
  // ROJO sin el fix: al aplicar `setSession(admin)` React re-renderiza con
  // `approval` = APPROVAL_PUBLIC (sin `gates`) y ApprovalPanel lanza
  // "TypeError: Cannot read properties of undefined (reading 'filter')".
  it('la sesión resuelve a admin DESPUÉS de que llegó la proyección de cliente (orden perdedor de la carrera)', async () => {
    await renderProduction();

    // 1) Se deja llegar la respuesta que exista para la URL vigente durante `loading`.
    await settle();

    // 2) La sesión resuelve a admin: `isClientView` pasa a false y el panel de research
    //    se monta. Si el hook arrastrase el `data` de la URL anterior, revienta aquí.
    await act(async () => {
      sessionStore.set({ data: { user: { role: 'admin' } }, status: 'authenticated' });
    });
    await settle();

    // 3) El panel debe mostrar los gates del documento ÍNTEGRO — 5 de 6 (el DSR falla).
    //    No basta con "no reventó": si el panel no llega a montarse, esto también cae.
    const panel = await screen.findByText(/Gates Vote 1 · 5\/6 pasaron/, undefined, { timeout: 5000 });
    expect(panel).toBeInTheDocument();
  });

  // ROJO sin el gate de sesión: la primera URL consultada es la de cliente.
  it('siendo admin NO se consulta la proyección de cliente ni una sola vez', async () => {
    await renderProduction();
    await settle();
    await act(async () => {
      sessionStore.set({ data: { user: { role: 'admin' } }, status: 'authenticated' });
    });
    await settle();
    await screen.findByText(/Gates Vote 1/, undefined, { timeout: 5000 });

    expect(
      seen.filter((u) => /\/api\/production\/status/.test(u)),
      `un admin no debe pedir la allowlist de cliente; URLs vistas: ${seen.join(' | ')}`,
    ).toHaveLength(0);
  });

  // La vista de cliente sigue funcionando: pide la allowlist y NO monta ApprovalPanel.
  it('rol free: consulta la proyección sanitizada y no monta el panel de gates (RBAC §8)', async () => {
    await renderProduction();
    await act(async () => {
      sessionStore.set({ data: { user: { role: 'free' } }, status: 'authenticated' });
    });
    await settle();
    await screen.findByTestId('prod-kpis', undefined, { timeout: 5000 });

    await waitFor(() => {
      expect(seen.some((u) => /\/api\/production\/status/.test(u))).toBe(true);
    });
    expect(seen.some((u) => /\/api\/production\/approval/.test(u))).toBe(false);
    expect(screen.queryByText(/Gates Vote 1/)).toBeNull();
  });
});

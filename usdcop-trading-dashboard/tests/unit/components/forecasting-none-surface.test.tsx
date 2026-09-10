/**
 * C034 / CXD-549 — `forecast_mode: 'none'` es una rama EXHAUSTIVA, no "lo que no es zoo".
 *
 * POR QUÉ EXISTE ESTE FICHERO. El 2026-08-05 se añadió el valor `'none'` al contrato con el
 * significado «este activo no tiene superficie de forecasting publicada» (medido: `spx500`
 * tenía 0 artefactos frente a los 459 de `xauusd`/`btcusdt`). Pero AMBAS vistas ramificaban
 * con un booleano `isModelZoo`, así que **todo lo que no era `model_zoo` caía en weekly** —
 * incluido `'none'`. Resultado: `?asset=spx500` no declaraba ausencia, **afirmaba la rama
 * weekly** y pedía `/api/forecasting/spx500/{index,forward,weekly_inference_<year>}.json`,
 * artefactos que no existen. El contrato decía una cosa y el código hacía otra.
 *
 * CODEX rechazó ese estado (CXD-549) y tenía razón: declarar el residuo en la ficha no lo
 * arregla. Este test fija el DONE-WHEN literal que pidió:
 *   1. estado explícito «sin superficie publicada»,
 *   2. CERO fetch de artefactos zoo y CERO de weekly,
 *   3. y que falle si aparece `Weekly Inference` o si se pide `/api/forecasting/spx500/*`.
 *
 * Lo que este test NO hace: comprobar la forma del texto. Comprueba las dos PROPOSICIONES
 * que pueden mentirle al usuario —qué afirma la superficie y qué artefactos pide— porque un
 * candado sobre prosa se sortea reescribiendo la prosa.
 */
import React from 'react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, cleanup, act } from '@testing-library/react';

import { installConsoleGate } from '../../support/console-gate';
import { stubChartLayout } from '../../support/chart-layout';

vi.mock('next/font/google', () => ({
  JetBrains_Mono: () => ({ variable: 'font-jb-mono', className: 'font-jb-mono', style: { fontFamily: 'monospace' } }),
  Inter: () => ({ variable: 'font-inter', className: 'font-inter', style: { fontFamily: 'sans-serif' } }),
}));

vi.mock('next-auth/react', () => ({
  useSession: () => ({ data: { user: { role: 'admin' } }, status: 'authenticated' }),
  SessionProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

let currentQs = 'asset=spx500';
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), prefetch: vi.fn() }),
  useSearchParams: () => new URLSearchParams(currentQs),
}));

/**
 * Se inyecta un catálogo con los TRES modos vivos a la vez. Es deliberado: si el fixture
 * sólo trajera el activo `none`, un bug que mandara *todo* a la rama de ausencia pasaría
 * verde. Con `usdcop` (zoo) y un activo weekly presentes, la rama tiene que discriminar.
 */
vi.mock('@/lib/contracts/analysis-assets', () => {
  const ASSETS = [
    { asset_id: 'usdcop', symbol: 'USD/COP', chart_symbol: 'USDCOP', display_name: 'USD/COP', asset_class: 'fx', forecast_mode: 'model_zoo' },
    { asset_id: 'rulesfixture', symbol: 'RUL/USD', chart_symbol: 'RULUSD', display_name: 'Activo por reglas (fixture)', asset_class: 'commodity', forecast_mode: 'weekly_inference' },
    { asset_id: 'spx500', symbol: 'SPX500', chart_symbol: 'SPX500', display_name: 'S&P 500', asset_class: 'equity_index', forecast_mode: 'none' },
  ];
  const ids = ASSETS.map((a) => a.asset_id);
  return {
    ANALYSIS_ASSETS: ASSETS,
    ANALYSIS_ASSET_IDS: ids,
    DEFAULT_ANALYSIS_ASSET: 'usdcop',
    isValidAnalysisAsset: (id: string | null | undefined) => !!id && ids.includes(id),
    resolveAnalysisAsset: (id: string | null | undefined) => (id && ids.includes(id) ? id : 'usdcop'),
    getAnalysisAsset: (id: string | null | undefined) => ASSETS.find((a) => a.asset_id === id) ?? ASSETS[0],
  };
});

stubChartLayout();
installConsoleGate();

/** Toda URL pedida queda registrada: el aserto duro se hace sobre ESTA lista. */
let requested: string[] = [];

beforeEach(() => {
  requested = [];
  currentQs = 'asset=spx500';
  (global as any).fetch = vi.fn(async (input: unknown) => {
    const url = String((input as any)?.url ?? input);
    requested.push(url);
    // Se responde 404 a todo a propósito: si la vista pidiera algo, el fallo llegaría
    // por la lista de URLs (aserto explícito) y no disfrazado de "sin datos".
    return new Response('not found', { status: 404, headers: { 'content-type': 'text/plain' } });
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

async function renderView() {
  const { ForecastingView } = await import('@/components/gm/views/ForecastingView');
  // `act` envuelve el render Y el drenaje de efectos: los fetch de la rama weekly resuelven
  // dentro del mismo acto, así que sus setState no escapan y la consola queda limpia (el
  // console-gate es tolerancia cero y un warning de act sería ruido fabricado por el test).
  await act(async () => {
    render(<ForecastingView />);
    await new Promise((r) => setTimeout(r, 0));
  });
}

describe("C034 — forecast_mode 'none' declara ausencia y NO pide artefactos", () => {
  it('monta el estado explícito de superficie ausente', async () => {
    await renderView();
    expect(screen.getByTestId('forecasting-no-surface')).toBeTruthy();
  });

  it('NO afirma Weekly Inference ni ML Model Zoo sobre un activo sin superficie', async () => {
    await renderView();
    // Son afirmaciones DE HECHO sobre una superficie que no existe: el defecto que CXD-549
    // rechazó era exactamente que spx500 mostrara el badge weekly.
    expect(screen.queryByText(/Weekly Inference/i)).toBeNull();
    expect(screen.queryByText(/ML Model Zoo/i)).toBeNull();
  });

  it('NO pide NINGÚN artefacto de /api/forecasting/spx500/*', async () => {
    await renderView();
    const forecasting = requested.filter((u) => u.includes('/api/forecasting/'));
    expect(
      forecasting,
      `la rama 'none' no puede pedir artefactos; pidió: ${forecasting.join(' | ')}`,
    ).toHaveLength(0);
  });

  it('en concreto: ni index.json, ni forward.json, ni weekly_inference_<year>.json, ni el CSV del zoo', async () => {
    await renderView();
    const prohibidas = [
      'index.json',
      'forward.json',
      'weekly_inference',
      'bi_dashboard_unified.csv',
    ];
    for (const frag of prohibidas) {
      const hits = requested.filter((u) => u.includes('spx500') && u.includes(frag));
      expect(hits, `pidió ${frag} para un activo sin superficie: ${hits.join(' | ')}`).toHaveLength(0);
    }
  });

  it('CONTROL: el activo weekly del MISMO fixture sí pide sus artefactos (la rama discrimina)', async () => {
    // Sin este control, mandar TODO a la rama de ausencia pasaría los cuatro tests de arriba.
    currentQs = 'asset=rulesfixture';
    await renderView();
    expect(screen.queryByTestId('forecasting-no-surface')).toBeNull();
    const pedidos = requested.filter((u) => u.includes('/api/forecasting/rulesfixture/'));
    expect(pedidos.length, 'el activo weekly debe seguir pidiendo su índice/forward').toBeGreaterThan(0);
  });
});

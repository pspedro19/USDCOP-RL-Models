/**
 * D1 (invariante del hook) — `data` de useGmQuery PERTENECE a su `path`.
 *
 * El crash de /production (ver ProductionView.approval-projection-race.test.tsx) fue el
 * síntoma visible; la causa estructural es esta: al cambiar `path`, el hook conservaba
 * el `data` de la URL anterior, de modo que la vista renderizaba la carga útil de un
 * endpoint bajo el contrato de otro. Cuando los dos endpoints comparten forma el fallo
 * es silencioso —y peor—: los KPIs de la estrategia anterior se pintan bajo el nombre
 * de la recién seleccionada, es decir números publicados equivocados en la superficie
 * que sostiene el Voto 2 (quant-constitution §7).
 *
 * `stale-while-error` NO se toca: es para un refresco fallido del MISMO path, y este
 * fichero lo fija como garantía para que el arreglo no lo erosione.
 *
 * MUTACIÓN QUE LO PONE ROJO (probada): quitar el bloque `if (syncedPath !== path)` de
 * components/gm/useGmQuery.ts ⇒ "tras cambiar de path el dato viejo sigue publicado".
 */
import React from 'react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, act } from '@testing-library/react';

import { useGmQuery } from '@/components/gm/useGmQuery';

/** Resolvers pendientes por URL: el test decide CUÁNDO responde cada endpoint. */
let pending: Record<string, (body: unknown, status?: number) => void>;

function installFetch() {
  pending = {};
  (global as any).fetch = vi.fn((input: unknown) => {
    const url = String(input);
    return new Promise((resolve) => {
      pending[url] = (body: unknown, status = 200) =>
        resolve({ ok: status < 400, status, json: async () => body } as unknown as Response);
    });
  });
}

function Probe({ path }: { path: string | null }) {
  const q = useGmQuery<{ marca: string }>(path);
  return (
    <div>
      <span data-testid="marca">{q.data ? q.data.marca : 'SIN-DATO'}</span>
      <span data-testid="loading">{String(q.loading)}</span>
      <span data-testid="error">{q.error ? 'ERR' : '-'}</span>
      <button data-testid="reload" onClick={q.reload}>reload</button>
    </div>
  );
}

/** Responde una URL pendiente y deja que React aplique el estado. */
async function respond(url: string, body: unknown, status = 200) {
  await act(async () => {
    pending[url]?.(body, status);
    await Promise.resolve();
    await Promise.resolve();
  });
}

beforeEach(installFetch);
afterEach(() => vi.clearAllMocks());

describe('useGmQuery — identidad path↔data', () => {
  it('al cambiar de path el dato anterior NUNCA se publica bajo la URL nueva', async () => {
    const { rerender } = render(<Probe path="/api/a" />);
    await respond('/api/a', { marca: 'DE-A' });
    expect(screen.getByTestId('marca').textContent).toBe('DE-A');

    // Cambio de URL: la respuesta de /api/b todavía no ha llegado.
    rerender(<Probe path="/api/b" />);

    // En esta ventana el consumidor DEBE ver "sin dato" (y su AsyncBoundary, el
    // esqueleto). Publicar 'DE-A' aquí es servir la respuesta de otro contrato.
    expect(
      screen.getByTestId('marca').textContent,
      'tras cambiar de path el dato viejo sigue publicado',
    ).toBe('SIN-DATO');
    expect(screen.getByTestId('loading').textContent).toBe('true');

    await respond('/api/b', { marca: 'DE-B' });
    expect(screen.getByTestId('marca').textContent).toBe('DE-B');
  });

  it('path → null (consulta desactivada) tampoco deja el dato anterior colgando', async () => {
    const { rerender } = render(<Probe path="/api/a" />);
    await respond('/api/a', { marca: 'DE-A' });
    expect(screen.getByTestId('marca').textContent).toBe('DE-A');

    rerender(<Probe path={null} />);
    expect(screen.getByTestId('marca').textContent).toBe('SIN-DATO');
    expect(screen.getByTestId('loading').textContent).toBe('false');
  });

  it('stale-while-error SIGUE vivo: un refresco fallido del MISMO path conserva el dato', async () => {
    render(<Probe path="/api/a" />);
    await respond('/api/a', { marca: 'DE-A' });

    // Mismo path: `reload` manual (no un cambio de URL) que falla con 500.
    await act(async () => { screen.getByTestId('reload').click(); });
    await respond('/api/a', { error: { code: 'BOOM', message: 'kaput' } }, 500);

    expect(screen.getByTestId('marca').textContent).toBe('DE-A');
    expect(screen.getByTestId('error').textContent).toBe('ERR');
  });
});

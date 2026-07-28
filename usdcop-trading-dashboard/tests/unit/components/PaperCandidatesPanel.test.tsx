/**
 * BL-05 — render test del panel Candidatas A/B (paper ledger).
 *
 * Cubre los 3 puntos del rechazo de Codex:
 *  1. "días al juez": derivados del bundle publicado (starts_after / anchor)
 *     contra el reloj del cliente — NUNCA métricas recomputadas (constitución §7).
 *  2. Tabla móvil/a11y: caption semántico (accessible name), th scope="col",
 *     contenedor de scroll horizontal focusable (role="region" + tabIndex=0),
 *     min-width para que el overflow scrollee en vez de romper el layout.
 *  3. Read-only: cero botones (invariante approval-gates: /production sin acciones).
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';

import {
  PaperCandidatesPanel,
  judgeElapsedDays,
  type PaperCandidatesLedger,
} from '@/components/gm/views/PaperCandidatesPanel';

/** Subconjunto fiel de public/data/production/paper/candidates_ledger_2026.json. */
const LEDGER: PaperCandidatesLedger = {
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
      ret_2026_ytd_pct: 3.36,
      n_trades: 11,
      note_n: 'N<20 => solo conteo y PnL',
      judge_window: null,
    },
    smart_simple_v12: {
      ret_2026_ytd_pct: 3.12,
      n_trades: 11,
      note_n: 'N<20 => solo conteo y PnL',
      judge_window: {
        starts_after: '2026-07-21',
        n_trades: 0,
        pnl_pct_compound: 0.0,
        note: 'N<20 => solo conteo y PnL',
      },
    },
  },
};

// Reloj del cliente congelado: 2026-07-27 10:00 COT.
const NOW = new Date('2026-07-27T10:00:00-05:00');

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: false });
  vi.setSystemTime(NOW);
});

afterEach(() => {
  vi.useRealTimers();
});

describe('judgeElapsedDays (helper puro — fechas del bundle vs reloj del cliente)', () => {
  it('cuenta los días transcurridos desde starts_after', () => {
    expect(judgeElapsedDays('2026-07-21', NOW)).toBe(6);
  });

  it('devuelve negativo si el juez aún no arranca (faltan días)', () => {
    expect(judgeElapsedDays('2026-08-01', NOW)).toBe(-5);
  });

  it('parsea la fecha ancla aunque traiga sufijo textual', () => {
    // anchor del ledger real: "2026-01-01 (directiva operador 2026-07-22)"
    expect(judgeElapsedDays('2026-01-01 (directiva operador 2026-07-22)', NOW)).toBe(207);
  });

  it('es null-safe (nunca NaN en la UI)', () => {
    expect(judgeElapsedDays(null, NOW)).toBeNull();
    expect(judgeElapsedDays('sin-fecha', NOW)).toBeNull();
  });
});

describe('PaperCandidatesPanel (BL-05)', () => {
  it('muestra los días al juez por candidata, derivados del bundle', () => {
    render(<PaperCandidatesPanel ledger={LEDGER} />);

    // Columna presente
    expect(screen.getByRole('columnheader', { name: /días al juez/i })).toBeInTheDocument();
    // v12: juez sellado desde 2026-07-21 → día 6 del forward post-freeze
    expect(screen.getByText('6 d')).toBeInTheDocument();
    // v11: forward real anclado 2026-01-01 → 207 días corriendo
    expect(screen.getByText('207 d')).toBeInTheDocument();
  });

  it('tabla accesible: caption con nombre, headers semánticos con scope="col"', () => {
    render(<PaperCandidatesPanel ledger={LEDGER} />);

    const table = screen.getByRole('table', { name: /candidatas/i });
    expect(table).toBeInTheDocument();

    const headers = screen.getAllByRole('columnheader');
    expect(headers.length).toBeGreaterThanOrEqual(7);
    for (const th of headers) {
      expect(th).toHaveAttribute('scope', 'col');
    }
  });

  it('móvil: el scroll horizontal vive en una región focusable, sin romper layout', () => {
    render(<PaperCandidatesPanel ledger={LEDGER} />);

    // axe: scrollable-region-focusable — la región de scroll es alcanzable por teclado
    const region = screen.getByRole('region', { name: /candidatas/i });
    expect(region).toHaveAttribute('tabindex', '0');
    expect(region.className).toContain('overflow-x-auto');
    // la tabla fija un ancho mínimo → en viewport angosto scrollea dentro del
    // contenedor en vez de aplastar columnas / desbordar la página
    const table = screen.getByRole('table', { name: /candidatas/i });
    expect(table.className).toMatch(/min-w-/);
  });

  it('constitución §6: con N<20 muestra la nota del ledger (solo conteo y PnL)', () => {
    render(<PaperCandidatesPanel ledger={LEDGER} />);
    expect(screen.getAllByText(/N<20 => solo conteo y PnL/).length).toBeGreaterThan(0);
  });

  it('read-only: cero botones/acciones (invariante /production)', () => {
    render(<PaperCandidatesPanel ledger={LEDGER} />);
    expect(screen.queryAllByRole('button')).toHaveLength(0);
  });
});

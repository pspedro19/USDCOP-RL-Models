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
    // Candidata recién sellada: TODAS sus métricas son null todavía. Es la fila que
    // ejercita el render de "celda sin dato" (símbolo '—' + equivalente textual).
    smart_simple_v14: {
      ret_2026_ytd_pct: null,
      n_trades: null,
      judge_window: {
        starts_after: '2026-07-24',
        n_trades: null,
        pnl_pct_compound: null,
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

  it('CXD-022: la celda de nombre de cada fila es <th scope="row"> (row header)', () => {
    render(<PaperCandidatesPanel ledger={LEDGER} />);

    // Un lector de pantalla debe poder anclar cada dato a SU candidata: eso exige
    // rowheader real, no un td con texto en negrita.
    const rowHeaders = screen.getAllByRole('rowheader');
    expect(rowHeaders).toHaveLength(Object.keys(LEDGER.strategies).length);
    for (const th of rowHeaders) {
      expect(th).toHaveAttribute('scope', 'row');
    }
    // El id de la estrategia vive DENTRO del rowheader (accessible name de la fila).
    expect(rowHeaders[0]).toHaveTextContent('smart_simple_v11');
  });

  it('CXD-022: tipografía de tabla relativa (rem/clamp), jamás px fijos', () => {
    render(<PaperCandidatesPanel ledger={LEDGER} />);

    const table = screen.getByRole('table', { name: /candidatas/i });
    // px fijo ignora la preferencia de tamaño de fuente del usuario (WCAG 1.4.4).
    expect(table.className).not.toMatch(/text-\[\d+(\.\d+)?px\]/);
    expect(table.className).toMatch(/text-\[(clamp\(|[\d.]+rem)/);
  });

  it('CXD-022: "sin dato" tiene texto equivalente; el guión es decorativo (aria-hidden)', () => {
    const { container } = render(<PaperCandidatesPanel ledger={LEDGER} />);

    // v14 llega con 4 celdas vacías (ret, trades, trades juez, PnL juez). Sin texto
    // equivalente, un lector de pantalla anuncia "guión" o nada: el usuario no vidente
    // no distingue "vacío" de "fallo de carga".
    expect(screen.getAllByText('sin dato').length).toBeGreaterThanOrEqual(4);

    // Y el símbolo visible queda fuera del árbol de accesibilidad (no se duplica el anuncio).
    const dashes = Array.from(container.querySelectorAll('span')).filter(
      (el) => el.textContent === '—',
    );
    expect(dashes.length).toBeGreaterThan(0);
    for (const dash of dashes) {
      expect(dash).toHaveAttribute('aria-hidden', 'true');
    }
  });

  it('CXD-022: el estado producción/paper se anuncia con texto Sí/No, no solo por color', () => {
    render(<PaperCandidatesPanel ledger={LEDGER} />);

    // El tono del badge (accent vs neutral) es un canal cromático: sin equivalente
    // textual, "en producción" vs "paper" se pierde sin visión del color (WCAG 1.4.1).
    const prodRow = screen.getAllByRole('rowheader')[0];
    expect(prodRow).toHaveTextContent(/En producción: Sí/);
    expect(prodRow).toHaveTextContent(/Juez sellado: No/);

    const paperRow = screen.getAllByRole('rowheader')[1];
    expect(paperRow).toHaveTextContent(/En producción: No/);
    expect(paperRow).toHaveTextContent(/Juez sellado: Sí/);
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

  // OJO: este test es de ECO — comprueba que el string `note_n` del FIXTURE se
  // renderiza. Es necesario pero NO suficiente: la nota puede aparecer con un Sharpe
  // al lado y sigue verde. La prohibición real la impone el test siguiente.
  it('constitución §6: con N<20 muestra la nota del ledger (solo conteo y PnL)', () => {
    render(<PaperCandidatesPanel ledger={LEDGER} />);
    expect(screen.getAllByText(/N<20 => solo conteo y PnL/).length).toBeGreaterThan(0);
  });

  /**
   * PROHIBICIÓN (no eco) de quant-constitution §6 y strategy-contract §6:
   * «con N < 20 trades no se reportan Sharpe ni p-value — solo conteo y PnL».
   *
   * Hueco que cierra (verificación por mutación, 2026-07-28): se añadió una celda
   * literal "Sharpe 3.35 · p=0.006" a las filas con n_trades: 11 y la suite salió
   * 13 passed / 0 failed. El único test §6 que existía afirmaba que se ecoaba un
   * string del propio fixture, no que NO hubiera un ratio junto a él.
   *
   * ROJO con: añadir <td>Sharpe 3.35 · p=0.006</td> (o cualquier celda Calmar /
   * Sortino / "p<0.05") a las filas de PaperCandidatesPanel.tsx.
   */
  it('constitución §6: con N<20 la FILA no publica Sharpe/p-value/Calmar/Sortino', () => {
    const { container } = render(<PaperCandidatesPanel ledger={LEDGER} />);

    // Vocabulario prohibido con N insuficiente: ratios y significancia. `p\s*[=<]`
    // captura "p=0.006", "p < 0.05", "p =0.01" — cualquier forma de publicar el p-value.
    const FORBIDDEN = /sharpe|p\s*[=<]|p-value|calmar|sortino/i;

    const sids = Object.keys(LEDGER.strategies);
    const rows = Array.from(container.querySelectorAll('tbody tr'));
    expect(rows, 'una fila por candidata del ledger').toHaveLength(sids.length);

    rows.forEach((row, i) => {
      const sid = sids[i];
      const n = LEDGER.strategies[sid].n_trades;
      // Precondición del fixture: ninguna candidata tiene N suficiente (11 o sin dato).
      expect(n == null || n < 20, `el fixture de ${sid} ya no ejercita el caso N<20`).toBe(true);

      const text = row.textContent ?? '';
      const hit = text.match(FORBIDDEN)?.[0] ?? null;
      expect(
        hit,
        `la fila ${sid} (n_trades=${n ?? 'sin dato'}) publica "${hit}" con N<20 — `
        + 'quant-constitution §6 solo permite conteo y PnL',
      ).toBeNull();
    });
  });

  it('read-only: cero botones/acciones (invariante /production)', () => {
    render(<PaperCandidatesPanel ledger={LEDGER} />);
    expect(screen.queryAllByRole('button')).toHaveLength(0);
  });
});

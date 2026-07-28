/**
 * FAIL-ON-CONSOLE — un test que imprime `act(...)` o "width(0) and height(0)"
 * NO está verde: está pasando mientras el render que dice ejercitar corre fuera
 * del acto de React o se monta sin layout.
 *
 * Por qué existe (CODEX P1, calidad de test): la suite de caveat devolvía
 * `Vitest 0` con 41 verdes y a la vez escupía avisos de `act(...)` y gráficos de
 * tamaño 0 en cada corrida. El escenario BDD "consola limpia" seguía rojo aunque
 * el exit code dijera lo contrario, así que el exit code había dejado de ser
 * evidencia. Este gate convierte esa discrepancia en un fallo.
 *
 * Contrato:
 *  - Se instala por SUITE (no globalmente): cada suite declara qué ruido acepta,
 *    en vez de que un allowlist global amnistíe a todas.
 *  - El allowlist es EXPLÍCITO y se declara con motivo. Vacío = tolerancia cero.
 *  - Cubre `console.error` y `console.warn` (los dos canales que usan React y
 *    recharts para avisar de un test mal montado).
 *
 * Uso:
 *
 *     import { installConsoleGate } from '../../support/console-gate';
 *     installConsoleGate();                                  // tolerancia cero
 *     installConsoleGate({ allow: [/motivo declarado/] });    // excepción con motivo
 */
import { afterEach, beforeEach, expect, vi } from 'vitest';

export interface ConsoleGateOptions {
  /** Patrones tolerados. Cada entrada debe llevar su motivo en un comentario. */
  allow?: RegExp[];
  /** Canales vigilados (por defecto error + warn). */
  channels?: Array<'error' | 'warn'>;
}

function formatArgs(args: unknown[]): string {
  return args
    .map((a) => {
      if (a instanceof Error) return `${a.name}: ${a.message}`;
      if (typeof a === 'string') return a;
      try {
        return JSON.stringify(a);
      } catch {
        return String(a);
      }
    })
    .join(' ');
}

export function installConsoleGate(options: ConsoleGateOptions = {}): void {
  const allow = options.allow ?? [];
  const channels = options.channels ?? (['error', 'warn'] as const).slice();
  let captured: string[] = [];

  beforeEach(() => {
    captured = [];
    for (const channel of channels) {
      vi.spyOn(console, channel).mockImplementation((...args: unknown[]) => {
        const message = formatArgs(args);
        if (allow.some((re) => re.test(message))) return;
        captured.push(`[console.${channel}] ${message}`);
      });
    }
  });

  afterEach(() => {
    const found = captured;
    captured = [];
    // Los spies los restaura el `vi.clearAllMocks()` global + el aislamiento de
    // Vitest; lo que este gate garantiza es que nada quede sin cuenta.
    expect(
      found,
      'la consola no quedó limpia: un aviso de React/recharts significa que el ' +
        'render bajo prueba no es el que ve el usuario (act fuera de tiempo, ' +
        'gráfico sin layout). Arregla la causa o declara el patrón en el allowlist ' +
        'con su motivo.',
    ).toEqual([]);
  });
}

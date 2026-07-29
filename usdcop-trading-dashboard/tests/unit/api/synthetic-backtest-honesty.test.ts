/**
 * Honestidad del ORIGEN DE DATOS en las superficies de backtest / replay.
 * ========================================================================
 *
 * DEFECTO REPRODUCIDO (vivo en `main`)
 * ------------------------------------
 * Tres rutas fabrican una curva de equity con `lib/services/synthetic-backtest.service.ts`
 * cuando el backend de inferencia no responde, y la sirven con `success: true` y HTTP 200,
 * indistinguible de un backtest real:
 *
 *   1. `app/api/backtest/route.ts:27-48`
 *      `catch { }` VACÍO sobre el fetch al backend (también cae aquí si el backend
 *      responde !ok, porque el `return` sólo ocurre dentro de `if (response.ok)`).
 *      Devuelve 200 · `{ success: true, source: 'generated', trades, summary, ... }`.
 *
 *   2. `app/api/backtest/stream/route.ts:120-139` (GET) y `:203-222` (POST)
 *      Mismo `catch {}` vacío. Devuelve 200 `text/event-stream` con un evento
 *      `result` cuyo payload es `{ success: true, source: 'generated', trades, summary }`
 *      (`synthetic-backtest.service.ts:555-570`).
 *
 *   3. `app/api/replay/load-trades/route.ts:110-164` (`buildSyntheticFallback`)
 *      Se dispara en tres condiciones: backend `!response.ok` (:215-221), `AbortError`
 *      por timeout (:256-259) y `ECONNREFUSED` / `fetch failed` (:262-265).
 *      Devuelve 200 con `createApiResponse(data, 'fallback')`, y como
 *      `lib/types/api.ts:66` fija `success = data !== null && !errorMessage`, el cuerpo
 *      sale otra vez con **`success: true`**.
 *
 * POR QUÉ `source: 'generated'` NO ES UN MARCADOR
 * -----------------------------------------------
 * No es sólo que esté enterrado dentro del JSON: es que **colisiona con el contrato del
 * backend real**. En `load-trades/route.ts:55`, `InferenceServiceResponse.source` es
 * `'database' | 'generated' | 'error'`, donde `'generated'` significa *"el servicio de
 * inferencia EJECUTÓ el modelo en vez de leer caché"* — es decir, DATO REAL. El fallback
 * reutiliza exactamente ese valor para decir lo contrario. Un consumidor que discrimine
 * por `source` no puede distinguir "modelo real recién ejecutado" de "números inventados".
 * El único bit honesto que existe hoy, `metadata.isRealData: false` (:159), vive en
 * segundo nivel, sólo en una de las tres rutas, y es una negación en vez de una
 * declaración positiva. `grep -riE "DEMO|SYNTHETIC" components/ app/` sobre la UI no
 * devuelve ningún badge: en pantalla no queda nada.
 *
 * Y los números fabricados no son ruido inocuo: `synthetic-backtest.service.ts:53-74`
 * ("Investor Demo Mode") apunta a ~33% anual, ~60% win rate y Sharpe ~1.8-2.2. Es una
 * curva DISEÑADA para parecer buena, servida como si fuera evidencia.
 * Choca con `.claude/rules/quant-constitution.md` §7 (las decisiones se toman sobre los
 * números del bundle publicado) y §6 (desconfianza de la magia).
 *
 * QUÉ SE ESPERA DEL FIX (cualquiera de las dos formas pasa este test)
 * -------------------------------------------------------------------
 *   (A) FAIL-CLOSED — recomendado. La ruta responde **>= 500** con motivo y **NO emite
 *       ni un solo trade fabricado**. El frontend ya sabe renderizar error; una
 *       superficie de decisión debe quedarse vacía, no rellenarse sola.
 *
 *   (B) ETIQUETADO INEQUÍVOCO. Si se conserva el fallback, el cuerpo debe cumplir LAS DOS:
 *         - `success` !== `true`  (un cuerpo fabricado no declara éxito), y
 *         - marcador POSITIVO de PRIMER NIVEL, uno de:
 *              `synthetic: true` · `is_synthetic: true` · `data_origin: 'SYNTHETIC'`
 *           (o `dataOrigin`; también se acepta la cabecera `X-Data-Origin: SYNTHETIC`,
 *           imprescindible en SSE porque ahí no hay un cuerpo JSON único).
 *       `source: 'generated'` NO cuenta, por la colisión de contrato explicada arriba.
 *       Nota: (B) sólo tapa el agujero del BACKEND; sigue haciendo falta el badge en la
 *       UI para cerrar el defecto de cara al usuario. Este test cubre el backend.
 *
 * Este fichero NO toca código de producción: sólo ejerce los handlers reales con el
 * backend caído (`fetch` rechazando con `TypeError: fetch failed`, que es literalmente
 * lo que lanza undici ante un ECONNREFUSED).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { NextRequest, NextResponse } from 'next/server';

// `withAuth` real exige sesión NextAuth + rate limiter + Postgres. Se sustituye por un
// passthrough que REPLICA su contrato de errores (`api-auth.ts:224-236`: si el handler
// lanza, responde 500 con motivo) para que la variante (A) del fix implementada como
// `throw` se evalúe correctamente y no como un error del test.
vi.mock('@/lib/auth/api-auth', () => ({
  withAuth:
    (handler: (req: unknown, ctx: unknown) => Promise<Response>) =>
    async (request: unknown): Promise<Response> => {
      try {
        return await handler(request, {
          user: { id: 'test-admin', email: 'admin@test.local', username: 'admin', role: 'admin' },
        });
      } catch (error) {
        return NextResponse.json(
          {
            error: 'Internal server error',
            message: error instanceof Error ? error.message : 'Unknown error',
          },
          { status: 500 },
        );
      }
    },
}));

import { POST as backtestPOST } from '@/app/api/backtest/route';
import { GET as streamGET, POST as streamPOST } from '@/app/api/backtest/stream/route';
import { POST as loadTradesPOST } from '@/app/api/replay/load-trades/route';

// ── Fixture ─────────────────────────────────────────────────────────────────────
const START = '2025-01-01';
const END = '2025-02-01';
/** No está en FORECAST_STRATEGIES ⇒ la ruta intenta el backend, no un JSON pre-computado. */
const MODEL = 'ppo_v20';
/** 16x recorta el delay entre eventos del SSE (300ms/speed) sin cambiar el payload. */
const SPEED = 16;

type SseEvent = { type: string; data: Record<string, unknown> };

const originalFetch = globalThis.fetch;

beforeEach(() => {
  // Backend de inferencia CAÍDO: undici lanza exactamente este error ante ECONNREFUSED.
  globalThis.fetch = vi.fn().mockRejectedValue(new TypeError('fetch failed')) as unknown as typeof fetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

// ── Predicados de honestidad ────────────────────────────────────────────────────

/** Marcador POSITIVO, de primer nivel, que declara el dato como fabricado. */
function hasSyntheticMarker(body: unknown): boolean {
  if (!body || typeof body !== 'object') return false;
  const b = body as Record<string, unknown>;
  if (b.synthetic === true || b.is_synthetic === true || b.isSynthetic === true) return true;
  const origin = String(b.data_origin ?? b.dataOrigin ?? '').toUpperCase();
  return origin === 'SYNTHETIC' || origin === 'FABRICATED' || origin === 'SIMULATED';
}

function headerDeclaresSynthetic(res: Response): boolean {
  const raw = (res.headers.get('x-data-origin') ?? res.headers.get('x-synthetic') ?? '').toUpperCase();
  return raw === 'SYNTHETIC' || raw === 'FABRICATED' || raw === 'TRUE' || raw === '1';
}

/** Trades fabricados presentes en un cuerpo, mire donde mire (raíz o `data`). */
function tradesIn(body: unknown): unknown[] {
  if (!body || typeof body !== 'object') return [];
  const b = body as Record<string, unknown>;
  const direct = b.trades;
  if (Array.isArray(direct)) return direct;
  const nested = (b.data as Record<string, unknown> | undefined)?.trades;
  return Array.isArray(nested) ? nested : [];
}

async function readSse(res: Response): Promise<SseEvent[]> {
  const raw = await res.text();
  return raw
    .split('\n\n')
    .map((chunk) => chunk.trim())
    .filter((chunk) => chunk.startsWith('data: '))
    .map((chunk) => JSON.parse(chunk.slice('data: '.length)) as SseEvent);
}

// ── Aserción compartida ─────────────────────────────────────────────────────────

/**
 * Una respuesta construida con datos fabricados es aceptable SI Y SÓLO SI:
 *   (A) falla limpiamente (>=500) y no cuela ni un trade, o
 *   (B) no declara `success: true` Y lleva marcador inequívoco de primer nivel.
 */
function expectFailClosedOrLabelled(
  where: string,
  res: Response,
  body: unknown,
  markerCarriers: unknown[] = [body],
): void {
  const trades = tradesIn(body);

  if (res.status >= 500) {
    expect(
      trades.length,
      `${where}: respondió ${res.status} (fail-closed, correcto) pero AÚN ASÍ adjuntó ` +
        `${trades.length} trades fabricados en el cuerpo de error.`,
    ).toBe(0);
    return;
  }

  const success = (body as Record<string, unknown> | null)?.success;
  expect(
    success,
    `${where}: HTTP ${res.status} con ${trades.length} trades FABRICADOS por ` +
      `synthetic-backtest.service.ts y aun así declara success:true. Un cuerpo inventado ` +
      `no puede declarar éxito: o falla limpiamente (>=500) o dice que es sintético.`,
  ).not.toBe(true);

  const labelled = headerDeclaresSynthetic(res) || markerCarriers.some(hasSyntheticMarker);
  expect(
    labelled,
    `${where}: HTTP ${res.status} con ${trades.length} trades FABRICADOS y NINGÚN marcador ` +
      `de primer nivel (se esperaba synthetic:true / data_origin:'SYNTHETIC' o la cabecera ` +
      `X-Data-Origin). 'source:"generated"' NO sirve: en el contrato del backend real ` +
      `(load-trades/route.ts:55) ese mismo valor significa DATO REAL recién computado.`,
  ).toBe(true);
}

// ── Tests ───────────────────────────────────────────────────────────────────────

describe('Origen de datos: el backend caído no puede producir un backtest que parezca real', () => {
  it('POST /api/backtest — el fallback sintético no se declara success:true ni sin marcar', async () => {
    const req = new NextRequest('http://localhost:3001/api/backtest', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ start_date: START, end_date: END, model_id: MODEL }),
    });

    const res = await backtestPOST(req);
    const body = res.status >= 500 ? await res.json().catch(() => null) : await res.json();

    expectFailClosedOrLabelled('POST /api/backtest', res, body);
  });

  it('GET /api/backtest/stream — el evento SSE `result` no se declara success:true ni sin marcar', async () => {
    const url =
      `http://localhost:3001/api/backtest/stream?startDate=${START}&endDate=${END}` +
      `&modelId=${MODEL}&speed=${SPEED}`;
    const res = await streamGET(new NextRequest(url));

    if (res.status >= 500) {
      const text = await res.text();
      expect(
        text.includes('"trades"'),
        'GET /api/backtest/stream: falló limpiamente pero coló trades fabricados en el cuerpo.',
      ).toBe(false);
      return;
    }

    const events = await readSse(res);
    const result = events.find((e) => e.type === 'result');
    const fabricatedTrades = events.filter((e) => e.type === 'trade');

    if (!result) {
      expect(
        fabricatedTrades.length,
        'GET /api/backtest/stream: sin evento `result`, pero emitió ' +
          `${fabricatedTrades.length} eventos \`trade\` fabricados.`,
      ).toBe(0);
      return;
    }

    expectFailClosedOrLabelled(
      'GET /api/backtest/stream (evento `result`)',
      res,
      result.data,
      // El marcador puede viajar en cualquier evento del stream (p.ej. un `meta` inicial).
      events.map((e) => e.data),
    );
  });

  it('POST /api/backtest/stream — el evento SSE `result` no se declara success:true ni sin marcar', async () => {
    const req = new NextRequest('http://localhost:3001/api/backtest/stream', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        start_date: START,
        end_date: END,
        model_id: MODEL,
        replay_speed: SPEED,
        emit_bar_events: false,
      }),
    });

    const res = await streamPOST(req);

    if (res.status >= 500) {
      const text = await res.text();
      expect(
        text.includes('"trades"'),
        'POST /api/backtest/stream: falló limpiamente pero coló trades fabricados.',
      ).toBe(false);
      return;
    }

    const events = await readSse(res);
    const result = events.find((e) => e.type === 'result');
    const fabricatedTrades = events.filter((e) => e.type === 'trade');

    if (!result) {
      expect(
        fabricatedTrades.length,
        'POST /api/backtest/stream: sin evento `result`, pero emitió ' +
          `${fabricatedTrades.length} eventos \`trade\` fabricados.`,
      ).toBe(0);
      return;
    }

    expectFailClosedOrLabelled(
      'POST /api/backtest/stream (evento `result`)',
      res,
      result.data,
      events.map((e) => e.data),
    );
  });

  it('POST /api/replay/load-trades — `buildSyntheticFallback` no se declara success:true ni sin marcar', async () => {
    const req = new NextRequest('http://localhost:3001/api/replay/load-trades', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ startDate: START, endDate: END, modelId: MODEL }),
    });

    const res = await loadTradesPOST(req);
    const body = await res.json().catch(() => null);

    // `metadata.isRealData: false` existe, pero es de SEGUNDO nivel y es una negación:
    // no habilita ningún badge y no sobrevive a que un consumidor lea sólo `data`.
    // Se acepta igualmente como marcador si además `success` deja de ser `true`.
    expectFailClosedOrLabelled('POST /api/replay/load-trades', res, body, [
      body,
      (body as Record<string, unknown> | null)?.data,
    ]);
  });
});

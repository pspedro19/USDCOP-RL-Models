/**
 * Honestidad del ORIGEN DE DATOS en las superficies de backtest / replay.
 * ========================================================================
 *
 * EL DEFECTO QUE ESTE FICHERO CUSTODIA
 * ------------------------------------
 * Tres rutas sabían fabricar una curva de equity con
 * `lib/services/synthetic-backtest.service.ts` cuando el backend de inferencia no
 * respondía, y la servían con HTTP 200 + `success: true`, indistinguible de un backtest
 * real. Los números no son ruido inocuo: el generador ("Investor Demo Mode",
 * `synthetic-backtest.service.ts:53-74`) apunta a ~33% anual, ~60% win rate y Sharpe
 * ~1.8-2.2 — una curva DISEÑADA para parecer buena, servida como si fuera evidencia, en
 * la superficie donde se emite el Vote 2 (`.claude/rules/quant-constitution.md` §6/§7).
 *
 * EL CONTRATO QUE SE EXIGE AQUÍ (no hay alternativa "etiquetada")
 * ---------------------------------------------------------------
 * La versión anterior de este test aceptaba DOS formas de arreglo: (A) fail-closed, o
 * (B) conservar el fallback siempre que llevara marcador y no dijera `success: true`.
 * Ese predicado permisivo dejaba VIVO el fallback automático: bastaba devolver 200 con
 * `success:false` + `synthetic:true` para pasar en verde, que es exactamente el estado
 * que el fix prometió eliminar. **(B) queda derogada.** Para los CUATRO handlers, sin
 * `mode=demo`, se exige literalmente:
 *
 *   1. status **503** — ni 200, ni 500, ni 502.
 *   2. `error === 'inference_backend_unavailable'` (en SSE, en el frame).
 *   3. **CERO** `trade` / `result` / `summary` en la respuesta, comprobado recorriendo la
 *      ESTRUCTURA (`findFabricationStructures`), nunca con `text.includes('"trades"')`.
 *      Buscar subcadenas es el defecto que este repo persigue: un frame
 *      `type:"result"` con `summary:{sharpe:2.2}` no contiene la subcadena `"trades"` y
 *      colaba entero.
 *   4. En SSE: **exactamente un** frame, y de tipo `error`. Ni uno más, ni de otro tipo.
 *   5. Nada de fuga: el cuerpo no puede contener la URL interna del upstream ni su
 *      cuerpo de error; sólo `reason` (conjunto cerrado) + `correlation_id`.
 *
 * TRES DISPAROS, NO UNO
 * ---------------------
 * Cada handler se ejerce con las tres condiciones que el fix declara cubiertas:
 * `fetch` rechazando (ECONNREFUSED), upstream respondiendo `!ok`, y timeout real
 * (el `setTimeout(() => controller.abort(), …)` de la propia ruta, disparado con
 * timers falsos). 4 handlers × 3 disparos = 12 combinaciones.
 *
 * Y EL OPT-IN, POR EL LADO POSITIVO
 * ---------------------------------
 * Un fail-closed que además rompiera el generador no sería el fix pedido. Se comprueba
 * que `mode=demo` (en query Y en body) SIGUE siendo la puerta al generador, que produce
 * trades de verdad, y que declara su origen en raíz (`synthetic: true` /
 * `data_origin: 'SYNTHETIC'`) y en la cabecera `X-Data-Origin`. Y que es la ÚNICA
 * puerta: cualquier otro valor de `mode` cae al 503.
 *
 * Este fichero NO toca código de producción: sólo ejerce los handlers reales.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { NextRequest, NextResponse } from 'next/server';

// `withAuth` real exige sesión NextAuth + rate limiter + Postgres. Se sustituye por un
// passthrough que REPLICA su contrato de errores (`api-auth.ts:224-236`: si el handler
// lanza, responde 500 con motivo) para que un fix implementado como `throw` se evalúe
// como lo que es (un 500, que NO cumple el contrato de 503) y no como un error del test.
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
/** 16x recorta el delay entre eventos del SSE sin cambiar el payload. */
const SPEED = 16;

/**
 * Cuerpo que devuelve el upstream cuando responde `!ok`. Marcador improbable a propósito:
 * si aparece en la respuesta al cliente, es que la ruta está reenviando texto del upstream.
 */
const UPSTREAM_BODY_SENTINEL = 'UPSTREAM-INTERNAL-TRACEBACK-a7f31c';

/** Conjunto CERRADO de motivos que el cliente puede llegar a ver. */
const ALLOWED_REASONS = [
  'backend_unreachable',
  'backend_error_response',
  'backend_timeout',
  'backend_empty_response',
] as const;

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.useRealTimers();
  vi.restoreAllMocks();
});

// ── Lectura estructural (NUNCA por subcadena) ───────────────────────────────────

type SseFrame = {
  /** `type` del payload JSON, o el `event:` del protocolo, o `unparseable`. */
  type: string;
  body: Record<string, unknown> | null;
  raw: string;
};

/**
 * Parser SSE deliberadamente TOLERANTE: acepta `event:` además de `data:`, y no descarta
 * un frame por no ser JSON válido. Un parser estricto que ignorase lo que no entiende
 * sería un agujero — un frame malformado con trades seguiría contando como "no hay frames".
 */
function parseSse(raw: string): SseFrame[] {
  return raw
    .split(/\n\n+/)
    .map((block) => block.trim())
    .filter((block) => block.length > 0)
    // Los comentarios SSE (`: heartbeat`) no transportan datos: no son frames.
    .filter((block) => !block.split('\n').every((line) => line.startsWith(':')))
    .map((block) => {
      const lines = block.split('\n').map((l) => l.trim());
      const eventName = lines.find((l) => l.startsWith('event:'))?.slice('event:'.length).trim();
      const dataPayload = lines
        .filter((l) => l.startsWith('data:'))
        .map((l) => l.slice('data:'.length).trim())
        .join('\n');

      let body: Record<string, unknown> | null = null;
      try {
        const parsed = JSON.parse(dataPayload);
        body = parsed && typeof parsed === 'object' ? (parsed as Record<string, unknown>) : null;
      } catch {
        body = null;
      }

      const type = String(body?.type ?? eventName ?? 'unparseable');
      return { type, body, raw: block };
    });
}

/**
 * Claves cuya sola PRESENCIA (con contenido) significa que la respuesta transporta un
 * backtest. No se busca su texto: se recorre el objeto y se devuelve la RUTA de cada
 * aparición, para que el fallo diga exactamente dónde estaba.
 */
const FABRICATION_KEYS = new Set([
  'trade',
  'trades',
  'result',
  'results',
  'summary',
  'equity_curve',
  'equityCurve',
]);

function isMeaningful(value: unknown): boolean {
  if (value === null || value === undefined) return false;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === 'object') return Object.keys(value as object).length > 0;
  return Boolean(value);
}

function findFabricationStructures(value: unknown, path = '$'): string[] {
  if (Array.isArray(value)) {
    return value.flatMap((item, i) => findFabricationStructures(item, `${path}[${i}]`));
  }
  if (!value || typeof value !== 'object') return [];

  const hits: string[] = [];
  for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
    const childPath = `${path}.${key}`;
    if (FABRICATION_KEYS.has(key) && isMeaningful(child)) {
      hits.push(`${childPath} (${Array.isArray(child) ? `${child.length} elementos` : typeof child})`);
    }
    hits.push(...findFabricationStructures(child, childPath));
  }
  return hits;
}

/**
 * Objetos con FORMA de trade, mire como se llame la clave que los contiene. Cierra el
 * hueco obvio de `FABRICATION_KEYS`: renombrar `trades` a `rows` no deja de fabricar.
 */
function countTradeShapedObjects(value: unknown): number {
  if (Array.isArray(value)) return value.reduce<number>((n, v) => n + countTradeShapedObjects(v), 0);
  if (!value || typeof value !== 'object') return 0;

  const obj = value as Record<string, unknown>;
  const looksLikeTrade =
    ('entry_price' in obj && 'side' in obj) ||
    ('entry_time' in obj && 'exit_time' in obj) ||
    ('pnl' in obj && 'entry_price' in obj);

  return (
    (looksLikeTrade ? 1 : 0) +
    Object.values(obj).reduce<number>((n, v) => n + countTradeShapedObjects(v), 0)
  );
}

/** Fugas de infraestructura que jamás deben viajar al cliente. */
const LEAK_PATTERNS: Array<[RegExp, string]> = [
  [/https?:\/\//i, 'una URL absoluta'],
  [/localhost:\d{2,5}/i, 'un host:puerto interno'],
  [/\/v1\/backtest/i, 'la ruta interna del upstream'],
  [new RegExp(UPSTREAM_BODY_SENTINEL, 'i'), 'el CUERPO de error del upstream'],
];

// ── Aserción única y estricta ───────────────────────────────────────────────────

type HandlerKind = 'json' | 'sse';

/**
 * El contrato completo del fail-closed. Sin ramas: si un handler llega aquí, o cumple
 * las cinco condiciones o falla.
 */
async function expectFailClosed(
  where: string,
  res: Response,
  kind: HandlerKind,
  expectedReason: (typeof ALLOWED_REASONS)[number],
): Promise<void> {
  const rawBody = await res.text();

  // (1) 503, exactamente.
  expect(
    res.status,
    `${where}: el commit prometió 503 fail-closed y respondió ${res.status}.\n` +
      `Cuerpo recibido:\n${rawBody.slice(0, 800)}`,
  ).toBe(503);

  // (4) SSE: exactamente un frame, de tipo error.
  let carrier: Record<string, unknown> | null;
  if (kind === 'sse') {
    const frames = parseSse(rawBody);
    expect(
      frames.map((f) => f.type),
      `${where}: el contrato SSE del 503 es EXACTAMENTE UN frame de tipo "error". ` +
        `Se recibieron ${frames.length} frame(s) de tipo [${frames.map((f) => f.type).join(', ')}].\n` +
        `Cuerpo:\n${rawBody.slice(0, 800)}`,
    ).toEqual(['error']);
    expect(
      frames[0].body,
      `${where}: el único frame SSE no es JSON parseable — un consumidor no puede leer el motivo.\n` +
        `Frame crudo:\n${frames[0].raw.slice(0, 400)}`,
    ).not.toBeNull();
    carrier = frames[0].body;
  } else {
    try {
      carrier = JSON.parse(rawBody) as Record<string, unknown>;
    } catch {
      carrier = null;
    }
    expect(carrier, `${where}: el cuerpo del 503 no es JSON parseable:\n${rawBody.slice(0, 400)}`).not.toBeNull();
  }

  // (2) Código de error estable y machine-readable.
  expect(
    carrier?.error,
    `${where}: se esperaba error === 'inference_backend_unavailable' y llegó ` +
      `${JSON.stringify(carrier?.error)}. Es el código sobre el que discrimina el cliente.`,
  ).toBe('inference_backend_unavailable');

  // Un 503 jamás declara éxito.
  expect(
    carrier?.success,
    `${where}: un 503 fail-closed no puede declarar success:true.`,
  ).not.toBe(true);

  // (3) CERO estructura de backtest — comprobado sobre el objeto, no sobre el texto.
  const fabricated = findFabricationStructures(carrier);
  expect(
    fabricated,
    `${where}: la respuesta 503 ARRASTRA estructura de backtest en ${fabricated.join(' · ')}. ` +
      `Un fail-closed no adjunta trades, ni result, ni summary: la superficie se queda VACÍA.`,
  ).toEqual([]);

  const tradeShaped = countTradeShapedObjects(carrier);
  expect(
    tradeShaped,
    `${where}: la respuesta 503 contiene ${tradeShaped} objeto(s) con forma de trade ` +
      `(entry_price/side/pnl), aunque la clave que los envuelve no se llame "trades".`,
  ).toBe(0);

  // (5) Motivo saneado: conjunto cerrado + correlación, sin fugas de infraestructura.
  expect(
    ALLOWED_REASONS as readonly string[],
    `${where}: 'reason' debe pertenecer al conjunto cerrado ${ALLOWED_REASONS.join('|')} ` +
      `y llegó ${JSON.stringify(carrier?.reason)}.`,
  ).toContain(carrier?.reason);

  expect(
    carrier?.reason,
    `${where}: el disparo era "${expectedReason}" y la ruta lo clasificó como ` +
      `${JSON.stringify(carrier?.reason)} — el motivo que ve el operador no corresponde al fallo real.`,
  ).toBe(expectedReason);

  expect(
    String(carrier?.correlation_id ?? ''),
    `${where}: falta un correlation_id con el que un operador pueda encontrar la línea de log ` +
      `que sí tiene el detalle completo.`,
  ).toMatch(/^[0-9a-f]{8}$/);

  expect(
    carrier,
    `${where}: el cuerpo conserva 'detail'. El detalle (URL interna + cuerpo upstream) va al ` +
      `log del servidor; al cliente sólo reason + correlation_id.`,
  ).not.toHaveProperty('detail');

  for (const [pattern, label] of LEAK_PATTERNS) {
    expect(
      pattern.test(rawBody),
      `${where}: la respuesta al cliente filtra ${label} (patrón ${pattern}).\n` +
        `Cuerpo:\n${rawBody.slice(0, 800)}`,
    ).toBe(false);
  }
}

// ── Handlers bajo prueba ────────────────────────────────────────────────────────

type HandlerCase = {
  name: string;
  kind: HandlerKind;
  /** Timeout que la propia ruta programa con `setTimeout(() => controller.abort(), N)`. */
  timeoutMs: number;
  invoke: () => Promise<Response>;
};

const HANDLERS: HandlerCase[] = [
  {
    name: 'POST /api/backtest',
    kind: 'json',
    timeoutMs: 30_000,
    invoke: () =>
      backtestPOST(
        new NextRequest('http://localhost:3001/api/backtest', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ start_date: START, end_date: END, model_id: MODEL }),
        }),
      ),
  },
  {
    name: 'GET /api/backtest/stream',
    kind: 'sse',
    timeoutMs: 5_000,
    invoke: () =>
      streamGET(
        new NextRequest(
          `http://localhost:3001/api/backtest/stream?startDate=${START}&endDate=${END}` +
            `&modelId=${MODEL}&speed=${SPEED}`,
        ),
      ),
  },
  {
    name: 'POST /api/backtest/stream',
    kind: 'sse',
    timeoutMs: 5_000,
    invoke: () =>
      streamPOST(
        new NextRequest('http://localhost:3001/api/backtest/stream', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            start_date: START,
            end_date: END,
            model_id: MODEL,
            replay_speed: SPEED,
            emit_bar_events: false,
          }),
        }),
      ),
  },
  {
    name: 'POST /api/replay/load-trades',
    kind: 'json',
    timeoutMs: 300_000,
    invoke: () =>
      loadTradesPOST(
        new NextRequest('http://localhost:3001/api/replay/load-trades', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ startDate: START, endDate: END, modelId: MODEL }),
        }),
      ),
  },
];

// ── Las TRES condiciones de disparo ─────────────────────────────────────────────

type Trigger = {
  name: string;
  reason: (typeof ALLOWED_REASONS)[number];
  run: (handler: HandlerCase) => Promise<Response>;
};

const TRIGGERS: Trigger[] = [
  {
    // undici lanza literalmente este error ante un ECONNREFUSED.
    name: 'fetch rechaza (backend caído / ECONNREFUSED)',
    reason: 'backend_unreachable',
    run: (handler) => {
      globalThis.fetch = vi
        .fn()
        .mockRejectedValue(new TypeError('fetch failed')) as unknown as typeof fetch;
      return handler.invoke();
    },
  },
  {
    // El backend responde, pero mal: 502 con un traceback en el cuerpo.
    name: 'upstream responde !ok (502 con cuerpo de error)',
    reason: 'backend_error_response',
    run: (handler) => {
      globalThis.fetch = vi.fn().mockResolvedValue(
        new Response(`${UPSTREAM_BODY_SENTINEL}: model file missing at /srv/models/ppo_v20.zip`, {
          status: 502,
          statusText: 'Bad Gateway',
        }),
      ) as unknown as typeof fetch;
      return handler.invoke();
    },
  },
  {
    // Timeout REAL de la ruta: el fetch nunca resuelve, y es el `setTimeout(...abort())`
    // de la propia ruta el que dispara. Timers falsos para no esperar 30s / 5min.
    name: 'timeout (el AbortController de la ruta dispara)',
    reason: 'backend_timeout',
    run: async (handler) => {
      globalThis.fetch = vi.fn((_url: unknown, init?: { signal?: AbortSignal }) => {
        return new Promise<Response>((_resolve, reject) => {
          const signal = init?.signal;
          if (!signal) return; // cuelga: si la ruta no pasa signal, el test expira y lo delata
          const onAbort = () => {
            const err = new Error('This operation was aborted');
            err.name = 'AbortError';
            reject(err);
          };
          if (signal.aborted) onAbort();
          else signal.addEventListener('abort', onAbort, { once: true });
        });
      }) as unknown as typeof fetch;

      vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
      const pending = handler.invoke();
      // Deja que el handler llegue a programar su setTimeout antes de avanzar el reloj.
      await vi.advanceTimersByTimeAsync(0);
      await vi.advanceTimersByTimeAsync(handler.timeoutMs + 1_000);
      const res = await pending;
      vi.useRealTimers();
      return res;
    },
  },
];

// ── Tests: fail-closed en 4 handlers × 3 disparos ───────────────────────────────

describe('Sin `mode=demo`, un backend caído produce 503 y NADA que parezca un backtest', () => {
  for (const handler of HANDLERS) {
    for (const trigger of TRIGGERS) {
      it(`${handler.name} — ${trigger.name}`, async () => {
        const res = await trigger.run(handler);
        await expectFailClosed(`${handler.name} [${trigger.name}]`, res, handler.kind, trigger.reason);
      });
    }
  }
});

// ── Tests: el opt-in explícito, por el lado POSITIVO ────────────────────────────

/** Marcador POSITIVO, de primer nivel, que declara el dato como fabricado. */
function declaresSynthetic(body: unknown): boolean {
  if (!body || typeof body !== 'object') return false;
  const b = body as Record<string, unknown>;
  if (b.synthetic === true || b.is_synthetic === true || b.isSynthetic === true) return true;
  const origin = String(b.data_origin ?? b.dataOrigin ?? '').toUpperCase();
  return origin === 'SYNTHETIC';
}

describe('`mode=demo` es la ÚNICA puerta al generador, y se declara como fabricado', () => {
  beforeEach(() => {
    // El backend sigue caído: si el demo funciona, es porque NO depende de él.
    globalThis.fetch = vi
      .fn()
      .mockRejectedValue(new TypeError('fetch failed')) as unknown as typeof fetch;
  });

  it('POST /api/backtest?mode=demo (query) — sirve el generador, marcado', async () => {
    const res = await backtestPOST(
      new NextRequest('http://localhost:3001/api/backtest?mode=demo', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ start_date: START, end_date: END, model_id: MODEL }),
      }),
    );
    expect(res.status, 'el opt-in explícito debe servir 200, no el 503 del fail-closed').toBe(200);
    const body = await res.json();

    expect(Array.isArray(body.trades) && body.trades.length > 0, 'el opt-in no produjo trades: el generador quedó inalcanzable').toBe(true);
    expect(declaresSynthetic(body), 'el cuerpo del demo no declara `synthetic:true` / `data_origin:"SYNTHETIC"` en raíz').toBe(true);
    expect(res.headers.get('x-data-origin'), 'falta la cabecera X-Data-Origin: SYNTHETIC').toBe('SYNTHETIC');
  });

  it('POST /api/backtest con {mode:"demo"} (body) — misma puerta', async () => {
    const res = await backtestPOST(
      new NextRequest('http://localhost:3001/api/backtest', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ start_date: START, end_date: END, model_id: MODEL, mode: 'demo' }),
      }),
    );
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(Array.isArray(body.trades) && body.trades.length > 0).toBe(true);
    expect(declaresSynthetic(body)).toBe(true);
    expect(res.headers.get('x-data-origin')).toBe('SYNTHETIC');
  });

  it('GET /api/backtest/stream?mode=demo — el frame `result` se declara fabricado', async () => {
    const res = await streamGET(
      new NextRequest(
        `http://localhost:3001/api/backtest/stream?startDate=${START}&endDate=${END}` +
          `&modelId=${MODEL}&speed=${SPEED}&mode=demo`,
      ),
    );
    expect(res.status).toBe(200);
    expect(res.headers.get('x-data-origin'), 'el stream demo debe anunciarse en la cabecera').toBe('SYNTHETIC');

    const frames = parseSse(await res.text());
    const result = frames.find((f) => f.type === 'result');
    expect(result, 'el stream demo no emitió `result`: el generador quedó inalcanzable').toBeDefined();

    // El protocolo del stream es `{ type, data }`: el primer nivel del payload es `data`.
    const payload = result?.body?.data as Record<string, unknown> | undefined;
    expect(declaresSynthetic(payload), 'el `result` del demo no se declara sintético en primer nivel').toBe(true);
    expect(
      Array.isArray(payload?.trades) && (payload!.trades as unknown[]).length > 0,
      'el `result` del demo no trae trades',
    ).toBe(true);
  });

  it('POST /api/backtest/stream con {mode:"demo"} — misma puerta', async () => {
    const res = await streamPOST(
      new NextRequest('http://localhost:3001/api/backtest/stream', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          start_date: START,
          end_date: END,
          model_id: MODEL,
          replay_speed: SPEED,
          emit_bar_events: false,
          mode: 'demo',
        }),
      }),
    );
    expect(res.status).toBe(200);
    expect(res.headers.get('x-data-origin')).toBe('SYNTHETIC');

    const frames = parseSse(await res.text());
    const result = frames.find((f) => f.type === 'result');
    expect(result).toBeDefined();
    expect(declaresSynthetic(result?.body?.data)).toBe(true);
  });

  it('POST /api/replay/load-trades con {mode:"demo"} — sirve el generador, marcado', async () => {
    const res = await loadTradesPOST(
      new NextRequest('http://localhost:3001/api/replay/load-trades', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ startDate: START, endDate: END, modelId: MODEL, mode: 'demo' }),
      }),
    );
    expect(res.status).toBe(200);
    const body = await res.json();

    expect(Array.isArray(body?.data?.trades) && body.data.trades.length > 0).toBe(true);
    expect(declaresSynthetic(body), 'el cuerpo del demo no declara su origen en primer nivel').toBe(true);
    expect(res.headers.get('x-data-origin')).toBe('SYNTHETIC');
  });

  it('un `mode` que NO es exactamente "demo" no abre la puerta — cae al 503', async () => {
    // 'DEMO' en mayúsculas, el clásico casi-acierto. Si esto sirviera trades, el opt-in
    // sería una coladera y volveríamos a fabricar sin que nadie lo pidiera.
    const res = await backtestPOST(
      new NextRequest('http://localhost:3001/api/backtest?mode=DEMO', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ start_date: START, end_date: END, model_id: MODEL, mode: 'fallback' }),
      }),
    );
    await expectFailClosed('POST /api/backtest [mode=DEMO/fallback]', res, 'json', 'backend_unreachable');
  });
});

// ── Test: un fallo NO clasificado tampoco puede fabricar ────────────────────────

describe('Un error de fetch no clasificado tampoco fabrica un backtest', () => {
  it('POST /api/replay/load-trades — error inesperado ⇒ error, nunca trades', async () => {
    // `load-trades` sólo reconoce AbortError y ECONNREFUSED/"fetch failed". Cualquier
    // otro error se re-lanza al catch externo. Ese camino debe seguir siendo estéril:
    // puede no ser 503, pero no puede colar ni un trade.
    globalThis.fetch = vi
      .fn()
      .mockRejectedValue(new Error('ENOTFOUND usdcop-backtest-api')) as unknown as typeof fetch;

    const res = await loadTradesPOST(
      new NextRequest('http://localhost:3001/api/replay/load-trades', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ startDate: START, endDate: END, modelId: MODEL }),
      }),
    );

    expect(res.status, 'un error desconocido debe fallar, no responder 200').toBeGreaterThanOrEqual(500);
    const body = await res.json().catch(() => null);
    expect(body?.success, 'un fallo no puede declarar éxito').not.toBe(true);
    expect(
      findFabricationStructures(body),
      'un fallo no clasificado coló estructura de backtest en la respuesta',
    ).toEqual([]);
    expect(countTradeShapedObjects(body)).toBe(0);
  });
});

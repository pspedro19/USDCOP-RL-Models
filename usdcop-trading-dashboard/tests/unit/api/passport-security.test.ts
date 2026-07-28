/**
 * CODEX C-00x — seguridad de la superficie Passport / Control Tower.
 * ===================================================================
 *
 * Cubre 1:1 los dos hallazgos de CODEX sobre código de CLAUDE:
 *
 *  H1 (P0, fuga RBAC) — `public/data/control-tower/governance.json` publicaba
 *      trials, DSR y gates INTERNOS bajo `public/`. `middleware.ts` solo exige
 *      SESIÓN para `/data/**` (GATED_STATIC_PREFIXES), mientras
 *      `rbac.contract.ts` reserva el Passport a `research:read`: un `free` o
 *      `subscriber` pedía el fichero directo y se saltaba `/api/passport`
 *      entero. Viola `rbac.md` §"nada monetizado anónimo" y §8 "subscribers ven
 *      OUTPUTS, no INTERNALS". Es el MISMO defecto que CODEX ya rechazó en C-006
 *      con los artefactos SHAP.
 *
 *      Remedio (precedente C-006/CXD-040, `data/interpretability/`): el artefacto
 *      sale de `public/` a `<repo>/data/control-tower/` y la ÚNICA vía es la API
 *      con permiso. Por eso aquí se prueban DOS cosas y no una:
 *        (a) AUSENCIA FÍSICA bajo `public/` — la defensa estructural. Mientras el
 *            fichero exista ahí, `/data/**` lo sirve a cualquier sesión y ningún
 *            handler puede impedirlo.
 *        (b) La vía canónica `/api/passport/**` niega a free/subscriber (403) y
 *            deja pasar a admin/developer (200) — ejecutando el middleware REAL.
 *
 *  H2 (P1, fuga de información interna) — los handlers devolvían
 *      `(e as Error).message` al cliente: un error de ruta/parseo filtraba rutas
 *      del filesystem al navegador. Mismo patrón en `/api/cart/checkout`
 *      (`{detail: String(e)}`).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { NextRequest } from 'next/server';
import fs from 'node:fs';
import path from 'node:path';

import { ROLE_PERMISSIONS } from '@/lib/contracts/rbac.contract';

// ── middleware real con sesión simulada (mismo patrón que middleware-replay-authz)
const getTokenMock = vi.fn();
vi.mock('next-auth/jwt', () => ({
  getToken: (...args: unknown[]) => getTokenMock(...args),
}));

// ── composer y billing mockeados para forzar el throw con una ruta de filesystem
const composeMock = vi.hoisted(() => ({
  tower: vi.fn(),
  strategy: vi.fn(),
  list: vi.fn(),
}));
vi.mock('@/lib/passport/compose', () => ({
  composeControlTower: composeMock.tower,
  composeStrategyPassport: composeMock.strategy,
  listPassportStrategies: composeMock.list,
}));

const billingMock = vi.hoisted(() => ({ createCheckout: vi.fn() }));
vi.mock('@/lib/billing', () => ({
  getBillingProvider: () => ({ createCheckout: billingMock.createCheckout }),
}));
vi.mock('@/lib/db/postgres-client', () => ({
  query: vi.fn(async (sql: string) => {
    if (String(sql).includes('sb_users')) return { rows: [{ email: 'x@y.co' }] };
    return { rows: [] };
  }),
}));
vi.mock('@/lib/auth/entitlements', () => ({
  getEntitlements: vi.fn(async () => ({ plan: 'free', assets: [] })),
}));

import { middleware } from '../../../middleware';
import { GET as towerGET } from '@/app/api/passport/tower/route';
import { GET as passportGET } from '@/app/api/passport/[strategyId]/route';
import { POST as checkoutPOST } from '@/app/api/cart/checkout/route';

/** Ruta de filesystem que JAMÁS puede aparecer en un cuerpo servido al cliente. */
const SECRET_PATH = 'C:\\secret\\usdcop\\registries\\ledger.jsonl';

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..', '..');
const PUBLIC_DIR = path.join(REPO_ROOT, 'usdcop-trading-dashboard', 'public');

function req(p: string, init?: RequestInit): NextRequest {
  return new NextRequest(`http://localhost:3001${p}`, init as never);
}

function tokenFor(role: keyof typeof ROLE_PERMISSIONS) {
  return { id: `user-${role}`, role, permissions: [...ROLE_PERMISSIONS[role]] };
}

beforeEach(() => {
  getTokenMock.mockReset();
  composeMock.tower.mockReset();
  composeMock.strategy.mockReset();
  composeMock.list.mockReset();
  billingMock.createCheckout.mockReset();
  delete process.env.AUTH_BYPASS_ENABLED;
});

afterEach(() => vi.restoreAllMocks());

// ───────────────────────────────────────────────────────────────────── H1 (P0)

describe('H1 · el artefacto de gobernanza NO vive bajo public/ (ausencia física)', () => {
  it('no existe public/data/control-tower/', () => {
    expect(fs.existsSync(path.join(PUBLIC_DIR, 'data', 'control-tower'))).toBe(false);
  });

  it('ningún fichero bajo public/ es la proyección de gobernanza', () => {
    const offenders: string[] = [];
    const walk = (dir: string) => {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) { walk(full); continue; }
        if (!entry.name.endsWith('.json')) continue;
        if (entry.name === 'governance.json') { offenders.push(full); continue; }
        // Marcador del artefacto: `kind: governance_projection` (export_control_tower.py).
        if (fs.statSync(full).size > 2 * 1024 * 1024) continue; // cap: la proyección es pequeña
        try {
          const parsed = JSON.parse(fs.readFileSync(full, 'utf-8')) as { kind?: unknown };
          if (parsed?.kind === 'governance_projection') offenders.push(full);
        } catch { /* no-JSON o corrupto: no es el artefacto */ }
      }
    };
    walk(PUBLIC_DIR);
    expect(offenders).toEqual([]);
  });
});

describe('H1 · tras la mudanza, la proyección SE SIGUE LEYENDO (no es un null silencioso)', () => {
  it('el composer real la lee desde data/control-tower y lo declara en source.path', async () => {
    // Sin este test, "sacarlo de public/" pasaría igual dejando la torre vacía: el
    // composer devuelve null ante artefacto ausente (degradación por diseño).
    vi.doUnmock('@/lib/passport/compose');
    vi.resetModules();
    const real = await vi.importActual<typeof import('@/lib/passport/compose')>(
      '@/lib/passport/compose',
    );
    const tower = await real.composeControlTower();
    expect(tower.data.n_global.source.status).toBe('published');
    expect(tower.data.n_global.source.path).toBe('data/control-tower/governance.json');
    expect(tower.data.n_global.source.path).not.toContain('public');
    expect(tower.data.trials_by_family.value?.length ?? 0).toBeGreaterThan(0);
  });
});

describe('H1 · la vía canónica /api/passport exige research:read (middleware real)', () => {
  it.each(['free', 'subscriber'] as const)(
    'rol cliente %s → 403 en /api/passport/tower (nunca ve trials/DSR/gates)',
    async (role) => {
      getTokenMock.mockResolvedValue(tokenFor(role));
      const res = await middleware(req('/api/passport/tower'));
      expect(res.status).toBe(403);
      const body = await res.json();
      expect(body.required).toBe('research:read');
    },
  );

  it.each(['admin', 'developer'] as const)(
    'rol de research %s → pasa (200) en /api/passport/tower',
    async (role) => {
      getTokenMock.mockResolvedValue(tokenFor(role));
      const res = await middleware(req('/api/passport/tower'));
      expect(res.status).toBe(200);
    },
  );

  it('anónimo → 401 (nada monetizado anónimo)', async () => {
    getTokenMock.mockResolvedValue(null);
    const res = await middleware(req('/api/passport/tower'));
    expect(res.status).toBe(401);
  });

  it('/data/** solo exige SESIÓN: por eso la defensa es la ausencia física', async () => {
    // Este test documenta POR QUÉ (a) es la defensa y no un handler: el edge deja
    // pasar a un `free` sobre cualquier estático de /data/**. Si algún día esto
    // cambiara a 403, este test lo avisará y (a) dejaría de ser la única defensa.
    getTokenMock.mockResolvedValue(tokenFor('free'));
    const res = await middleware(req('/data/control-tower/governance.json'));
    expect(res.status).toBe(200);
  });
});

// ───────────────────────────────────────────────────────────────────── H2 (P1)

describe('H2 · los handlers no filtran detalles internos al cliente', () => {
  it('GET /api/passport/tower → 500 genérico, sin la ruta del filesystem', async () => {
    composeMock.tower.mockRejectedValue(new Error(SECRET_PATH));
    composeMock.list.mockResolvedValue([]);
    const res = await towerGET();
    expect(res.status).toBe(500);
    const text = await res.text();
    expect(text).not.toContain('C:\\secret');
    expect(text).not.toContain('ledger.jsonl');
    expect(JSON.parse(text).error.code).toBe('TOWER_COMPOSE_FAILED');
  });

  it('GET /api/passport/[strategyId] → 500 genérico, sin la ruta del filesystem', async () => {
    composeMock.strategy.mockRejectedValue(new Error(SECRET_PATH));
    const res = await passportGET(req('/api/passport/v11'), {
      params: Promise.resolve({ strategyId: 'v11' }),
    });
    expect(res.status).toBe(500);
    const text = await res.text();
    expect(text).not.toContain('C:\\secret');
    expect(text).not.toContain('ledger.jsonl');
    expect(JSON.parse(text).error.code).toBe('PASSPORT_COMPOSE_FAILED');
  });

  it('POST /api/cart/checkout → 503 sin `detail: String(e)`', async () => {
    billingMock.createCheckout.mockRejectedValue(new Error(SECRET_PATH));
    const res = await checkoutPOST(req('/api/cart/checkout', {
      method: 'POST',
      headers: { 'x-user-id': 'u1', 'x-user-role': 'subscriber', 'content-type': 'application/json' },
      body: JSON.stringify({ plan: 'signals' }),
    }));
    const text = await res.text();
    expect(text).not.toContain('C:\\secret');
    expect(text).not.toContain('ledger.jsonl');
    expect(JSON.parse(text).error.details).toBeUndefined();
  });
});

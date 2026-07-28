/**
 * CXD-057 (P0, fuga RBAC) — el estado de aprobación NO puede vivir bajo `public/`.
 * ================================================================================
 *
 * Hallazgo confirmado por CODEX: los cinco
 * `usdcop-trading-dashboard/public/data/production/approval_state*.json` publican
 * `gates`, el gate `deflated_sharpe` (DSR trial-aware) y `backtest_metrics`.
 * `middleware.ts` solo exige SESIÓN para `/data/**` (GATED_STATIC_PREFIXES) y
 * `/api/data/**` resuelve a `authenticated` — de modo que un `free` o `subscriber`
 * autenticado se saltaba entero el `research:read` que el SSOT reserva para
 * Backtest/Experimentos pidiendo el JSON directo.
 *
 * SSOT que lo decide (los tres, verificados literalmente):
 *   · `docs/rbac/VISUAL-SPEC-CHECKLIST.md` §B — "subscriber sin gates/votos/jerga L4/PENDING".
 *   · `.claude/specs/platform/frontend-backend-contract.md` §6 — Backtest/Experimentos
 *     ✖ para free y subscriber, ✔ admin/dev (`research:read`).
 *   · `.claude/specs/platform/ux-navigation.md` P3 — "cliente jamás ve jerga interna
 *     (L4/votos/gates)".
 *
 * Remedio (precedente C-006 `data/interpretability/` y H1 `data/control-tower/`):
 *   1. El artefacto ÍNTEGRO sale de `public/` a `<repo>/data/approvals/`.
 *   2. La única vía a la proyección íntegra es `/api/production/approval` con
 *      `research:read` (admin/developer).
 *   3. La superficie de cliente (`/api/production/status`, `signals:read`) sirve una
 *      proyección sanitizada por ALLOWLIST de campos — nunca por blacklist.
 *   4. El consumidor del Voto 2 (`/api/production/approve` + DAG H5-L4b) sigue leyendo
 *      el privado; eso es lo que no puede romperse.
 *
 * Fail-closed: si el artefacto no está, se responde 404 con motivo declarado — jamás
 * un `DEFAULT_STATE` fabricado que un humano pueda confundir con "sin gates".
 */
import { describe, it, expect, vi, beforeEach, afterEach, beforeAll, afterAll } from 'vitest';
import { NextRequest } from 'next/server';
import realFs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { ROLE_PERMISSIONS } from '@/lib/contracts/rbac.contract';

// ── middleware real con sesión simulada (mismo patrón que middleware-replay-authz)
const getTokenMock = vi.fn();
vi.mock('next-auth/jwt', () => ({
  getToken: (...args: unknown[]) => getTokenMock(...args),
}));

// ── el Voto 2 se autentica con protectApiRoute: se simula el principal aprobador.
const authMock = vi.hoisted(() => ({ protect: vi.fn() }));
vi.mock('@/lib/auth/api-auth', () => ({
  protectApiRoute: (...a: unknown[]) => authMock.protect(...a),
}));

import { middleware } from '../../../middleware';
import { GET as approvalGET } from '@/app/api/production/approval/route';
import { GET as statusGET } from '@/app/api/production/status/route';
import { POST as approvePOST } from '@/app/api/production/approve/route';
import {
  PUBLIC_APPROVAL_FIELDS,
  approvalsRoot,
  readApprovalState,
  toPublicApproval,
} from '@/lib/approvals/store';

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..', '..');
const PUBLIC_DIR = path.join(REPO_ROOT, 'usdcop-trading-dashboard', 'public');
const REAL_APPROVALS_DIR = path.join(REPO_ROOT, 'data', 'approvals');

/** Claves INTERNAS que ningún fichero servible desde `public/` puede llevar. */
const LEAK_KEYS = ['gates', 'deflated_sharpe', 'backtest_metrics'] as const;

function req(p: string, init?: { method?: string }): NextRequest {
  return new NextRequest(`http://localhost:3001${p}`, init as never);
}
function tokenFor(role: keyof typeof ROLE_PERMISSIONS) {
  return { id: `user-${role}`, role, permissions: [...ROLE_PERMISSIONS[role]] };
}
const hdr = (role: string) => ({ 'x-user-role': role, 'x-user-perms': [...ROLE_PERMISSIONS[role as 'admin']].join(',') });

beforeEach(() => {
  getTokenMock.mockReset();
  authMock.protect.mockReset();
  delete process.env.AUTH_BYPASS_ENABLED;
});
afterEach(() => vi.restoreAllMocks());

// ═══════════════════════════════════════════ BDD 1 · barrido físico de public/**

describe('BDD-1 · barrido físico: ninguna clave interna sobrevive bajo public/', () => {
  it('public/data/production/approval_state*.json ya NO existe (git mv a data/approvals/)', () => {
    const prod = path.join(PUBLIC_DIR, 'data', 'production');
    const left = realFs.existsSync(prod)
      ? realFs.readdirSync(prod).filter((f) => /^approval_state.*\.json$/.test(f))
      : [];
    expect(left).toEqual([]);
  });

  it('los cinco artefactos viven en <repo>/data/approvals/', () => {
    expect(realFs.existsSync(REAL_APPROVALS_DIR)).toBe(true);
    const files = realFs.readdirSync(REAL_APPROVALS_DIR).filter((f) => /^approval_state.*\.json$/.test(f));
    expect(files).toContain('approval_state.json');
    expect(files.length).toBeGreaterThanOrEqual(5);
  });

  it('ningún JSON bajo public/** es un documento de aprobación (marcador estructural)', () => {
    const offenders: string[] = [];
    const walk = (dir: string) => {
      for (const e of realFs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, e.name);
        if (e.isDirectory()) { walk(full); continue; }
        if (!e.name.endsWith('.json')) continue;
        if (/^approval_state.*\.json$/.test(e.name)) { offenders.push(full); continue; }
        if (realFs.statSync(full).size > 4 * 1024 * 1024) continue;
        try {
          const p = JSON.parse(realFs.readFileSync(full, 'utf-8')) as Record<string, unknown>;
          // Marcador del artefacto: estado de aprobación + métricas del backtest juntos.
          if (p && typeof p === 'object' && 'status' in p && 'backtest_metrics' in p) offenders.push(full);
        } catch { /* no-JSON/corrupto: no es el artefacto */ }
      }
    };
    walk(PUBLIC_DIR);
    expect(offenders).toEqual([]);
  });

  it('ningún JSON de public/data/production/** lleva gates / deflated_sharpe / backtest_metrics', () => {
    const offenders: string[] = [];
    const walk = (dir: string) => {
      if (!realFs.existsSync(dir)) return;
      for (const e of realFs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, e.name);
        if (e.isDirectory()) { walk(full); continue; }
        if (!e.name.endsWith('.json')) continue;
        const raw = realFs.readFileSync(full, 'utf-8');
        if (LEAK_KEYS.some((k) => raw.includes(`"${k}"`))) offenders.push(path.relative(PUBLIC_DIR, full));
      }
    };
    walk(path.join(PUBLIC_DIR, 'data', 'production'));
    expect(offenders).toEqual([]);
  });

  /**
   * RESIDUO DECLARADO (no es CXD-057): el árbol inmutable de bundles
   * `public/data/strategies/**` publica las MISMAS claves (gates/dsr) y sigue
   * físicamente bajo `public/`. No se mueve aquí (lo escribe BundlePublisher y lo
   * leen replay/registry/passport); se cierra el bypass por ACCESO: `research:read`
   * tanto en el estático `/data/strategies/**` como en `/api/data/strategies/**`.
   * Este test fija el perímetro: si aparece un fichero con esas claves FUERA de ese
   * árbol, falla.
   */
  it('los ficheros con claves internas restantes están TODOS en el árbol de bundles gateado', () => {
    const outside: string[] = [];
    const walk = (dir: string) => {
      for (const e of realFs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, e.name);
        if (e.isDirectory()) { walk(full); continue; }
        if (!e.name.endsWith('.json')) continue;
        if (realFs.statSync(full).size > 4 * 1024 * 1024) continue;
        const rel = path.relative(PUBLIC_DIR, full).replace(/\\/g, '/');
        if (rel.startsWith('data/strategies/')) continue; // árbol gateado a research:read
        const raw = realFs.readFileSync(full, 'utf-8');
        if (LEAK_KEYS.some((k) => raw.includes(`"${k}"`))) outside.push(rel);
      }
    };
    walk(PUBLIC_DIR);
    expect(outside).toEqual([]);
  });
});

// ═════════════════════════════════ BDD 2 · el edge niega el documento privado

describe('BDD-2/3/4 · /api/production/approval — 401 anónimo, 403 cliente, 200 research', () => {
  it('anónimo ⇒ 401 (middleware real)', async () => {
    getTokenMock.mockResolvedValue(null);
    const res = await middleware(req('/api/production/approval'));
    expect(res.status).toBe(401);
  });

  it.each(['free', 'subscriber'] as const)('%s ⇒ 403 con required=research:read', async (role) => {
    getTokenMock.mockResolvedValue(tokenFor(role));
    const res = await middleware(req('/api/production/approval'));
    expect(res.status).toBe(403);
    const body = (await res.json()) as { required?: string };
    expect(body.required).toBe('research:read');
  });

  it.each(['admin', 'developer'] as const)('%s ⇒ el edge deja pasar', async (role) => {
    getTokenMock.mockResolvedValue(tokenFor(role));
    const res = await middleware(req('/api/production/approval'));
    expect(res.status).toBe(200);
  });

  it('el estático /data/production/approval_state.json ya no puede servir nada (fichero ausente)', () => {
    expect(realFs.existsSync(path.join(PUBLIC_DIR, 'data', 'production', 'approval_state.json'))).toBe(false);
  });

  it('/api/data/strategies/** exige research:read (residuo de bundles cerrado por acceso)', async () => {
    getTokenMock.mockResolvedValue(tokenFor('subscriber'));
    const res = await middleware(req('/api/data/strategies/btc_hodl_b1/manifest.json'));
    expect(res.status).toBe(403);
  });

  it('el estático /data/strategies/** exige research:read', async () => {
    getTokenMock.mockResolvedValue(tokenFor('subscriber'));
    const res = await middleware(req('/data/strategies/btc_hodl_b1/manifest.json'));
    expect(res.status).toBe(403);
  });
});

// ═════════════════════════════ handler: re-check en el handler, no solo el edge

describe('BDD-3 · el handler re-valida (defensa en profundidad)', () => {
  it('sin cabecera de identidad ⇒ 401', async () => {
    const res = await approvalGET(new Request('http://t/api/production/approval'));
    expect(res.status).toBe(401);
  });

  it.each(['free', 'subscriber'] as const)('%s ⇒ 403', async (role) => {
    const res = await approvalGET(new Request('http://t/api/production/approval', { headers: hdr(role) }));
    expect(res.status).toBe(403);
  });

  it.each(['admin', 'developer'] as const)('%s ⇒ 200 con gates + DSR + backtest_metrics íntegros', async (role) => {
    const res = await approvalGET(new Request('http://t/api/production/approval', { headers: hdr(role) }));
    expect(res.status).toBe(200);
    const body = (await res.json()) as { ok: boolean; data: Record<string, unknown> };
    expect(body.ok).toBe(true);
    expect(Array.isArray(body.data.gates)).toBe(true);
    expect(JSON.stringify(body.data)).toContain('deflated_sharpe');
    expect(body.data.backtest_metrics).toBeTruthy();
  });

  it('strategy_id inválido ⇒ 400 genérico (sin eco de rutas)', async () => {
    const res = await approvalGET(
      new Request('http://t/api/production/approval?strategy_id=..%2F..%2Fetc', { headers: hdr('admin') }),
    );
    expect(res.status).toBe(400);
    expect(JSON.stringify(await res.json())).not.toMatch(/[A-Za-z]:\\|\/etc\/|approvals/i);
  });

  it('estrategia inexistente ⇒ 404 con motivo DECLARADO (fail-closed, no default fabricado)', async () => {
    const res = await approvalGET(
      new Request('http://t/api/production/approval?strategy_id=no_such_strategy', { headers: hdr('admin') }),
    );
    expect(res.status).toBe(404);
    const body = (await res.json()) as { error?: { code?: string } };
    expect(body.error?.code).toBe('APPROVAL_ARTIFACT_MISSING');
  });
});

// ═════════════════ BDD-4 · proyección pública = ALLOWLIST (nunca blacklist)

describe('BDD-4 · /api/production/status sirve SOLO la allowlist', () => {
  it('la allowlist es explícita y no contiene ninguna clave interna', () => {
    expect([...PUBLIC_APPROVAL_FIELDS]).toEqual([
      'status', 'strategy', 'strategy_name', 'approved_at', 'created_at', 'last_updated',
    ]);
    for (const k of LEAK_KEYS) expect(PUBLIC_APPROVAL_FIELDS as readonly string[]).not.toContain(k);
  });

  it('toPublicApproval descarta TODO lo no enumerado (campo nuevo desconocido incluido)', () => {
    const pub = toPublicApproval({
      status: 'PENDING_APPROVAL', strategy: 's', strategy_name: 'S',
      created_at: 'c', last_updated: 'u',
      gates: [{ gate: 'deflated_sharpe', label: 'x', passed: false, value: 0.05, threshold: 0.95 }],
      backtest_metrics: { sharpe: 1 }, backtest_recommendation: 'REVIEW', backtest_confidence: 0.67,
      deploy_manifest: { script: 'x' }, un_campo_futuro_interno: 'secreto',
    } as never);
    expect(Object.keys(pub).sort()).toEqual([...PUBLIC_APPROVAL_FIELDS].sort());
    const raw = JSON.stringify(pub);
    for (const k of [...LEAK_KEYS, 'deploy_manifest', 'backtest_recommendation',
      'backtest_confidence', 'un_campo_futuro_interno', 'secreto']) {
      expect(raw).not.toContain(k);
    }
  });

  it.each(['subscriber', 'developer', 'admin'] as const)(
    'rol %s (signals:read) ⇒ 200 y el cuerpo no lleva gates/DSR/metrics',
    async (role) => {
      const res = await statusGET(new Request('http://t/api/production/status', { headers: hdr(role) }));
      expect(res.status).toBe(200);
      const raw = JSON.stringify(await res.json());
      for (const k of LEAK_KEYS) expect(raw).not.toContain(k);
      expect(raw).not.toContain('deploy_manifest');
    },
  );

  it('free (sin signals:read) ⇒ 403 incluso en la proyección sanitizada', async () => {
    const res = await statusGET(new Request('http://t/api/production/status', { headers: hdr('free') }));
    expect(res.status).toBe(403);
  });

  it('estado ausente ⇒ 404 declarado, jamás un DEFAULT_STATE fabricado', async () => {
    const res = await statusGET(
      new Request('http://t/api/production/status?strategy_id=no_such_strategy', { headers: hdr('admin') }),
    );
    expect(res.status).toBe(404);
  });
});

// ═════════════ BDD-5 · el consumidor del Voto 2 sigue leyendo el privado

describe('BDD-5 · Vote 2 (/api/production/approve) opera sobre el artefacto PRIVADO', () => {
  let tmp: string;

  beforeAll(() => {
    tmp = realFs.mkdtempSync(path.join(os.tmpdir(), 'approvals-'));
    realFs.writeFileSync(
      path.join(tmp, 'approval_state.json'),
      JSON.stringify({
        status: 'PENDING_APPROVAL', strategy: 'smart_simple_v11', strategy_name: 'SS',
        backtest_recommendation: 'REVIEW', backtest_confidence: 0.67,
        gates: [{ gate: 'deflated_sharpe', label: 'DSR', passed: false, value: 0.05, threshold: 0.95 }],
        backtest_metrics: { sharpe: 0.94 },
        created_at: 'c', last_updated: 'u',
      }, null, 2),
    );
    process.env.APPROVALS_DATA_DIR = tmp;
  });
  afterAll(() => {
    delete process.env.APPROVALS_DATA_DIR;
    realFs.rmSync(tmp, { recursive: true, force: true });
  });

  it('approvalsRoot() apunta al directorio privado (override por env para contenedor/test)', () => {
    expect(path.resolve(approvalsRoot())).toBe(path.resolve(tmp));
  });

  it('readApprovalState lee el privado íntegro (gates incluidos)', async () => {
    const s = await readApprovalState('smart_simple_v11');
    expect(s?.state.status).toBe('PENDING_APPROVAL');
    expect(s?.state.gates?.[0]?.gate).toBe('deflated_sharpe');
  });

  it('APPROVE escribe en el privado y /api/production/status refleja el nuevo estado', async () => {
    // El principal DEBE llevar rol: `approval:vote` se exige ahora DENTRO del handler
    // (approval-gates §4/§5). Sin rol, el Voto 2 es 403 — fail-closed. Este mock sin
    // rol era justamente el hueco de cobertura que dejó pasar el P0 de autorización;
    // el caso `subscriber` directo vive en `approval-vote2-authz-cas.test.ts`.
    authMock.protect.mockResolvedValue({
      authenticated: true,
      user: { id: 'u-admin', email: 'op@x.co', username: 'op', role: 'admin' },
    });
    const res = await approvePOST(
      new NextRequest('http://localhost:3001/api/production/approve', {
        method: 'POST',
        headers: { 'content-type': 'application/json', ...hdr('admin') },
        body: JSON.stringify({ action: 'APPROVE', strategy_id: 'smart_simple_v11', notes: 'ok' }),
      }) as never,
    );
    expect(res.status).toBe(200);
    expect((await res.json()).status).toBe('APPROVED');

    // El fichero PRIVADO es el que cambió — y nada se escribió bajo public/.
    const onDisk = JSON.parse(realFs.readFileSync(path.join(tmp, 'approval_state.json'), 'utf-8'));
    expect(onDisk.status).toBe('APPROVED');
    expect(onDisk.approved_by).toBe('op@x.co');
    expect(onDisk.gates?.[0]?.gate).toBe('deflated_sharpe'); // el privado conserva TODO

    const st = await statusGET(new Request('http://t/api/production/status', { headers: hdr('subscriber') }));
    const body = (await st.json()) as { ok: boolean; data: Record<string, unknown> };
    expect(body.data.status).toBe('APPROVED');
    expect(JSON.stringify(body)).not.toContain('gates');
  });
});

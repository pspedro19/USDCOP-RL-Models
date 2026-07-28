/**
 * Voto 2/2 — autorización DENTRO del handler + transición compare-and-set.
 * =========================================================================
 *
 * Tres agujeros en la ruta más sensible del sistema (el voto humano que promueve a
 * producción), encontrados por CODEX y reproducidos aquí ANTES de arreglarlos:
 *
 *  P0-1 · `app/api/production/approve/route.ts` llamaba a `protectApiRoute(request)`
 *         SIN `approval:vote`: solo comprobaba que hubiera SESIÓN. El middleware sí
 *         exige el permiso, pero `approval-gates.md` §4 dice que el deploy re-valida
 *         server-side y que **la UI no es la autoridad**; §5 dice que **solo `admin`**
 *         emite el Voto 2. Un `subscriber` autenticado que invocara el handler directo
 *         (sin pasar por el edge) EJECUTABA el voto. La cobertura previa probaba el
 *         middleware y un handler mockeado COMO ADMIN — nunca un `subscriber` directo.
 *         Idéntico en `deploy/route.ts`.
 *
 *  P0-2 · La ruta hacía read(PENDING) → mutación en memoria → `fs.writeFile`, sin CAS,
 *         sin lock y sin publicación atómica. Dos peticiones simultáneas devolvían
 *         AMBAS 200; la última ganaba el JSON, pero el APPROVE perdedor **ya había
 *         disparado el deploy fire-and-forget**. Además, un crash a mitad de
 *         `writeFile` truncaba el SSOT.
 *
 *  P0-3 · (tercer agujero, hallado durante el fix) el auto-deploy construía su URL con
 *         `request.nextUrl.origin` — es decir, con la cabecera `Host` que controla el
 *         CLIENTE — y le reenviaba la **cookie de sesión del aprobador**. `Host:
 *         evil.tld` ⇒ exfiltración de la sesión de un admin. El destino interno tiene
 *         que venir de configuración del servidor, jamás de la petición.
 *
 * Invariante transversal: **fail-closed**. Ante ambigüedad (rol desconocido, lock no
 * adquirido, precondición no verificable) NO se aprueba.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { NextRequest } from 'next/server';
import realFs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { ROLE_PERMISSIONS } from '@/lib/contracts/rbac.contract';

const authMock = vi.hoisted(() => ({ protect: vi.fn() }));
vi.mock('@/lib/auth/api-auth', () => ({
  protectApiRoute: (...a: unknown[]) => authMock.protect(...a),
}));

// El audit_log del dashboard vive en Postgres; en unit no hay DB. Se simula para poder
// AFIRMAR que la fila se intenta (regla 5 de approval-gates.md) sin levantar servicios.
const dbMock = vi.hoisted(() => ({ query: vi.fn() }));
vi.mock('@/lib/db/postgres-client', () => ({
  query: (...a: unknown[]) => dbMock.query(...a),
  default: { query: (...a: unknown[]) => dbMock.query(...a) },
}));

import { POST as approvePOST } from '@/app/api/production/approve/route';
import { POST as deployPOST } from '@/app/api/production/deploy/route';

const PENDING = (strategy = 'smart_simple_v11') => ({
  status: 'PENDING_APPROVAL',
  strategy,
  strategy_name: 'Smart Simple v1.1',
  backtest_recommendation: 'REVIEW',
  backtest_confidence: 0.67,
  gates: [{ gate: 'deflated_sharpe', label: 'DSR', passed: false, value: 0.05, threshold: 0.95 }],
  backtest_metrics: { sharpe: 0.94 },
  created_at: '2026-07-01T00:00:00Z',
  last_updated: '2026-07-01T00:00:00Z',
});

let tmp: string;
const statePath = () => path.join(tmp, 'approval_state.json');
const onDisk = () => JSON.parse(realFs.readFileSync(statePath(), 'utf-8'));

function seed(state: Record<string, unknown> = PENDING()) {
  realFs.writeFileSync(statePath(), JSON.stringify(state, null, 2), 'utf-8');
}

/** Principal AUTENTICADO con el rol dado (lo que devuelve `protectApiRoute`). */
function principal(role: string | undefined) {
  return {
    authenticated: true,
    user: { id: `user-${role}`, email: `${role}@x.co`, username: String(role), role },
  };
}

function approveReq(
  body: Record<string, unknown>,
  opts: { headers?: Record<string, string>; origin?: string } = {},
): NextRequest {
  return new NextRequest(`${opts.origin ?? 'http://localhost:3001'}/api/production/approve`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', cookie: 'next-auth.session-token=SECRET', ...(opts.headers ?? {}) },
    body: JSON.stringify(body),
  } as never);
}

function deployReq(body: Record<string, unknown> = {}): NextRequest {
  return new NextRequest('http://localhost:3001/api/production/deploy', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  } as never);
}

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  tmp = realFs.mkdtempSync(path.join(os.tmpdir(), 'vote2-'));
  process.env.APPROVALS_DATA_DIR = tmp;
  delete process.env.AUTH_BYPASS_ENABLED;
  // El auto-deploy y el trigger de Airflow salen por `fetch`: se observa, no se ejecuta.
  fetchMock = vi.fn(async () => new Response(JSON.stringify({ ok: true }), { status: 200 }));
  vi.stubGlobal('fetch', fetchMock);
  authMock.protect.mockReset();
  dbMock.query.mockReset();
  dbMock.query.mockResolvedValue({ rows: [] });
  seed();
});

afterEach(() => {
  delete process.env.APPROVALS_DATA_DIR;
  realFs.rmSync(tmp, { recursive: true, force: true });
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// ═══════════════════════ P0-1 · el permiso se exige DENTRO del handler ═══════════════

describe('P0-1 · approval:vote se exige en el handler, no solo en el edge', () => {
  /**
   * El rojo original: `subscriber` autenticado, handler llamado DIRECTO (sin
   * middleware) ⇒ 200 + fichero mutado. Es exactamente el caso que la cobertura
   * previa no tenía: probaba el middleware, o el handler mockeado como admin.
   */
  it.each(['subscriber', 'free', 'developer'] as const)(
    '%s autenticado llamando al handler DIRECTO ⇒ 403 y el SSOT NO cambia',
    async (role) => {
      expect(ROLE_PERMISSIONS[role]).not.toContain('approval:vote'); // premisa del contrato
      authMock.protect.mockResolvedValue(principal(role));

      const res = await approvePOST(approveReq({ action: 'APPROVE', strategy_id: 'smart_simple_v11' }));

      expect(res.status).toBe(403);
      expect(onDisk().status).toBe('PENDING_APPROVAL');
      expect(onDisk().approved_by).toBeUndefined();
      expect(fetchMock).not.toHaveBeenCalled(); // ni un deploy disparado
    },
  );

  it('REJECT de un subscriber tampoco pasa (el rechazo también es una decisión de gobierno)', async () => {
    authMock.protect.mockResolvedValue(principal('subscriber'));
    const res = await approvePOST(approveReq({ action: 'REJECT', strategy_id: 'smart_simple_v11' }));
    expect(res.status).toBe(403);
    expect(onDisk().status).toBe('PENDING_APPROVAL');
  });

  it('sesión SIN rol reconocible ⇒ 403 (fail-closed, no "por defecto admin")', async () => {
    authMock.protect.mockResolvedValue(principal(undefined));
    const res = await approvePOST(approveReq({ action: 'APPROVE', strategy_id: 'smart_simple_v11' }));
    expect(res.status).toBe(403);
    expect(onDisk().status).toBe('PENDING_APPROVAL');
  });

  it('admin PREVISUALIZANDO como subscriber (x-user-perms rebajado) ⇒ 403', async () => {
    // Migración 056: el middleware sella el set EFECTIVO; el "Ver como" solo rebaja.
    // Si el set efectivo no trae approval:vote, el voto no se emite aunque el rol sea admin.
    authMock.protect.mockResolvedValue(principal('admin'));
    const res = await approvePOST(
      approveReq({ action: 'APPROVE', strategy_id: 'smart_simple_v11' }, {
        headers: { 'x-user-role': 'subscriber', 'x-user-perms': [...ROLE_PERMISSIONS.subscriber].join(',') },
      }),
    );
    expect(res.status).toBe(403);
    expect(onDisk().status).toBe('PENDING_APPROVAL');
  });

  it('admin ⇒ 200, muta el SSOT y deja fila en audit_log', async () => {
    authMock.protect.mockResolvedValue(principal('admin'));
    const res = await approvePOST(approveReq({ action: 'APPROVE', strategy_id: 'smart_simple_v11', notes: 'ok' }));
    expect(res.status).toBe(200);
    expect((await res.json()).status).toBe('APPROVED');

    const d = onDisk();
    expect(d.status).toBe('APPROVED');
    expect(d.approved_by).toBe('admin@x.co');
    expect(d.gates?.[0]?.gate).toBe('deflated_sharpe'); // el privado conserva TODO

    // audit_log append-only (rule 5): la fila se intenta con la acción y el objeto.
    const sql = dbMock.query.mock.calls.map((c) => String(c[0])).join('\n');
    expect(sql).toMatch(/INSERT INTO audit_log/i);
    expect(JSON.stringify(dbMock.query.mock.calls)).toContain('approval_vote2_approve');
  });

  it('/api/production/deploy también exige approval:vote en el handler', async () => {
    seed({ ...PENDING(), status: 'APPROVED' });
    authMock.protect.mockResolvedValue(principal('subscriber'));
    const res = await deployPOST(deployReq({ strategy_id: 'smart_simple_v11' }));
    expect(res.status).toBe(403);
    expect(fetchMock).not.toHaveBeenCalled(); // no se disparó ningún DAG de deploy
  });
});

// ═════════════════════════ P0-2 · transición compare-and-set ════════════════════════

describe('P0-2 · APPROVE ↔ REJECT concurrentes: exactamente un ganador', () => {
  /**
   * Barrera: se retrasa TODA escritura de fichero para forzar el entrelazado que en
   * producción ocurre por latencia real. Sin CAS, ambos leen PENDING y ambos escriben.
   */
  function slowWrites(ms = 60) {
    const real = realFs.promises.writeFile.bind(realFs.promises);
    vi.spyOn(realFs.promises, 'writeFile').mockImplementation((async (...args: unknown[]) => {
      await new Promise((r) => setTimeout(r, ms));
      return (real as (...a: unknown[]) => Promise<void>)(...args);
    }) as never);
  }

  it('APPROVE y REJECT simultáneos ⇒ un 200 y un 409 (hoy: dos 200 contradictorios)', async () => {
    authMock.protect.mockResolvedValue(principal('admin'));
    slowWrites();

    const [a, b] = await Promise.all([
      approvePOST(approveReq({ action: 'APPROVE', strategy_id: 'smart_simple_v11' })),
      approvePOST(approveReq({ action: 'REJECT', strategy_id: 'smart_simple_v11', notes: 'no' })),
    ]);

    expect([a.status, b.status].sort()).toEqual([200, 409]);

    // El SSOT queda en UN estado coherente, sin mezcla de campos de ambas decisiones.
    const d = onDisk();
    expect(['APPROVED', 'REJECTED']).toContain(d.status);
    if (d.status === 'APPROVED') {
      expect(d.approved_by).toBeTruthy();
      expect(d.rejected_by).toBeUndefined();
    } else {
      expect(d.rejected_by).toBeTruthy();
      expect(d.approved_by).toBeUndefined();
    }
  });

  it('el deploy se dispara SOLO tras el commit ganador, y solo si el ganador fue APPROVE', async () => {
    authMock.protect.mockResolvedValue(principal('admin'));
    slowWrites();

    const [a, b] = await Promise.all([
      approvePOST(approveReq({ action: 'APPROVE', strategy_id: 'smart_simple_v11' })),
      approvePOST(approveReq({ action: 'REJECT', strategy_id: 'smart_simple_v11' })),
    ]);
    await new Promise((r) => setTimeout(r, 50)); // deja correr el fire-and-forget

    const deployCalls = fetchMock.mock.calls.filter((c) => String(c[0]).includes('/api/production/deploy'));
    const approveWon = onDisk().status === 'APPROVED';
    expect(deployCalls.length).toBe(approveWon ? 1 : 0);
    expect([a.status, b.status].sort()).toEqual([200, 409]);
  });

  it('dos APPROVE simultáneos ⇒ un 200 y un 409, y UN solo deploy', async () => {
    authMock.protect.mockResolvedValue(principal('admin'));
    slowWrites();

    const res = await Promise.all([
      approvePOST(approveReq({ action: 'APPROVE', strategy_id: 'smart_simple_v11' })),
      approvePOST(approveReq({ action: 'APPROVE', strategy_id: 'smart_simple_v11' })),
    ]);
    await new Promise((r) => setTimeout(r, 50));

    expect(res.map((r) => r.status).sort()).toEqual([200, 409]);
    expect(fetchMock.mock.calls.filter((c) => String(c[0]).includes('/api/production/deploy')).length).toBe(1);
  });

  it('publicación ATÓMICA: el destino se publica por rename, nunca por writeFile directo', async () => {
    authMock.protect.mockResolvedValue(principal('admin'));
    const renameSpy = vi.spyOn(realFs.promises, 'rename');
    const writeSpy = vi.spyOn(realFs.promises, 'writeFile');

    const res = await approvePOST(approveReq({ action: 'APPROVE', strategy_id: 'smart_simple_v11' }));
    expect(res.status).toBe(200);

    const target = realFs.realpathSync(statePath());
    // Un crash a mitad de escritura no puede truncar el SSOT: se escribe a temporal…
    const wroteTargetDirectly = writeSpy.mock.calls.some(
      (c) => typeof c[0] === 'string' && path.resolve(c[0] as string) === target,
    );
    expect(wroteTargetDirectly).toBe(false);
    // …y se publica con un rename atómico sobre el destino.
    expect(renameSpy.mock.calls.some((c) => path.resolve(String(c[1])) === target)).toBe(true);
  });

  it('un estado ya resuelto (APPROVED) rechaza una segunda transición con 409', async () => {
    seed({ ...PENDING(), status: 'APPROVED' });
    authMock.protect.mockResolvedValue(principal('admin'));
    const res = await approvePOST(approveReq({ action: 'REJECT', strategy_id: 'smart_simple_v11' }));
    expect(res.status).toBe(409);
    expect(onDisk().status).toBe('APPROVED');
  });
});

// ═══════════ P0-3 · el destino interno del auto-deploy no lo elige el cliente ════════

describe('P0-3 · el auto-deploy no reenvía la cookie del aprobador a un host del cliente', () => {
  it('Host controlado por el cliente ⇒ el fetch interno NO va a ese origen', async () => {
    authMock.protect.mockResolvedValue(principal('admin'));
    // `NEXTAUTH_URL` es la URL de cara al navegador (puerto del HOST) y tampoco puede
    // ser el destino interno: en compose es :5000 y el contenedor escucha en :3000.
    process.env.NEXTAUTH_URL = 'http://public.example.com:5000';

    const res = await approvePOST(
      approveReq({ action: 'APPROVE', strategy_id: 'smart_simple_v11' }, { origin: 'http://evil.example.com' }),
    );
    expect(res.status).toBe(200);
    await new Promise((r) => setTimeout(r, 50));

    const targets = fetchMock.mock.calls.map((c) => String(c[0]));
    expect(targets.some((t) => t.includes('evil.example.com'))).toBe(false);
    expect(targets.some((t) => t.includes('public.example.com'))).toBe(false);
    // La cookie de sesión solo puede viajar al origen INTERNO configurado.
    for (const [url, init] of fetchMock.mock.calls) {
      const headers = (init as { headers?: Record<string, string> } | undefined)?.headers ?? {};
      if (JSON.stringify(headers).includes('SECRET')) {
        expect(String(url)).toMatch(/^http:\/\/(localhost|127\.0\.0\.1)/);
      }
    }
    delete process.env.NEXTAUTH_URL;
  });
});

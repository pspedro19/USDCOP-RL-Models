/**
 * BL-34 — autorización SERVER-SIDE de /replay y sus APIs (edge middleware).
 * ==========================================================================
 * El ACK de Codex a C-002 fue explícito: la entrada RBAC `/replay → research:read`
 * "no aprueba por si solo BL-34 ni sustituye pruebas de autorizacion server-side".
 * Este test ejecuta el middleware REAL (middleware.ts) con sesiones simuladas
 * (getToken mockeado) y verifica el comportamiento observable del edge:
 *
 *  - anónimo:    /replay → redirect /login (con callbackUrl), /api/registry → 401.
 *  - sin permiso (subscriber/free, sin research:read): /replay → bounce a /hub,
 *                /api/registry → 403 con `required: research:read`.
 *  - con permiso (developer/admin): /replay pasa (sin redirect).
 *  - Vote-2 sigue admin-only server-side: POST /api/production/approve → 403 para
 *    developer aunque /replay le renderice (approval-gates.md invariante 4/5).
 *
 * La UI nunca es la autoridad — este archivo prueba la defensa real.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { NextRequest } from 'next/server';

import { ROLE_PERMISSIONS } from '@/lib/contracts/rbac.contract';

const getTokenMock = vi.fn();
vi.mock('next-auth/jwt', () => ({
  getToken: (...args: unknown[]) => getTokenMock(...args),
}));

// Import AFTER the mock so middleware.ts binds to the mocked getToken.
import { middleware } from '../../middleware';

function req(path: string, init?: { method?: string }): NextRequest {
  return new NextRequest(`http://localhost:3001${path}`, init);
}

function tokenFor(role: keyof typeof ROLE_PERMISSIONS, id = `user-${role}`) {
  return { id, role, permissions: [...ROLE_PERMISSIONS[role]] };
}

beforeEach(() => {
  getTokenMock.mockReset();
  // El bypass de auth jamás debe estar activo en estas pruebas.
  delete process.env.AUTH_BYPASS_ENABLED;
});

describe('BL-34 — /replay page (edge RBAC, deny-by-default)', () => {
  it('anónimo → redirect a /login con callbackUrl=/replay', async () => {
    getTokenMock.mockResolvedValue(null);
    const res = await middleware(req('/replay'));
    expect([302, 307]).toContain(res.status);
    const loc = res.headers.get('location') ?? '';
    expect(loc).toContain('/login');
    expect(loc).toContain('callbackUrl=%2Freplay');
  });

  it.each(['subscriber', 'free'] as const)(
    'rol sin research:read (%s) → rebotado a /hub (nunca se sirve la página)',
    async (role) => {
      getTokenMock.mockResolvedValue(tokenFor(role));
      const res = await middleware(req('/replay'));
      expect([302, 307]).toContain(res.status);
      expect(res.headers.get('location') ?? '').toContain('/hub');
    },
  );

  it.each(['developer', 'admin'] as const)(
    'rol con research:read (%s) → la página pasa sin redirect',
    async (role) => {
      getTokenMock.mockResolvedValue(tokenFor(role));
      const res = await middleware(req('/replay'));
      expect(res.status).toBe(200);
      expect(res.headers.get('location')).toBeNull();
    },
  );
});

describe('BL-34 — APIs del replay (rechazo server-side real, no solo contrato)', () => {
  it.each(['/api/registry', '/api/backtest', '/api/replay', '/api/strategies'])(
    'anónimo en %s → 401 JSON',
    async (path) => {
      getTokenMock.mockResolvedValue(null);
      const res = await middleware(req(path));
      expect(res.status).toBe(401);
      const body = await res.json();
      expect(body.error).toBe('Authentication required');
    },
  );

  it.each(['/api/registry', '/api/backtest', '/api/replay', '/api/strategies'])(
    'subscriber (sin research:read) en %s → 403 con required=research:read',
    async (path) => {
      getTokenMock.mockResolvedValue(tokenFor('subscriber'));
      const res = await middleware(req(path));
      expect(res.status).toBe(403);
      const body = await res.json();
      expect(body.required).toBe('research:read');
    },
  );

  it('developer en /api/registry → pasa con identidad + perms asertadas por el edge', async () => {
    getTokenMock.mockResolvedValue(tokenFor('developer'));
    const res = await middleware(req('/api/registry'));
    expect(res.status).toBe(200);
    expect(res.headers.get('location')).toBeNull();
  });
});

describe('BL-34 — Vote-2 permanece admin-only server-side (aunque /replay renderice)', () => {
  it('developer POST /api/production/approve → 403 (approval:vote es admin-only)', async () => {
    getTokenMock.mockResolvedValue(tokenFor('developer'));
    const res = await middleware(req('/api/production/approve', { method: 'POST' }));
    expect(res.status).toBe(403);
    const body = await res.json();
    expect(body.required).toBe('approval:vote');
  });

  it('subscriber POST /api/production/deploy → 403', async () => {
    getTokenMock.mockResolvedValue(tokenFor('subscriber'));
    const res = await middleware(req('/api/production/deploy', { method: 'POST' }));
    expect(res.status).toBe(403);
  });

  it('admin POST /api/production/approve → el edge lo deja pasar (el handler re-valida)', async () => {
    getTokenMock.mockResolvedValue(tokenFor('admin'));
    const res = await middleware(req('/api/production/approve', { method: 'POST' }));
    expect(res.status).toBe(200);
    expect(res.headers.get('location')).toBeNull();
  });
});

/**
 * /api/admin/impersonate — "Ver como" admin-only, READ-ONLY (admin:all).
 *
 * POST {user_id | role, motivo} → valida admin:all, resuelve el rol simulado, deja
 * fila 'view_as_start' en audit_log y setea DOS cookies:
 *   - `gm-view-as` httpOnly FIRMADA (HMAC NEXTAUTH_SECRET): `role|exp|sig` — la fuente
 *     de verdad; el navegador no puede forjar el rol.
 *   - `gm-view-as-role` legible (no sensible): espejo del rol para banner/hooks cliente.
 * DELETE limpia ambas y audita 'view_as_end'.
 *
 * NADA de esto concede permisos: la simulación es solo de NAVEGACIÓN/lectura; toda
 * mutación sigue exigiendo el rol REAL del admin (las cookies no tocan x-user-role).
 */
import crypto from 'crypto';

import { ok, fail } from '@/lib/api/envelope';
import { requirePermission } from '@/lib/api/relay';
import {
  VIEW_AS_COOKIE, VIEW_AS_ROLE_COOKIE, VIEW_AS_TTL_SECONDS,
  type ImpersonateRequest, type ImpersonateResponse,
} from '@/lib/contracts/admin-console.contract';
import { ROLES, type Role } from '@/lib/contracts/rbac.contract';
import { query } from '@/lib/db/postgres-client';

function secret(): string {
  return process.env.NEXTAUTH_SECRET || process.env.AUTH_SECRET || 'dev-insecure-secret';
}

function sign(role: Role, expEpoch: number): string {
  const payload = `${role}|${expEpoch}`;
  const sig = crypto.createHmac('sha256', secret()).update(payload).digest('hex');
  return `${payload}|${sig}`;
}

/** `role|exp|sig` → role si la firma es válida y no venció, si no null. */
export function verifyViewAs(token: string | null | undefined): Role | null {
  if (!token) return null;
  const parts = token.split('|');
  if (parts.length !== 3) return null;
  const [role, expStr, sig] = parts;
  const expected = crypto.createHmac('sha256', secret()).update(`${role}|${expStr}`).digest('hex');
  const a = Buffer.from(sig);
  const b = Buffer.from(expected);
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) return null;
  if (Number(expStr) * 1000 < Date.now()) return null;
  return (ROLES as readonly string[]).includes(role) ? (role as Role) : null;
}

function cookieHeader(name: string, value: string, maxAge: number, httpOnly: boolean): string {
  const attrs = [
    `${name}=${value}`, 'Path=/', `Max-Age=${maxAge}`, 'SameSite=Lax',
    httpOnly ? 'HttpOnly' : '', process.env.NODE_ENV === 'production' ? 'Secure' : '',
  ].filter(Boolean);
  return attrs.join('; ');
}

export async function POST(req: Request) {
  const gate = requirePermission(req, 'admin:all');
  if (gate instanceof Response) return gate;

  let payload: ImpersonateRequest;
  try { payload = await req.json(); } catch { return fail('BAD_REQUEST', 'Body JSON inválido.', 400); }

  const motivo = (payload.motivo ?? '').trim();
  if (motivo.length < 3) return fail('MOTIVO_REQUIRED', 'El motivo es obligatorio (queda en auditoría).', 400);

  // Rol simulado: explícito, o resuelto desde el usuario objetivo.
  let role: Role | null = null;
  let targetUserId: string | null = payload.user_id ?? null;
  if (payload.role && (ROLES as readonly string[]).includes(payload.role)) {
    role = payload.role;
  } else if (payload.user_id) {
    try {
      const res = await query('SELECT role FROM sb_users WHERE id = $1', [payload.user_id]);
      const raw = res.rows[0]?.role as string | undefined;
      role = (ROLES as readonly string[]).includes(raw ?? '') ? (raw as Role) : 'free';
    } catch (e) {
      return fail('DB_UNAVAILABLE', `No se pudo resolver el usuario: ${String((e as Error)?.message ?? e)}`, 503);
    }
  }
  if (!role) return fail('BAD_REQUEST', 'Indica user_id o role.', 400);

  const expEpoch = Math.floor(Date.now() / 1000) + VIEW_AS_TTL_SECONDS;
  const signed = sign(role, expEpoch);

  try {
    await query(
      `INSERT INTO audit_log (user_id, action, object_type, object_id, detail, ip)
       VALUES ($1, 'view_as_start', 'user', $2, $3::jsonb, $4)`,
      [gate.userId, targetUserId, JSON.stringify({ role, motivo }),
        req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() ?? null],
    );
  } catch { /* best-effort: no bloquear el "ver como" si el ledger está caído */ }

  const body: ImpersonateResponse = { role, expires_at: new Date(expEpoch * 1000).toISOString() };
  const res = ok(body);
  res.headers.append('set-cookie', cookieHeader(VIEW_AS_COOKIE, signed, VIEW_AS_TTL_SECONDS, true));
  res.headers.append('set-cookie', cookieHeader(VIEW_AS_ROLE_COOKIE, role, VIEW_AS_TTL_SECONDS, false));
  return res;
}

export async function DELETE(req: Request) {
  const gate = requirePermission(req, 'admin:all');
  if (gate instanceof Response) return gate;

  try {
    await query(
      `INSERT INTO audit_log (user_id, action, object_type, object_id, detail, ip)
       VALUES ($1, 'view_as_end', 'user', NULL, '{}'::jsonb, $2)`,
      [gate.userId, req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() ?? null],
    );
  } catch { /* best-effort */ }

  const res = ok({ cleared: true });
  res.headers.append('set-cookie', cookieHeader(VIEW_AS_COOKIE, '', 0, true));
  res.headers.append('set-cookie', cookieHeader(VIEW_AS_ROLE_COOKIE, '', 0, false));
  return res;
}

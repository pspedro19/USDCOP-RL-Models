/**
 * /api/admin/roles — dynamic role→permission matrix (admin:all, CTR-RBAC-001 / migration 056).
 *
 *   GET   → { roles, permissions, matrix }  (matrix = role → permission[], DB-backed w/ static fallback)
 *   PATCH → toggle ONE cell { role, permission, enabled } → writes rbac_role_permissions,
 *           audits `role_perm_change`, returns the updated matrix.
 *
 * The route→permission matrices + the ROLES/PERMISSIONS enums stay STATIC in
 * rbac.contract.ts (the rbac:check invariant + deny-by-default floor); only the
 * role→permission MAPPING is data. Effective permissions apply on each user's next
 * login (JWT re-mint) and within the resolver's 60s cache for server-side reads.
 */
import { ok, fail } from '@/lib/api/envelope';
import { adminActor, requireAdminRole } from '@/lib/admin/guard';
import { PERMISSIONS, ROLES, type Permission, type Role } from '@/lib/contracts/rbac.contract';
import { allRolePermissions, invalidateRbacCaches } from '@/lib/auth/rbac-resolver';
import { query } from '@/lib/db/postgres-client';

function isRole(v: unknown): v is Role {
  return typeof v === 'string' && (ROLES as readonly string[]).includes(v);
}
function isPermission(v: unknown): v is Permission {
  return typeof v === 'string' && (PERMISSIONS as readonly string[]).includes(v);
}

export async function GET(req: Request) {
  const denied = requireAdminRole(req);
  if (denied) return denied;
  try {
    const matrix = await allRolePermissions();
    return ok({ roles: ROLES, permissions: PERMISSIONS, matrix });
  } catch (e) {
    return fail('DB_UNAVAILABLE', `No se pudo leer la matriz de roles: ${String((e as Error)?.message ?? e)}`, 503);
  }
}

export async function PATCH(req: Request) {
  const denied = requireAdminRole(req);
  if (denied) return denied;

  let payload: { role?: unknown; permission?: unknown; enabled?: unknown };
  try { payload = await req.json(); } catch { return fail('BAD_REQUEST', 'Body JSON inválido.', 400); }

  if (!isRole(payload.role)) return fail('BAD_REQUEST', `role inválido — uno de: ${ROLES.join(', ')}.`, 400);
  if (!isPermission(payload.permission)) return fail('BAD_REQUEST', `permission inválido — uno de: ${PERMISSIONS.join(', ')}.`, 400);
  if (typeof payload.enabled !== 'boolean') return fail('BAD_REQUEST', 'enabled debe ser booleano.', 400);

  const role = payload.role;
  const permission = payload.permission;
  const enabled = payload.enabled;

  // Guardrail: an admin can never strip `admin:all` from the `admin` role (self-lockout).
  if (role === 'admin' && permission === 'admin:all' && !enabled) {
    return fail('FORBIDDEN', 'No se puede quitar admin:all del rol admin (bloqueo de seguridad).', 403);
  }

  try {
    if (enabled) {
      await query(
        `INSERT INTO rbac_role_permissions (role, permission) VALUES ($1, $2)
         ON CONFLICT (role, permission) DO NOTHING`,
        [role, permission],
      );
    } else {
      await query('DELETE FROM rbac_role_permissions WHERE role = $1 AND permission = $2', [role, permission]);
    }
    invalidateRbacCaches();

    const actor = adminActor(req);
    try {
      await query(
        `INSERT INTO audit_log (user_id, action, object_type, object_id, detail, ip)
         VALUES ($1, 'role_perm_change', 'role', $2, $3::jsonb, $4)`,
        [actor.id, role, JSON.stringify({ permission, enabled, via: '/api/admin/roles' }), actor.ip],
      );
    } catch (auditErr) {
      console.error('[roles PATCH] audit insert failed:', auditErr);
    }

    const matrix = await allRolePermissions();
    return ok({ roles: ROLES, permissions: PERMISSIONS, matrix });
  } catch (e) {
    return fail('DB_UNAVAILABLE', `No se pudo actualizar la matriz: ${String((e as Error)?.message ?? e)}`, 503);
  }
}

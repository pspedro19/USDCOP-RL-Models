/**
 * /api/admin/users/:id/overrides — per-user permission grants/denies (admin:all,
 * CTR-RBAC-001 / migration 056). Layered on top of the user's role permissions:
 * effective = rolePerms ∪ grants − denies (lib/auth/rbac-resolver).
 *
 *   GET   → { overrides: [{permission, effect, expires_at, nda_reference}] }
 *   PATCH → { permission, effect: 'grant' | 'deny' | 'clear', expires_at?, nda_reference? }
 *           → upsert/delete one override, audit `user_override_change`, return the list.
 *
 * TEMPORARY GRANTS (the NDA data room): a grant may carry `expires_at`, after which the
 * resolver stops applying it — no sweeper job sits between the deadline and the loss of
 * access. This is how a prospect gets `research:read` for the length of a deal without
 * anyone having to remember to take it back. `nda_reference` records which signed
 * agreement authorised it. A DENY may never expire (a revocation that lapses would
 * silently restore what someone deliberately removed — enforced by migration 091).
 *
 * Applies on the user's next login (JWT re-mint) / within the resolver's 60s cache.
 */
import { ok, fail } from '@/lib/api/envelope';
import { adminActor, requireAdminRole } from '@/lib/admin/guard';
import { PERMISSIONS, type Permission } from '@/lib/contracts/rbac.contract';
import { getUserOverrides, invalidateRbacCaches } from '@/lib/auth/rbac-resolver';
import { query } from '@/lib/db/postgres-client';

function isPermission(v: unknown): v is Permission {
  return typeof v === 'string' && (PERMISSIONS as readonly string[]).includes(v);
}
type Effect = 'grant' | 'deny' | 'clear';
function isEffect(v: unknown): v is Effect {
  return v === 'grant' || v === 'deny' || v === 'clear';
}

export async function GET(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const denied = requireAdminRole(req);
  if (denied) return denied;
  const { id } = await ctx.params;
  try {
    return ok({ overrides: await getUserOverrides(id) });
  } catch (e) {
    return fail('DB_UNAVAILABLE', `No se pudieron leer los overrides: ${String((e as Error)?.message ?? e)}`, 503);
  }
}

export async function PATCH(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const denied = requireAdminRole(req);
  if (denied) return denied;
  const { id } = await ctx.params;

  let payload: {
    permission?: unknown; effect?: unknown; reason?: unknown;
    expires_at?: unknown; nda_reference?: unknown;
  };
  try { payload = await req.json(); } catch { return fail('BAD_REQUEST', 'Body JSON inválido.', 400); }

  if (!isPermission(payload.permission)) return fail('BAD_REQUEST', `permission inválido — uno de: ${PERMISSIONS.join(', ')}.`, 400);
  if (!isEffect(payload.effect)) return fail('BAD_REQUEST', 'effect debe ser grant | deny | clear.', 400);

  const permission = payload.permission;
  const effect = payload.effect;
  const actor = adminActor(req);

  // Temporary-grant fields. Validated here so a malformed date can never reach the column
  // and become an override that silently never expires.
  let expiresAt: string | null = null;
  if (payload.expires_at != null) {
    const iso = String(payload.expires_at);
    if (Number.isNaN(new Date(iso).getTime())) {
      return fail('BAD_REQUEST', 'expires_at debe ser una fecha ISO válida o null.', 400);
    }
    if (effect !== 'grant') {
      return fail('BAD_REQUEST', 'Solo un grant puede expirar: un deny que caduca reabriría un permiso retirado a propósito.', 400);
    }
    expiresAt = iso;
  }
  const ndaRef = typeof payload.nda_reference === 'string' ? payload.nda_reference : null;

  // Research internals behind a time-boxed grant must be traceable to an agreement.
  if (effect === 'grant' && permission === 'research:read' && expiresAt && !ndaRef) {
    return fail('BAD_REQUEST', 'Un acceso temporal a research exige nda_reference (el acuerdo que lo autoriza).', 400);
  }

  // Guardrail: an admin can never deny their OWN admin:all (self-lockout).
  if (effect === 'deny' && permission === 'admin:all' && id === actor.id) {
    return fail('FORBIDDEN', 'No puedes denegarte admin:all a ti mismo (bloqueo de seguridad).', 403);
  }

  try {
    if (effect === 'clear') {
      await query('DELETE FROM rbac_user_overrides WHERE user_id = $1 AND permission = $2', [id, permission]);
    } else {
      await query(
        `INSERT INTO rbac_user_overrides
           (user_id, permission, effect, reason, updated_by, updated_at, expires_at, nda_reference)
         VALUES ($1, $2, $3, $4, $5, NOW(), $6, $7)
         ON CONFLICT (user_id, permission)
         DO UPDATE SET effect = EXCLUDED.effect, reason = EXCLUDED.reason,
                       updated_by = EXCLUDED.updated_by, updated_at = NOW(),
                       expires_at = EXCLUDED.expires_at,
                       nda_reference = EXCLUDED.nda_reference`,
        [id, permission, effect, typeof payload.reason === 'string' ? payload.reason : null,
         actor.id, expiresAt, ndaRef],
      );
    }
    invalidateRbacCaches();

    try {
      await query(
        `INSERT INTO audit_log (user_id, action, object_type, object_id, detail, ip)
         VALUES ($1, 'user_override_change', 'user', $2, $3::jsonb, $4)`,
        [actor.id, id, JSON.stringify({
          permission, effect, expires_at: expiresAt, nda_reference: ndaRef,
          via: '/api/admin/users/:id/overrides',
        }), actor.ip],
      );
    } catch (auditErr) {
      console.error('[overrides PATCH] audit insert failed:', auditErr);
    }

    return ok({ overrides: await getUserOverrides(id) });
  } catch (e) {
    return fail('DB_UNAVAILABLE', `No se pudo actualizar el override: ${String((e as Error)?.message ?? e)}`, 503);
  }
}

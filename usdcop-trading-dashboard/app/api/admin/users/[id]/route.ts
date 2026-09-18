/**
 * PATCH /api/admin/users/:id — edit a user's plan/entitlements and/or role (admin:all).
 *
 * The RBAC rule "entitlements only via webhook/admin" (rbac.md §7) names THIS route as the
 * sole human path to change a plan. Writes the canonical entitlements object (derived from
 * PLAN_DEFAULTS[plan], the SSOT ceilings — a partial row would later crash consumers) with
 * the chosen expiry into sb_users.entitlements, and/or updates the role. Every change lands
 * an append-only audit_log row (plan_change / role_change) recording the REAL acting admin.
 *
 * ASSETS ARE UNIONED, NEVER REPLACED (the same CXD-056 rule the billing webhook follows).
 * Every PLAN_DEFAULTS entry carries `assets: ['usdcop']`, so writing the plan object
 * verbatim used to DELETE every per-asset add-on the user had paid for — and the admin UI
 * re-sends `plan` just to edit an expiry date, which made "extend this customer" silently
 * destroy their purchase. An admin who genuinely wants to set the asset list passes
 * `assets` explicitly; omitting it preserves what the user holds.
 *
 * Server-side validation is authoritative: plan ∈ PlanId, role ∈ Role — the client select is
 * only a convenience, never trusted.
 */
import { ok, fail } from '@/lib/api/envelope';
import { adminActor, requireAdminRole } from '@/lib/admin/guard';
import { PLAN_DEFAULTS, ROLES, type Entitlements, type PlanId, type Role }
  from '@/lib/contracts/rbac.contract';
import { getClient } from '@/lib/db/postgres-client';

const PLAN_IDS = Object.keys(PLAN_DEFAULTS) as PlanId[];

function isPlanId(v: unknown): v is PlanId {
  return typeof v === 'string' && (PLAN_IDS as string[]).includes(v);
}
function isRole(v: unknown): v is Role {
  return typeof v === 'string' && (ROLES as readonly string[]).includes(v);
}

export async function PATCH(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const denied = requireAdminRole(req);
  if (denied) return denied;
  const { id } = await ctx.params;

  let payload: { plan?: unknown; expires_at?: unknown; role?: unknown; assets?: unknown };
  try { payload = await req.json(); } catch { return fail('BAD_REQUEST', 'Body JSON inválido.', 400); }

  const wantsPlan = payload.plan !== undefined;
  const wantsRole = payload.role !== undefined;
  if (!wantsPlan && !wantsRole) {
    return fail('BAD_REQUEST', 'Nada que actualizar: envía plan y/o role.', 400);
  }
  if (wantsPlan && !isPlanId(payload.plan)) {
    return fail('BAD_REQUEST', `plan inválido — debe ser uno de: ${PLAN_IDS.join(', ')}.`, 400);
  }
  if (wantsRole && !isRole(payload.role)) {
    return fail('BAD_REQUEST', `role inválido — debe ser uno de: ${ROLES.join(', ')}.`, 400);
  }
  // Explicit asset override — the ONLY way to shrink a user's asset list from here.
  const assetsOverride = payload.assets === undefined ? null : payload.assets;
  if (assetsOverride !== null
      && !(Array.isArray(assetsOverride) && assetsOverride.every((a) => typeof a === 'string'))) {
    return fail('BAD_REQUEST', 'assets debe ser un arreglo de ids de activo.', 400);
  }

  // expires_at: ISO string or null (never trust an arbitrary shape).
  let expiresAt: string | null = null;
  if (payload.expires_at != null) {
    const iso = String(payload.expires_at);
    if (Number.isNaN(new Date(iso).getTime())) {
      return fail('BAD_REQUEST', 'expires_at debe ser una fecha ISO válida o null.', 400);
    }
    expiresAt = iso;
  }

  // The plan write is read-modify-write over a JSONB column, so it runs inside a
  // transaction with `FOR UPDATE` — exactly like the billing webhook. Without the lock an
  // admin edit landing next to a paid webhook is last-write-wins and one of them vanishes.
  let client;
  try {
    client = await getClient();
  } catch (e) {
    return fail('DB_UNAVAILABLE', `No se pudo conectar: ${String((e as Error)?.message ?? e)}`, 503);
  }

  try {
    await client.query('BEGIN');

    const prevRes = await client.query<{ entitlements: unknown }>(
      'SELECT entitlements FROM sb_users WHERE id = $1 FOR UPDATE',
      [id],
    );
    if (prevRes.rowCount === 0) {
      await client.query('ROLLBACK');
      return fail('NOT_FOUND', 'Usuario no encontrado.', 404);
    }

    // Stored assets are the PURCHASE RECORD and survive a lapse; access is separately
    // denied while the row is expired, because every reader resolves through
    // `effectiveEntitlements` (expired ⇒ free). Destroying the list here would make a
    // renewal unable to restore what the customer already paid for.
    const storedRaw = prevRes.rows[0].entitlements;
    const stored = (storedRaw && typeof storedRaw === 'object'
      ? (storedRaw as Partial<Entitlements>)
      : null);
    const previousAssets = Array.isArray(stored?.assets) ? (stored!.assets as string[]) : [];

    const sets: string[] = [];
    const vals: unknown[] = [];
    let i = 1;
    let entitlements: Entitlements | null = null;
    if (wantsPlan) {
      const base = PLAN_DEFAULTS[payload.plan as PlanId];
      const assets = assetsOverride !== null
        ? [...new Set(assetsOverride as string[])]
        : [...new Set([...previousAssets, ...base.assets])];
      entitlements = { ...base, assets, expires_at: expiresAt };
      sets.push(`entitlements = $${i}::jsonb`); vals.push(JSON.stringify(entitlements)); i++;
    }
    if (wantsRole) {
      sets.push(`role = $${i}`); vals.push(payload.role as Role); i++;
    }
    vals.push(id);

    const res = await client.query(
      `UPDATE sb_users SET ${sets.join(', ')} WHERE id = $${i}
       RETURNING id, email, role, entitlements->>'plan' AS plan, entitlements->>'expires_at' AS expires_at`,
      vals,
    );
    const row = res.rows[0];
    const actor = adminActor(req);

    // One audit row per dimension changed (append-only, loud on failure). `previous_assets`
    // makes an accidental asset change recoverable from the trail alone, the same way the
    // billing webhook records what the user held before a purchase.
    const auditRows: Array<[string, Record<string, unknown>]> = [];
    if (wantsPlan) {
      auditRows.push(['plan_change', {
        target_email: row.email, plan: payload.plan, expires_at: expiresAt,
        assets: entitlements?.assets, previous_assets: previousAssets,
        assets_replaced: assetsOverride !== null,
      }]);
    }
    if (wantsRole) auditRows.push(['role_change', { target_email: row.email, role: payload.role }]);
    for (const [act, detail] of auditRows) {
      try {
        await client.query(
          `INSERT INTO audit_log (user_id, action, object_type, object_id, detail, ip)
           VALUES ($1, $2, 'user', $3, $4::jsonb, $5)`,
          [actor.id, act, id, JSON.stringify({ ...detail, via: '/api/admin/users/:id' }), actor.ip],
        );
      } catch (auditErr) {
        console.error(`[users PATCH] audit insert (${act}) failed:`, auditErr);
      }
    }

    await client.query('COMMIT');
    return ok({
      id: row.id, email: row.email, role: row.role,
      plan: row.plan, expires_at: row.expires_at,
      ...(entitlements ? { assets: entitlements.assets } : {}),
    });
  } catch (e) {
    try { await client.query('ROLLBACK'); } catch { /* connection already gone */ }
    return fail('DB_UNAVAILABLE', `No se pudo actualizar el usuario: ${String((e as Error)?.message ?? e)}`, 503);
  } finally {
    client.release();
  }
}

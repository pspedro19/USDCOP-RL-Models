/**
 * POST /api/admin/risk/keys/:id — approve / reject a pending exchange key (admin:all).
 *
 * Turns the Riesgo "Conexiones API por aprobar" table from view-only into an audited
 * action. Mirrors the /api/admin/risk kill-switch POST: typed confirm + append-only
 * audit_log row. SignalBridge exposes only self-service key registration (/tenant/me/keys,
 * which already REJECTS withdraw-enabled keys and parks unverifiable keys as 'pending');
 * there is no admin key-state endpoint, so the console transitions the key's state
 * directly in `user_exchange_keys` — the same table the GET reads. Status vocabulary
 * matches migration 055: pending → verified (approve) | revoked (reject).
 *
 * Withdraw-permission gate (rbac.md §5, fail-closed): a key can only be APPROVED when the
 * admin asserts withdraw is disabled (`withdraw_disabled_confirmed: true`). The assertion
 * is recorded in the audit detail — the system never custodies withdraw-capable keys.
 */
import { ok, fail } from '@/lib/api/envelope';
import { requirePermission } from '@/lib/api/relay';
import { query } from '@/lib/db/postgres-client';

type Action = 'approve' | 'reject';

export async function POST(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const gate = requirePermission(req, 'admin:all');
  if (gate instanceof Response) return gate;
  const { id } = await ctx.params;

  let payload: { action?: Action; confirm?: string; withdraw_disabled_confirmed?: boolean; notes?: string };
  try { payload = await req.json(); } catch { return fail('BAD_REQUEST', 'Body JSON inválido.', 400); }

  const action = payload.action;
  if (action !== 'approve' && action !== 'reject') {
    return fail('BAD_REQUEST', 'action debe ser "approve" o "reject".', 400);
  }
  // Typed confirm proportional to a live-key decision.
  const token = action === 'approve' ? 'APPROVE' : 'REJECT';
  if (payload.confirm !== token) {
    return fail('CONFIRMATION_REQUIRED', `Escribe ${token} para confirmar.`, 400);
  }
  // Withdraw gate — fail closed: no approval without an explicit anti-withdraw assertion.
  if (action === 'approve' && payload.withdraw_disabled_confirmed !== true) {
    return fail(
      'WITHDRAW_CHECK_REQUIRED',
      'Confirma que la llave NO tiene permiso de retiro antes de aprobarla (jamás custodiamos retiros).',
      400,
    );
  }

  const nextStatus = action === 'approve' ? 'verified' : 'revoked';
  try {
    const res = await query(
      `UPDATE user_exchange_keys
         SET status = $1, last_verified_at = NOW()
       WHERE id = $2 AND status = 'pending'
       RETURNING id, user_id, exchange, status`,
      [nextStatus, id],
    );
    if (res.rowCount === 0) {
      // Either not found or not pending anymore — surface a precise 409/404.
      const exists = await query(`SELECT status FROM user_exchange_keys WHERE id = $1`, [id]);
      if (exists.rowCount === 0) return fail('NOT_FOUND', 'Llave no encontrada.', 404);
      return fail('CONFLICT', `La llave ya no está pendiente (estado: ${exists.rows[0].status}).`, 409);
    }
    const row = res.rows[0];

    try {
      await query(
        `INSERT INTO audit_log (user_id, action, object_type, object_id, detail, ip)
         VALUES ($1, $2, 'exchange_key', $3, $4::jsonb, $5)`,
        [gate.userId, action === 'approve' ? 'key_approve' : 'key_reject', String(row.id),
          JSON.stringify({
            key_id: String(row.id),
            target_user_id: String(row.user_id),
            exchange: row.exchange,
            new_status: nextStatus,
            withdraw_disabled_confirmed: action === 'approve' ? true : undefined,
            notes: payload.notes ?? null,
            via: '/api/admin/risk/keys',
          }),
          req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() ?? null],
      );
    } catch (auditErr) {
      console.error('[risk/keys] audit insert failed:', auditErr);
    }
    return ok({ id: String(row.id), status: nextStatus });
  } catch (e) {
    return fail('DB_UNAVAILABLE', `No se pudo actualizar la llave: ${String((e as Error)?.message ?? e)}`, 503);
  }
}

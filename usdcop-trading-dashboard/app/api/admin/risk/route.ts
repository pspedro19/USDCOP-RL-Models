/**
 * /api/admin/risk — pestaña "Riesgo y bloqueos" (admin:all).
 *
 * GET: DB directa (user_risk_limits_v2: kill por usuario + modos; user_exchange_keys
 * status='pending'; sb_users suspendidos/rechazados) + relay a SignalBridge
 * GET /api/signal-bridge/kill-switch/status para el kill GLOBAL — tolerante a fallo:
 * cada sub-check que cae aterriza en partial_errors, el resto se sirve (C5).
 *
 * POST: kill global — reutiliza el endpoint SB /api/tenant/system/kill vía relay con
 * el bearer del ADMIN real (SB audita con su identidad). Requiere confirmación
 * tipada `confirm: "STOP"` desde la UI y deja fila propia en audit_log.
 */
import { ok, fail, UpstreamError } from '@/lib/api/envelope';
import { requirePermission, relayUpstream, bearerFrom } from '@/lib/api/relay';
import type {
  AdminRiskApiPending, AdminRiskResponse, AdminRiskSuspended, AdminRiskUserKill,
} from '@/lib/contracts/admin-console.contract';
import { query } from '@/lib/db/postgres-client';

const SB_URL = process.env.SIGNALBRIDGE_BACKEND_URL || 'http://usdcop-signalbridge:8000';

/** Techos de sistema — espejo de services/signalbridge_api/app/api/routes/tenant.py. */
const CEILINGS = { max_notional_usd: 5000, max_open_positions: 2, max_daily_loss_pct: 3.0 };

export async function GET(req: Request) {
  const gate = requirePermission(req, 'admin:all');
  if (gate instanceof Response) return gate;

  const partial: string[] = [];
  const settle = async <T>(label: string, p: Promise<T>): Promise<T | null> => {
    try { return await p; } catch (e) { partial.push(`${label}: ${String((e as Error)?.message ?? e)}`); return null; }
  };

  const [kills, pendingKeys, suspended, sbKill] = await Promise.all([
    settle('user_risk_limits_v2', query(`
      SELECT rl.user_id, rl.mode, rl.kill_switch, u.email
      FROM user_risk_limits_v2 rl
      LEFT JOIN sb_users u ON u.id = rl.user_id
      ORDER BY rl.updated_at DESC
      LIMIT 200`)),
    settle('user_exchange_keys', query(`
      SELECT k.id, k.user_id, k.exchange, k.created_at, u.email
      FROM user_exchange_keys k
      LEFT JOIN sb_users u ON u.id = k.user_id
      WHERE k.status = 'pending'
      ORDER BY k.created_at ASC
      LIMIT 100`)),
    settle('sb_users_suspended', query(`
      SELECT id, email, status, created_at
      FROM sb_users
      WHERE status IN ('suspended', 'rejected') OR is_active = FALSE
      ORDER BY created_at DESC
      LIMIT 100`)),
    settle('signalbridge kill-switch', relayUpstream<{ active?: boolean; enabled?: boolean; kill_switch?: boolean; reason?: string }>(
      `${SB_URL}/api/signal-bridge/kill-switch/status`,
      { bearer: bearerFrom(req), incomingHeaders: new Headers(req.headers) },
    )),
  ]);

  const user_kills: AdminRiskUserKill[] = (kills?.rows ?? []).map((r) => ({
    user_id: String(r.user_id),
    email: r.email ?? null,
    mode: r.mode ?? 'paper',
    kill_switch: !!r.kill_switch,
  }));

  const api_pending: AdminRiskApiPending[] = (pendingKeys?.rows ?? []).map((r) => ({
    id: String(r.id),
    user_id: String(r.user_id),
    email: r.email ?? null,
    exchange: r.exchange,
    created_at: r.created_at instanceof Date ? r.created_at.toISOString() : String(r.created_at),
  }));

  const suspendedRows: AdminRiskSuspended[] = (suspended?.rows ?? []).map((r) => ({
    id: String(r.id),
    email: r.email,
    status: r.status,
    created_at: r.created_at instanceof Date ? r.created_at.toISOString() : (r.created_at ?? null),
  }));

  const sbData = sbKill?.data ?? null;
  const globalActive = sbData
    ? Boolean(sbData.active ?? sbData.enabled ?? sbData.kill_switch ?? false)
    : null;

  const body: AdminRiskResponse = {
    global_kill: { active: globalActive, detail: sbData?.reason ?? (globalActive === null ? 'SignalBridge no disponible' : null) },
    user_kills,
    modes: {
      paper: user_kills.filter((k) => k.mode !== 'live').length,
      live: user_kills.filter((k) => k.mode === 'live').length,
    },
    api_pending,
    suspended: suspendedRows,
    limits: CEILINGS,
    partial_errors: partial,
  };
  return ok(body, { meta: { asOf: new Date().toISOString() } });
}

/** Kill global ON/OFF — confirmación tipada "STOP" + audit_log + relay al SB del admin. */
export async function POST(req: Request) {
  const gate = requirePermission(req, 'admin:all');
  if (gate instanceof Response) return gate;

  let payload: { enable?: boolean; confirm?: string };
  try { payload = await req.json(); } catch { return fail('BAD_REQUEST', 'Body JSON inválido.', 400); }
  if (payload.confirm !== 'STOP') {
    return fail('CONFIRMATION_REQUIRED', 'Escribe STOP para confirmar la acción de kill global.', 400);
  }
  const enable = payload.enable !== false;

  const bearer = bearerFrom(req);
  if (!bearer) {
    return fail('UNAUTHENTICATED', 'Falta el token de SignalBridge — vuelve a iniciar sesión.', 401);
  }

  try {
    const { data } = await relayUpstream<Record<string, unknown>>(
      `${SB_URL}/api/tenant/system/kill?enable=${enable}`,
      { method: 'POST', bearer, incomingHeaders: new Headers(req.headers) },
    );
    // Fila propia en audit_log del dashboard (SB ya auditó 'kill_global' por su lado).
    try {
      await query(
        `INSERT INTO audit_log (user_id, action, object_type, object_id, detail, ip)
         VALUES ($1, $2, 'system', 'global_kill', $3::jsonb, $4)`,
        [gate.userId, enable ? 'kill_global_on' : 'kill_global_off',
          JSON.stringify({ enable, via: '/api/admin/risk' }),
          req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() ?? null],
      );
    } catch { /* best-effort: SB ya dejó su fila kill_global */ }
    return ok({ enabled: enable, upstream: data ?? null });
  } catch (e) {
    if (e instanceof UpstreamError) return e.toResponse();
    return fail('UPSTREAM_UNAVAILABLE', 'SignalBridge no disponible.', 502);
  }
}

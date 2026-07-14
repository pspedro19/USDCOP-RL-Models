/**
 * POST /api/admin/system/recover — trigger a data-recovery DAG via Airflow REST (admin:all).
 *
 * Turns the Sistema "Reintentar backfill" button from a clipboard no-op into a real,
 * audited action. Mirrors the /api/admin/risk kill-switch POST pattern: typed confirm
 * (write the dag_id) + append-only audit_log row + upstream relay with the admin's
 * identity. Fail-closed: unknown DAGs and missing confirmation are rejected before any
 * upstream call. The dag_id is whitelisted to the documented recovery procedures
 * (`.claude/rules/data-freshness.md`) — the console never triggers arbitrary DAGs.
 *
 * Degrades honestly: if Airflow REST is not configured (host dev / no env), the route
 * returns 503 and the UI falls back to the clipboard command.
 */
import { ok, fail } from '@/lib/api/envelope';
import { adminActor, requireAdminRole } from '@/lib/admin/guard';
import { query } from '@/lib/db/postgres-client';

const AIRFLOW_API_URL = process.env.AIRFLOW_API_URL || '';
const AIRFLOW_API_USER = process.env.AIRFLOW_API_USER || '';
const AIRFLOW_API_PASSWORD = process.env.AIRFLOW_API_PASSWORD || '';

/**
 * Whitelist of recovery DAGs the console may trigger (data-freshness.md — recovery
 * procedures). Keys mirror SystemSection's RECOVERY_DAG map; extend both together.
 */
const ALLOWED_RECOVERY_DAGS = new Set<string>([
  'core_l0_01_ohlcv_backfill',
  'core_l0_02_ohlcv_realtime',
  'core_l0_03_macro_backfill',
  'core_l0_04_macro_update',
  'core_l0_05_seed_backup',
  'news_daily_pipeline',
  'forecast_h5_l3_weekly_training',
]);

async function triggerDag(dagId: string): Promise<{ dag_run_id?: string; state?: string }> {
  const base = AIRFLOW_API_URL.replace(/\/$/, '');
  const auth = 'Basic ' + Buffer.from(`${AIRFLOW_API_USER}:${AIRFLOW_API_PASSWORD}`).toString('base64');
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  try {
    const res = await fetch(`${base}/api/v1/dags/${dagId}/dagRuns`, {
      method: 'POST',
      headers: { Authorization: auth, 'content-type': 'application/json' },
      body: JSON.stringify({ conf: {}, note: 'admin-console recovery' }),
      signal: controller.signal,
      cache: 'no-store',
    });
    const text = await res.text();
    if (!res.ok) throw new Error(`Airflow HTTP ${res.status}: ${text.slice(0, 200)}`);
    const j = text ? JSON.parse(text) : {};
    return { dag_run_id: j?.dag_run_id, state: j?.state };
  } finally {
    clearTimeout(timer);
  }
}

export async function POST(req: Request) {
  const denied = requireAdminRole(req);
  if (denied) return denied;

  let payload: { dag_id?: string; confirm?: string };
  try { payload = await req.json(); } catch { return fail('BAD_REQUEST', 'Body JSON inválido.', 400); }

  const dagId = (payload.dag_id ?? '').trim();
  if (!dagId) return fail('BAD_REQUEST', 'Falta dag_id.', 400);
  if (!ALLOWED_RECOVERY_DAGS.has(dagId)) {
    return fail('DAG_NOT_ALLOWED', `El DAG ${dagId} no está en la lista de recuperación permitida.`, 400);
  }
  // Typed confirm proportional to the action: escribe el dag_id exacto (§ audit gate).
  if (payload.confirm !== dagId) {
    return fail('CONFIRMATION_REQUIRED', `Escribe ${dagId} para confirmar el disparo del DAG.`, 400);
  }
  if (!AIRFLOW_API_URL || !AIRFLOW_API_USER) {
    return fail('AIRFLOW_UNCONFIGURED', 'Airflow REST no está configurado — usa el comando manual.', 503);
  }

  const actor = adminActor(req);
  try {
    const run = await triggerDag(dagId);
    // Append-only audit row (best-effort loud): the trigger is legit even if the ledger is down.
    try {
      await query(
        `INSERT INTO audit_log (user_id, action, object_type, object_id, detail, ip)
         VALUES ($1, 'dag_trigger', 'system', $2, $3::jsonb, $4)`,
        [actor.id, dagId,
          JSON.stringify({ dag_id: dagId, dag_run_id: run.dag_run_id ?? null, via: '/api/admin/system/recover' }),
          actor.ip],
      );
    } catch (auditErr) {
      console.error('[system/recover] audit insert failed:', auditErr);
    }
    return ok({ dag_id: dagId, dag_run_id: run.dag_run_id ?? null, state: run.state ?? 'queued' });
  } catch (e) {
    return fail('AIRFLOW_UNAVAILABLE', `No se pudo disparar el DAG: ${String((e as Error)?.message ?? e)}`, 502);
  }
}

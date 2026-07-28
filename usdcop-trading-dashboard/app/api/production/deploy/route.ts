/**
 * POST /api/production/deploy — One-click production deploy
 *
 * Validates approval status, then spawns a detached Python process to retrain
 * with full data (2020-2025) and export to production.
 *
 * File-based state: public/data/production/deploy_status.json
 */
import { NextRequest, NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { requireApprovalVote } from '@/lib/auth/approval-authz';
import { readApprovalState } from '@/lib/approvals/store';
import { query } from '@/lib/db/postgres-client';
import { resolveDeployCommand, type DeployCommandResolution } from '@/lib/security/deploy-command';
import type {
  ApprovalState,
  DeployStatus,
  DeployResponse,
} from '@/lib/contracts/production-approval.contract';

const DATA_DIR = path.join(process.cwd(), 'public', 'data', 'production');
/** Approval state lives OUTSIDE `public/` (CXD-057) — see `lib/approvals/store.ts`.
 *  The per-strategy → singleton resolution is now shared verbatim with the approve
 *  route and the H5-L4b DAG; they must agree or approval succeeds and deploy 404s. */
const DEPLOY_FILE = path.join(DATA_DIR, 'deploy_status.json');

// Project root is one level above the dashboard
const PROJECT_ROOT = path.resolve(process.cwd(), '..');

async function readJsonFile<T>(filePath: string): Promise<T | null> {
  try {
    const raw = await fs.readFile(filePath, 'utf-8');
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

async function writeDeployStatus(status: DeployStatus): Promise<void> {
  await fs.writeFile(DEPLOY_FILE, JSON.stringify(status, null, 2), 'utf-8');
}

// ── Airflow REST trigger (container-native deploy path, QA ledger #40) ─────────
// The node container has no python3, so the in-process spawn can never retrain there.
// When AIRFLOW_API_* env is configured, trigger the H5-L4b deploy DAG instead: it
// re-checks APPROVED (hard gate), runs the manifest command, and mirrors progress
// into deploy_status.json — same panel UX, real deploy.
const AIRFLOW_URL = process.env.AIRFLOW_API_URL ?? '';
const AIRFLOW_API_USER = process.env.AIRFLOW_API_USER ?? '';
const AIRFLOW_API_PASSWORD = process.env.AIRFLOW_API_PASSWORD ?? '';
const DEPLOY_DAG_ID = process.env.DEPLOY_DAG_ID ?? 'forecast_h5_l4b_production_deploy';

/**
 * Incidente de seguridad: el manifiesto APROBADO pide algo que no se puede ejecutar.
 *
 * Es append-only en `audit_log` (best-effort: la DB puede estar caída) + un `console.error`
 * que SIEMPRE queda, porque este evento significa una de dos cosas y ambas se investigan:
 * el artefacto de aprobación fue manipulado, o el pipeline empezó a escribir un manifiesto
 * que ya no cumple el contrato. En ninguno de los dos casos se ejecuta nada.
 */
async function auditManifestRejection(
  userId: string | null,
  strategy: string,
  denial: Extract<DeployCommandResolution, { ok: false }>,
  manifest: unknown,
  req: NextRequest,
): Promise<void> {
  const detail = {
    field: denial.field,
    reason: denial.reason,
    manifest_script: (manifest as { script?: unknown } | undefined)?.script ?? null,
    manifest_args: (manifest as { args?: unknown } | undefined)?.args ?? null,
    via: '/api/production/deploy',
    executed: false,
  };
  console.error(`[Deploy] SECURITY — deploy manifest REJECTED (${denial.field}): ${denial.reason}`, detail);
  try {
    await query(
      `INSERT INTO audit_log (user_id, action, object_type, object_id, detail, ip)
       VALUES ($1, 'deploy_manifest_rejected', 'deploy', $2, $3::jsonb, $4)`,
      [
        userId,
        strategy,
        JSON.stringify(detail),
        req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() ?? null,
      ],
    );
  } catch (e) {
    console.error('[Deploy] audit_log row failed for rejected manifest (console trail stands):', e);
  }
}

async function triggerAirflowDeploy(strategyId: string | null): Promise<{ ok: boolean; detail: string }> {
  if (!AIRFLOW_URL || !AIRFLOW_API_USER || !AIRFLOW_API_PASSWORD) {
    return { ok: false, detail: 'airflow api env not configured' };
  }
  try {
    const r = await fetch(`${AIRFLOW_URL}/api/v1/dags/${DEPLOY_DAG_ID}/dagRuns`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Basic ${Buffer.from(`${AIRFLOW_API_USER}:${AIRFLOW_API_PASSWORD}`).toString('base64')}`,
      },
      body: JSON.stringify({ conf: { source: 'dashboard-deploy-api', strategy_id: strategyId } }),
    });
    if (r.ok) {
      const d = (await r.json()) as { dag_run_id?: string };
      return { ok: true, detail: d.dag_run_id ?? 'triggered' };
    }
    return { ok: false, detail: `airflow HTTP ${r.status}: ${(await r.text()).slice(0, 150)}` };
  } catch (e) {
    return { ok: false, detail: `airflow unreachable: ${e instanceof Error ? e.message : String(e)}` };
  }
}

export async function POST(request: NextRequest) {
  try {
    // Spawns a production retrain+deploy — `approval-gates.md` §4/§5: el servidor
    // re-valida en CADA capa y **solo `admin`** promueve. Antes bastaba con estar
    // autenticado (`protectApiRoute` sin permiso): un `subscriber` que invocara el
    // handler sin atravesar el middleware lanzaba el deploy de producción sobre un
    // bundle ya APPROVED. El permiso se exige aquí, no solo en el edge.
    // La ruta approve reenvía la cookie del aprobador, así que su auto-deploy pasa.
    const gate = await requireApprovalVote(request);
    if (!gate.ok) {
      return NextResponse.json(
        { success: false, message: gate.message } as DeployResponse,
        { status: gate.status }
      );
    }

    // 0. Optional multi-strategy target (per-sid approval + Airflow conf)
    let strategyId: string | null = null;
    try {
      const body = await request.json();
      strategyId = typeof body?.strategy_id === 'string' ? body.strategy_id : null;
    } catch { /* empty body = default strategy */ }

    // 1. Validate approval state
    const approvalRecord = await readApprovalState(strategyId);
    const approval: ApprovalState | null = approvalRecord?.state ?? null;
    if (!approval) {
      return NextResponse.json(
        { success: false, status: 'idle', message: 'No approval state found. Run backtest first.' } as DeployResponse,
        { status: 404 }
      );
    }

    if (approval.status !== 'APPROVED') {
      return NextResponse.json(
        { success: false, status: 'idle', message: `Cannot deploy — approval status is ${approval.status}. Must be APPROVED.` } as DeployResponse,
        { status: 409 }
      );
    }

    // 2. Check if a deploy is already running
    const existing = await readJsonFile<DeployStatus>(DEPLOY_FILE);
    if (existing?.status === 'running') {
      // Airflow-run deploys have no local pid — trust the status file (the DAG mirrors
      // completion/failure into it; max_active_runs=1 also guards on the Airflow side).
      if (existing.runner === 'airflow') {
        return NextResponse.json(
          { success: false, status: 'running', message: `A deploy is already in progress (Airflow ${existing.dag_run_id ?? ''}).` } as DeployResponse,
          { status: 409 }
        );
      }
      // Check if the process is actually still alive
      if (existing.pid) {
        try {
          process.kill(existing.pid, 0); // Signal 0 = just check existence
          return NextResponse.json(
            { success: false, status: 'running', message: 'A deploy is already in progress.' } as DeployResponse,
            { status: 409 }
          );
        } catch {
          // Process is dead — mark as failed and allow re-deploy
          await writeDeployStatus({
            ...existing,
            status: 'failed',
            error: 'Previous deploy process terminated unexpectedly.',
            completed_at: new Date().toISOString(),
          });
        }
      }
    }

    // 3. Resolver el COMANDO antes de cualquier efecto (fail-closed, K-040).
    //
    // El manifiesto viaja dentro del artefacto de aprobación; antes se pasaba tal cual a
    // `spawn(..., { shell: true })`, así que un `;`/`&&`/`$( )` en el JSON ejecutaba
    // comandos arbitrarios. Ahora `script` y `args` pasan por allowlist y el proceso se
    // lanza con argv explícito y SIN shell.
    //
    // Se valida ANTES de tocar `deploy_status.json` y ANTES de disparar Airflow **a
    // propósito**: el DAG H5-L4b ejecuta ESTE MISMO manifiesto, así que delegar un
    // manifiesto inválido sería mover el problema de contenedor, no cerrarlo.
    const resolution = await resolveDeployCommand(PROJECT_ROOT, approval.deploy_manifest);
    if (!resolution.ok) {
      const at = new Date().toISOString();
      await writeDeployStatus({
        status: 'failed',
        strategy_id: approval.strategy,
        strategy_name: approval.strategy_name,
        started_at: at,
        completed_at: at,
        error: `Deploy manifest rejected (${resolution.field}): ${resolution.reason}`,
      });
      await auditManifestRejection(gate.userId, approval.strategy, resolution, approval.deploy_manifest, request);
      return NextResponse.json(
        {
          success: false,
          status: 'failed',
          message: `Deploy manifest rejected (${resolution.field}): ${resolution.reason}. Nothing was executed.`,
        } as DeployResponse,
        { status: 400 }
      );
    }
    const command = resolution.command;

    // 3b. Write initial deploy status
    const deployStatus: DeployStatus = {
      status: 'running',
      strategy_id: approval.strategy,
      strategy_name: approval.strategy_name,
      started_at: new Date().toISOString(),
      phase: 'initializing',
    };
    await writeDeployStatus(deployStatus);

    // 4a. PREFERRED: container-native deploy via the Airflow H5-L4b DAG (when configured).
    // The DAG re-validates APPROVED, runs the manifest command with a real Python env,
    // and mirrors progress into deploy_status.json for the panel.
    const airflow = await triggerAirflowDeploy(strategyId);
    if (airflow.ok) {
      await writeDeployStatus({ ...deployStatus, phase: 'retraining', runner: 'airflow',
        dag_run_id: airflow.detail } as DeployStatus);
      return NextResponse.json({
        success: true,
        status: 'running',
        message: `Deploy delegated to Airflow (${DEPLOY_DAG_ID}): ${airflow.detail}`,
      } as DeployResponse);
    }
    console.warn(`[Deploy] Airflow path unavailable (${airflow.detail}) — falling back to local spawn`);

    // 4b. Lanzamiento local: argv EXPLÍCITO, SIN shell.
    //
    // `shell: true` estaba aquí "para resolver el PATH en Windows"; el precio era que la
    // línea de comandos la interpretaba `cmd.exe`. La resolución del ejecutable ahora es
    // explícita (`DEPLOY_PYTHON_BIN`/`PYTHON_BIN`, o `python` en Windows y `python3` en
    // el resto — `spawn` sin shell ya busca en el PATH), y `script`/`args` vienen ya
    // validados por allowlist en `lib/security/deploy-command.ts`.
    const child = spawn(
      command.interpreter,
      [command.scriptPath, ...command.args],
      {
        cwd: PROJECT_ROOT,
        detached: true,
        stdio: ['ignore', 'pipe', 'pipe'],
      }
    );

    // Update status with PID
    deployStatus.pid = child.pid;
    deployStatus.phase = 'retraining';
    await writeDeployStatus(deployStatus);

    // Collect output for error reporting
    let stdout = '';
    let stderr = '';

    child.stdout?.on('data', (data: Buffer) => {
      stdout += data.toString();
      // Update phase based on output keywords
      const text = data.toString().toLowerCase();
      if (text.includes('seeding_db') || text.includes('seeding db')) {
        readJsonFile<DeployStatus>(DEPLOY_FILE).then(current => {
          if (current?.status === 'running') {
            writeDeployStatus({ ...current, phase: 'seeding_db' });
          }
        });
      } else if (text.includes('export') || text.includes('writing') || text.includes('json')) {
        readJsonFile<DeployStatus>(DEPLOY_FILE).then(current => {
          if (current?.status === 'running') {
            writeDeployStatus({ ...current, phase: 'exporting' });
          }
        });
      }
    });

    child.stderr?.on('data', (data: Buffer) => {
      stderr += data.toString();
    });

    // Handle process completion
    child.on('close', async (code: number | null) => {
      const current = await readJsonFile<DeployStatus>(DEPLOY_FILE);
      if (!current || current.status !== 'running') return;

      if (code === 0) {
        await writeDeployStatus({
          ...current,
          status: 'completed',
          phase: 'done',
          completed_at: new Date().toISOString(),
        });
      } else {
        const errorMsg = stderr.slice(-500) || `Process exited with code ${code}`;
        await writeDeployStatus({
          ...current,
          status: 'failed',
          error: errorMsg,
          completed_at: new Date().toISOString(),
        });
      }
    });

    child.on('error', async (err: Error) => {
      const current = await readJsonFile<DeployStatus>(DEPLOY_FILE);
      if (!current || current.status !== 'running') return;

      await writeDeployStatus({
        ...current,
        status: 'failed',
        error: `Failed to start process: ${err.message}`,
        completed_at: new Date().toISOString(),
      });
    });

    // Unref so the parent can exit independently
    child.unref();

    return NextResponse.json({
      success: true,
      status: 'running',
      message: 'Deploy started. Retraining with full data (2020-2025)...',
    } as DeployResponse);

  } catch (error) {
    console.error('[Deploy] Error:', error);
    return NextResponse.json(
      {
        success: false,
        status: 'failed',
        message: error instanceof Error ? error.message : 'Internal server error',
      } as DeployResponse,
      { status: 500 }
    );
  }
}

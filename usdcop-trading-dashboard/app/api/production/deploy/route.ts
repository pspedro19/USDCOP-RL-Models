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
import { protectApiRoute } from '@/lib/auth/api-auth';
import type {
  ApprovalState,
  DeployManifest,
  DeployStatus,
  DeployResponse,
} from '@/lib/contracts/production-approval.contract';

const DATA_DIR = path.join(process.cwd(), 'public', 'data', 'production');
const APPROVAL_FILE = path.join(DATA_DIR, 'approval_state.json');
function approvalFileFor(sid?: string | null): string {
  if (!sid || !/^[A-Za-z0-9_-]+$/.test(sid)) return APPROVAL_FILE;
  return path.join(DATA_DIR, `approval_state_${sid}.json`);
}
const DEPLOY_FILE = path.join(DATA_DIR, 'deploy_status.json');

// Project root is one level above the dashboard
const PROJECT_ROOT = path.resolve(process.cwd(), '..');
// Legacy fallback script (used when no deploy_manifest is present)
const LEGACY_SCRIPT = path.join(PROJECT_ROOT, 'scripts', 'pipeline', 'train_and_export_smart_simple.py');

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
    // Spawns a production retrain+deploy — must be authenticated (audit A4-13).
    // The approve route forwards its session cookie so its auto-deploy is allowed.
    const auth = await protectApiRoute(request);
    if (!auth.authenticated) {
      return NextResponse.json(
        { success: false, message: auth.error || 'Unauthorized' } as DeployResponse,
        { status: auth.status || 401 }
      );
    }

    // 0. Optional multi-strategy target (per-sid approval + Airflow conf)
    let strategyId: string | null = null;
    try {
      const body = await request.json();
      strategyId = typeof body?.strategy_id === 'string' ? body.strategy_id : null;
    } catch { /* empty body = default strategy */ }

    // 1. Validate approval state
    const approval = await readJsonFile<ApprovalState>(approvalFileFor(strategyId));
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

    // 3. Write initial deploy status
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

    // 4b. Resolve deploy command from manifest (or legacy fallback)
    const manifest: DeployManifest | undefined = approval.deploy_manifest;
    let scriptPath: string;
    let scriptArgs: string[];

    if (manifest) {
      scriptPath = path.join(PROJECT_ROOT, manifest.script);
      scriptArgs = [...manifest.args];
    } else {
      // Legacy fallback: hardcoded Smart Simple deploy
      scriptPath = LEGACY_SCRIPT;
      scriptArgs = ['--phase', 'production', '--no-png', '--seed-db'];
    }

    // Spawn detached Python process (python3 on Debian/Ubuntu)
    const child = spawn(
      'python3',
      [scriptPath, ...scriptArgs],
      {
        cwd: PROJECT_ROOT,
        detached: true,
        stdio: ['ignore', 'pipe', 'pipe'],
        shell: true, // Required on Windows for PATH resolution
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

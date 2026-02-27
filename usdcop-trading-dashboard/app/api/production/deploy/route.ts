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
import type {
  ApprovalState,
  DeployManifest,
  DeployStatus,
  DeployResponse,
} from '@/lib/contracts/production-approval.contract';

const DATA_DIR = path.join(process.cwd(), 'public', 'data', 'production');
const APPROVAL_FILE = path.join(DATA_DIR, 'approval_state.json');
const DEPLOY_FILE = path.join(DATA_DIR, 'deploy_status.json');

// Project root is one level above the dashboard
const PROJECT_ROOT = path.resolve(process.cwd(), '..');
// Legacy fallback script (used when no deploy_manifest is present)
const LEGACY_SCRIPT = path.join(PROJECT_ROOT, 'scripts', 'train_and_export_smart_simple.py');

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

export async function POST(_request: NextRequest) {
  try {
    // 1. Validate approval state
    const approval = await readJsonFile<ApprovalState>(APPROVAL_FILE);
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

    // 4. Resolve deploy command from manifest (or legacy fallback)
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

/**
 * GET /api/production/deploy/status — Poll deploy progress
 *
 * Reads deploy_status.json and performs dual completion detection:
 * (a) Process exit handler already wrote "completed"
 * (b) summary.json mtime > started_at → mark completed
 * (c) Elapsed > 10 min → mark failed (timeout)
 */
import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';
import type { DeployStatus } from '@/lib/contracts/production-approval.contract';

const DATA_DIR = path.join(process.cwd(), 'public', 'data', 'production');
const DEPLOY_FILE = path.join(DATA_DIR, 'deploy_status.json');
const SUMMARY_FILE = path.join(DATA_DIR, 'summary.json');

const TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes

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

async function getFileMtime(filePath: string): Promise<Date | null> {
  try {
    const stat = await fs.stat(filePath);
    return stat.mtime;
  } catch {
    return null;
  }
}

export async function GET() {
  try {
    const status = await readJsonFile<DeployStatus>(DEPLOY_FILE);

    if (!status) {
      return NextResponse.json({
        status: 'idle',
      } as DeployStatus);
    }

    // If already completed or failed, return as-is
    if (status.status !== 'running') {
      return NextResponse.json(status);
    }

    // === Dual completion detection for running deploys ===

    const startedAt = status.started_at ? new Date(status.started_at).getTime() : 0;
    const now = Date.now();
    const elapsed = now - startedAt;

    // (b) Check if summary.json was updated after deploy started
    const summaryMtime = await getFileMtime(SUMMARY_FILE);
    if (summaryMtime && summaryMtime.getTime() > startedAt) {
      const completed: DeployStatus = {
        ...status,
        status: 'completed',
        phase: 'done',
        completed_at: summaryMtime.toISOString(),
      };
      await writeDeployStatus(completed);
      return NextResponse.json(completed);
    }

    // (c) Timeout check
    if (elapsed > TIMEOUT_MS) {
      const failed: DeployStatus = {
        ...status,
        status: 'failed',
        error: 'Deploy timed out after 10 minutes.',
        completed_at: new Date().toISOString(),
      };
      await writeDeployStatus(failed);
      return NextResponse.json(failed);
    }

    // (d) Check if process is still alive
    if (status.pid) {
      try {
        process.kill(status.pid, 0);
      } catch {
        // Process is dead but didn't write completion — check summary.json one more time
        const summaryExists = await getFileMtime(SUMMARY_FILE);
        if (summaryExists && summaryExists.getTime() > startedAt) {
          const completed: DeployStatus = {
            ...status,
            status: 'completed',
            phase: 'done',
            completed_at: new Date().toISOString(),
          };
          await writeDeployStatus(completed);
          return NextResponse.json(completed);
        }

        const failed: DeployStatus = {
          ...status,
          status: 'failed',
          error: 'Deploy process terminated unexpectedly.',
          completed_at: new Date().toISOString(),
        };
        await writeDeployStatus(failed);
        return NextResponse.json(failed);
      }
    }

    // Estimate phase from elapsed time
    let phase = status.phase || 'initializing';
    if (elapsed > 30_000 && phase === 'initializing') {
      phase = 'retraining';
    }
    if (elapsed > 120_000 && phase === 'retraining') {
      phase = 'exporting';
    }

    return NextResponse.json({
      ...status,
      phase,
    } as DeployStatus);

  } catch (error) {
    console.error('[Deploy Status] Error:', error);
    return NextResponse.json(
      { status: 'idle' } as DeployStatus,
      { status: 500 }
    );
  }
}

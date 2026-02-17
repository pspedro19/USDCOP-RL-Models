/**
 * GET /api/production/status — Read approval state
 *
 * File-based: reads public/data/production/approval_state.json
 * No database required.
 */
import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';
import type { ApprovalState } from '@/lib/contracts/production-approval.contract';

const APPROVAL_FILE = path.join(
  process.cwd(),
  'public',
  'data',
  'production',
  'approval_state.json'
);

const DEFAULT_STATE: ApprovalState = {
  status: 'PENDING_APPROVAL',
  strategy: 'smart_simple_v11',
  backtest_recommendation: 'REVIEW',
  backtest_confidence: 0,
  gates: [],
  created_at: new Date().toISOString(),
  last_updated: new Date().toISOString(),
};

export async function GET() {
  try {
    const raw = await fs.readFile(APPROVAL_FILE, 'utf-8');
    const state: ApprovalState = JSON.parse(raw);
    return NextResponse.json(state);
  } catch {
    // File doesn't exist or parse error — return default
    return NextResponse.json(DEFAULT_STATE);
  }
}

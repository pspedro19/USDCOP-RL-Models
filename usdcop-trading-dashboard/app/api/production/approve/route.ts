/**
 * POST /api/production/approve — Approve or reject the production strategy
 *
 * File-based: reads + writes public/data/production/approval_state.json
 * This is the human "second vote" in the 2-vote promotion system.
 *
 * Body: { action: 'APPROVE' | 'REJECT', notes?: string, reviewer?: string }
 */
import { NextRequest, NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';
import { protectApiRoute } from '@/lib/auth/api-auth';
import type { ApprovalState, ApproveRequest, ApproveResponse } from '@/lib/contracts/production-approval.contract';

const PROD_DIR = path.join(process.cwd(), 'public', 'data', 'production');
const APPROVAL_FILE = path.join(PROD_DIR, 'approval_state.json');

/** Per-strategy approval files (multi-strategy production): approval_state_<sid>.json.
 *  The singleton stays the ACTIVE/default strategy's file (COP).
 *
 *  The fallback is load-bearing, not defensive: the export pipeline writes the ACTIVE
 *  strategy to the unsuffixed `approval_state.json`, while the dashboard always posts a
 *  `strategy_id`. Without it, `smart_simple_v11` resolved to a file that is never written
 *  and Vote 2 returned 404 — the production strategy could not be approved from the UI at
 *  all, while Gold and BTC (which do have suffixed files) worked.
 *
 *  It only falls back when the singleton actually belongs to the requested strategy, so a
 *  stale or mismatched id can never approve somebody else's bundle.
 */
async function approvalFileFor(strategyId?: string | null): Promise<string> {
  if (!strategyId || !/^[A-Za-z0-9_-]+$/.test(strategyId)) return APPROVAL_FILE;
  const scoped = path.join(PROD_DIR, `approval_state_${strategyId}.json`);
  try {
    await fs.access(scoped);
    return scoped;
  } catch {
    // No per-strategy file: use the singleton only if it IS this strategy.
    try {
      const raw = await fs.readFile(APPROVAL_FILE, 'utf-8');
      const singleton = JSON.parse(raw) as { strategy?: string };
      if (singleton.strategy === strategyId) return APPROVAL_FILE;
    } catch { /* singleton unreadable — fall through to the scoped path so the 404 is honest */ }
    return scoped;
  }
}

export async function POST(request: NextRequest) {
  try {
    // Vote 2/2 is a privileged action that fire-and-forget triggers a production
    // retrain+deploy. It MUST be authenticated — an unauthenticated caller must
    // not be able to approve or trigger a deploy (audit A4-13).
    const auth = await protectApiRoute(request);
    if (!auth.authenticated) {
      return NextResponse.json(
        { success: false, status: 'PENDING_APPROVAL', message: auth.error || 'Unauthorized' } as ApproveResponse,
        { status: auth.status || 401 }
      );
    }

    const body: ApproveRequest & { strategy_id?: string } = await request.json();
    const approvalFile = await approvalFileFor(body.strategy_id);

    if (!body.action || !['APPROVE', 'REJECT'].includes(body.action)) {
      return NextResponse.json(
        { success: false, status: 'PENDING_APPROVAL', message: 'Invalid action. Must be APPROVE or REJECT.' } as ApproveResponse,
        { status: 400 }
      );
    }

    // Read current state
    let state: ApprovalState;
    try {
      const raw = await fs.readFile(approvalFile, 'utf-8');
      state = JSON.parse(raw);
    } catch {
      return NextResponse.json(
        { success: false, status: 'PENDING_APPROVAL', message: 'No approval state file found. Run backtest first.' } as ApproveResponse,
        { status: 404 }
      );
    }

    // Can only approve/reject from PENDING_APPROVAL
    if (state.status !== 'PENDING_APPROVAL') {
      return NextResponse.json(
        { success: false, status: state.status, message: `Cannot ${body.action.toLowerCase()} — current status is ${state.status}` } as ApproveResponse,
        { status: 409 }
      );
    }

    const now = new Date().toISOString();
    // Record the AUTHENTICATED principal, never a client-supplied name (audit A4-13).
    const reviewer = auth.user?.email || auth.user?.username || auth.user?.id || 'operator';

    if (body.action === 'APPROVE') {
      state.status = 'APPROVED';
      state.approved_by = reviewer;
      state.approved_at = now;
      state.reviewer_notes = body.notes || '';
    } else {
      state.status = 'REJECTED';
      state.rejected_by = reviewer;
      state.rejected_at = now;
      state.rejection_reason = body.notes || '';
    }

    state.last_updated = now;

    // Write back
    await fs.writeFile(approvalFile, JSON.stringify(state, null, 2), 'utf-8');

    const response: ApproveResponse = {
      success: true,
      status: state.status,
      message: body.action === 'APPROVE'
        ? `Strategy approved by ${reviewer}. Deploy starting automatically...`
        : `Strategy rejected by ${reviewer}. Daily runner will remain in dry-run mode.`,
    };

    // Auto-trigger deploy after APPROVE (fire-and-forget)
    if (body.action === 'APPROVE') {
      try {
        const baseUrl = request.nextUrl.origin;
        // Forward the caller's session cookie so the internal deploy call is
        // authenticated as the same approving principal (deploy is auth-gated).
        fetch(`${baseUrl}/api/production/deploy`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            cookie: request.headers.get('cookie') ?? '',
          },
          body: JSON.stringify({ strategy_id: body.strategy_id ?? null }),
        }).catch(() => {
          // Non-blocking: deploy failure doesn't affect approval
        });
      } catch {
        // Non-blocking: if fetch itself throws, approval still succeeds
      }
    }

    return NextResponse.json(response);
  } catch (error) {
    console.error('Error processing approval:', error);
    return NextResponse.json(
      { success: false, status: 'PENDING_APPROVAL', message: 'Internal server error' } as ApproveResponse,
      { status: 500 }
    );
  }
}

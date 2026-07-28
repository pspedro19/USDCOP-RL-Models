/**
 * POST /api/production/approve — Approve or reject the production strategy
 *
 * File-based: reads + writes `<repo>/data/approvals/approval_state*.json` (CXD-057 —
 * the artifact moved OUT of `public/` because it carries gates / DSR / backtest metrics,
 * which the SSOT reserves to `research:read`; under `public/` the static `/data/**`
 * path served it to any session). Store SSOT: `lib/approvals/store.ts`.
 *
 * This is the human "second vote" in the 2-vote promotion system and it is UNCHANGED:
 * same surface (`/dashboard`), same permission (`approval:vote`), same semantics —
 * only the file location moved.
 *
 * Body: { action: 'APPROVE' | 'REJECT', notes?: string, reviewer?: string }
 */
import { NextRequest, NextResponse } from 'next/server';
import { protectApiRoute } from '@/lib/auth/api-auth';
import { readApprovalState, writeApprovalState } from '@/lib/approvals/store';
import type { ApprovalState, ApproveRequest, ApproveResponse } from '@/lib/contracts/production-approval.contract';

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

    if (!body.action || !['APPROVE', 'REJECT'].includes(body.action)) {
      return NextResponse.json(
        { success: false, status: 'PENDING_APPROVAL', message: 'Invalid action. Must be APPROVE or REJECT.' } as ApproveResponse,
        { status: 400 }
      );
    }

    // Read current state from the PRIVATE store. The per-strategy → singleton fallback
    // (load-bearing: the export pipeline writes the ACTIVE strategy unsuffixed while the
    // dashboard always posts a strategy_id) lives in `readApprovalState` and is shared
    // verbatim with the H5-L4b deploy DAG — they must agree or approval succeeds and
    // deploy 404s.
    const record = await readApprovalState(body.strategy_id ?? null);
    if (!record) {
      return NextResponse.json(
        { success: false, status: 'PENDING_APPROVAL', message: 'No approval state file found. Run backtest first.' } as ApproveResponse,
        { status: 404 }
      );
    }
    const approvalFile = record.file;
    const state: ApprovalState = record.state;

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
    await writeApprovalState(approvalFile, state);

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

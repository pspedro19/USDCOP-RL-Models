'use client';

/**
 * /passport — Control Tower + Strategy Passport (BL-32, FABRIC §24.4-§24.5).
 *
 * READ-ONLY DIAGNOSTIC surface: it shows identity, governance, lineage, performance
 * across five environments, live state and risk. It does NOT approve, promote, deploy
 * or execute anything — Vote 2 lives exclusively on /dashboard (approval-gates.md
 * invariante 3), and that separation is enforced by the contract, not by convention.
 *
 * RBAC: `research:read` (admin/developer) via PAGE_ROUTES in rbac.contract.ts.
 * Spec: .claude/specs/platform/passport-control-tower.md
 */
import { TerminalShell } from '@/components/gm';
import { PassportView } from '@/components/gm/views/PassportView';

export default function PassportPage() {
  return (
    <TerminalShell active="passport" width="wide">
      <PassportView />
    </TerminalShell>
  );
}

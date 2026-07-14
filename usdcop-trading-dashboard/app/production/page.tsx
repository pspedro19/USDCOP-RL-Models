'use client';

/**
 * /production — Producción / Señales (GlobalMarkets Terminal, CTR-GM-UI-001).
 * La vista vive en components/gm/views/ProductionView.tsx; la versión anterior
 * quedó preservada en /legacy/production (components/legacy/ProductionLegacy.tsx).
 *
 * Spec: .claude/specs/platform/dashboard-integration.md
 * Contratos: lib/contracts/{strategy,production-approval,production-monitor}.contract.ts
 */
import { TerminalShell } from '@/components/gm';
import { ProductionView } from '@/components/gm/views/ProductionView';

export default function ProductionPage() {
  return (
    <TerminalShell active="production">
      <ProductionView />
    </TerminalShell>
  );
}

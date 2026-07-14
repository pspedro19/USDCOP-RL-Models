'use client';

/**
 * /analysis — Weekly Analysis Dashboard (SDD-08) under the GlobalMarkets chrome
 * (CTR-GM-UI-001: TerminalShell replaces the legacy GlobalNavbar; the analysis
 * content itself keeps its as-built components until its own re-skin increment).
 *
 * Spec: .claude/specs/tracks/news-analysis/ · Contract: lib/contracts/weekly-analysis.contract.ts
 */

import { Suspense, lazy } from 'react';

import { AnalysisPage } from '@/components/analysis/AnalysisPage';
import { GmPageHeader } from '@/components/gm';
import { TerminalShell } from '@/components/gm/TerminalShell';

const FloatingChatWidget = lazy(() =>
  import('@/components/analysis/FloatingChatWidget').then(m => ({ default: m.FloatingChatWidget }))
);

export default function AnalysisRoute() {
  return (
    <TerminalShell active="analysis" width="wide">
      <GmPageHeader
        kicker="Inteligencia de mercado"
        title="Análisis semanal"
        subtitle="Análisis AI multi-activo (USD/COP, Oro, Bitcoin) con indicadores macro, señales y timeline diario"
      />
      <AnalysisPage />
      <Suspense fallback={null}>
        <FloatingChatWidget />
      </Suspense>
    </TerminalShell>
  );
}

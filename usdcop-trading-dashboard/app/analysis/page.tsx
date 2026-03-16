'use client';

/**
 * /analysis — Weekly Analysis Dashboard (SDD-08)
 * ================================================
 * AI-powered weekly/daily analysis with macro charts, daily timeline,
 * signal summaries, and economic calendar.
 *
 * Spec: .claude/Featurechatnews/08_dashboard_frontend_spec.md
 * Contract: lib/contracts/weekly-analysis.contract.ts
 */

import { Suspense, lazy } from 'react';
import { GlobalNavbar } from '@/components/navigation/GlobalNavbar';
import { AnalysisPage } from '@/components/analysis/AnalysisPage';
import { RefreshCw } from 'lucide-react';

const FloatingChatWidget = lazy(() =>
  import('@/components/analysis/FloatingChatWidget').then(m => ({ default: m.FloatingChatWidget }))
);

export default function AnalysisRoute() {
  return (
    <div className="min-h-screen bg-[#030712]">
      <GlobalNavbar currentPage="analysis" />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-12">
        {/* Page header */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-white">Analisis Semanal</h1>
          <p className="text-sm text-gray-500 mt-1">
            Analisis AI del USD/COP con indicadores macro, señales y timeline diario
          </p>
        </div>

        <AnalysisPage />
      </main>

      {/* Floating chat */}
      <Suspense fallback={null}>
        <FloatingChatWidget />
      </Suspense>
    </div>
  );
}

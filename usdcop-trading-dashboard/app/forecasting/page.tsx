'use client';

/**
 * /forecasting — GlobalMarkets Terminal (CTR-GM-UI-001).
 * Monta la vista GM (components/gm/views/ForecastingView) dentro del shell.
 * La página anterior está archivada en /legacy/forecasting (admin-only).
 *
 * Suspense: ForecastingView usa useSearchParams (estado de filtros en la URL),
 * que en Next 15 exige un boundary durante el prerender.
 */

import { Suspense } from 'react';

import { GmSkeleton } from '@/components/gm';
import { TerminalShell } from '@/components/gm/TerminalShell';
import { ForecastingView } from '@/components/gm/views/ForecastingView';

export default function ForecastingPage() {
  return (
    <TerminalShell active="forecasting">
      <Suspense fallback={<GmSkeleton label="Cargando forecasting…" />}>
        <ForecastingView />
      </Suspense>
    </TerminalShell>
  );
}

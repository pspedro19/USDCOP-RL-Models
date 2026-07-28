'use client';

/**
 * Dashboard Page — vista BACKTEST con APROBACIÓN (GlobalMarkets Terminal)
 * ========================================================================
 * Superficie de decisión: incluye el flujo Voto 2/2 intacto (admin-only,
 * approval-gates.md) — los botones de aprobación viven SOLO en esta ruta.
 * Contenido compartido con /replay (variante read-only) en
 * `components/production/BacktestTerminalPage.tsx` (BL-34, DRY).
 */

import { BacktestTerminalPage } from '@/components/production/BacktestTerminalPage';

export default function DashboardPage() {
  return <BacktestTerminalPage variant="dashboard" />;
}

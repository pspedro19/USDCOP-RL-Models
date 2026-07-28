'use client';

/**
 * /replay — investigación READ-ONLY del backtest (BL-34, plan 00 §2 / FABRIC §24.2)
 * ==================================================================================
 * Remediación del rechazo Codex (2026-07-27): la versión anterior re-exportaba
 * /dashboard completo y exhibía los botones admin Aprobar/Rechazar. Ahora monta la
 * MISMA sección (selector estrategia/versión, candle+replay, equity, gates Voto 1)
 * en variante `replay` (readOnly):
 *
 *  - CERO superficie de Vote-2 (ni Aprobar/Rechazar ni deploy, ni siquiera para
 *    admin) — los botones de aprobación viven exclusivamente en /dashboard
 *    (approval-gates.md invariante 3).
 *  - Las cifras recomputadas por el replay se etiquetan PREVIEW; el Voto 2 y los
 *    gates deciden sobre el bundle publicado (quant-constitution §7).
 *  - RBAC: research:read en el edge (C-002); el deploy re-valida server-side
 *    (`/api/production/approve` exige `approval:vote`) — la UI nunca es la autoridad.
 *
 * Pruebas: tests/unit/middleware-replay-authz.test.ts (autorización server-side) +
 * tests/e2e/replay-read-only.spec.ts (Playwright real).
 */

import { BacktestTerminalPage } from '@/components/production/BacktestTerminalPage';

export default function ReplayPage() {
  return <BacktestTerminalPage variant="replay" />;
}

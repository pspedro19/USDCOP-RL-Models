/**
 * /replay — alias de navegación del Backtest (BL-34, plan 00 §2 / FABRIC §24.2)
 * ============================================================================
 * Monta EXACTAMENTE la misma página que /dashboard (ForecastingBacktestSection:
 * selector estrategia/versión, candle+replay, equity, gates Voto 1). Cero lógica
 * nueva — es un alias de nomenclatura hacia la constitución.
 *
 * Vote-2 permanece en /dashboard (approval-gates.md): el panel de aprobación es
 * ADMIN-ONLY dentro de ForecastingBacktestSection (`canPromote`), los no-admin
 * ven /replay como lectura pura, y el deploy re-valida server-side
 * (`/api/production/approve` exige `approval:vote`) — la UI nunca es la autoridad.
 */
export { default } from '../dashboard/page';

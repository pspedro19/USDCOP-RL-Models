/**
 * Real-Time Risk Engine Service
 * ==============================
 *
 * DEPRECATED: This file is maintained for backwards compatibility only.
 *
 * The risk engine has been refactored into modular components following
 * the Single Responsibility Principle. Please use the new module structure:
 *
 *   import { realTimeRiskEngine } from '@/lib/services/risk';
 *
 * New modular structure in lib/services/risk/:
 * - PortfolioTracker.ts       - Position tracking
 * - RiskMetricsCalculator.ts  - Risk metric calculations
 * - RiskAlertSystem.ts        - Alert generation and management
 * - RealTimeRiskEngine.ts     - Orchestrator component
 * - types.ts                  - Type definitions
 * - index.ts                  - Public API exports
 *
 * Migration guide:
 * 1. Replace imports from './real-time-risk-engine' with '@/lib/services/risk'
 * 2. API remains the same - no code changes needed
 * 3. Test thoroughly after migration
 * 4. Remove this file once all imports are updated
 */

// Re-export everything from the new modular structure
export type {
  Position,
  RiskAlert,
  RealTimeRiskMetrics,
  RiskMetrics,
} from './risk';

export {
  realTimeRiskEngine,
  getRiskMetrics,
} from './risk';

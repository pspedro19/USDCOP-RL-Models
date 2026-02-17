/**
 * Production Approval Contract
 * ==============================
 * Types for the file-based 2-vote approval system.
 * Vote 1: Backtest validates 5 gates (automatic).
 * Vote 2: User clicks Approve (human review).
 *
 * Strategy types are imported from the universal strategy.contract.ts (SDD).
 *
 * Shared contract between:
 * - Python: train_and_export_smart_simple.py writes approval_state.json
 * - Next.js: /api/production/status reads, /api/production/approve writes
 * - Dashboard: /production page renders gates + approval panel
 */

import type {
  StrategyTrade,
  StrategyStats,
  StrategySummary,
  StrategyTradeFile,
} from './strategy.contract';

// Re-export universal types as production aliases
export type ProductionTrade = StrategyTrade;
export type ProductionTradeFile = StrategyTradeFile;
export type ProductionSummary = StrategySummary;
export type { StrategyStats };

// -----------------------------------------------------------------------------
// Approval Status
// -----------------------------------------------------------------------------

export type ProductionStatus = 'PENDING_APPROVAL' | 'APPROVED' | 'REJECTED' | 'LIVE';

// -----------------------------------------------------------------------------
// Gate Result (one of the 5 backtest gates)
// -----------------------------------------------------------------------------

export interface GateResult {
  gate: string;        // e.g. "min_return_pct"
  label: string;       // e.g. "Retorno Minimo"
  passed: boolean;
  value: number;       // actual metric value
  threshold: number;   // gate threshold
}

// -----------------------------------------------------------------------------
// Approval State (persisted as JSON file)
// -----------------------------------------------------------------------------

export interface ApprovalState {
  status: ProductionStatus;
  strategy: string;                          // e.g. "smart_simple_v11"
  strategy_name?: string;                    // e.g. "Smart Simple v1.1.0"
  backtest_recommendation: 'PROMOTE' | 'REJECT' | 'REVIEW';
  backtest_confidence: number;               // fraction of gates passed (0-1)
  gates: GateResult[];

  // Deploy manifest: written by Python during --phase backtest
  // Tells the deploy API which script/args to run for production
  deploy_manifest?: DeployManifest;

  // Set on APPROVED
  approved_by?: string;
  approved_at?: string;
  reviewer_notes?: string;

  // Set on REJECTED
  rejected_by?: string;
  rejected_at?: string;
  rejection_reason?: string;

  created_at: string;
  last_updated: string;
}

// -----------------------------------------------------------------------------
// API Request/Response
// -----------------------------------------------------------------------------

export interface ApproveRequest {
  action: 'APPROVE' | 'REJECT';
  notes?: string;
  reviewer?: string;
}

export interface ApproveResponse {
  success: boolean;
  status: ProductionStatus;
  message: string;
}

// -----------------------------------------------------------------------------
// Deploy Manifest (embedded in approval_state.json by Python backtest scripts)
// -----------------------------------------------------------------------------

export interface DeployManifest {
  pipeline_type: 'ml_forecasting' | 'rl';
  script: string;           // Relative to project root
  args: string[];            // CLI args for deploy
  config_path: string;       // SSOT config used during backtest
  db_tables: string[];       // Tables to seed on deploy
}

// -----------------------------------------------------------------------------
// Deploy Types (One-Click Production Deploy)
// -----------------------------------------------------------------------------

export type DeployPhase = 'initializing' | 'retraining' | 'exporting' | 'seeding_db' | 'done';

export interface DeployStatus {
  status: 'idle' | 'running' | 'completed' | 'failed';
  strategy_id?: string;
  strategy_name?: string;
  started_at?: string;
  completed_at?: string;
  error?: string;
  phase?: DeployPhase;
  pid?: number;
}

export interface DeployRequest {
  strategy_id?: string;
}

export interface DeployResponse {
  success: boolean;
  status: DeployStatus['status'];
  message: string;
}

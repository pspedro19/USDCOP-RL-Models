/**
 * Experiments Contract
 * ====================
 * Types and interfaces for experiment approval workflow.
 */

// ============================================================================
// Experiment Status
// ============================================================================

export type ExperimentStatus = 'PENDING_APPROVAL' | 'APPROVED' | 'REJECTED' | 'EXPIRED';

export type PromotionRecommendation = 'PROMOTE' | 'REJECT' | 'REVIEW';

// ============================================================================
// Experiment Types
// ============================================================================

export interface Experiment {
  id: number;
  proposalId: string;
  modelId: string;
  experimentName: string;
  recommendation: PromotionRecommendation;
  confidence: number;
  reason: string;
  metrics: ExperimentMetrics;
  vsBaseline: BaselineComparison;
  criteriaResults: CriteriaResult[];
  lineage: ExperimentLineage;
  status: ExperimentStatus;
  reviewer?: string;
  reviewerEmail?: string;
  reviewerNotes?: string;
  reviewedAt?: string;
  createdAt: string;
  expiresAt?: string;
  hoursUntilExpiry?: number | null;
  auditLog?: AuditLogEntry[];
}

export interface ExperimentMetrics {
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  avgTradeReturn: number;
  sortinoRatio?: number;
  calmarRatio?: number;
}

export interface BaselineComparison {
  returnDelta: number;
  sharpeDelta: number;
  drawdownDelta: number;
  winRateDelta: number;
}

export interface CriteriaResult {
  criterion: string;
  passed: boolean;
  value: number;
  threshold: number;
  weight: number;
}

export interface ExperimentLineage {
  configHash: string;
  featureOrderHash: string;
  modelHash: string;
  datasetHash: string;
  normStatsHash: string;
  rewardConfigHash: string;
  modelPath?: string;
  trainingStart?: string;
  trainingEnd?: string;
}

export interface AuditLogEntry {
  action: 'APPROVE' | 'REJECT' | 'CREATE';
  reviewer: string;
  notes?: string;
  createdAt: string;
}

// ============================================================================
// API Request/Response Types
// ============================================================================

export interface ExperimentsListResponse {
  experiments: Experiment[];
  total: number;
  statusCounts: Record<ExperimentStatus, number>;
}

export interface PendingExperimentsResponse {
  pending: Experiment[];
  count: number;
}

export interface ExperimentDetailResponse {
  experiment: Experiment;
}

export interface ApproveRequest {
  notes?: string;
  promoteToProduction?: boolean;
}

export interface RejectRequest {
  notes?: string;
  reason?: string;
}

export interface ApproveResponse {
  success: boolean;
  modelId: string;
  newStage: string;
  previousModelArchived?: string;
  approvedBy: string;
  message: string;
}

export interface RejectResponse {
  success: boolean;
  modelId: string;
  status: string;
  rejectedBy: string;
  message: string;
}

// ============================================================================
// Helpers
// ============================================================================

export function getRecommendationColor(rec: PromotionRecommendation): string {
  switch (rec) {
    case 'PROMOTE': return 'text-green-400 bg-green-500/20 border-green-500/50';
    case 'REJECT': return 'text-red-400 bg-red-500/20 border-red-500/50';
    case 'REVIEW': return 'text-amber-400 bg-amber-500/20 border-amber-500/50';
  }
}

export function getStatusColor(status: ExperimentStatus): string {
  switch (status) {
    case 'PENDING_APPROVAL': return 'text-amber-400 bg-amber-500/20';
    case 'APPROVED': return 'text-green-400 bg-green-500/20';
    case 'REJECTED': return 'text-red-400 bg-red-500/20';
    case 'EXPIRED': return 'text-gray-400 bg-gray-500/20';
  }
}

export function formatMetricDelta(delta: number): { text: string; isPositive: boolean } {
  const sign = delta >= 0 ? '+' : '';
  return {
    text: `${sign}${(delta * 100).toFixed(2)}%`,
    isPositive: delta >= 0,
  };
}

export function getCriterionIcon(passed: boolean): string {
  return passed ? '✓' : '✗';
}

export function formatExpiryTime(hoursUntilExpiry: number | null): string {
  if (hoursUntilExpiry === null) return 'No expiry';
  if (hoursUntilExpiry <= 0) return 'Expired';
  if (hoursUntilExpiry < 1) return `${Math.round(hoursUntilExpiry * 60)}m left`;
  if (hoursUntilExpiry < 24) return `${Math.round(hoursUntilExpiry)}h left`;
  return `${Math.round(hoursUntilExpiry / 24)}d left`;
}

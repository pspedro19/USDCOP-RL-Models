/**
 * Model Contract
 * ==============
 * Single Source of Truth for model types.
 *
 * These types MUST mirror the model_registry table schema in PostgreSQL.
 * Any changes to the DB schema should be reflected here.
 *
 * SOLID Principles Applied:
 * - Single Responsibility: Only model-related types
 * - Open/Closed: Extensible via generics, closed for modification
 * - Interface Segregation: Separate interfaces for different use cases
 */

// ============================================================================
// Core Types (mirror model_registry table)
// ============================================================================

/**
 * Model status from model_registry.status
 * CHECK constraint: status IN ('registered', 'deployed', 'retired')
 */
export type ModelStatus = 'registered' | 'deployed' | 'retired';

/**
 * Core model identity from model_registry
 * These fields uniquely identify a model
 */
export interface ModelIdentity {
  /** Primary identifier - model_registry.model_id (UNIQUE) */
  model_id: string;
  /** Version string - model_registry.model_version */
  model_version: string;
  /** Current status */
  status: ModelStatus;
}

/**
 * Model metadata from model_registry
 * Technical configuration for inference
 */
export interface ModelMetadata {
  /** Path to model file */
  model_path: string;
  /** SHA256 hash for integrity */
  model_hash: string;
  /** Observation space dimension */
  observation_dim: number;
  /** Action space size (typically 3: BUY, SELL, HOLD) */
  action_space: number;
  /** Ordered list of feature names */
  feature_order: string[];
}

/**
 * Model performance metrics from model_registry
 * Backtest/validation results
 */
export interface ModelPerformance {
  test_sharpe?: number;
  test_max_drawdown?: number;
  test_win_rate?: number;
  validation_metrics?: Record<string, unknown>;
}

/**
 * Model temporal info from model_registry
 */
export interface ModelTimestamps {
  created_at: string;
  deployed_at?: string;
  retired_at?: string;
  training_start_date?: string;
  training_end_date?: string;
}

/**
 * Complete model record from model_registry table
 * This is the canonical representation
 */
export interface ModelRegistryRecord extends
  ModelIdentity,
  ModelMetadata,
  ModelPerformance,
  ModelTimestamps {
  /** Auto-increment primary key */
  id: number;
  /** Normalization stats hash */
  norm_stats_hash: string;
  /** Config hash for reproducibility */
  config_hash?: string;
  /** Reference to training dataset */
  training_dataset_id?: number;
}

// ============================================================================
// Frontend Types (derived from ModelRegistryRecord)
// ============================================================================

/**
 * Model configuration for frontend display
 * Derived from ModelRegistryRecord with UI-specific fields
 */
export interface FrontendModelConfig {
  /** model_registry.model_id - used as primary key in frontend */
  id: string;
  /** Display name (derived from model_id + version) */
  name: string;
  /** Algorithm type (extracted from model_id, e.g., 'PPO') */
  algorithm: string;
  /** Version for display */
  version: string;
  /** UI status: 'production' | 'testing' | 'deprecated' */
  displayStatus: 'production' | 'testing' | 'deprecated';
  /** Actual DB status */
  dbStatus: ModelStatus;
  /** Model type */
  type: 'rl' | 'ml' | 'ensemble';
  /** UI color for charts/badges */
  color: string;
  /** Human-readable description */
  description: string;
  /** Whether model uses real production data */
  isRealData: boolean;
  /** Observation dimension */
  observationDim: number;
  /** Backtest metrics for display */
  backtest?: {
    sharpe: number;
    maxDrawdown: number;
    winRate: number;
    totalTrades?: number;
  };
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Transform ModelRegistryRecord to FrontendModelConfig
 * Factory function that derives UI config from DB record
 */
export function createFrontendModelConfig(record: ModelRegistryRecord): FrontendModelConfig {
  const algorithm = extractAlgorithm(record.model_id);
  const version = record.model_version.toUpperCase();

  return {
    id: record.model_id,
    name: formatModelName(record.model_id, version, record.status),
    algorithm,
    version,
    displayStatus: mapStatus(record.status),
    dbStatus: record.status,
    type: algorithm === 'PPO' || algorithm === 'SAC' ? 'rl' : 'ml',
    color: getModelColor(algorithm, version),
    description: `${algorithm} model ${version} - ${record.observation_dim}-dim observation`,
    isRealData: record.status === 'deployed',
    observationDim: record.observation_dim,
    backtest: record.test_sharpe ? {
      sharpe: record.test_sharpe,
      maxDrawdown: record.test_max_drawdown || 0,
      winRate: record.test_win_rate || 0,
    } : undefined,
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

function extractAlgorithm(modelId: string): string {
  const id = modelId.toLowerCase();
  if (id.includes('ppo')) return 'PPO';
  if (id.includes('sac')) return 'SAC';
  if (id.includes('td3')) return 'TD3';
  if (id.includes('a2c')) return 'A2C';
  if (id.includes('dqn')) return 'DQN';
  if (id.includes('lgbm')) return 'LGBM';
  if (id.includes('xgb')) return 'XGB';
  return 'Unknown';
}

function formatModelName(modelId: string, version: string, status: ModelStatus): string {
  const algorithm = extractAlgorithm(modelId);
  const statusLabel = status === 'deployed' ? '(Production)'
    : status === 'retired' ? '(Retired)'
    : '(Testing)';
  return `${algorithm} ${version} ${statusLabel}`;
}

function mapStatus(dbStatus: ModelStatus): 'production' | 'testing' | 'deprecated' {
  switch (dbStatus) {
    case 'deployed': return 'production';
    case 'retired': return 'deprecated';
    default: return 'testing';
  }
}

function getModelColor(algorithm: string, version: string): string {
  // Color by algorithm first
  const algorithmColors: Record<string, string> = {
    'PPO': '#10B981',  // Emerald
    'SAC': '#8B5CF6',  // Violet
    'TD3': '#F59E0B',  // Amber
    'A2C': '#EF4444',  // Red
    'LGBM': '#3B82F6', // Blue
    'XGB': '#EC4899',  // Pink
  };

  return algorithmColors[algorithm] || '#6B7280'; // Gray default
}

// ============================================================================
// Validation
// ============================================================================

/**
 * Validate that a model_id exists and is usable
 */
export function isValidModelId(modelId: string | null | undefined): modelId is string {
  return typeof modelId === 'string' && modelId.length > 0;
}

/**
 * Get default model ID (should be fetched from API, but this is the fallback)
 */
export const DEFAULT_MODEL_ID = 'ppo_primary';

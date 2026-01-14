/**
 * Model Service
 * =============
 * Service layer for model operations.
 *
 * Follows Clean Architecture:
 * - This is the Application layer service
 * - Uses contracts from lib/contracts
 * - Abstracts data source (API/DB)
 *
 * SOLID Principles:
 * - Single Responsibility: Only model-related operations
 * - Dependency Inversion: Depends on abstractions (contracts)
 */

import {
  FrontendModelConfig,
  ModelRegistryRecord,
  ModelStatus,
  createFrontendModelConfig,
  DEFAULT_MODEL_ID,
  isValidModelId,
} from '@/lib/contracts/model.contract';

// ============================================================================
// Types
// ============================================================================

export interface ModelServiceResponse {
  models: FrontendModelConfig[];
  defaultModelId: string;
  source: 'database' | 'inference-service' | 'fallback';
}

export interface ModelServiceError {
  message: string;
  code: 'NETWORK_ERROR' | 'INVALID_RESPONSE' | 'NO_MODELS';
}

// ============================================================================
// Service Implementation
// ============================================================================

/**
 * Fetch available models from the API
 * Returns models from model_registry via the backend
 */
export async function fetchAvailableModels(): Promise<ModelServiceResponse> {
  try {
    const response = await fetch('/api/models', {
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();

    if (!data.models || !Array.isArray(data.models) || data.models.length === 0) {
      throw new Error('No models returned');
    }

    // Find default model (deployed/production)
    const defaultModel = data.models.find(
      (m: FrontendModelConfig) => m.displayStatus === 'production' || m.dbStatus === 'deployed'
    );

    return {
      models: data.models,
      defaultModelId: data.defaultModel || defaultModel?.id || data.models[0]?.id || DEFAULT_MODEL_ID,
      source: data.source || 'database',
    };
  } catch (error) {
    console.warn('[ModelService] Failed to fetch models, using fallback');
    return getFallbackModels();
  }
}

/**
 * Get a specific model by ID
 */
export async function getModelById(modelId: string): Promise<FrontendModelConfig | null> {
  if (!isValidModelId(modelId)) {
    return null;
  }

  const { models } = await fetchAvailableModels();
  return models.find(m => m.id === modelId) || null;
}

/**
 * Get the default/production model
 */
export async function getDefaultModel(): Promise<FrontendModelConfig | null> {
  const { models, defaultModelId } = await fetchAvailableModels();
  return models.find(m => m.id === defaultModelId) || models[0] || null;
}

/**
 * Check if a model ID is valid (exists in registry)
 */
export async function validateModelId(modelId: string): Promise<boolean> {
  const model = await getModelById(modelId);
  return model !== null;
}

// ============================================================================
// Fallback Data
// ============================================================================

/**
 * Fallback models when API is unavailable
 * These should match what's in model_registry
 */
function getFallbackModels(): ModelServiceResponse {
  const fallbackModels: FrontendModelConfig[] = [
    {
      id: 'ppo_v20',
      name: 'PPO V20 (Production)',
      algorithm: 'PPO',
      version: 'V20',
      displayStatus: 'production',
      dbStatus: 'deployed',
      type: 'rl',
      color: '#10B981',
      description: 'PPO model V20 - 15-dim observation',
      isRealData: true,
      observationDim: 15,
      backtest: {
        sharpe: 1.19,
        maxDrawdown: 7.96,
        winRate: 49.2,
      },
    },
    {
      id: 'ppo_v1',
      name: 'PPO V1 (Testing)',
      algorithm: 'PPO',
      version: 'V1',
      displayStatus: 'testing',
      dbStatus: 'registered',
      type: 'rl',
      color: '#06B6D4',
      description: 'PPO model V1 - 32-dim observation',
      isRealData: true,
      observationDim: 32,
    },
  ];

  return {
    models: fallbackModels,
    defaultModelId: 'ppo_v20',
    source: 'fallback',
  };
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get model display color by ID
 * Uses the model's configured color or derives from algorithm
 */
export function getModelDisplayColor(modelId: string, models: FrontendModelConfig[]): string {
  const model = models.find(m => m.id === modelId);
  return model?.color || '#6B7280';
}

/**
 * Filter models by status
 */
export function filterModelsByStatus(
  models: FrontendModelConfig[],
  status: 'production' | 'testing' | 'deprecated' | 'all'
): FrontendModelConfig[] {
  if (status === 'all') return models;
  return models.filter(m => m.displayStatus === status);
}

/**
 * Sort models: production first, then by name
 */
export function sortModels(models: FrontendModelConfig[]): FrontendModelConfig[] {
  return [...models].sort((a, b) => {
    // Production first
    if (a.displayStatus === 'production' && b.displayStatus !== 'production') return -1;
    if (b.displayStatus === 'production' && a.displayStatus !== 'production') return 1;
    // Then by name
    return a.name.localeCompare(b.name);
  });
}

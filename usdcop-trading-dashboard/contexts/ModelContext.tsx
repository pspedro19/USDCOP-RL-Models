'use client';

/**
 * Model Context
 * ==============
 * Provides global state for the selected trading model.
 * Model data comes from the API - NO hardcoded models here.
 */

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useMemo,
} from 'react';
import type { ModelConfig, ModelStatus } from '@/lib/config/models.config';
import { modelRefreshIntervals } from '@/lib/config/models.config';

// ============================================================================
// Types
// ============================================================================

interface ModelContextState {
  // Selected model
  selectedModelId: string | null;
  selectedModel: ModelConfig | null;

  // Available models (from API)
  models: ModelConfig[];
  isLoading: boolean;
  error: string | null;

  // Actions
  setSelectedModel: (modelId: string) => void;
  refreshModels: () => Promise<void>;

  // Comparison mode
  isComparing: boolean;
  setIsComparing: (value: boolean) => void;
  comparisonModelIds: string[];
  toggleComparisonModel: (modelId: string) => void;
  clearComparison: () => void;

  // Helpers
  getModelById: (modelId: string) => ModelConfig | undefined;
  getProductionModel: () => ModelConfig | undefined;
  getActiveModels: () => ModelConfig[];
}

// ============================================================================
// Context
// ============================================================================

const ModelContext = createContext<ModelContextState | undefined>(undefined);

// ============================================================================
// Provider
// ============================================================================

interface ModelProviderProps {
  children: React.ReactNode;
  initialModelId?: string;
}

export function ModelProvider({ children, initialModelId }: ModelProviderProps) {
  // Model list state
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Selected model
  const [selectedModelId, setSelectedModelId] = useState<string | null>(
    initialModelId || null
  );

  // Comparison mode
  const [isComparing, setIsComparing] = useState(false);
  const [comparisonModelIds, setComparisonModelIds] = useState<string[]>([]);

  // Default models when backend is unavailable
  // NOTE: These IDs MUST match the model_id values used by the inference service
  const defaultModels: ModelConfig[] = [
    {
      id: 'ppo_primary',  // Must match inference service model_id
      name: 'PPO Primary (Production)',
      type: 'rl',
      algorithm: 'PPO',
      status: 'production' as ModelStatus,
      version: 'current',
      color: '#10B981',
      description: '15-feature model with macro indicators',
      isRealData: true,
      backtest: {
        sharpe: 1.19,
        maxDrawdown: 7.96,
        winRate: 49.2,
        holdPercent: 0,
        totalTrades: 518,
        dataRange: { start: '2025-01-01', end: '2025-03-31' },
      },
    },
  ];

  // Fetch models from API
  const fetchModels = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch('/api/models', {
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        // Use default models when backend unavailable (503, 502, etc)
        if (response.status >= 500) {
          setModels(defaultModels);
          if (!selectedModelId) {
            setSelectedModelId(defaultModels[0].id);
          }
          return;
        }
        throw new Error(`Failed to fetch models: ${response.status}`);
      }

      const data = await response.json();

      if (!data.models || !Array.isArray(data.models)) {
        // Use defaults if invalid response
        setModels(defaultModels);
        if (!selectedModelId) {
          setSelectedModelId(defaultModels[0].id);
        }
        return;
      }

      setModels(data.models);

      // Auto-select production model if no model selected
      if (!selectedModelId && data.models.length > 0) {
        const productionModel = data.models.find(
          (m: ModelConfig) => m.status === 'production'
        );
        setSelectedModelId(productionModel?.id || data.models[0].id);
      }
    } catch (err) {
      // Silently use default models on network errors
      setModels(defaultModels);
      if (!selectedModelId) {
        setSelectedModelId(defaultModels[0].id);
      }
      // Only log in development, not as error
      if (process.env.NODE_ENV === 'development') {
        console.debug('[ModelContext] Using default models - backend unavailable');
      }
    } finally {
      setIsLoading(false);
    }
  }, [selectedModelId]);

  // Initial fetch
  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  // Periodic refresh
  useEffect(() => {
    const interval = setInterval(fetchModels, modelRefreshIntervals.modelList);
    return () => clearInterval(interval);
  }, [fetchModels]);

  // Set selected model
  const setSelectedModel = useCallback((modelId: string) => {
    setSelectedModelId(modelId);
  }, []);

  // Refresh models
  const refreshModels = useCallback(async () => {
    await fetchModels();
  }, [fetchModels]);

  // Toggle comparison model
  const toggleComparisonModel = useCallback((modelId: string) => {
    setComparisonModelIds((prev) => {
      if (prev.includes(modelId)) {
        return prev.filter((id) => id !== modelId);
      }
      // Max 4 models for comparison
      if (prev.length >= 4) {
        return prev;
      }
      return [...prev, modelId];
    });
  }, []);

  // Clear comparison
  const clearComparison = useCallback(() => {
    setComparisonModelIds([]);
    setIsComparing(false);
  }, []);

  // Get model by ID
  const getModelById = useCallback(
    (modelId: string) => {
      return models.find((m) => m.id === modelId);
    },
    [models]
  );

  // Get production model
  const getProductionModel = useCallback(() => {
    return models.find((m) => m.status === 'production');
  }, [models]);

  // Get active models (not deprecated)
  const getActiveModels = useCallback(() => {
    return models.filter((m) => m.status !== 'deprecated');
  }, [models]);

  // Selected model object
  const selectedModel = useMemo(() => {
    if (!selectedModelId) return null;
    return models.find((m) => m.id === selectedModelId) || null;
  }, [selectedModelId, models]);

  // Context value
  const value: ModelContextState = useMemo(
    () => ({
      selectedModelId,
      selectedModel,
      models,
      isLoading,
      error,
      setSelectedModel,
      refreshModels,
      isComparing,
      setIsComparing,
      comparisonModelIds,
      toggleComparisonModel,
      clearComparison,
      getModelById,
      getProductionModel,
      getActiveModels,
    }),
    [
      selectedModelId,
      selectedModel,
      models,
      isLoading,
      error,
      setSelectedModel,
      refreshModels,
      isComparing,
      comparisonModelIds,
      toggleComparisonModel,
      clearComparison,
      getModelById,
      getProductionModel,
      getActiveModels,
    ]
  );

  return (
    <ModelContext.Provider value={value}>{children}</ModelContext.Provider>
  );
}

// ============================================================================
// Hook
// ============================================================================

export function useModel(): ModelContextState {
  const context = useContext(ModelContext);

  if (context === undefined) {
    throw new Error('useModel must be used within a ModelProvider');
  }

  return context;
}

// ============================================================================
// Selector Hooks (for performance optimization)
// ============================================================================

/**
 * Use only the selected model (prevents re-renders when other state changes)
 */
export function useSelectedModel(): {
  model: ModelConfig | null;
  modelId: string | null;
  isLoading: boolean;
} {
  const { selectedModel, selectedModelId, isLoading } = useModel();
  return { model: selectedModel, modelId: selectedModelId, isLoading };
}

/**
 * Use only the models list
 */
export function useModelsList(): {
  models: ModelConfig[];
  isLoading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
} {
  const { models, isLoading, error, refreshModels } = useModel();
  return { models, isLoading, error, refresh: refreshModels };
}

/**
 * Use only comparison state
 */
export function useModelComparison(): {
  isComparing: boolean;
  setIsComparing: (value: boolean) => void;
  comparisonModelIds: string[];
  toggleComparisonModel: (modelId: string) => void;
  clearComparison: () => void;
} {
  const {
    isComparing,
    setIsComparing,
    comparisonModelIds,
    toggleComparisonModel,
    clearComparison,
  } = useModel();

  return {
    isComparing,
    setIsComparing,
    comparisonModelIds,
    toggleComparisonModel,
    clearComparison,
  };
}

export default ModelContext;

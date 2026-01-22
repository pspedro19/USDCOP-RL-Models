/**
 * SSOT API Hook
 * =============
 *
 * React hook for fetching and validating SSOT from the backend API.
 * This ensures frontend SSOT values stay in sync with the backend.
 *
 * Usage:
 *   const { ssot, isValid, errors, refetch } = useSSOT();
 *
 * Features:
 * - Fetches SSOT from /api/v1/ssot endpoint
 * - Validates local SSOT against backend
 * - Caches results to avoid excessive API calls
 * - Provides error handling for API failures
 *
 * @version 1.0.0
 * @date 2026-01-18
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  FEATURE_ORDER,
  OBSERVATION_DIM,
  ACTION_COUNT,
  FEATURE_CONTRACT_VERSION,
} from '../contracts/ssot.contract';

// ============================================================================
// Types
// ============================================================================

/**
 * Backend SSOT response structure
 */
export interface BackendSSOT {
  feature_contract: {
    version: string;
    feature_order: string[];
    feature_order_hash: string;
    observation_dim: number;
    market_features_count: number;
    state_features_count: number;
    feature_specs: Record<string, {
      index: number;
      clip_min: number;
      clip_max: number;
      unit: string;
    }>;
  };
  action_contract: {
    version: string;
    actions: Record<string, number>;
    action_count: number;
    action_names: Record<number, string>;
    valid_actions: number[];
  };
  indicators: {
    rsi_period: number;
    atr_period: number;
    adx_period: number;
    warmup_bars: number;
  };
  normalization: {
    clip_min: number;
    clip_max: number;
  };
  thresholds: {
    long: number;
    short: number;
  };
  risk: {
    min_confidence: number;
    high_confidence: number;
    max_position_size: number;
    default_stop_loss_pct: number;
    max_drawdown_pct: number;
  };
  market_hours: {
    timezone: string;
    start_hour: number;
    end_hour: number;
    utc_offset: number;
  };
  ssot_hash: string;
}

/**
 * Validation result from backend
 */
export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  backend_hash: string;
}

/**
 * Hook state
 */
export interface SSOTState {
  /** Backend SSOT data */
  ssot: BackendSSOT | null;
  /** Whether local SSOT matches backend */
  isValid: boolean;
  /** Validation errors */
  errors: string[];
  /** Validation warnings */
  warnings: string[];
  /** Whether data is loading */
  isLoading: boolean;
  /** Error message if API call failed */
  error: string | null;
  /** Backend SSOT hash */
  backendHash: string | null;
  /** Last fetch timestamp */
  lastFetch: Date | null;
}

// ============================================================================
// Configuration
// ============================================================================

const SSOT_API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const SSOT_CACHE_TTL = 5 * 60 * 1000; // 5 minutes cache TTL

// ============================================================================
// Hook Implementation
// ============================================================================

/**
 * Compute a simple hash for the frontend feature order
 * This should match the backend's FEATURE_ORDER_HASH computation
 */
function computeFeatureOrderHash(features: readonly string[]): string {
  // Simple hash based on joined string - backend uses SHA256
  // For validation, we compare the full feature order instead
  return features.join(',');
}

/**
 * Custom hook for SSOT management
 */
export function useSSOT(options: { autoValidate?: boolean; cacheTTL?: number } = {}) {
  const { autoValidate = true, cacheTTL = SSOT_CACHE_TTL } = options;

  const [state, setState] = useState<SSOTState>({
    ssot: null,
    isValid: false,
    errors: [],
    warnings: [],
    isLoading: false,
    error: null,
    backendHash: null,
    lastFetch: null,
  });

  // Cache ref to persist across renders
  const cacheRef = useRef<{ ssot: BackendSSOT | null; timestamp: number | null }>({
    ssot: null,
    timestamp: null,
  });

  /**
   * Check if cache is still valid
   */
  const isCacheValid = useCallback(() => {
    if (!cacheRef.current.ssot || !cacheRef.current.timestamp) {
      return false;
    }
    return Date.now() - cacheRef.current.timestamp < cacheTTL;
  }, [cacheTTL]);

  /**
   * Fetch SSOT from backend
   */
  const fetchSSOT = useCallback(async (): Promise<BackendSSOT | null> => {
    try {
      const response = await fetch(`${SSOT_API_BASE}/api/v1/ssot`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`API responded with status ${response.status}`);
      }

      const data: BackendSSOT = await response.json();
      return data;
    } catch (err) {
      console.error('Failed to fetch SSOT:', err);
      throw err;
    }
  }, []);

  /**
   * Validate local SSOT against backend
   */
  const validateSSOT = useCallback(async (): Promise<ValidationResult> => {
    try {
      const response = await fetch(`${SSOT_API_BASE}/api/v1/ssot/validate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          feature_order_hash: computeFeatureOrderHash(FEATURE_ORDER),
          observation_dim: OBSERVATION_DIM,
          action_count: ACTION_COUNT,
        }),
      });

      if (!response.ok) {
        throw new Error(`API responded with status ${response.status}`);
      }

      return await response.json();
    } catch (err) {
      console.error('Failed to validate SSOT:', err);
      throw err;
    }
  }, []);

  /**
   * Refresh SSOT data from backend
   */
  const refetch = useCallback(async (force = false) => {
    // Check cache first unless forced
    if (!force && isCacheValid()) {
      setState(prev => ({
        ...prev,
        ssot: cacheRef.current.ssot,
        isLoading: false,
      }));
      return;
    }

    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const ssot = await fetchSSOT();

      // Update cache
      cacheRef.current = {
        ssot,
        timestamp: Date.now(),
      };

      // Validate if autoValidate is enabled
      let validation: ValidationResult | null = null;
      if (autoValidate && ssot) {
        validation = await validateSSOT();
      }

      setState({
        ssot,
        isValid: validation?.valid ?? false,
        errors: validation?.errors ?? [],
        warnings: validation?.warnings ?? [],
        isLoading: false,
        error: null,
        backendHash: ssot?.ssot_hash ?? null,
        lastFetch: new Date(),
      });
    } catch (err) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: err instanceof Error ? err.message : 'Failed to fetch SSOT',
      }));
    }
  }, [autoValidate, fetchSSOT, isCacheValid, validateSSOT]);

  /**
   * Compare local feature order with backend
   */
  const compareFeatureOrder = useCallback((backendFeatures: string[]): boolean => {
    if (backendFeatures.length !== FEATURE_ORDER.length) {
      return false;
    }
    return backendFeatures.every((f, i) => f === FEATURE_ORDER[i]);
  }, []);

  /**
   * Get detailed comparison between local and backend SSOT
   */
  const getComparison = useCallback(() => {
    if (!state.ssot) return null;

    return {
      featureOrder: {
        local: [...FEATURE_ORDER],
        backend: state.ssot.feature_contract.feature_order,
        match: compareFeatureOrder(state.ssot.feature_contract.feature_order),
      },
      observationDim: {
        local: OBSERVATION_DIM,
        backend: state.ssot.feature_contract.observation_dim,
        match: OBSERVATION_DIM === state.ssot.feature_contract.observation_dim,
      },
      actionCount: {
        local: ACTION_COUNT,
        backend: state.ssot.action_contract.action_count,
        match: ACTION_COUNT === state.ssot.action_contract.action_count,
      },
      version: {
        local: FEATURE_CONTRACT_VERSION,
        backend: state.ssot.feature_contract.version,
        match: FEATURE_CONTRACT_VERSION === state.ssot.feature_contract.version,
      },
    };
  }, [state.ssot, compareFeatureOrder]);

  // Initial fetch on mount
  useEffect(() => {
    refetch();
  }, [refetch]);

  return {
    ...state,
    refetch,
    getComparison,
    compareFeatureOrder,
  };
}

/**
 * Simple hook to just check if SSOT is valid
 * Use this for quick validation checks without full SSOT data
 */
export function useSSOTValidation() {
  const [isValid, setIsValid] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function validate() {
      try {
        const response = await fetch(`${SSOT_API_BASE}/api/v1/ssot/validate`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            feature_order_hash: FEATURE_ORDER.join(','),
            observation_dim: OBSERVATION_DIM,
            action_count: ACTION_COUNT,
          }),
        });

        if (!response.ok) {
          setIsValid(false);
          setError(`API error: ${response.status}`);
          return;
        }

        const data: ValidationResult = await response.json();
        setIsValid(data.valid);
        if (!data.valid && data.errors.length > 0) {
          setError(data.errors.join('; '));
        }
      } catch (err) {
        setIsValid(false);
        setError(err instanceof Error ? err.message : 'Validation failed');
      }
    }

    validate();
  }, []);

  return { isValid, error };
}

export default useSSOT;

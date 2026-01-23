/**
 * Forecasting Service
 * ===================
 *
 * Service for interacting with the Forecasting API.
 * Supports both backend API and CSV fallback.
 *
 * @version 1.0.0
 * @lastSync 2026-01-22
 */

import {
  FORECASTING_API_BASE,
  FORECASTING_ENDPOINTS,
  type DashboardResponse,
  type ForecastListResponse,
  type ConsensusResponse,
  type ModelListResponse,
  type ModelDetailResponse,
  type ModelRankingResponse,
  type ImageListResponse,
  type ForecastQueryParams,
  DashboardResponseSchema,
  ForecastListResponseSchema,
  ConsensusResponseSchema,
  ModelListResponseSchema,
} from '@/lib/contracts/forecasting.contract';

class ForecastingService {
  private baseUrl: string;
  private useBackend: boolean;

  constructor() {
    this.baseUrl = FORECASTING_API_BASE;
    // Check if backend is available, otherwise use local data
    this.useBackend = process.env.NEXT_PUBLIC_USE_FORECASTING_BACKEND === 'true';
  }

  /**
   * Make API request with error handling
   */
  private async fetchApi<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }

      return response.json();
    } catch (error) {
      console.error(`Forecasting API error: ${endpoint}`, error);
      throw error;
    }
  }

  /**
   * Load data from local CSV (fallback)
   */
  private async loadLocalData(): Promise<DashboardResponse> {
    try {
      // Fetch the CSV from public folder
      const response = await fetch('/forecasting/bi_dashboard_unified.csv');
      if (!response.ok) {
        throw new Error('CSV not found');
      }

      const csvText = await response.text();
      const data = this.parseCSV(csvText);

      // Calculate consensus
      const consensus = this.calculateConsensus(data);

      // Calculate metrics
      const metrics = this.calculateMetrics(data);

      return {
        source: 'csv',
        forecasts: data,
        consensus,
        metrics,
        last_update: new Date().toISOString(),
      };
    } catch (error) {
      console.error('Error loading local data:', error);
      return {
        source: 'none',
        forecasts: [],
        consensus: [],
        metrics: [],
        last_update: new Date().toISOString(),
        error: String(error),
      };
    }
  }

  /**
   * Parse CSV text to array of objects
   */
  private parseCSV(csvText: string): Record<string, any>[] {
    const lines = csvText.trim().split('\n');
    if (lines.length < 2) return [];

    const headers = lines[0].split(',').map(h => h.trim());
    const data: Record<string, any>[] = [];

    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',');
      const row: Record<string, any> = {};

      headers.forEach((header, index) => {
        const value = values[index]?.trim() || '';
        // Try to parse numbers
        const num = parseFloat(value);
        row[header] = isNaN(num) ? value : num;
      });

      data.push(row);
    }

    return data;
  }

  /**
   * Calculate consensus from forecast data
   */
  private calculateConsensus(data: Record<string, any>[]): Record<string, any>[] {
    const horizons = [...new Set(data.map(d => d.horizon))].filter(h => h);
    const consensus: Record<string, any>[] = [];

    for (const horizon of horizons) {
      const horizonData = data.filter(d => d.horizon === horizon);
      const bullish = horizonData.filter(d => d.direction === 'UP').length;
      const bearish = horizonData.filter(d => d.direction === 'DOWN').length;
      const total = horizonData.length;

      consensus.push({
        horizon_id: horizon,
        bullish_count: bullish,
        bearish_count: bearish,
        total_models: total,
        consensus_direction: bullish > bearish ? 'UP' : 'DOWN',
        agreement_pct: total > 0 ? (Math.max(bullish, bearish) / total) * 100 : 0,
        avg_predicted_price: this.average(horizonData.map(d => d.predicted_price)),
      });
    }

    return consensus.sort((a, b) => a.horizon_id - b.horizon_id);
  }

  /**
   * Calculate model metrics from forecast data
   */
  private calculateMetrics(data: Record<string, any>[]): Record<string, any>[] {
    const models = [...new Set(data.map(d => d.model_name))].filter(m => m);
    const metrics: Record<string, any>[] = [];

    for (const model of models) {
      const modelData = data.filter(d => d.model_name === model);

      metrics.push({
        model_id: model,
        sample_count: modelData.length,
        avg_direction_accuracy: this.average(modelData.map(d => d.direction_accuracy)),
        avg_rmse: this.average(modelData.map(d => d.rmse)),
        avg_mae: this.average(modelData.map(d => d.mae)),
      });
    }

    return metrics.sort((a, b) => (b.avg_direction_accuracy || 0) - (a.avg_direction_accuracy || 0));
  }

  /**
   * Calculate average of array, filtering out NaN
   */
  private average(arr: number[]): number {
    const valid = arr.filter(n => !isNaN(n) && n !== null && n !== undefined);
    if (valid.length === 0) return 0;
    return valid.reduce((a, b) => a + b, 0) / valid.length;
  }

  // ============================================================================
  // PUBLIC API METHODS
  // ============================================================================

  /**
   * Get complete dashboard data
   */
  async getDashboard(): Promise<DashboardResponse> {
    if (this.useBackend) {
      try {
        const data = await this.fetchApi<DashboardResponse>(FORECASTING_ENDPOINTS.DASHBOARD);
        return DashboardResponseSchema.parse(data);
      } catch (error) {
        console.warn('Backend unavailable, using local data');
        return this.loadLocalData();
      }
    }
    return this.loadLocalData();
  }

  /**
   * Get forecasts with optional filtering
   */
  async getForecasts(params?: ForecastQueryParams): Promise<ForecastListResponse> {
    if (this.useBackend) {
      try {
        const queryString = params
          ? '?' + new URLSearchParams(
              Object.entries(params)
                .filter(([_, v]) => v !== undefined)
                .map(([k, v]) => [k, String(v)])
            ).toString()
          : '';
        return await this.fetchApi<ForecastListResponse>(
          FORECASTING_ENDPOINTS.FORECASTS + queryString
        );
      } catch (error) {
        console.warn('Backend unavailable, using local data');
      }
    }

    // Fallback to local data
    const dashboard = await this.loadLocalData();
    let data = dashboard.forecasts;

    // Apply filters
    if (params?.model) {
      data = data.filter(d => d.model_name === params.model);
    }
    if (params?.horizon) {
      data = data.filter(d => d.horizon === params.horizon);
    }
    if (params?.limit) {
      data = data.slice(0, params.limit);
    }

    return {
      source: dashboard.source,
      count: data.length,
      data,
    };
  }

  /**
   * Get latest forecasts per model/horizon
   */
  async getLatestForecasts(): Promise<ForecastListResponse> {
    if (this.useBackend) {
      try {
        return await this.fetchApi<ForecastListResponse>(FORECASTING_ENDPOINTS.FORECASTS_LATEST);
      } catch (error) {
        console.warn('Backend unavailable, using local data');
      }
    }

    const dashboard = await this.loadLocalData();
    // Get unique model/horizon combinations
    const seen = new Set<string>();
    const latest = dashboard.forecasts.filter(d => {
      const key = `${d.model_name}-${d.horizon}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });

    return {
      source: dashboard.source,
      count: latest.length,
      data: latest,
    };
  }

  /**
   * Get consensus forecasts
   */
  async getConsensus(): Promise<ConsensusResponse> {
    if (this.useBackend) {
      try {
        return await this.fetchApi<ConsensusResponse>(FORECASTING_ENDPOINTS.FORECASTS_CONSENSUS);
      } catch (error) {
        console.warn('Backend unavailable, using local data');
      }
    }

    const dashboard = await this.loadLocalData();
    return {
      source: dashboard.source,
      count: dashboard.consensus.length,
      data: dashboard.consensus as any,
    };
  }

  /**
   * Get all models
   */
  async getModels(): Promise<ModelListResponse> {
    if (this.useBackend) {
      try {
        return await this.fetchApi<ModelListResponse>(FORECASTING_ENDPOINTS.MODELS);
      } catch (error) {
        console.warn('Backend unavailable, using local data');
      }
    }

    const dashboard = await this.loadLocalData();
    const models = dashboard.metrics.map(m => ({
      model_id: m.model_id,
      model_name: m.model_id,
      model_type: this.getModelType(m.model_id),
      requires_scaling: ['ridge', 'bayesian_ridge', 'ard'].some(t => m.model_id?.includes(t)),
      supports_early_stopping: ['xgboost', 'lightgbm', 'catboost'].some(t => m.model_id?.includes(t)),
      is_active: true,
      avg_direction_accuracy: m.avg_direction_accuracy,
      avg_rmse: m.avg_rmse,
    }));

    return {
      models: models as any,
      count: models.length,
    };
  }

  /**
   * Get model type from model ID
   */
  private getModelType(modelId: string): 'linear' | 'boosting' | 'hybrid' {
    if (modelId?.includes('hybrid')) return 'hybrid';
    if (['xgboost', 'lightgbm', 'catboost'].some(t => modelId?.includes(t))) return 'boosting';
    return 'linear';
  }

  /**
   * Get model details
   */
  async getModelDetail(modelId: string): Promise<ModelDetailResponse | null> {
    if (this.useBackend) {
      try {
        return await this.fetchApi<ModelDetailResponse>(
          FORECASTING_ENDPOINTS.MODEL_DETAIL(modelId)
        );
      } catch (error) {
        console.warn('Backend unavailable');
      }
    }
    return null;
  }

  /**
   * Get model comparison
   */
  async getModelComparison(modelId: string): Promise<any> {
    if (this.useBackend) {
      try {
        return await this.fetchApi(FORECASTING_ENDPOINTS.MODEL_COMPARISON(modelId));
      } catch (error) {
        console.warn('Backend unavailable');
      }
    }
    return null;
  }

  /**
   * Get model rankings
   */
  async getModelRankings(metric?: string, horizon?: number): Promise<ModelRankingResponse | null> {
    if (this.useBackend) {
      try {
        const params = new URLSearchParams();
        if (metric) params.set('metric', metric);
        if (horizon) params.set('horizon', String(horizon));
        return await this.fetchApi<ModelRankingResponse>(
          `${FORECASTING_ENDPOINTS.MODEL_RANKING}?${params.toString()}`
        );
      } catch (error) {
        console.warn('Backend unavailable');
      }
    }
    return null;
  }

  /**
   * Get available images
   */
  async getImages(): Promise<ImageListResponse | null> {
    if (this.useBackend) {
      try {
        return await this.fetchApi<ImageListResponse>(FORECASTING_ENDPOINTS.IMAGES);
      } catch (error) {
        console.warn('Backend unavailable');
      }
    }
    return null;
  }

  /**
   * Get backtest image URL
   */
  getBacktestImageUrl(model: string, horizon: number): string {
    if (this.useBackend) {
      return `${this.baseUrl}${FORECASTING_ENDPOINTS.IMAGE_BACKTEST(model, horizon)}`;
    }
    // Local fallback
    return `/forecasting/backtest_${model}_h${horizon}.png`;
  }

  /**
   * Get forecast image URL
   */
  getForecastImageUrl(model: string): string {
    if (this.useBackend) {
      return `${this.baseUrl}${FORECASTING_ENDPOINTS.IMAGE_FORECAST(model)}`;
    }
    // Local fallback
    return `/forecasting/forward_forecast_${model}.png`;
  }

  /**
   * Check service health
   */
  async checkHealth(): Promise<{ status: string; data_source: string }> {
    if (this.useBackend) {
      try {
        return await this.fetchApi(FORECASTING_ENDPOINTS.HEALTH);
      } catch (error) {
        return { status: 'unavailable', data_source: 'none' };
      }
    }
    return { status: 'local', data_source: 'csv' };
  }
}

// Export singleton instance
export const forecastingService = new ForecastingService();
export default forecastingService;

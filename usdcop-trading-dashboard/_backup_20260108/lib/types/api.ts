/**
 * API Response Types
 * Standard response format for all API routes
 */

export type DataSource = 'live' | 'cached' | 'mock' | 'none' | 'postgres' | 'minio' | 'demo';

export interface ApiMetadata {
  dataSource: DataSource;
  timestamp: string;
  isRealData: boolean;
  latency?: number;
  cacheHit?: boolean;
  backendUrl?: string;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  metadata: ApiMetadata;
}

/**
 * Helper function to create standardized API responses
 *
 * @param data - The response data (null for errors)
 * @param dataSource - Source of the data ('postgres' | 'minio' | 'live' | 'none')
 * @param errorMessage - Optional error message
 * @param options - Additional metadata options
 */
export function createApiResponse<T>(
  data: T | null,
  dataSource: DataSource,
  errorMessage?: string,
  options: {
    message?: string;
    latency?: number;
    cacheHit?: boolean;
    backendUrl?: string;
  } = {}
): ApiResponse<T> {
  const isRealData = dataSource === 'live' || dataSource === 'postgres' || dataSource === 'minio';
  // Note: 'demo' and 'mock' are NOT real data
  const success = data !== null && !errorMessage;

  return {
    success,
    data: data ?? undefined,
    error: errorMessage,
    message: options.message,
    metadata: {
      dataSource,
      timestamp: new Date().toISOString(),
      isRealData,
      latency: options.latency,
      cacheHit: options.cacheHit,
      backendUrl: options.backendUrl,
    },
  };
}

/**
 * Helper to measure API latency
 */
export function measureLatency(startTime: number): number {
  return Date.now() - startTime;
}

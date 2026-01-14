/**
 * API Response Utilities
 * ======================
 *
 * Helper functions for creating standardized API responses.
 * Uses canonical types from types/schemas.ts
 *
 * @module lib/types/api
 */

import type { ApiMetadata, DataSource } from '../../types/schemas';

// Re-export canonical types for convenience
export type { ApiMetadata, DataSource } from '../../types/schemas';

/**
 * Generic API Response structure
 * This is the canonical response format for all API routes
 */
export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T | null;
  error?: string;
  message?: string;
  metadata: ApiMetadata;
}

/**
 * Options for createApiResponse helper
 */
export interface CreateApiResponseOptions {
  message?: string;
  latency?: number;
  cacheHit?: boolean;
  requestId?: string;
}

/**
 * Creates a standardized API response
 *
 * @param data - The response data (null for errors)
 * @param dataSource - Source of the data
 * @param errorMessage - Optional error message
 * @param options - Additional metadata options
 *
 * @example
 * // Success response
 * return NextResponse.json(
 *   createApiResponse(data, 'postgres', undefined, { latency: 45 })
 * );
 *
 * @example
 * // Error response
 * return NextResponse.json(
 *   createApiResponse(null, 'none', 'Database connection failed'),
 *   { status: 500 }
 * );
 */
export function createApiResponse<T>(
  data: T | null,
  dataSource: DataSource,
  errorMessage?: string,
  options: CreateApiResponseOptions = {}
): ApiResponse<T> {
  const isRealData = ['live', 'postgres', 'minio'].includes(dataSource);
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
      requestId: options.requestId,
    },
  };
}

/**
 * Creates a success response
 */
export function createSuccessResponse<T>(
  data: T,
  dataSource: DataSource,
  options: CreateApiResponseOptions = {}
): ApiResponse<T> {
  return createApiResponse(data, dataSource, undefined, options);
}

/**
 * Creates an error response
 */
export function createErrorResponse(
  errorMessage: string,
  dataSource: DataSource = 'none',
  options: CreateApiResponseOptions = {}
): ApiResponse<null> {
  return createApiResponse(null, dataSource, errorMessage, options);
}

/**
 * Measures API latency from start time
 */
export function measureLatency(startTime: number): number {
  return Date.now() - startTime;
}

/**
 * Wraps an async handler with latency measurement
 */
export async function withLatency<T>(
  handler: () => Promise<T>,
  onComplete: (latency: number) => void
): Promise<T> {
  const startTime = Date.now();
  try {
    return await handler();
  } finally {
    onComplete(measureLatency(startTime));
  }
}

/**
 * Type guard to check if response is successful
 */
export function isSuccessResponse<T>(
  response: ApiResponse<T>
): response is ApiResponse<T> & { success: true; data: T } {
  return response.success && response.data !== undefined && response.data !== null;
}

/**
 * Type guard to check if response is an error
 */
export function isErrorResponse<T>(
  response: ApiResponse<T>
): response is ApiResponse<T> & { success: false; error: string } {
  return !response.success && typeof response.error === 'string';
}

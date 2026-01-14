/**
 * API Module Index
 * ================
 *
 * Central export point for API client and utilities.
 *
 * @module lib/api
 *
 * @example
 * import { apiClient, ApiClientError } from '@/lib/api';
 *
 * const data = await apiClient.getRealtimePrice();
 */

export { apiClient, ApiClientError, createFetcher } from './client';

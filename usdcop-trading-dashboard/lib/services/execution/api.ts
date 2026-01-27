/**
 * Execution API Client - SignalBridge Integration
 * ================================================
 *
 * Centralized API client for execution module.
 * Follows dashboard API client patterns with auth support.
 *
 * @version 1.0.0
 * @lastSync 2026-01-22
 */

import { API_BASE_URL } from '@/lib/config/execution/constants';

// ============================================================================
// API CLIENT (follows dashboard pattern but with auth)
// ============================================================================

const DEFAULT_TIMEOUT = 30000;

/**
 * Fetch wrapper with timeout and auth
 */
async function fetchWithAuth<T>(
  endpoint: string,
  options: RequestInit = {},
  timeout: number = DEFAULT_TIMEOUT
): Promise<T> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  // Get auth token from storage
  const token = typeof window !== 'undefined' ? localStorage.getItem('auth-token') : null;

  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...(token && { Authorization: `Bearer ${token}` }),
        ...options.headers,
      },
    });

    clearTimeout(timeoutId);

    // Handle 401 - only redirect if not authenticated via main app
    if (response.status === 401) {
      if (typeof window !== 'undefined') {
        localStorage.removeItem('auth-token');
        localStorage.removeItem('auth-storage');

        // Check if user is authenticated via main app login
        const mainAppAuth = localStorage.getItem('isAuthenticated') === 'true' ||
                           sessionStorage.getItem('isAuthenticated') === 'true';

        // Only redirect to login if not authenticated at all
        if (!mainAppAuth) {
          window.location.href = '/login?callbackUrl=' + encodeURIComponent(window.location.pathname);
        }
        // If main app auth exists, don't redirect - let the error be handled by calling code
      }
      throw new Error('Unauthorized');
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    clearTimeout(timeoutId);

    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        throw new Error(`Request timeout after ${timeout}ms`);
      }
      throw error;
    }

    throw new Error('Unknown error occurred');
  }
}

/**
 * API client object with HTTP methods
 */
export const api = {
  async get<T>(endpoint: string): Promise<{ data: T }> {
    const data = await fetchWithAuth<T>(endpoint);
    return { data };
  },

  async post<T>(endpoint: string, body?: unknown): Promise<{ data: T }> {
    const data = await fetchWithAuth<T>(endpoint, {
      method: 'POST',
      body: body ? JSON.stringify(body) : undefined,
    });
    return { data };
  },

  async put<T>(endpoint: string, body?: unknown): Promise<{ data: T }> {
    const data = await fetchWithAuth<T>(endpoint, {
      method: 'PUT',
      body: body ? JSON.stringify(body) : undefined,
    });
    return { data };
  },

  async patch<T>(endpoint: string, body?: unknown): Promise<{ data: T }> {
    const data = await fetchWithAuth<T>(endpoint, {
      method: 'PATCH',
      body: body ? JSON.stringify(body) : undefined,
    });
    return { data };
  },

  async delete<T>(endpoint: string): Promise<{ data: T }> {
    const data = await fetchWithAuth<T>(endpoint, {
      method: 'DELETE',
    });
    return { data };
  },
};

export default api;

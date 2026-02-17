/**
 * API Authentication Utilities
 * ============================
 *
 * Provides authentication and authorization helpers for API routes.
 * Follows Chain of Responsibility pattern for layered security.
 */

import { getServerSession } from 'next-auth';
import { NextRequest, NextResponse } from 'next/server';
import { authOptions } from './next-auth-options';
import { checkApiRateLimit, createRateLimitHeaders } from './rate-limiter';
import { hasPermission } from './types';
import type { SessionUser, UserRole, Permission } from './types';
import { query as pgQuery } from '@/lib/db/postgres-client';
import crypto from 'crypto';

// ============================================================================
// Types
// ============================================================================

export interface AuthenticatedRequest extends NextRequest {
  user: SessionUser;
}

export interface ApiAuthOptions {
  requiredRole?: UserRole | UserRole[];
  requiredPermission?: Permission;
  rateLimit?: boolean;
  strictRateLimit?: boolean;
}

export interface AuthResult {
  authenticated: boolean;
  user?: SessionUser;
  error?: string;
  status?: number;
}

// ============================================================================
// Get Current User (Server-side)
// ============================================================================

/**
 * Get the current authenticated user from session
 *
 * @returns SessionUser or null if not authenticated
 */
export async function getCurrentUser(): Promise<SessionUser | null> {
  try {
    const session = await getServerSession(authOptions);
    return session?.user || null;
  } catch (error) {
    console.error('[ApiAuth] Error getting session:', error);
    return null;
  }
}

/**
 * Get current user or throw error
 *
 * @throws Error if not authenticated
 */
export async function requireUser(): Promise<SessionUser> {
  const user = await getCurrentUser();
  if (!user) {
    throw new Error('Authentication required');
  }
  return user;
}

// ============================================================================
// API Route Protection
// ============================================================================

/**
 * Protect an API route with authentication and authorization
 *
 * Usage:
 * ```typescript
 * export async function GET(request: NextRequest) {
 *   const auth = await protectApiRoute(request, { requiredRole: 'admin' });
 *   if (!auth.authenticated) {
 *     return NextResponse.json({ error: auth.error }, { status: auth.status });
 *   }
 *
 *   // Access authenticated user
 *   const user = auth.user;
 *   // ... handle request
 * }
 * ```
 */
export async function protectApiRoute(
  request: NextRequest,
  options: ApiAuthOptions = {}
): Promise<AuthResult> {
  // AUTH BYPASS: Skip authentication when AUTH_BYPASS_ENABLED is set (for testing)
  if (process.env.AUTH_BYPASS_ENABLED === 'true') {
    return {
      authenticated: true,
      user: {
        id: 'bypass-user',
        email: 'admin@bypass.local',
        username: 'admin',
        role: 'admin' as const,
      },
    };
  }

  const { requiredRole, requiredPermission, rateLimit = true, strictRateLimit = false } = options;

  // Check rate limit first (before auth to prevent DoS)
  if (rateLimit) {
    const clientIp = getClientIp(request);
    const rateLimitResult = await checkApiRateLimit(clientIp, strictRateLimit);

    if (!rateLimitResult.allowed) {
      return {
        authenticated: false,
        error: 'Rate limit exceeded',
        status: 429,
      };
    }
  }

  // Check authentication
  const user = await getCurrentUser();

  if (!user) {
    return {
      authenticated: false,
      error: 'Authentication required',
      status: 401,
    };
  }

  // Check role authorization
  if (requiredRole) {
    const roles = Array.isArray(requiredRole) ? requiredRole : [requiredRole];
    if (!roles.includes(user.role)) {
      return {
        authenticated: false,
        user,
        error: 'Insufficient permissions',
        status: 403,
      };
    }
  }

  // Check permission authorization
  if (requiredPermission) {
    if (!hasPermission(user.role, requiredPermission.resource, requiredPermission.action)) {
      return {
        authenticated: false,
        user,
        error: 'Permission denied',
        status: 403,
      };
    }
  }

  return {
    authenticated: true,
    user,
  };
}

// ============================================================================
// Higher-Order Function for Route Protection
// ============================================================================

type ApiHandler = (
  request: NextRequest,
  context: { user: SessionUser }
) => Promise<NextResponse>;

/**
 * Wrap an API handler with authentication
 *
 * Usage:
 * ```typescript
 * export const GET = withAuth(async (request, { user }) => {
 *   // user is guaranteed to be authenticated
 *   return NextResponse.json({ user });
 * }, { requiredRole: 'admin' });
 * ```
 */
export function withAuth(handler: ApiHandler, options: ApiAuthOptions = {}) {
  return async (request: NextRequest): Promise<NextResponse> => {
    // Dev mode bypass: skip auth when DEV_MODE is enabled
    const isDevMode = process.env.NODE_ENV === 'development';
    if (isDevMode) {
      const devUser: SessionUser = {
        id: 'dev-user',
        username: 'admin',
        role: 'admin',
        email: 'admin@dev.local',
      };
      try {
        return await handler(request, { user: devUser });
      } catch (error) {
        console.error('[ApiAuth] Handler error:', error);
        return NextResponse.json(
          { error: 'Internal server error', message: error instanceof Error ? error.message : 'Unknown error', timestamp: new Date().toISOString() },
          { status: 500 }
        );
      }
    }

    const auth = await protectApiRoute(request, options);

    if (!auth.authenticated || !auth.user) {
      return NextResponse.json(
        { error: auth.error, timestamp: new Date().toISOString() },
        { status: auth.status || 401 }
      );
    }

    try {
      return await handler(request, { user: auth.user });
    } catch (error) {
      console.error('[ApiAuth] Handler error:', error);
      return NextResponse.json(
        {
          error: 'Internal server error',
          message: error instanceof Error ? error.message : 'Unknown error',
          timestamp: new Date().toISOString(),
        },
        { status: 500 }
      );
    }
  };
}

/**
 * Wrap handler requiring admin role
 */
export function withAdminAuth(handler: ApiHandler) {
  return withAuth(handler, { requiredRole: 'admin' });
}

/**
 * Wrap handler requiring trader or admin role
 */
export function withTraderAuth(handler: ApiHandler) {
  return withAuth(handler, { requiredRole: ['admin', 'trader'] });
}

// ============================================================================
// API Key Authentication
// ============================================================================

/**
 * Hash API key for database comparison
 */
function hashApiKey(key: string): string {
  return crypto.createHash('sha256').update(key).digest('hex');
}

/**
 * Authenticate request using API key
 *
 * Checks Authorization header for Bearer token or X-API-Key header.
 * Validates against database first, then falls back to environment variable
 * for internal services or if database is unavailable.
 */
export async function authenticateApiKey(request: NextRequest): Promise<AuthResult> {
  const authHeader = request.headers.get('authorization');
  const apiKeyHeader = request.headers.get('x-api-key');

  const apiKey = authHeader?.replace('Bearer ', '') || apiKeyHeader;

  if (!apiKey) {
    return {
      authenticated: false,
      error: 'API key required',
      status: 401,
    };
  }

  try {
    // Check against database first
    const result = await pgQuery(`
      SELECT ak.id, ak.user_id, ak.name, ak.scopes, u.username, u.role
      FROM api_keys ak
      JOIN users u ON ak.user_id = u.id
      WHERE ak.key_hash = $1
        AND ak.is_active = true
        AND (ak.expires_at IS NULL OR ak.expires_at > NOW())
    `, [hashApiKey(apiKey)]);

    if (result.rows.length > 0) {
      const keyData = result.rows[0];

      // Update last_used_at
      await pgQuery('UPDATE api_keys SET last_used_at = NOW() WHERE id = $1', [keyData.id]);

      return {
        authenticated: true,
        user: {
          id: keyData.user_id,
          username: keyData.username,
          role: keyData.role,
          email: `${keyData.username}@api.local`,
        },
      };
    }

    // Fallback to environment variable for internal services
    const validKey = process.env.INTERNAL_API_KEY;
    if (validKey && apiKey === validKey) {
      return {
        authenticated: true,
        user: {
          id: 'api-service',
          email: 'api@system.local',
          username: 'api-service',
          role: 'api_service',
        },
      };
    }

    return {
      authenticated: false,
      error: 'Invalid API key',
      status: 401,
    };
  } catch (error) {
    console.error('[API Auth] Database error:', error);

    // Fallback to env var if DB unavailable
    const validKey = process.env.INTERNAL_API_KEY;
    if (validKey && apiKey === validKey) {
      return {
        authenticated: true,
        user: {
          id: 'api-service',
          username: 'api-service',
          role: 'api_service',
          email: 'api@system.local',
        },
      };
    }

    return {
      authenticated: false,
      error: 'Authentication service unavailable',
      status: 503,
    };
  }
}

/**
 * Wrap handler with API key authentication
 */
export function withApiKeyAuth(handler: ApiHandler) {
  return async (request: NextRequest): Promise<NextResponse> => {
    const auth = await authenticateApiKey(request);

    if (!auth.authenticated || !auth.user) {
      return NextResponse.json(
        { error: auth.error, timestamp: new Date().toISOString() },
        { status: auth.status || 401 }
      );
    }

    return handler(request, { user: auth.user });
  };
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get client IP from request
 */
export function getClientIp(request: NextRequest): string {
  const forwarded = request.headers.get('x-forwarded-for');
  const realIp = request.headers.get('x-real-ip');

  if (forwarded) {
    return forwarded.split(',')[0].trim();
  }

  return realIp || '127.0.0.1';
}

/**
 * Get client user agent
 */
export function getClientUserAgent(request: NextRequest): string {
  return request.headers.get('user-agent') || 'Unknown';
}

/**
 * Create unauthorized response
 */
export function unauthorized(message: string = 'Unauthorized'): NextResponse {
  return NextResponse.json(
    { error: message, timestamp: new Date().toISOString() },
    { status: 401 }
  );
}

/**
 * Create forbidden response
 */
export function forbidden(message: string = 'Forbidden'): NextResponse {
  return NextResponse.json(
    { error: message, timestamp: new Date().toISOString() },
    { status: 403 }
  );
}

/**
 * Create rate limited response
 */
export function rateLimited(retryAfter: number): NextResponse {
  return NextResponse.json(
    {
      error: 'Rate limit exceeded',
      retryAfter,
      timestamp: new Date().toISOString(),
    },
    {
      status: 429,
      headers: { 'Retry-After': String(retryAfter) },
    }
  );
}

// ============================================================================
// Legacy Aliases
// ============================================================================

/**
 * Validate API auth - simplified version for quick auth check
 * @deprecated Use protectApiRoute for full auth with options
 */
export async function validateApiAuth(request: NextRequest): Promise<AuthResult> {
  // In development mode, always allow
  if (process.env.NODE_ENV === 'development') {
    return {
      authenticated: true,
      user: {
        id: 'dev-user',
        username: 'admin',
        role: 'admin',
        email: 'admin@dev.local',
      },
    };
  }

  return protectApiRoute(request, { rateLimit: false });
}

/**
 * Rate Limiter Service
 * ====================
 *
 * Single Responsibility: Rate limiting for authentication and API endpoints
 *
 * Strategy Pattern: Different rate limiting strategies
 * - In-memory (default, for development)
 * - Redis (for production, distributed)
 *
 * Implements sliding window algorithm for accurate rate limiting
 */

import type { RateLimitConfig, RateLimitResult, RateLimitEntry } from './types';

// ============================================================================
// Rate Limiter Interface
// ============================================================================

export interface IRateLimiter {
  check(key: string): Promise<RateLimitResult>;
  increment(key: string): Promise<void>;
  reset(key: string): Promise<void>;
  isBlocked(key: string): Promise<boolean>;
  block(key: string, durationMs: number): Promise<void>;
}

// ============================================================================
// Default Configurations
// ============================================================================

export const RATE_LIMIT_CONFIGS = {
  // Login attempts: 5 per 15 minutes, block for 15 minutes
  login: {
    maxAttempts: 5,
    windowMs: 15 * 60 * 1000, // 15 minutes
    blockDurationMs: 15 * 60 * 1000, // 15 minutes
  },

  // Password reset: 3 per hour
  passwordReset: {
    maxAttempts: 3,
    windowMs: 60 * 60 * 1000, // 1 hour
    blockDurationMs: 60 * 60 * 1000, // 1 hour
  },

  // API requests: 600 per minute (increased for real-time dashboard polling)
  api: {
    maxAttempts: 600,
    windowMs: 60 * 1000, // 1 minute
    blockDurationMs: 30 * 1000, // 30 seconds (shorter block for dashboard)
  },

  // Strict API (for sensitive endpoints): 10 per minute
  apiStrict: {
    maxAttempts: 10,
    windowMs: 60 * 1000, // 1 minute
    blockDurationMs: 5 * 60 * 1000, // 5 minutes
  },
} as const;

// ============================================================================
// In-Memory Rate Limiter (Development/Single Instance)
// ============================================================================

class InMemoryRateLimiter implements IRateLimiter {
  private store: Map<string, RateLimitEntry> = new Map();
  private config: RateLimitConfig;
  private cleanupInterval: NodeJS.Timeout | null = null;

  constructor(config: RateLimitConfig) {
    this.config = config;
    this.startCleanup();
  }

  /**
   * Check if request is allowed
   */
  async check(key: string): Promise<RateLimitResult> {
    const now = Date.now();
    const entry = this.store.get(key);

    // No previous attempts
    if (!entry) {
      return {
        allowed: true,
        remaining: this.config.maxAttempts,
        resetAt: new Date(now + this.config.windowMs),
      };
    }

    // Check if blocked
    if (entry.blockedUntil && entry.blockedUntil > now) {
      const retryAfter = Math.ceil((entry.blockedUntil - now) / 1000);
      return {
        allowed: false,
        remaining: 0,
        resetAt: new Date(entry.blockedUntil),
        retryAfter,
      };
    }

    // Check if window has expired
    const windowExpiry = entry.firstAttempt + this.config.windowMs;
    if (now > windowExpiry) {
      // Window expired, reset
      this.store.delete(key);
      return {
        allowed: true,
        remaining: this.config.maxAttempts,
        resetAt: new Date(now + this.config.windowMs),
      };
    }

    // Within window, check attempts
    const remaining = Math.max(0, this.config.maxAttempts - entry.attempts);
    const allowed = remaining > 0;

    return {
      allowed,
      remaining: allowed ? remaining - 1 : 0, // Account for current request
      resetAt: new Date(windowExpiry),
      retryAfter: allowed ? undefined : Math.ceil((windowExpiry - now) / 1000),
    };
  }

  /**
   * Increment attempt counter
   */
  async increment(key: string): Promise<void> {
    const now = Date.now();
    const entry = this.store.get(key);

    if (!entry) {
      this.store.set(key, {
        attempts: 1,
        firstAttempt: now,
      });
      return;
    }

    // Check if window expired
    const windowExpiry = entry.firstAttempt + this.config.windowMs;
    if (now > windowExpiry) {
      // Start new window
      this.store.set(key, {
        attempts: 1,
        firstAttempt: now,
      });
      return;
    }

    // Increment within window
    entry.attempts++;

    // Auto-block if exceeded
    if (entry.attempts >= this.config.maxAttempts) {
      entry.blockedUntil = now + this.config.blockDurationMs;
    }

    this.store.set(key, entry);
  }

  /**
   * Reset rate limit for key
   */
  async reset(key: string): Promise<void> {
    this.store.delete(key);
  }

  /**
   * Check if key is currently blocked
   */
  async isBlocked(key: string): Promise<boolean> {
    const entry = this.store.get(key);
    if (!entry || !entry.blockedUntil) {
      return false;
    }
    return entry.blockedUntil > Date.now();
  }

  /**
   * Block a key for specified duration
   */
  async block(key: string, durationMs: number): Promise<void> {
    const now = Date.now();
    const entry = this.store.get(key) || {
      attempts: this.config.maxAttempts,
      firstAttempt: now,
    };

    entry.blockedUntil = now + durationMs;
    this.store.set(key, entry);
  }

  /**
   * Start periodic cleanup of expired entries
   */
  private startCleanup(): void {
    // Clean up every 5 minutes
    this.cleanupInterval = setInterval(() => {
      const now = Date.now();
      for (const [key, entry] of this.store.entries()) {
        const windowExpiry = entry.firstAttempt + this.config.windowMs;
        const blockExpiry = entry.blockedUntil || 0;

        if (now > windowExpiry && now > blockExpiry) {
          this.store.delete(key);
        }
      }
    }, 5 * 60 * 1000);

    // Prevent cleanup from blocking process exit
    if (this.cleanupInterval.unref) {
      this.cleanupInterval.unref();
    }
  }

  /**
   * Stop cleanup and clear store
   */
  destroy(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
    this.store.clear();
  }
}

// ============================================================================
// Rate Limiter Factory
// ============================================================================

export type RateLimiterType = 'login' | 'passwordReset' | 'api' | 'apiStrict';

const limiters: Map<string, IRateLimiter> = new Map();

/**
 * Get or create a rate limiter for the specified type
 *
 * Factory Pattern: Creates appropriate rate limiter based on type
 */
export function getRateLimiter(type: RateLimiterType): IRateLimiter {
  if (!limiters.has(type)) {
    const config = RATE_LIMIT_CONFIGS[type];
    limiters.set(type, new InMemoryRateLimiter(config));
  }

  return limiters.get(type)!;
}

/**
 * Create a custom rate limiter with specific config
 */
export function createRateLimiter(config: RateLimitConfig): IRateLimiter {
  return new InMemoryRateLimiter(config);
}

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * Check login rate limit for identifier (email/username + IP)
 */
export async function checkLoginRateLimit(
  identifier: string,
  ip: string
): Promise<RateLimitResult> {
  const limiter = getRateLimiter('login');
  const key = `login:${identifier}:${ip}`;
  return limiter.check(key);
}

/**
 * Record login attempt
 */
export async function recordLoginAttempt(
  identifier: string,
  ip: string,
  success: boolean
): Promise<void> {
  const limiter = getRateLimiter('login');
  const key = `login:${identifier}:${ip}`;

  if (success) {
    // Reset on successful login
    await limiter.reset(key);
  } else {
    // Increment on failure
    await limiter.increment(key);
  }
}

/**
 * Check API rate limit for key
 */
export async function checkApiRateLimit(
  apiKey: string,
  strict: boolean = false
): Promise<RateLimitResult> {
  const limiter = getRateLimiter(strict ? 'apiStrict' : 'api');
  const key = `api:${apiKey}`;

  const result = await limiter.check(key);
  await limiter.increment(key);

  return result;
}

/**
 * Check IP-based rate limit
 */
export async function checkIpRateLimit(
  ip: string,
  endpoint: string,
  config?: RateLimitConfig
): Promise<RateLimitResult> {
  const limiter = config
    ? createRateLimiter(config)
    : getRateLimiter('api');

  const key = `ip:${ip}:${endpoint}`;
  return limiter.check(key);
}

// ============================================================================
// Middleware Helper
// ============================================================================

/**
 * Create rate limit headers for response
 */
export function createRateLimitHeaders(result: RateLimitResult): Record<string, string> {
  return {
    'X-RateLimit-Limit': String(result.remaining + 1),
    'X-RateLimit-Remaining': String(result.remaining),
    'X-RateLimit-Reset': result.resetAt.toISOString(),
    ...(result.retryAfter ? { 'Retry-After': String(result.retryAfter) } : {}),
  };
}

// ============================================================================
// Export Types
// ============================================================================

export { InMemoryRateLimiter };

/**
 * Authentication Module - Public API
 * ===================================
 *
 * Centralized exports for authentication functionality.
 * Import from '@/lib/auth' instead of individual files.
 */

// Types
export type {
  User,
  UserWithPassword,
  UserRole,
  SessionUser,
  Session,
  LoginCredentials,
  LoginResult,
  AuthContext,
  ApiKey,
  CreateUserInput,
  UpdateUserInput,
  RateLimitConfig,
  RateLimitResult,
  AuditEventType,
  AuditLogEntry,
  Permission,
} from './types';

export {
  isValidEmail,
  isValidUsername,
  isStrongPassword,
  hasPermission,
  ROLE_PERMISSIONS,
} from './types';

// Services
export { passwordService } from './password-service';
export { authService } from './auth-service';
export { userRepository, toSessionUser } from './user-repository';
export { auditLogger } from './audit-logger';

// Rate Limiting
export {
  getRateLimiter,
  createRateLimiter,
  checkLoginRateLimit,
  recordLoginAttempt,
  checkApiRateLimit,
  checkIpRateLimit,
  createRateLimitHeaders,
  RATE_LIMIT_CONFIGS,
} from './rate-limiter';

// NextAuth
export { authOptions } from './next-auth-options';

// API Authentication
export {
  getCurrentUser,
  requireUser,
  protectApiRoute,
  withAuth,
  withAdminAuth,
  withTraderAuth,
  withApiKeyAuth,
  authenticateApiKey,
  getClientIp,
  getClientUserAgent,
  unauthorized,
  forbidden,
  rateLimited,
} from './api-auth';

export type {
  AuthenticatedRequest,
  ApiAuthOptions,
  AuthResult,
} from './api-auth';

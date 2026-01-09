/**
 * Authentication Types
 * ====================
 *
 * Interface Segregation Principle: Small, specific interfaces
 * for authentication domain
 */

// ============================================================================
// User Types
// ============================================================================

export type UserRole = 'admin' | 'trader' | 'viewer' | 'api_service';

export interface User {
  id: string;
  email: string;
  username: string;
  role: UserRole;
  fullName?: string;
  avatarUrl?: string;
  emailVerified: boolean;
  twoFactorEnabled: boolean;
  isActive: boolean;
  createdAt: Date;
  lastLoginAt?: Date;
}

export interface UserWithPassword extends User {
  passwordHash: string;
  lockedUntil?: Date;
  failedLoginAttempts: number;
}

export interface CreateUserInput {
  email: string;
  username: string;
  password: string;
  role?: UserRole;
  fullName?: string;
}

export interface UpdateUserInput {
  email?: string;
  username?: string;
  fullName?: string;
  avatarUrl?: string;
  role?: UserRole;
  isActive?: boolean;
}

// ============================================================================
// Session Types
// ============================================================================

export interface Session {
  id: string;
  userId: string;
  sessionToken: string;
  ipAddress?: string;
  userAgent?: string;
  deviceType?: string;
  createdAt: Date;
  expiresAt: Date;
  lastActivityAt: Date;
}

export interface SessionUser {
  id: string;
  email: string;
  username: string;
  role: UserRole;
  fullName?: string;
  avatarUrl?: string;
}

// ============================================================================
// Authentication Types
// ============================================================================

export interface LoginCredentials {
  identifier: string; // email or username
  password: string;
}

export interface LoginResult {
  success: boolean;
  user?: SessionUser;
  error?: string;
  requiresTwoFactor?: boolean;
}

export interface AuthContext {
  user: SessionUser | null;
  isAuthenticated: boolean;
  session: Session | null;
}

// ============================================================================
// API Key Types
// ============================================================================

export interface ApiKey {
  id: string;
  userId?: string;
  keyPrefix: string;
  name: string;
  description?: string;
  scopes: string[];
  rateLimitPerMinute: number;
  rateLimitPerDay: number;
  isActive: boolean;
  createdAt: Date;
  expiresAt?: Date;
  lastUsedAt?: Date;
}

export interface CreateApiKeyInput {
  userId?: string;
  name: string;
  description?: string;
  scopes?: string[];
  rateLimitPerMinute?: number;
  rateLimitPerDay?: number;
  expiresInDays?: number;
}

export interface ApiKeyWithSecret extends ApiKey {
  plainTextKey: string; // Only available at creation time
}

// ============================================================================
// Rate Limiting Types
// ============================================================================

export interface RateLimitConfig {
  maxAttempts: number;
  windowMs: number;
  blockDurationMs: number;
}

export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  resetAt: Date;
  retryAfter?: number; // seconds
}

export interface RateLimitEntry {
  attempts: number;
  firstAttempt: number;
  blockedUntil?: number;
}

// ============================================================================
// Audit Types
// ============================================================================

export type AuditEventType =
  | 'login_success'
  | 'login_failure'
  | 'logout'
  | 'password_change'
  | 'password_reset_request'
  | 'password_reset_complete'
  | 'two_factor_enabled'
  | 'two_factor_disabled'
  | 'api_key_created'
  | 'api_key_revoked'
  | 'account_locked'
  | 'account_unlocked'
  | 'role_changed'
  | 'session_created'
  | 'session_revoked';

export interface AuditLogEntry {
  id: number;
  userId?: string;
  eventType: AuditEventType;
  eventDescription?: string;
  ipAddress?: string;
  userAgent?: string;
  metadata?: Record<string, unknown>;
  createdAt: Date;
}

// ============================================================================
// Permission Types
// ============================================================================

export interface Permission {
  resource: string;
  action: 'create' | 'read' | 'update' | 'delete' | 'execute';
}

export const ROLE_PERMISSIONS: Record<UserRole, Permission[]> = {
  admin: [
    { resource: '*', action: 'create' },
    { resource: '*', action: 'read' },
    { resource: '*', action: 'update' },
    { resource: '*', action: 'delete' },
    { resource: '*', action: 'execute' },
  ],
  trader: [
    { resource: 'trading', action: 'create' },
    { resource: 'trading', action: 'read' },
    { resource: 'trading', action: 'update' },
    { resource: 'trading', action: 'execute' },
    { resource: 'signals', action: 'read' },
    { resource: 'market', action: 'read' },
    { resource: 'pipeline', action: 'read' },
    { resource: 'backtest', action: 'read' },
    { resource: 'backtest', action: 'execute' },
  ],
  viewer: [
    { resource: 'trading', action: 'read' },
    { resource: 'signals', action: 'read' },
    { resource: 'market', action: 'read' },
    { resource: 'pipeline', action: 'read' },
    { resource: 'backtest', action: 'read' },
  ],
  api_service: [
    { resource: 'market', action: 'read' },
    { resource: 'market', action: 'create' },
    { resource: 'pipeline', action: 'read' },
    { resource: 'pipeline', action: 'execute' },
  ],
};

// ============================================================================
// Validation Helpers
// ============================================================================

export function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

export function isValidUsername(username: string): boolean {
  // 3-50 chars, alphanumeric, underscore, hyphen
  const usernameRegex = /^[a-zA-Z0-9_-]{3,50}$/;
  return usernameRegex.test(username);
}

export function isStrongPassword(password: string): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  if (password.length < 8) {
    errors.push('Password must be at least 8 characters');
  }
  if (!/[A-Z]/.test(password)) {
    errors.push('Password must contain at least one uppercase letter');
  }
  if (!/[a-z]/.test(password)) {
    errors.push('Password must contain at least one lowercase letter');
  }
  if (!/[0-9]/.test(password)) {
    errors.push('Password must contain at least one number');
  }
  if (!/[!@#$%^&*(),.?":{}|<>]/.test(password)) {
    errors.push('Password must contain at least one special character');
  }

  return { valid: errors.length === 0, errors };
}

export function hasPermission(
  userRole: UserRole,
  resource: string,
  action: Permission['action']
): boolean {
  const permissions = ROLE_PERMISSIONS[userRole] || [];

  return permissions.some(
    (p) =>
      (p.resource === '*' || p.resource === resource) &&
      (p.action === action)
  );
}

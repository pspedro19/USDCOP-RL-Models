/**
 * Authentication Service
 * ======================
 *
 * Single Responsibility: Authentication logic and session management
 *
 * Facade Pattern: Provides unified interface for authentication operations
 * Coordinates between password service, user repository, and rate limiter
 */

import { passwordService } from './password-service';
import { userRepository, toSessionUser } from './user-repository';
import {
  checkLoginRateLimit,
  recordLoginAttempt,
  createRateLimitHeaders,
} from './rate-limiter';
import { auditLogger } from './audit-logger';
import type {
  LoginCredentials,
  LoginResult,
  SessionUser,
  CreateUserInput,
  User,
} from './types';
import { isValidEmail, isValidUsername, isStrongPassword } from './types';

// ============================================================================
// Authentication Service Interface
// ============================================================================

export interface IAuthService {
  login(credentials: LoginCredentials, ip: string, userAgent?: string): Promise<LoginResult>;
  validateSession(sessionToken: string): Promise<SessionUser | null>;
  logout(userId: string, sessionToken: string): Promise<void>;
  register(input: CreateUserInput): Promise<User>;
  changePassword(userId: string, currentPassword: string, newPassword: string): Promise<boolean>;
}

// ============================================================================
// Authentication Service Implementation
// ============================================================================

class AuthService implements IAuthService {
  /**
   * Authenticate user with credentials
   *
   * @param credentials - Login credentials (email/username + password)
   * @param ip - Client IP address for rate limiting
   * @param userAgent - Client user agent for audit logging
   * @returns Login result with user or error
   */
  async login(
    credentials: LoginCredentials,
    ip: string,
    userAgent?: string
  ): Promise<LoginResult> {
    const { identifier, password } = credentials;

    // Input validation
    if (!identifier || !password) {
      return { success: false, error: 'Email/username and password are required' };
    }

    // Check rate limit
    const rateLimit = await checkLoginRateLimit(identifier, ip);
    if (!rateLimit.allowed) {
      await auditLogger.log({
        eventType: 'login_failure',
        eventDescription: 'Rate limit exceeded',
        ipAddress: ip,
        userAgent,
        metadata: { identifier, retryAfter: rateLimit.retryAfter },
      });

      return {
        success: false,
        error: `Too many login attempts. Please try again in ${rateLimit.retryAfter} seconds.`,
      };
    }

    try {
      // Find user by identifier
      const user = await userRepository.findByIdentifier(identifier);

      if (!user) {
        await recordLoginAttempt(identifier, ip, false);
        await auditLogger.log({
          eventType: 'login_failure',
          eventDescription: 'User not found',
          ipAddress: ip,
          userAgent,
          metadata: { identifier },
        });

        // Don't reveal if user exists
        return { success: false, error: 'Invalid credentials' };
      }

      // Check if account is active
      if (!user.isActive) {
        await recordLoginAttempt(identifier, ip, false);
        await auditLogger.log({
          userId: user.id,
          eventType: 'login_failure',
          eventDescription: 'Account inactive',
          ipAddress: ip,
          userAgent,
        });

        return { success: false, error: 'Account is disabled. Please contact support.' };
      }

      // Check if account is locked
      const isLocked = await userRepository.isLocked(user.id);
      if (isLocked) {
        await auditLogger.log({
          userId: user.id,
          eventType: 'login_failure',
          eventDescription: 'Account locked',
          ipAddress: ip,
          userAgent,
        });

        return {
          success: false,
          error: 'Account is temporarily locked due to too many failed attempts. Please try again later.',
        };
      }

      // Verify password
      const passwordValid = await passwordService.verify(password, user.passwordHash);

      if (!passwordValid) {
        await recordLoginAttempt(identifier, ip, false);
        await userRepository.recordFailedLogin(user.id);
        await auditLogger.log({
          userId: user.id,
          eventType: 'login_failure',
          eventDescription: 'Invalid password',
          ipAddress: ip,
          userAgent,
        });

        return { success: false, error: 'Invalid credentials' };
      }

      // Check for 2FA
      if (user.twoFactorEnabled) {
        return {
          success: false,
          requiresTwoFactor: true,
          user: toSessionUser(user),
        };
      }

      // Success!
      await recordLoginAttempt(identifier, ip, true);
      await userRepository.recordSuccessfulLogin(user.id);
      await auditLogger.log({
        userId: user.id,
        eventType: 'login_success',
        eventDescription: 'User logged in successfully',
        ipAddress: ip,
        userAgent,
      });

      return {
        success: true,
        user: toSessionUser(user),
      };
    } catch (error) {
      console.error('[AuthService] Login error:', error);
      return { success: false, error: 'An error occurred during login' };
    }
  }

  /**
   * Validate session token and return user
   */
  async validateSession(sessionToken: string): Promise<SessionUser | null> {
    // This would typically query the sessions table
    // For NextAuth, this is handled by the session callback
    return null;
  }

  /**
   * Log out user and invalidate session
   */
  async logout(userId: string, sessionToken: string): Promise<void> {
    await auditLogger.log({
      userId,
      eventType: 'logout',
      eventDescription: 'User logged out',
    });
  }

  /**
   * Register a new user
   */
  async register(input: CreateUserInput): Promise<User> {
    // Validate email
    if (!isValidEmail(input.email)) {
      throw new Error('Invalid email format');
    }

    // Validate username
    if (!isValidUsername(input.username)) {
      throw new Error('Username must be 3-50 characters, alphanumeric with _ or -');
    }

    // Validate password strength
    const passwordCheck = isStrongPassword(input.password);
    if (!passwordCheck.valid) {
      throw new Error(passwordCheck.errors.join('. '));
    }

    // Check if password is compromised
    const isCompromised = await passwordService.isCompromised(input.password);
    if (isCompromised) {
      throw new Error('This password has been found in data breaches. Please choose a different password.');
    }

    // Create user
    const user = await userRepository.create(input);

    await auditLogger.log({
      userId: user.id,
      eventType: 'login_success', // Using existing type
      eventDescription: 'New user registered',
      metadata: { email: user.email, username: user.username },
    });

    return user;
  }

  /**
   * Change user password
   */
  async changePassword(
    userId: string,
    currentPassword: string,
    newPassword: string
  ): Promise<boolean> {
    // Get user with password
    const user = await userRepository.findById(userId);
    if (!user) {
      throw new Error('User not found');
    }

    // Get user with password hash
    const userWithPassword = await userRepository.findByEmail(user.email);
    if (!userWithPassword) {
      throw new Error('User not found');
    }

    // Verify current password
    const currentValid = await passwordService.verify(
      currentPassword,
      userWithPassword.passwordHash
    );
    if (!currentValid) {
      throw new Error('Current password is incorrect');
    }

    // Validate new password
    const passwordCheck = isStrongPassword(newPassword);
    if (!passwordCheck.valid) {
      throw new Error(passwordCheck.errors.join('. '));
    }

    // Check if new password is same as current
    const samePassword = await passwordService.verify(
      newPassword,
      userWithPassword.passwordHash
    );
    if (samePassword) {
      throw new Error('New password must be different from current password');
    }

    // Check if password is compromised
    const isCompromised = await passwordService.isCompromised(newPassword);
    if (isCompromised) {
      throw new Error('This password has been found in data breaches');
    }

    // Update password
    const success = await userRepository.updatePassword(userId, newPassword);

    if (success) {
      await auditLogger.log({
        userId,
        eventType: 'password_change',
        eventDescription: 'Password changed successfully',
      });
    }

    return success;
  }

  /**
   * Request password reset (sends email)
   */
  async requestPasswordReset(email: string): Promise<boolean> {
    const user = await userRepository.findByEmail(email);

    // Always return true to prevent email enumeration
    if (!user) {
      return true;
    }

    // Generate reset token
    const resetToken = passwordService.generateSecureToken();

    // TODO: Store reset token in database with expiry
    // TODO: Send email with reset link

    await auditLogger.log({
      userId: user.id,
      eventType: 'password_reset_request',
      eventDescription: 'Password reset requested',
    });

    return true;
  }

  /**
   * Verify password reset token
   */
  async verifyResetToken(token: string): Promise<SessionUser | null> {
    // TODO: Implement token verification from database
    return null;
  }

  /**
   * Complete password reset
   */
  async completePasswordReset(token: string, newPassword: string): Promise<boolean> {
    // TODO: Implement password reset completion
    return false;
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

export const authService = new AuthService();
export default authService;

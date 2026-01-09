/**
 * Password Service
 * ================
 *
 * Single Responsibility: Password hashing and verification
 *
 * Uses bcrypt with configurable cost factor for secure password storage.
 * This service is the ONLY place in the codebase that handles password operations.
 */

import bcrypt from 'bcryptjs';
import crypto from 'crypto';

// ============================================================================
// Configuration
// ============================================================================

const BCRYPT_COST_FACTOR = 12; // Higher = more secure but slower
const MIN_PASSWORD_LENGTH = 8;
const MAX_PASSWORD_LENGTH = 128;

// ============================================================================
// Password Service Interface
// ============================================================================

export interface IPasswordService {
  hash(password: string): Promise<string>;
  verify(password: string, hash: string): Promise<boolean>;
  generateSecureToken(length?: number): string;
  isCompromised(password: string): Promise<boolean>;
}

// ============================================================================
// Password Service Implementation
// ============================================================================

class PasswordService implements IPasswordService {
  private readonly costFactor: number;

  constructor(costFactor: number = BCRYPT_COST_FACTOR) {
    this.costFactor = costFactor;
  }

  /**
   * Hash a password using bcrypt
   *
   * @param password - Plain text password
   * @returns Hashed password
   * @throws Error if password is invalid
   */
  async hash(password: string): Promise<string> {
    this.validatePassword(password);

    const salt = await bcrypt.genSalt(this.costFactor);
    const hash = await bcrypt.hash(password, salt);

    return hash;
  }

  /**
   * Verify a password against a hash
   *
   * Uses timing-safe comparison to prevent timing attacks
   *
   * @param password - Plain text password
   * @param hash - Stored password hash
   * @returns True if password matches
   */
  async verify(password: string, hash: string): Promise<boolean> {
    if (!password || !hash) {
      return false;
    }

    try {
      return await bcrypt.compare(password, hash);
    } catch (error) {
      console.error('[PasswordService] Verification error:', error);
      return false;
    }
  }

  /**
   * Generate a cryptographically secure random token
   *
   * @param length - Token length in bytes (default 32)
   * @returns Hex-encoded token
   */
  generateSecureToken(length: number = 32): string {
    return crypto.randomBytes(length).toString('hex');
  }

  /**
   * Generate a secure API key
   *
   * Format: prefix_base64token
   *
   * @param prefix - Key prefix for identification
   * @returns API key string
   */
  generateApiKey(prefix: string = 'usdcop'): { key: string; hash: string } {
    const token = crypto.randomBytes(32).toString('base64url');
    const key = `${prefix}_${token}`;
    const hash = crypto.createHash('sha256').update(key).digest('hex');

    return { key, hash };
  }

  /**
   * Verify an API key against its hash
   *
   * @param key - Plain API key
   * @param hash - Stored hash
   * @returns True if key matches
   */
  verifyApiKey(key: string, hash: string): boolean {
    const keyHash = crypto.createHash('sha256').update(key).digest('hex');
    return crypto.timingSafeEqual(Buffer.from(keyHash), Buffer.from(hash));
  }

  /**
   * Check if password appears in known breach databases
   *
   * Uses k-Anonymity model with HaveIBeenPwned API
   * Only sends first 5 chars of SHA1 hash
   *
   * @param password - Password to check
   * @returns True if password is compromised
   */
  async isCompromised(password: string): Promise<boolean> {
    try {
      // Generate SHA1 hash of password
      const sha1Hash = crypto
        .createHash('sha1')
        .update(password)
        .digest('hex')
        .toUpperCase();

      const prefix = sha1Hash.substring(0, 5);
      const suffix = sha1Hash.substring(5);

      // Query HaveIBeenPwned API
      const response = await fetch(
        `https://api.pwnedpasswords.com/range/${prefix}`,
        {
          headers: { 'Add-Padding': 'true' },
          signal: AbortSignal.timeout(5000),
        }
      );

      if (!response.ok) {
        // If service unavailable, don't block user
        console.warn('[PasswordService] HIBP API unavailable');
        return false;
      }

      const text = await response.text();
      const hashes = text.split('\r\n');

      // Check if our suffix appears in the response
      for (const line of hashes) {
        const [hashSuffix] = line.split(':');
        if (hashSuffix === suffix) {
          return true;
        }
      }

      return false;
    } catch (error) {
      // Don't block login if breach check fails
      console.warn('[PasswordService] Breach check failed:', error);
      return false;
    }
  }

  /**
   * Validate password requirements
   *
   * @param password - Password to validate
   * @throws Error if password is invalid
   */
  private validatePassword(password: string): void {
    if (!password) {
      throw new Error('Password is required');
    }

    if (password.length < MIN_PASSWORD_LENGTH) {
      throw new Error(`Password must be at least ${MIN_PASSWORD_LENGTH} characters`);
    }

    if (password.length > MAX_PASSWORD_LENGTH) {
      throw new Error(`Password must be at most ${MAX_PASSWORD_LENGTH} characters`);
    }
  }

  /**
   * Check password strength
   *
   * @param password - Password to check
   * @returns Strength score (0-100) and feedback
   */
  checkStrength(password: string): { score: number; feedback: string[] } {
    const feedback: string[] = [];
    let score = 0;

    // Length check
    if (password.length >= 8) score += 20;
    if (password.length >= 12) score += 10;
    if (password.length >= 16) score += 10;

    // Character variety
    if (/[a-z]/.test(password)) score += 10;
    if (/[A-Z]/.test(password)) score += 10;
    if (/[0-9]/.test(password)) score += 10;
    if (/[!@#$%^&*(),.?":{}|<>]/.test(password)) score += 15;

    // Complexity patterns
    if (/(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])/.test(password)) score += 15;

    // Feedback
    if (password.length < 8) feedback.push('Use at least 8 characters');
    if (!/[A-Z]/.test(password)) feedback.push('Add uppercase letters');
    if (!/[a-z]/.test(password)) feedback.push('Add lowercase letters');
    if (!/[0-9]/.test(password)) feedback.push('Add numbers');
    if (!/[!@#$%^&*(),.?":{}|<>]/.test(password)) feedback.push('Add special characters');

    // Common patterns to avoid
    if (/^[a-zA-Z]+$/.test(password)) {
      feedback.push('Avoid using only letters');
      score -= 10;
    }
    if (/^[0-9]+$/.test(password)) {
      feedback.push('Avoid using only numbers');
      score -= 10;
    }
    if (/(.)\1{2,}/.test(password)) {
      feedback.push('Avoid repeated characters');
      score -= 10;
    }
    if (/^(123|abc|qwerty|password)/i.test(password)) {
      feedback.push('Avoid common patterns');
      score -= 20;
    }

    return {
      score: Math.max(0, Math.min(100, score)),
      feedback,
    };
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

export const passwordService = new PasswordService();
export default passwordService;

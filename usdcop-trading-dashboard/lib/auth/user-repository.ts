/**
 * User Repository
 * ===============
 *
 * Single Responsibility: User data access and persistence
 *
 * Repository Pattern: Abstracts database operations for users
 * Dependency Inversion: Depends on abstract pgQuery, not concrete implementation
 */

import { pgQuery } from '@/lib/db/postgres-client';
import { passwordService } from './password-service';
import type {
  User,
  UserWithPassword,
  CreateUserInput,
  UpdateUserInput,
  UserRole,
  SessionUser,
} from './types';

// ============================================================================
// User Repository Interface
// ============================================================================

export interface IUserRepository {
  findById(id: string): Promise<User | null>;
  findByEmail(email: string): Promise<UserWithPassword | null>;
  findByUsername(username: string): Promise<UserWithPassword | null>;
  findByIdentifier(identifier: string): Promise<UserWithPassword | null>;
  create(input: CreateUserInput): Promise<User>;
  update(id: string, input: UpdateUserInput): Promise<User | null>;
  delete(id: string): Promise<boolean>;
  recordFailedLogin(userId: string): Promise<void>;
  recordSuccessfulLogin(userId: string): Promise<void>;
  isLocked(userId: string): Promise<boolean>;
}

// ============================================================================
// User Repository Implementation
// ============================================================================

class UserRepository implements IUserRepository {
  /**
   * Find user by ID
   */
  async findById(id: string): Promise<User | null> {
    try {
      const result = await pgQuery(
        `SELECT
          id, email, username, role, full_name, avatar_url,
          email_verified, two_factor_enabled, is_active,
          created_at, last_login_at
        FROM users
        WHERE id = $1`,
        [id]
      );

      if (result.rows.length === 0) {
        return null;
      }

      return this.mapRowToUser(result.rows[0]);
    } catch (error) {
      console.error('[UserRepository] findById error:', error);
      throw error;
    }
  }

  /**
   * Find user by email (includes password hash for authentication)
   */
  async findByEmail(email: string): Promise<UserWithPassword | null> {
    try {
      const result = await pgQuery(
        `SELECT
          id, email, username, password_hash, role, full_name, avatar_url,
          email_verified, two_factor_enabled, is_active,
          locked_until, failed_login_attempts,
          created_at, last_login_at
        FROM users
        WHERE LOWER(email) = LOWER($1)`,
        [email]
      );

      if (result.rows.length === 0) {
        return null;
      }

      return this.mapRowToUserWithPassword(result.rows[0]);
    } catch (error) {
      console.error('[UserRepository] findByEmail error:', error);
      throw error;
    }
  }

  /**
   * Find user by username (includes password hash for authentication)
   */
  async findByUsername(username: string): Promise<UserWithPassword | null> {
    try {
      const result = await pgQuery(
        `SELECT
          id, email, username, password_hash, role, full_name, avatar_url,
          email_verified, two_factor_enabled, is_active,
          locked_until, failed_login_attempts,
          created_at, last_login_at
        FROM users
        WHERE LOWER(username) = LOWER($1)`,
        [username]
      );

      if (result.rows.length === 0) {
        return null;
      }

      return this.mapRowToUserWithPassword(result.rows[0]);
    } catch (error) {
      console.error('[UserRepository] findByUsername error:', error);
      throw error;
    }
  }

  /**
   * Find user by email OR username (for login)
   */
  async findByIdentifier(identifier: string): Promise<UserWithPassword | null> {
    // Check if identifier looks like an email
    if (identifier.includes('@')) {
      return this.findByEmail(identifier);
    }
    return this.findByUsername(identifier);
  }

  /**
   * Create a new user
   */
  async create(input: CreateUserInput): Promise<User> {
    try {
      // Hash password
      const passwordHash = await passwordService.hash(input.password);

      const result = await pgQuery(
        `INSERT INTO users (email, username, password_hash, role, full_name)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING
          id, email, username, role, full_name, avatar_url,
          email_verified, two_factor_enabled, is_active,
          created_at, last_login_at`,
        [
          input.email.toLowerCase(),
          input.username.toLowerCase(),
          passwordHash,
          input.role || 'trader',
          input.fullName || null,
        ]
      );

      return this.mapRowToUser(result.rows[0]);
    } catch (error: any) {
      if (error.code === '23505') {
        // Unique constraint violation
        if (error.constraint?.includes('email')) {
          throw new Error('Email already exists');
        }
        if (error.constraint?.includes('username')) {
          throw new Error('Username already exists');
        }
      }
      console.error('[UserRepository] create error:', error);
      throw error;
    }
  }

  /**
   * Update user
   */
  async update(id: string, input: UpdateUserInput): Promise<User | null> {
    try {
      const updates: string[] = [];
      const values: unknown[] = [];
      let paramIndex = 1;

      if (input.email !== undefined) {
        updates.push(`email = $${paramIndex++}`);
        values.push(input.email.toLowerCase());
      }
      if (input.username !== undefined) {
        updates.push(`username = $${paramIndex++}`);
        values.push(input.username.toLowerCase());
      }
      if (input.fullName !== undefined) {
        updates.push(`full_name = $${paramIndex++}`);
        values.push(input.fullName);
      }
      if (input.avatarUrl !== undefined) {
        updates.push(`avatar_url = $${paramIndex++}`);
        values.push(input.avatarUrl);
      }
      if (input.role !== undefined) {
        updates.push(`role = $${paramIndex++}`);
        values.push(input.role);
      }
      if (input.isActive !== undefined) {
        updates.push(`is_active = $${paramIndex++}`);
        values.push(input.isActive);
      }

      if (updates.length === 0) {
        return this.findById(id);
      }

      values.push(id);

      const result = await pgQuery(
        `UPDATE users
        SET ${updates.join(', ')}
        WHERE id = $${paramIndex}
        RETURNING
          id, email, username, role, full_name, avatar_url,
          email_verified, two_factor_enabled, is_active,
          created_at, last_login_at`,
        values
      );

      if (result.rows.length === 0) {
        return null;
      }

      return this.mapRowToUser(result.rows[0]);
    } catch (error) {
      console.error('[UserRepository] update error:', error);
      throw error;
    }
  }

  /**
   * Update user password
   */
  async updatePassword(id: string, newPassword: string): Promise<boolean> {
    try {
      const passwordHash = await passwordService.hash(newPassword);

      const result = await pgQuery(
        `UPDATE users
        SET password_hash = $1
        WHERE id = $2`,
        [passwordHash, id]
      );

      return result.rowCount === 1;
    } catch (error) {
      console.error('[UserRepository] updatePassword error:', error);
      throw error;
    }
  }

  /**
   * Delete user
   */
  async delete(id: string): Promise<boolean> {
    try {
      const result = await pgQuery(
        `DELETE FROM users WHERE id = $1`,
        [id]
      );

      return result.rowCount === 1;
    } catch (error) {
      console.error('[UserRepository] delete error:', error);
      throw error;
    }
  }

  /**
   * Record failed login attempt
   */
  async recordFailedLogin(userId: string): Promise<void> {
    try {
      await pgQuery(`SELECT record_failed_login($1)`, [userId]);
    } catch (error) {
      console.error('[UserRepository] recordFailedLogin error:', error);
      // Don't throw - this is not critical
    }
  }

  /**
   * Record successful login
   */
  async recordSuccessfulLogin(userId: string): Promise<void> {
    try {
      await pgQuery(`SELECT record_successful_login($1)`, [userId]);
    } catch (error) {
      console.error('[UserRepository] recordSuccessfulLogin error:', error);
      // Don't throw - this is not critical
    }
  }

  /**
   * Check if user account is locked
   */
  async isLocked(userId: string): Promise<boolean> {
    try {
      const result = await pgQuery(
        `SELECT is_user_locked($1) as locked`,
        [userId]
      );

      return result.rows[0]?.locked || false;
    } catch (error) {
      console.error('[UserRepository] isLocked error:', error);
      return false;
    }
  }

  /**
   * Get all users (admin only)
   */
  async findAll(limit: number = 100, offset: number = 0): Promise<User[]> {
    try {
      const result = await pgQuery(
        `SELECT
          id, email, username, role, full_name, avatar_url,
          email_verified, two_factor_enabled, is_active,
          created_at, last_login_at
        FROM users
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2`,
        [limit, offset]
      );

      return result.rows.map(this.mapRowToUser);
    } catch (error) {
      console.error('[UserRepository] findAll error:', error);
      throw error;
    }
  }

  /**
   * Count total users
   */
  async count(): Promise<number> {
    try {
      const result = await pgQuery(`SELECT COUNT(*) as count FROM users`);
      return parseInt(result.rows[0]?.count || '0');
    } catch (error) {
      console.error('[UserRepository] count error:', error);
      return 0;
    }
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private mapRowToUser(row: any): User {
    return {
      id: row.id,
      email: row.email,
      username: row.username,
      role: row.role as UserRole,
      fullName: row.full_name || undefined,
      avatarUrl: row.avatar_url || undefined,
      emailVerified: row.email_verified || false,
      twoFactorEnabled: row.two_factor_enabled || false,
      isActive: row.is_active !== false,
      createdAt: new Date(row.created_at),
      lastLoginAt: row.last_login_at ? new Date(row.last_login_at) : undefined,
    };
  }

  private mapRowToUserWithPassword(row: any): UserWithPassword {
    return {
      ...this.mapRowToUser(row),
      passwordHash: row.password_hash,
      lockedUntil: row.locked_until ? new Date(row.locked_until) : undefined,
      failedLoginAttempts: row.failed_login_attempts || 0,
    };
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Convert User to SessionUser (safe for client)
 */
export function toSessionUser(user: User): SessionUser {
  return {
    id: user.id,
    email: user.email,
    username: user.username,
    role: user.role,
    fullName: user.fullName,
    avatarUrl: user.avatarUrl,
  };
}

// ============================================================================
// Singleton Export
// ============================================================================

export const userRepository = new UserRepository();
export default userRepository;

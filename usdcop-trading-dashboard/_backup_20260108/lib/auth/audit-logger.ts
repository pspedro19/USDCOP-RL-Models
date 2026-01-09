/**
 * Audit Logger
 * ============
 *
 * Single Responsibility: Logging security and authentication events
 *
 * Records all authentication-related events for security monitoring
 * and compliance requirements.
 */

import { pgQuery } from '@/lib/db/postgres-client';
import type { AuditEventType, AuditLogEntry } from './types';

// ============================================================================
// Audit Logger Interface
// ============================================================================

export interface IAuditLogger {
  log(entry: Omit<AuditLogEntry, 'id' | 'createdAt'>): Promise<void>;
  getRecentLogs(userId?: string, limit?: number): Promise<AuditLogEntry[]>;
  getLoginHistory(userId: string, limit?: number): Promise<AuditLogEntry[]>;
}

// ============================================================================
// Audit Log Input Type
// ============================================================================

interface AuditLogInput {
  userId?: string;
  eventType: AuditEventType;
  eventDescription?: string;
  ipAddress?: string;
  userAgent?: string;
  metadata?: Record<string, unknown>;
}

// ============================================================================
// Audit Logger Implementation
// ============================================================================

class AuditLogger implements IAuditLogger {
  private fallbackLogs: AuditLogInput[] = [];
  private maxFallbackLogs = 1000;

  /**
   * Log an audit event
   */
  async log(entry: AuditLogInput): Promise<void> {
    try {
      await pgQuery(
        `INSERT INTO auth_audit_log (user_id, event_type, event_description, ip_address, user_agent, metadata)
        VALUES ($1, $2, $3, $4, $5, $6)`,
        [
          entry.userId || null,
          entry.eventType,
          entry.eventDescription || null,
          entry.ipAddress || null,
          entry.userAgent || null,
          entry.metadata ? JSON.stringify(entry.metadata) : null,
        ]
      );
    } catch (error) {
      // Fallback to in-memory if database unavailable
      console.warn('[AuditLogger] Database unavailable, using fallback:', error);
      this.logFallback(entry);
    }
  }

  /**
   * Fallback to in-memory logging
   */
  private logFallback(entry: AuditLogInput): void {
    this.fallbackLogs.push({
      ...entry,
      metadata: { ...entry.metadata, fallbackTimestamp: new Date().toISOString() },
    });

    // Keep only recent logs
    if (this.fallbackLogs.length > this.maxFallbackLogs) {
      this.fallbackLogs = this.fallbackLogs.slice(-this.maxFallbackLogs);
    }

    // Also log to console in development
    if (process.env.NODE_ENV === 'development') {
      console.log('[AuditLogger]', entry.eventType, entry.eventDescription, entry.metadata);
    }
  }

  /**
   * Get recent audit logs
   */
  async getRecentLogs(userId?: string, limit: number = 100): Promise<AuditLogEntry[]> {
    try {
      let query = `
        SELECT id, user_id, event_type, event_description, ip_address, user_agent, metadata, created_at
        FROM auth_audit_log
      `;
      const params: (string | number)[] = [];

      if (userId) {
        query += ` WHERE user_id = $1`;
        params.push(userId);
      }

      query += ` ORDER BY created_at DESC LIMIT $${params.length + 1}`;
      params.push(limit);

      const result = await pgQuery(query, params);

      return result.rows.map(this.mapRowToAuditEntry);
    } catch (error) {
      console.error('[AuditLogger] getRecentLogs error:', error);
      return [];
    }
  }

  /**
   * Get login history for a user
   */
  async getLoginHistory(userId: string, limit: number = 20): Promise<AuditLogEntry[]> {
    try {
      const result = await pgQuery(
        `SELECT id, user_id, event_type, event_description, ip_address, user_agent, metadata, created_at
        FROM auth_audit_log
        WHERE user_id = $1 AND event_type IN ('login_success', 'login_failure')
        ORDER BY created_at DESC
        LIMIT $2`,
        [userId, limit]
      );

      return result.rows.map(this.mapRowToAuditEntry);
    } catch (error) {
      console.error('[AuditLogger] getLoginHistory error:', error);
      return [];
    }
  }

  /**
   * Get security events (for admin dashboard)
   */
  async getSecurityEvents(hours: number = 24, limit: number = 100): Promise<AuditLogEntry[]> {
    try {
      const result = await pgQuery(
        `SELECT id, user_id, event_type, event_description, ip_address, user_agent, metadata, created_at
        FROM auth_audit_log
        WHERE created_at > NOW() - INTERVAL '${hours} hours'
          AND event_type IN ('login_failure', 'account_locked', 'password_reset_request')
        ORDER BY created_at DESC
        LIMIT $1`,
        [limit]
      );

      return result.rows.map(this.mapRowToAuditEntry);
    } catch (error) {
      console.error('[AuditLogger] getSecurityEvents error:', error);
      return [];
    }
  }

  /**
   * Count failed logins by IP
   */
  async countFailedLoginsByIp(ip: string, hours: number = 24): Promise<number> {
    try {
      const result = await pgQuery(
        `SELECT COUNT(*) as count
        FROM auth_audit_log
        WHERE ip_address = $1
          AND event_type = 'login_failure'
          AND created_at > NOW() - INTERVAL '${hours} hours'`,
        [ip]
      );

      return parseInt(result.rows[0]?.count || '0');
    } catch (error) {
      console.error('[AuditLogger] countFailedLoginsByIp error:', error);
      return 0;
    }
  }

  /**
   * Flush fallback logs to database (call periodically)
   */
  async flushFallbackLogs(): Promise<number> {
    if (this.fallbackLogs.length === 0) {
      return 0;
    }

    const logsToFlush = [...this.fallbackLogs];
    this.fallbackLogs = [];

    let flushed = 0;

    for (const entry of logsToFlush) {
      try {
        await pgQuery(
          `INSERT INTO auth_audit_log (user_id, event_type, event_description, ip_address, user_agent, metadata)
          VALUES ($1, $2, $3, $4, $5, $6)`,
          [
            entry.userId || null,
            entry.eventType,
            entry.eventDescription || null,
            entry.ipAddress || null,
            entry.userAgent || null,
            entry.metadata ? JSON.stringify(entry.metadata) : null,
          ]
        );
        flushed++;
      } catch {
        // Re-add to fallback if still failing
        this.fallbackLogs.push(entry);
      }
    }

    return flushed;
  }

  private mapRowToAuditEntry(row: any): AuditLogEntry {
    return {
      id: row.id,
      userId: row.user_id || undefined,
      eventType: row.event_type as AuditEventType,
      eventDescription: row.event_description || undefined,
      ipAddress: row.ip_address || undefined,
      userAgent: row.user_agent || undefined,
      metadata: row.metadata || undefined,
      createdAt: new Date(row.created_at),
    };
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

export const auditLogger = new AuditLogger();
export default auditLogger;

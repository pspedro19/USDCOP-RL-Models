// Audit Logger for USDCOP Trading System

import type { IAuditLogger, AuditLogEntry, AuditFilters } from './types';
import { getLogger } from './StructuredLogger';

/**
 * Audit logger for tracking critical trading events
 * Maintains immutable audit trail for compliance and debugging
 */
export class AuditLogger implements IAuditLogger {
  private auditTrail: AuditLogEntry[] = [];
  private maxTrailSize: number;
  private logger = getLogger({ service: 'AuditLogger' });

  constructor(maxTrailSize: number = 10000) {
    this.maxTrailSize = maxTrailSize;
  }

  private generateAuditId(): string {
    return `audit_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private createAuditEntry(
    eventType: AuditLogEntry['event_type'],
    symbol: string,
    action: string,
    metadata?: Record<string, any>,
    beforeState?: Record<string, any>,
    afterState?: Record<string, any>
  ): AuditLogEntry {
    return {
      audit_id: this.generateAuditId(),
      timestamp: new Date().toISOString(),
      event_type: eventType,
      symbol,
      action,
      before_state: beforeState,
      after_state: afterState,
      metadata,
    };
  }

  private addToTrail(entry: AuditLogEntry): void {
    this.auditTrail.unshift(entry);

    // Enforce max size
    if (this.auditTrail.length > this.maxTrailSize) {
      this.auditTrail.pop();
    }

    this.logger.info('Audit event logged', {
      audit_id: entry.audit_id,
      event_type: entry.event_type,
    }, entry);
  }

  async logSignalGenerated(signal: any, metadata?: Record<string, any>): Promise<void> {
    const entry = this.createAuditEntry(
      'SIGNAL_GENERATED',
      signal.symbol || 'UNKNOWN',
      `Signal ${signal.action} generated with confidence ${signal.confidence}`,
      {
        ...metadata,
        signal_id: signal.signal_id,
        confidence: signal.confidence,
        model_version: signal.model_version,
      },
      undefined,
      {
        action: signal.action,
        price: signal.price,
        features: signal.features,
      }
    );

    this.addToTrail(entry);
  }

  async logSignalExecuted(
    signal: any,
    execution: any,
    metadata?: Record<string, any>
  ): Promise<void> {
    const entry = this.createAuditEntry(
      'SIGNAL_EXECUTED',
      signal.symbol || 'UNKNOWN',
      `Signal executed: ${signal.action} at ${execution.price}`,
      {
        ...metadata,
        signal_id: signal.signal_id,
        execution_id: execution.execution_id,
        execution_price: execution.price,
        execution_time: execution.timestamp,
      },
      {
        signal_action: signal.action,
        signal_price: signal.price,
      },
      {
        executed_price: execution.price,
        quantity: execution.quantity,
        status: execution.status,
      }
    );

    this.addToTrail(entry);
  }

  async logPositionOpened(position: any, metadata?: Record<string, any>): Promise<void> {
    const entry = this.createAuditEntry(
      'POSITION_OPENED',
      position.symbol || 'UNKNOWN',
      `Position opened: ${position.quantity} @ ${position.entry_price}`,
      {
        ...metadata,
        position_id: position.position_id,
        entry_price: position.entry_price,
        quantity: position.quantity,
      },
      undefined,
      {
        position_id: position.position_id,
        entry_price: position.entry_price,
        quantity: position.quantity,
        entry_time: position.entry_time,
      }
    );

    this.addToTrail(entry);
  }

  async logPositionClosed(
    position: any,
    pnl: number,
    metadata?: Record<string, any>
  ): Promise<void> {
    const entry = this.createAuditEntry(
      'POSITION_CLOSED',
      position.symbol || 'UNKNOWN',
      `Position closed with PnL: ${pnl}`,
      {
        ...metadata,
        position_id: position.position_id,
        pnl,
        pnl_percent: position.pnl_percent,
        exit_price: position.exit_price,
      },
      {
        position_id: position.position_id,
        entry_price: position.entry_price,
        status: 'OPEN',
      },
      {
        exit_price: position.exit_price,
        pnl,
        pnl_percent: position.pnl_percent,
        status: 'CLOSED',
      }
    );

    this.addToTrail(entry);
  }

  async logRiskAlert(alert: any, metadata?: Record<string, any>): Promise<void> {
    const entry = this.createAuditEntry(
      'RISK_ALERT',
      alert.symbol || 'SYSTEM',
      `Risk alert: ${alert.message}`,
      {
        ...metadata,
        alert_type: alert.type,
        severity: alert.severity,
        threshold: alert.threshold,
        current_value: alert.current_value,
      },
      undefined,
      alert
    );

    this.addToTrail(entry);
  }

  async logSystemEvent(event: string, metadata?: Record<string, any>): Promise<void> {
    const entry = this.createAuditEntry(
      'SYSTEM_EVENT',
      'SYSTEM',
      event,
      metadata
    );

    this.addToTrail(entry);
  }

  async getAuditTrail(filters?: AuditFilters): Promise<AuditLogEntry[]> {
    let filtered = [...this.auditTrail];

    if (filters) {
      if (filters.startTime) {
        filtered = filtered.filter(
          (entry) => new Date(entry.timestamp) >= filters.startTime!
        );
      }

      if (filters.endTime) {
        filtered = filtered.filter(
          (entry) => new Date(entry.timestamp) <= filters.endTime!
        );
      }

      if (filters.eventType) {
        filtered = filtered.filter((entry) => entry.event_type === filters.eventType);
      }

      if (filters.symbol) {
        filtered = filtered.filter((entry) => entry.symbol === filters.symbol);
      }

      if (filters.userId) {
        filtered = filtered.filter((entry) => entry.user_id === filters.userId);
      }

      if (filters.limit) {
        filtered = filtered.slice(0, filters.limit);
      }
    }

    return filtered;
  }

  /**
   * Get audit statistics
   */
  async getAuditStats(): Promise<{
    total_events: number;
    events_by_type: Record<string, number>;
    events_by_symbol: Record<string, number>;
    recent_events: number;
  }> {
    const eventsByType: Record<string, number> = {};
    const eventsBySymbol: Record<string, number> = {};
    const oneHourAgo = new Date(Date.now() - 3600000);

    let recentEvents = 0;

    for (const entry of this.auditTrail) {
      // Count by type
      eventsByType[entry.event_type] = (eventsByType[entry.event_type] || 0) + 1;

      // Count by symbol
      eventsBySymbol[entry.symbol] = (eventsBySymbol[entry.symbol] || 0) + 1;

      // Count recent events
      if (new Date(entry.timestamp) >= oneHourAgo) {
        recentEvents++;
      }
    }

    return {
      total_events: this.auditTrail.length,
      events_by_type: eventsByType,
      events_by_symbol: eventsBySymbol,
      recent_events: recentEvents,
    };
  }

  /**
   * Export audit trail
   */
  async exportAuditTrail(format: 'json' | 'csv' = 'json'): Promise<string> {
    if (format === 'json') {
      return JSON.stringify(this.auditTrail, null, 2);
    }

    // CSV format
    const headers = [
      'audit_id',
      'timestamp',
      'event_type',
      'symbol',
      'action',
      'metadata',
    ].join(',');

    const rows = this.auditTrail.map((entry) =>
      [
        entry.audit_id,
        entry.timestamp,
        entry.event_type,
        entry.symbol,
        entry.action,
        JSON.stringify(entry.metadata || {}),
      ].join(',')
    );

    return [headers, ...rows].join('\n');
  }

  /**
   * Clear audit trail
   */
  async clearAuditTrail(): Promise<void> {
    this.logger.warn('Clearing audit trail', undefined, {
      entries_cleared: this.auditTrail.length,
    });
    this.auditTrail = [];
  }
}

// Export singleton instance
let auditLoggerInstance: AuditLogger | null = null;

export const getAuditLogger = (): AuditLogger => {
  if (!auditLoggerInstance) {
    auditLoggerInstance = new AuditLogger();
  }
  return auditLoggerInstance;
};

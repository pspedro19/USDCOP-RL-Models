'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { AuditLogEntry } from '@/lib/services/logging/types';

interface AuditLogViewerProps {
  limit?: number;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export function AuditLogViewer({
  limit = 50,
  autoRefresh = true,
  refreshInterval = 30000,
}: AuditLogViewerProps) {
  const [entries, setEntries] = useState<AuditLogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>('all');

  const fetchAuditLog = async () => {
    try {
      const params = new URLSearchParams({ limit: limit.toString() });

      if (filter !== 'all') {
        params.append('eventType', filter);
      }

      const response = await fetch(`/api/audit/signals?${params}`);

      if (!response.ok) {
        throw new Error('Failed to fetch audit log');
      }

      const data = await response.json();
      setEntries(data.entries || []);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAuditLog();

    if (autoRefresh) {
      const interval = setInterval(fetchAuditLog, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [limit, filter, autoRefresh, refreshInterval]);

  const getEventTypeColor = (eventType: string) => {
    switch (eventType) {
      case 'SIGNAL_GENERATED':
        return 'bg-blue-100 text-blue-800';
      case 'SIGNAL_EXECUTED':
        return 'bg-green-100 text-green-800';
      case 'POSITION_OPENED':
        return 'bg-purple-100 text-purple-800';
      case 'POSITION_CLOSED':
        return 'bg-indigo-100 text-indigo-800';
      case 'RISK_ALERT':
        return 'bg-red-100 text-red-800';
      case 'SYSTEM_EVENT':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getEventTypeIcon = (eventType: string) => {
    switch (eventType) {
      case 'SIGNAL_GENERATED':
        return '📊';
      case 'SIGNAL_EXECUTED':
        return '✅';
      case 'POSITION_OPENED':
        return '📈';
      case 'POSITION_CLOSED':
        return '📉';
      case 'RISK_ALERT':
        return '⚠️';
      case 'SYSTEM_EVENT':
        return '🔧';
      default:
        return '📝';
    }
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="border-red-500">
        <CardContent className="p-6">
          <p className="text-red-500 text-center">{error}</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Audit Log</span>
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="text-sm border rounded px-3 py-1 font-normal"
          >
            <option value="all">All Events</option>
            <option value="SIGNAL_GENERATED">Signal Generated</option>
            <option value="SIGNAL_EXECUTED">Signal Executed</option>
            <option value="POSITION_OPENED">Position Opened</option>
            <option value="POSITION_CLOSED">Position Closed</option>
            <option value="RISK_ALERT">Risk Alerts</option>
            <option value="SYSTEM_EVENT">System Events</option>
          </select>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {entries.length === 0 ? (
          <p className="text-center text-gray-500 py-8">No audit entries found</p>
        ) : (
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {entries.map((entry) => (
              <div
                key={entry.audit_id}
                className="border border-gray-200 rounded-lg p-3 hover:shadow-md transition-shadow"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="text-lg">{getEventTypeIcon(entry.event_type)}</span>
                    <span
                      className={`text-xs px-2 py-1 rounded ${getEventTypeColor(
                        entry.event_type
                      )}`}
                    >
                      {entry.event_type.replace(/_/g, ' ')}
                    </span>
                    <span className="text-sm font-medium text-gray-700">
                      {entry.symbol}
                    </span>
                  </div>
                  <span className="text-xs text-gray-500">
                    {new Date(entry.timestamp).toLocaleString()}
                  </span>
                </div>

                <p className="text-sm text-gray-800 mb-2">{entry.action}</p>

                {(entry.before_state || entry.after_state) && (
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    {entry.before_state && (
                      <div className="bg-gray-50 p-2 rounded">
                        <p className="font-semibold text-gray-600 mb-1">Before:</p>
                        <pre className="overflow-x-auto">
                          {JSON.stringify(entry.before_state, null, 2)}
                        </pre>
                      </div>
                    )}
                    {entry.after_state && (
                      <div className="bg-gray-50 p-2 rounded">
                        <p className="font-semibold text-gray-600 mb-1">After:</p>
                        <pre className="overflow-x-auto">
                          {JSON.stringify(entry.after_state, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                )}

                {entry.metadata && Object.keys(entry.metadata).length > 0 && (
                  <details className="mt-2">
                    <summary className="text-xs text-gray-600 cursor-pointer hover:text-gray-800">
                      Metadata
                    </summary>
                    <pre className="text-xs mt-1 p-2 bg-gray-50 rounded overflow-x-auto">
                      {JSON.stringify(entry.metadata, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

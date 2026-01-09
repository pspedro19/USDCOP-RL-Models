'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ServiceStatusCard } from './ServiceStatusCard';
import { LatencyChart } from './LatencyChart';
import { AuditLogViewer } from './AuditLogViewer';
import type { SystemHealth } from '@/lib/services/health/types';
import { AlertTriangle, Database } from 'lucide-react';

interface SystemHealthDashboardProps {
  refreshInterval?: number; // milliseconds
  showLatency?: boolean;
  showAuditLog?: boolean;
}

export function SystemHealthDashboard({
  refreshInterval = 30000,
  showLatency = true,
  showAuditLog = true,
}: SystemHealthDashboardProps) {
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const fetchSystemHealth = async () => {
    try {
      const response = await fetch('/api/health/services');
      if (!response.ok) {
        throw new Error('Failed to fetch system health');
      }

      const data = await response.json();
      setSystemHealth(data);
      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSystemHealth();

    const interval = setInterval(fetchSystemHealth, refreshInterval);

    return () => clearInterval(interval);
  }, [refreshInterval]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'text-green-500';
      case 'degraded':
        return 'text-yellow-500';
      case 'unhealthy':
        return 'text-red-500';
      default:
        return 'text-gray-500';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return '✓';
      case 'degraded':
        return '⚠';
      case 'unhealthy':
        return '✗';
      default:
        return '?';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading system health...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 bg-red-900/20 rounded-lg border border-red-500/30">
        <div className="flex items-center gap-2 text-red-400 mb-4">
          <AlertTriangle className="w-5 h-5" />
          <span className="font-semibold">Failed to load system health</span>
        </div>
        <p className="text-sm text-red-400/80 mb-4">{error}</p>
        <button
          onClick={fetchSystemHealth}
          className="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 rounded-lg text-sm text-white transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!systemHealth) {
    return (
      <div className="p-6 bg-fintech-dark-900 rounded-lg border border-fintech-dark-700 text-center">
        <div className="text-fintech-dark-400">
          <Database className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p>No system health data available</p>
          <button
            onClick={fetchSystemHealth}
            className="mt-4 px-4 py-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg text-sm text-white transition-colors"
          >
            Load System Health
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Overall System Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>System Health Overview</span>
            <span className={`text-2xl ${getStatusColor(systemHealth.overall_status)}`}>
              {getStatusIcon(systemHealth.overall_status)}
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <p className="text-2xl font-bold">{systemHealth.summary.total_services}</p>
              <p className="text-sm text-gray-600">Total Services</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-green-500">
                {systemHealth.summary.healthy_services}
              </p>
              <p className="text-sm text-gray-600">Healthy</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-yellow-500">
                {systemHealth.summary.degraded_services}
              </p>
              <p className="text-sm text-gray-600">Degraded</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-red-500">
                {systemHealth.summary.unhealthy_services}
              </p>
              <p className="text-sm text-gray-600">Unhealthy</p>
            </div>
          </div>

          {lastUpdate && (
            <p className="text-xs text-gray-500 mt-4 text-center">
              Last updated: {lastUpdate.toLocaleTimeString()}
            </p>
          )}
        </CardContent>
      </Card>

      {/* Individual Service Status */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Service Status</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {systemHealth.services.map((service) => (
            <ServiceStatusCard key={service.service} service={service} />
          ))}
        </div>
      </div>

      {/* Latency Chart */}
      {showLatency && (
        <div>
          <h3 className="text-lg font-semibold mb-4">Latency Monitoring</h3>
          <LatencyChart refreshInterval={refreshInterval} />
        </div>
      )}

      {/* Audit Log */}
      {showAuditLog && (
        <div>
          <h3 className="text-lg font-semibold mb-4">Recent Audit Events</h3>
          <AuditLogViewer limit={10} />
        </div>
      )}
    </div>
  );
}

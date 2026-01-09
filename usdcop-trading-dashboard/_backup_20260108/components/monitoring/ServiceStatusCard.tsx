'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { ServiceHealth } from '@/lib/services/health/types';

interface ServiceStatusCardProps {
  service: ServiceHealth;
}

export function ServiceStatusCard({ service }: ServiceStatusCardProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'bg-green-500';
      case 'degraded':
        return 'bg-yellow-500';
      case 'unhealthy':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getStatusTextColor = (status: string) => {
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

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);

    if (days > 0) {
      return `${days}d ${hours}h`;
    }
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  const formatMemory = (mb: number) => {
    if (mb > 1024) {
      return `${(mb / 1024).toFixed(2)} GB`;
    }
    return `${mb.toFixed(2)} MB`;
  };

  return (
    <Card className="hover:shadow-lg transition-shadow">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between text-base">
          <span className="capitalize">{service.service.replace(/-/g, ' ')}</span>
          <div className={`h-3 w-3 rounded-full ${getStatusColor(service.status)}`}></div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Status */}
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-600">Status</span>
          <span className={`text-sm font-semibold capitalize ${getStatusTextColor(service.status)}`}>
            {service.status}
          </span>
        </div>

        {/* Latency */}
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-600">Latency</span>
          <span className="text-sm font-medium">
            {service.latency_ms.toFixed(0)} ms
          </span>
        </div>

        {/* Uptime */}
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-600">Uptime</span>
          <span className="text-sm font-medium">
            {formatUptime(service.details.uptime)}
          </span>
        </div>

        {/* Error Rate */}
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-600">Error Rate</span>
          <span className="text-sm font-medium">
            {(service.details.error_rate * 100).toFixed(2)}%
          </span>
        </div>

        {/* Memory Usage */}
        {service.details.memory_usage_mb > 0 && (
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-600">Memory</span>
            <span className="text-sm font-medium">
              {formatMemory(service.details.memory_usage_mb)}
            </span>
          </div>
        )}

        {/* Requests per Minute */}
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-600">Req/min</span>
          <span className="text-sm font-medium">
            {service.details.requests_per_minute.toFixed(1)}
          </span>
        </div>

        {/* Last Check */}
        <div className="pt-2 border-t border-gray-200">
          <p className="text-xs text-gray-500 text-center">
            Last check: {new Date(service.last_check).toLocaleTimeString()}
          </p>
        </div>

        {/* Metadata */}
        {service.metadata && Object.keys(service.metadata).length > 0 && (
          <details className="text-xs">
            <summary className="cursor-pointer text-gray-600 hover:text-gray-800">
              Details
            </summary>
            <pre className="mt-2 p-2 bg-gray-50 rounded overflow-x-auto">
              {JSON.stringify(service.metadata, null, 2)}
            </pre>
          </details>
        )}
      </CardContent>
    </Card>
  );
}

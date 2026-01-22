'use client';

/**
 * AlertsPanel.tsx
 * ===============
 * Phase 2.2: Alerts Panel UI
 *
 * Sheet component for displaying system alerts with:
 * - Bell icon with badge counter
 * - Sheet component showing active/resolved alerts
 * - Color coding by severity (critical, warning, info)
 * - Auto-refresh every 30 seconds
 */

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Bell,
  AlertTriangle,
  Info,
  XCircle,
  CheckCircle,
  ExternalLink,
  RefreshCw,
  Clock
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

interface Alert {
  id: string;
  severity: 'critical' | 'warning' | 'info';
  title: string;
  description: string;
  timestamp: string;
  status: 'firing' | 'resolved';
  labels: Record<string, string>;
  runbook?: string;
}

interface AlertsPanelProps {
  className?: string;
}

export function AlertsPanel({ className }: AlertsPanelProps) {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  const fetchAlerts = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/v1/alerts');
      if (response.ok) {
        const data = await response.json();
        setAlerts(data.alerts || []);
        setLastRefresh(new Date());
      }
    } catch (error) {
      console.error('Failed to fetch alerts:', error);
      // Fallback to mock data for demo
      setAlerts([
        {
          id: '1',
          severity: 'warning',
          title: 'Low Sharpe Ratio',
          description: '30-day Sharpe ratio below 0.5',
          timestamp: new Date().toISOString(),
          status: 'firing',
          labels: { team: 'trading' },
          runbook: 'https://docs.internal/runbooks/low-sharpe'
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    // Initial fetch
    fetchAlerts();

    // Poll for alerts every 30 seconds
    const interval = setInterval(fetchAlerts, 30000);
    return () => clearInterval(interval);
  }, [fetchAlerts]);

  const firingAlerts = alerts.filter(a => a.status === 'firing');
  const resolvedAlerts = alerts.filter(a => a.status === 'resolved');
  const criticalCount = firingAlerts.filter(a => a.severity === 'critical').length;
  const warningCount = firingAlerts.filter(a => a.severity === 'warning').length;

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <XCircle className="h-4 w-4 text-red-400" />;
      case 'warning':
        return <AlertTriangle className="h-4 w-4 text-amber-400" />;
      default:
        return <Info className="h-4 w-4 text-cyan-400" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-500/10 border-red-500/30';
      case 'warning':
        return 'bg-amber-500/10 border-amber-500/30';
      default:
        return 'bg-cyan-500/10 border-cyan-500/30';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleString('es-CO', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return timestamp;
    }
  };

  return (
    <>
      {/* Trigger Button */}
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setIsOpen(true)}
        className={cn("relative", className)}
      >
        <Bell className="h-5 w-5" />

        {/* Badge Counter */}
        {firingAlerts.length > 0 && (
          <span className="absolute -top-1 -right-1 flex h-5 w-5 items-center justify-center">
            {/* Ping animation */}
            <span className={cn(
              "absolute inline-flex h-full w-full animate-ping rounded-full opacity-75",
              criticalCount > 0 ? "bg-red-400" : "bg-amber-400"
            )} />
            {/* Badge */}
            <span className={cn(
              "relative inline-flex h-4 w-4 items-center justify-center rounded-full text-[10px] font-bold text-white",
              criticalCount > 0 ? "bg-red-500" : "bg-amber-500"
            )}>
              {firingAlerts.length > 9 ? '9+' : firingAlerts.length}
            </span>
          </span>
        )}
      </Button>

      {/* Sheet Overlay */}
      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsOpen(false)}
              className="fixed inset-0 bg-black/40 backdrop-blur-sm z-50"
            />

            {/* Sheet */}
            <motion.div
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 30, stiffness: 300 }}
              className="fixed right-0 top-0 bottom-0 w-full max-w-md z-50 flex flex-col"
            >
              <div className="h-full bg-slate-900/95 backdrop-blur-xl border-l border-slate-700/50 shadow-2xl flex flex-col">
                {/* Header */}
                <div className="p-6 border-b border-slate-700/50">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-xl bg-cyan-500/20 border border-cyan-500/30">
                        <Bell className="h-5 w-5 text-cyan-400" />
                      </div>
                      <div>
                        <h2 className="text-lg font-bold text-slate-100">System Alerts</h2>
                        <p className="text-sm text-slate-400">
                          {firingAlerts.length === 0 ? (
                            'No active alerts'
                          ) : (
                            `${firingAlerts.length} active alert${firingAlerts.length > 1 ? 's' : ''}`
                          )}
                        </p>
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={fetchAlerts}
                      disabled={isLoading}
                      className="text-slate-400 hover:text-white"
                    >
                      <RefreshCw className={cn("h-4 w-4", isLoading && "animate-spin")} />
                    </Button>
                  </div>

                  {/* Summary Badges */}
                  <div className="flex items-center gap-2 mt-4">
                    {criticalCount > 0 && (
                      <Badge variant="destructive" className="gap-1">
                        <XCircle className="h-3 w-3" />
                        {criticalCount} Critical
                      </Badge>
                    )}
                    {warningCount > 0 && (
                      <Badge variant="warning" className="gap-1">
                        <AlertTriangle className="h-3 w-3" />
                        {warningCount} Warning
                      </Badge>
                    )}
                    {firingAlerts.length === 0 && (
                      <span className="text-sm text-emerald-400 flex items-center gap-1">
                        <CheckCircle className="h-4 w-4" />
                        All systems operational
                      </span>
                    )}
                  </div>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-6 space-y-4">
                  {/* Firing Alerts */}
                  {firingAlerts.length > 0 && (
                    <div className="space-y-3">
                      <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                        Active Alerts
                      </h3>
                      {firingAlerts.map((alert) => (
                        <motion.div
                          key={alert.id}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          className={cn(
                            "p-4 rounded-xl border",
                            getSeverityColor(alert.severity)
                          )}
                        >
                          <div className="flex items-start gap-3">
                            {getSeverityIcon(alert.severity)}
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center justify-between gap-2 mb-1">
                                <h4 className="font-medium text-slate-100 truncate">
                                  {alert.title}
                                </h4>
                                <Badge
                                  variant={alert.status === 'firing' ? 'destructive' : 'outline'}
                                  className="flex-shrink-0"
                                >
                                  {alert.status}
                                </Badge>
                              </div>
                              <p className="text-sm text-slate-300 mb-2">
                                {alert.description}
                              </p>
                              <div className="flex items-center gap-4 text-xs text-slate-500">
                                <span className="flex items-center gap-1">
                                  <Clock className="h-3 w-3" />
                                  {formatTimestamp(alert.timestamp)}
                                </span>
                                {alert.labels.team && (
                                  <span>Team: {alert.labels.team}</span>
                                )}
                              </div>
                              {alert.runbook && (
                                <a
                                  href={alert.runbook}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="inline-flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300 mt-2"
                                >
                                  View Runbook
                                  <ExternalLink className="h-3 w-3" />
                                </a>
                              )}
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  )}

                  {/* Resolved Alerts */}
                  {resolvedAlerts.length > 0 && (
                    <div className="space-y-3">
                      <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                        Recently Resolved
                      </h3>
                      {resolvedAlerts.slice(0, 5).map((alert) => (
                        <motion.div
                          key={alert.id}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="p-4 rounded-xl border bg-slate-800/30 border-slate-700/30 opacity-60"
                        >
                          <div className="flex items-start gap-3">
                            <CheckCircle className="h-4 w-4 text-emerald-400" />
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center justify-between gap-2 mb-1">
                                <h4 className="font-medium text-slate-400 truncate">
                                  {alert.title}
                                </h4>
                                <Badge variant="outline" className="text-emerald-400 flex-shrink-0">
                                  resolved
                                </Badge>
                              </div>
                              <p className="text-sm text-slate-500 mb-2">
                                {alert.description}
                              </p>
                              <span className="text-xs text-slate-600 flex items-center gap-1">
                                <Clock className="h-3 w-3" />
                                {formatTimestamp(alert.timestamp)}
                              </span>
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  )}

                  {/* Empty State */}
                  {alerts.length === 0 && !isLoading && (
                    <div className="flex flex-col items-center justify-center h-64 text-center">
                      <div className="p-4 rounded-full bg-emerald-500/10 border border-emerald-500/30 mb-4">
                        <CheckCircle className="h-8 w-8 text-emerald-400" />
                      </div>
                      <h3 className="text-lg font-medium text-slate-200 mb-1">
                        All Clear
                      </h3>
                      <p className="text-sm text-slate-500">
                        No alerts to display. System is operating normally.
                      </p>
                    </div>
                  )}

                  {/* Loading State */}
                  {isLoading && alerts.length === 0 && (
                    <div className="space-y-3">
                      {[1, 2, 3].map((i) => (
                        <div
                          key={i}
                          className="h-24 rounded-xl bg-slate-800/50 animate-pulse"
                        />
                      ))}
                    </div>
                  )}
                </div>

                {/* Footer */}
                <div className="p-4 border-t border-slate-700/50 flex items-center justify-between">
                  {lastRefresh && (
                    <span className="text-xs text-slate-500">
                      Last updated: {lastRefresh.toLocaleTimeString('es-CO')}
                    </span>
                  )}
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setIsOpen(false)}
                  >
                    Close
                  </Button>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}

export default AlertsPanel;

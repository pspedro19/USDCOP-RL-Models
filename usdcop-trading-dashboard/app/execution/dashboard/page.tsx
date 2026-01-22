'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  Activity,
  AlertTriangle,
  TrendingUp,
  Zap,
  Shield,
  RefreshCw,
  Clock,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Power,
} from 'lucide-react';
import { signalBridgeService } from '@/lib/services/execution';
import {
  type BridgeStatus,
  type BridgeStatistics,
  type TradingMode,
  TRADING_MODE_LABELS,
  TRADING_MODE_COLORS,
  formatUptime,
  isBridgeOperational,
} from '@/lib/contracts/execution/signal-bridge.contract';

export default function ExecutionDashboardPage() {
  const [status, setStatus] = useState<BridgeStatus | null>(null);
  const [statistics, setStatistics] = useState<BridgeStatistics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isKillSwitchLoading, setIsKillSwitchLoading] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      setError(null);
      const [statusData, statsData] = await Promise.all([
        signalBridgeService.getStatus(),
        signalBridgeService.getStatistics(7),
      ]);
      setStatus(statusData);
      setStatistics(statsData);
    } catch (err) {
      console.error('Failed to fetch bridge data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, [fetchData]);

  const handleKillSwitch = async () => {
    if (!status) return;

    const action = status.kill_switch_active ? 'deactivate' : 'activate';
    const confirmed = window.confirm(
      status.kill_switch_active
        ? 'Are you sure you want to DEACTIVATE the kill switch? Trading will resume.'
        : 'Are you sure you want to ACTIVATE the kill switch? All trading will stop immediately.'
    );

    if (!confirmed) return;

    setIsKillSwitchLoading(true);
    try {
      if (status.kill_switch_active) {
        await signalBridgeService.deactivateKillSwitch();
      } else {
        await signalBridgeService.activateKillSwitch('Manual activation from dashboard');
      }
      await fetchData();
    } catch (err) {
      console.error('Kill switch error:', err);
      alert('Failed to toggle kill switch');
    } finally {
      setIsKillSwitchLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <RefreshCw className="w-8 h-8 text-cyan-500 animate-spin" />
          <p className="text-gray-400">Loading bridge status...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-white mb-2">Connection Error</h2>
          <p className="text-gray-400 mb-4">{error}</p>
          <button
            onClick={fetchData}
            className="px-4 py-2 bg-cyan-500 text-white rounded-lg hover:bg-cyan-600 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const isOperational = status ? isBridgeOperational(status) : false;

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Execution Dashboard</h1>
          <p className="text-gray-400">Real-time SignalBridge monitoring</p>
        </div>
        <button
          onClick={fetchData}
          className="p-2 text-gray-400 hover:text-white transition-colors"
        >
          <RefreshCw className="w-5 h-5" />
        </button>
      </div>

      {/* Kill Switch Alert */}
      {status?.kill_switch_active && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-red-500/20 border border-red-500/50 rounded-xl p-4 flex items-center gap-4"
        >
          <AlertTriangle className="w-6 h-6 text-red-500 flex-shrink-0" />
          <div className="flex-1">
            <h3 className="font-bold text-red-400">Kill Switch Active</h3>
            <p className="text-sm text-red-300">{status.kill_switch_reason || 'Trading halted'}</p>
          </div>
          <button
            onClick={handleKillSwitch}
            disabled={isKillSwitchLoading}
            className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 disabled:opacity-50"
          >
            {isKillSwitchLoading ? 'Processing...' : 'Deactivate'}
          </button>
        </motion.div>
      )}

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Bridge Status */}
        <div className="bg-gray-900/50 border border-gray-800/50 rounded-xl p-5">
          <div className="flex items-center justify-between mb-3">
            <span className="text-gray-400 text-sm">Bridge Status</span>
            <Activity className={`w-5 h-5 ${isOperational ? 'text-green-400' : 'text-red-400'}`} />
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${isOperational ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
            <span className="text-xl font-bold text-white">
              {isOperational ? 'Operational' : 'Stopped'}
            </span>
          </div>
          {status && (
            <p className="text-xs text-gray-500 mt-2">
              Uptime: {formatUptime(status.uptime_seconds)}
            </p>
          )}
        </div>

        {/* Trading Mode */}
        <div className="bg-gray-900/50 border border-gray-800/50 rounded-xl p-5">
          <div className="flex items-center justify-between mb-3">
            <span className="text-gray-400 text-sm">Trading Mode</span>
            <Shield className="w-5 h-5 text-cyan-400" />
          </div>
          {status && (
            <>
              <span className={`text-xl font-bold ${TRADING_MODE_COLORS[status.trading_mode]?.split(' ')[0] || 'text-gray-400'}`}>
                {TRADING_MODE_LABELS[status.trading_mode]}
              </span>
              <p className="text-xs text-gray-500 mt-2">
                {status.inference_ws_connected ? 'WS Connected' : 'WS Disconnected'}
              </p>
            </>
          )}
        </div>

        {/* Executions Today */}
        <div className="bg-gray-900/50 border border-gray-800/50 rounded-xl p-5">
          <div className="flex items-center justify-between mb-3">
            <span className="text-gray-400 text-sm">Executions (7d)</span>
            <Zap className="w-5 h-5 text-amber-400" />
          </div>
          <span className="text-xl font-bold text-white">
            {statistics?.total_executions || 0}
          </span>
          <div className="flex items-center gap-2 mt-2 text-xs">
            <span className="text-green-400">{statistics?.successful_executions || 0} success</span>
            <span className="text-gray-500">|</span>
            <span className="text-red-400">{statistics?.failed_executions || 0} failed</span>
          </div>
        </div>

        {/* P&L */}
        <div className="bg-gray-900/50 border border-gray-800/50 rounded-xl p-5">
          <div className="flex items-center justify-between mb-3">
            <span className="text-gray-400 text-sm">Total P&L (7d)</span>
            <TrendingUp className="w-5 h-5 text-green-400" />
          </div>
          <span className={`text-xl font-bold ${(statistics?.total_pnl_usd || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            ${(statistics?.total_pnl_usd || 0).toFixed(2)}
          </span>
          <p className="text-xs text-gray-500 mt-2">
            Volume: ${(statistics?.total_volume_usd || 0).toLocaleString()}
          </p>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Kill Switch Control */}
        <div className="bg-gray-900/50 border border-gray-800/50 rounded-xl p-6">
          <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <Power className="w-5 h-5 text-cyan-400" />
            Emergency Controls
          </h2>

          <div className="space-y-4">
            <div className="p-4 bg-gray-800/50 rounded-lg">
              <div className="flex items-center justify-between mb-3">
                <span className="text-gray-300">Kill Switch</span>
                <span className={`text-sm px-2 py-1 rounded-full ${
                  status?.kill_switch_active
                    ? 'bg-red-500/20 text-red-400'
                    : 'bg-green-500/20 text-green-400'
                }`}>
                  {status?.kill_switch_active ? 'ACTIVE' : 'INACTIVE'}
                </span>
              </div>
              <button
                onClick={handleKillSwitch}
                disabled={isKillSwitchLoading}
                className={`w-full py-3 rounded-lg font-bold transition-all ${
                  status?.kill_switch_active
                    ? 'bg-green-500/20 text-green-400 hover:bg-green-500/30 border border-green-500/30'
                    : 'bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/30'
                } disabled:opacity-50`}
              >
                {isKillSwitchLoading
                  ? 'Processing...'
                  : status?.kill_switch_active
                    ? 'Resume Trading'
                    : 'Stop All Trading'
                }
              </button>
            </div>

            <p className="text-xs text-gray-500">
              The kill switch immediately halts all trading activity across all exchanges.
              Use in case of emergency or unexpected market conditions.
            </p>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="lg:col-span-2 bg-gray-900/50 border border-gray-800/50 rounded-xl p-6">
          <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5 text-cyan-400" />
            System Status
          </h2>

          <div className="space-y-3">
            {/* Connection Status Items */}
            {[
              {
                label: 'Inference WebSocket',
                status: status?.inference_ws_connected,
                icon: Activity,
              },
              {
                label: 'Connected Exchanges',
                status: (status?.connected_exchanges?.length || 0) > 0,
                detail: status?.connected_exchanges?.join(', ') || 'None',
                icon: Link2Icon,
              },
              {
                label: 'Pending Executions',
                status: true,
                detail: `${status?.pending_executions || 0} pending`,
                icon: Clock,
              },
            ].map((item, idx) => {
              const Icon = item.icon;
              return (
                <div
                  key={idx}
                  className="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg"
                >
                  <div className="flex items-center gap-3">
                    <Icon className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-300">{item.label}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {item.detail && (
                      <span className="text-sm text-gray-500">{item.detail}</span>
                    )}
                    {item.status ? (
                      <CheckCircle2 className="w-5 h-5 text-green-400" />
                    ) : (
                      <XCircle className="w-5 h-5 text-red-400" />
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Stats Summary */}
          {statistics && (
            <div className="mt-6 pt-4 border-t border-gray-800/50">
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <p className="text-2xl font-bold text-white">{statistics.total_signals_received}</p>
                  <p className="text-xs text-gray-500">Signals Received</p>
                </div>
                <div>
                  <p className="text-2xl font-bold text-amber-400">{statistics.blocked_by_risk}</p>
                  <p className="text-xs text-gray-500">Blocked by Risk</p>
                </div>
                <div>
                  <p className="text-2xl font-bold text-cyan-400">
                    {statistics.avg_execution_time_ms.toFixed(0)}ms
                  </p>
                  <p className="text-xs text-gray-500">Avg Exec Time</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Helper component for the link icon
function Link2Icon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M9 17H7A5 5 0 0 1 7 7h2" />
      <path d="M15 7h2a5 5 0 1 1 0 10h-2" />
      <line x1="8" x2="16" y1="12" y2="12" />
    </svg>
  );
}

'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, Power, PlayCircle, Shield, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useNotifications } from '@/components/ui/notification-manager';

interface KillSwitchProps {
  className?: string;
  compact?: boolean;
}

interface OperationsStatus {
  mode: 'normal' | 'paused' | 'killed';
  kill_switch_active: boolean;
  activated_at: string | null;
  activated_by: string | null;
  reason: string | null;
  timestamp: string;
}

export function KillSwitch({ className, compact = false }: KillSwitchProps) {
  const [status, setStatus] = useState<OperationsStatus | null>(null);
  const [reason, setReason] = useState('');
  const [confirmCode, setConfirmCode] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showKillDialog, setShowKillDialog] = useState(false);
  const [showResumeDialog, setShowResumeDialog] = useState(false);
  const { addNotification } = useNotifications();

  const isKilled = status?.kill_switch_active || false;

  // Fetch current status
  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/v1/operations/status');
      if (response.ok) {
        const data = await response.json();
        setStatus(data);
      }
    } catch (error) {
      console.error('Failed to fetch operations status:', error);
    }
  }, []);

  // Poll for status updates
  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 10000); // Poll every 10 seconds
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const activateKillSwitch = async () => {
    if (!reason.trim()) {
      addNotification({
        type: 'error',
        title: 'Error',
        message: 'Debe proporcionar una razon para activar el kill switch',
        duration: 5000
      });
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch('/api/v1/operations/kill-switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reason,
          activated_by: 'dashboard',
          close_positions: true,
          notify_team: true,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        setStatus({
          mode: 'killed',
          kill_switch_active: true,
          activated_at: result.activated_at,
          activated_by: 'dashboard',
          reason: reason,
          timestamp: new Date().toISOString()
        });
        setShowKillDialog(false);
        setReason('');
        addNotification({
          type: 'alert',
          title: 'KILL SWITCH ACTIVADO',
          message: `Trading detenido. ${result.positions_closed} posiciones cerradas.`,
          duration: 10000
        });
      } else {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to activate kill switch');
      }
    } catch (error: any) {
      addNotification({
        type: 'error',
        title: 'Error',
        message: error.message || 'No se pudo activar kill switch',
        duration: 5000
      });
    } finally {
      setIsLoading(false);
    }
  };

  const resumeTrading = async () => {
    if (confirmCode !== 'CONFIRM_RESUME') {
      addNotification({
        type: 'error',
        title: 'Error',
        message: 'Codigo de confirmacion incorrecto. Escriba CONFIRM_RESUME',
        duration: 5000
      });
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch('/api/v1/operations/resume', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          resumed_by: 'dashboard',
          confirmation_code: confirmCode,
        }),
      });

      if (response.ok) {
        setStatus({
          mode: 'normal',
          kill_switch_active: false,
          activated_at: null,
          activated_by: null,
          reason: null,
          timestamp: new Date().toISOString()
        });
        setShowResumeDialog(false);
        setConfirmCode('');
        addNotification({
          type: 'info',
          title: 'Trading Resumido',
          message: 'El sistema ha vuelto a operar normalmente.',
          duration: 5000
        });
      } else {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to resume trading');
      }
    } catch (error: any) {
      addNotification({
        type: 'error',
        title: 'Error',
        message: error.message || 'No se pudo resumir trading',
        duration: 5000
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Compact mode for header integration
  if (compact) {
    return (
      <div className={className}>
        {!isKilled ? (
          <Button
            variant="destructive"
            size="sm"
            onClick={() => setShowKillDialog(true)}
            className="gap-1.5 font-semibold"
          >
            <Power className="h-4 w-4" />
            KILL
          </Button>
        ) : (
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowResumeDialog(true)}
            className="gap-1.5 border-red-500/50 text-red-400 hover:bg-red-500/10"
          >
            <AlertCircle className="h-4 w-4 animate-pulse" />
            KILLED
          </Button>
        )}

        {/* Kill Switch Dialog */}
        <AnimatePresence>
          {showKillDialog && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
              onClick={() => setShowKillDialog(false)}
            >
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.9, opacity: 0 }}
                className="w-full max-w-md p-6 bg-slate-900 border border-red-500/30 rounded-xl shadow-xl"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex items-center gap-3 mb-4">
                  <div className="p-2 bg-red-500/20 rounded-lg">
                    <AlertTriangle className="h-6 w-6 text-red-500" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-red-400">Activar Kill Switch</h3>
                    <p className="text-sm text-slate-400">Esta accion detendra TODO el trading</p>
                  </div>
                </div>

                <div className="p-4 mb-4 bg-red-500/10 border border-red-500/20 rounded-lg">
                  <p className="text-sm text-red-300">
                    Esta accion detendera TODO el trading inmediatamente y cerrara
                    todas las posiciones abiertas. Esta accion es irreversible
                    hasta que se reactive manualmente con codigo de confirmacion.
                  </p>
                </div>

                <div className="mb-4">
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Razon (requerido):
                  </label>
                  <input
                    type="text"
                    value={reason}
                    onChange={(e) => setReason(e.target.value)}
                    placeholder="Ej: Perdida excesiva, error del sistema..."
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-slate-200 placeholder-slate-500 focus:outline-none focus:border-red-500"
                  />
                </div>

                <div className="flex gap-3">
                  <Button
                    variant="outline"
                    className="flex-1"
                    onClick={() => setShowKillDialog(false)}
                    disabled={isLoading}
                  >
                    Cancelar
                  </Button>
                  <Button
                    variant="destructive"
                    className="flex-1"
                    onClick={activateKillSwitch}
                    disabled={isLoading || !reason.trim()}
                    loading={isLoading}
                  >
                    {isLoading ? 'Activando...' : 'CONFIRMAR KILL SWITCH'}
                  </Button>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Resume Dialog */}
        <AnimatePresence>
          {showResumeDialog && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
              onClick={() => setShowResumeDialog(false)}
            >
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.9, opacity: 0 }}
                className="w-full max-w-md p-6 bg-slate-900 border border-emerald-500/30 rounded-xl shadow-xl"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex items-center gap-3 mb-4">
                  <div className="p-2 bg-emerald-500/20 rounded-lg">
                    <PlayCircle className="h-6 w-6 text-emerald-500" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-emerald-400">Resumir Trading</h3>
                    <p className="text-sm text-slate-400">Reactivar el sistema de trading</p>
                  </div>
                </div>

                {status?.reason && (
                  <div className="p-3 mb-4 bg-slate-800 border border-slate-600 rounded-lg">
                    <p className="text-xs text-slate-400 mb-1">Razon del kill switch:</p>
                    <p className="text-sm text-slate-200">{status.reason}</p>
                    <p className="text-xs text-slate-500 mt-1">
                      Por: {status.activated_by} | {status.activated_at && new Date(status.activated_at).toLocaleString()}
                    </p>
                  </div>
                )}

                <div className="mb-4">
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Para resumir, escriba <span className="font-mono text-emerald-400">CONFIRM_RESUME</span>:
                  </label>
                  <input
                    type="text"
                    value={confirmCode}
                    onChange={(e) => setConfirmCode(e.target.value.toUpperCase())}
                    placeholder="CONFIRM_RESUME"
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-slate-200 font-mono placeholder-slate-500 focus:outline-none focus:border-emerald-500"
                  />
                </div>

                <div className="flex gap-3">
                  <Button
                    variant="outline"
                    className="flex-1"
                    onClick={() => setShowResumeDialog(false)}
                    disabled={isLoading}
                  >
                    Cancelar
                  </Button>
                  <Button
                    variant="gradient"
                    className="flex-1"
                    onClick={resumeTrading}
                    disabled={isLoading || confirmCode !== 'CONFIRM_RESUME'}
                    loading={isLoading}
                  >
                    {isLoading ? 'Resumiendo...' : 'Resumir Trading'}
                  </Button>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  }

  // Full mode for dedicated panel
  return (
    <div className={`p-4 rounded-xl border-2 ${
      isKilled
        ? 'border-red-500/50 bg-red-500/10'
        : 'border-slate-600/50 bg-slate-800/50'
    } ${className}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${
            isKilled ? 'bg-red-500/20' : 'bg-amber-500/20'
          }`}>
            {isKilled ? (
              <AlertTriangle className="h-5 w-5 text-red-500 animate-pulse" />
            ) : (
              <Shield className="h-5 w-5 text-amber-500" />
            )}
          </div>
          <div>
            <span className={`font-semibold ${
              isKilled ? 'text-red-400' : 'text-slate-200'
            }`}>
              {isKilled ? 'TRADING DETENIDO' : 'Kill Switch'}
            </span>
            {isKilled && status?.reason && (
              <p className="text-xs text-slate-400 mt-0.5">
                Razon: {status.reason}
              </p>
            )}
          </div>
        </div>

        {!isKilled ? (
          <Button
            variant="destructive"
            size="lg"
            onClick={() => setShowKillDialog(true)}
            className="gap-2"
          >
            <Power className="h-5 w-5" />
            ACTIVAR KILL SWITCH
          </Button>
        ) : (
          <Button
            variant="outline"
            size="lg"
            onClick={() => setShowResumeDialog(true)}
            className="gap-2 border-emerald-500/50 text-emerald-400 hover:bg-emerald-500/10"
          >
            <PlayCircle className="h-5 w-5" />
            RESUMIR TRADING
          </Button>
        )}
      </div>

      {isKilled && status && (
        <div className="mt-4 pt-4 border-t border-red-500/20">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-slate-400">Activado por:</span>
              <span className="ml-2 text-slate-200">{status.activated_by}</span>
            </div>
            <div>
              <span className="text-slate-400">Fecha:</span>
              <span className="ml-2 text-slate-200">
                {status.activated_at && new Date(status.activated_at).toLocaleString()}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Kill Switch Dialog - Same as compact mode */}
      <AnimatePresence>
        {showKillDialog && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
            onClick={() => setShowKillDialog(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="w-full max-w-md p-6 bg-slate-900 border border-red-500/30 rounded-xl shadow-xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-red-500/20 rounded-lg">
                  <AlertTriangle className="h-6 w-6 text-red-500" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-red-400">Activar Kill Switch</h3>
                  <p className="text-sm text-slate-400">Esta accion detendra TODO el trading</p>
                </div>
              </div>

              <div className="p-4 mb-4 bg-red-500/10 border border-red-500/20 rounded-lg">
                <p className="text-sm text-red-300">
                  Esta accion detendera TODO el trading inmediatamente y cerrara
                  todas las posiciones abiertas. Esta accion es irreversible
                  hasta que se reactive manualmente con codigo de confirmacion.
                </p>
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Razon (requerido):
                </label>
                <input
                  type="text"
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                  placeholder="Ej: Perdida excesiva, error del sistema..."
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-slate-200 placeholder-slate-500 focus:outline-none focus:border-red-500"
                />
              </div>

              <div className="flex gap-3">
                <Button
                  variant="outline"
                  className="flex-1"
                  onClick={() => setShowKillDialog(false)}
                  disabled={isLoading}
                >
                  Cancelar
                </Button>
                <Button
                  variant="destructive"
                  className="flex-1"
                  onClick={activateKillSwitch}
                  disabled={isLoading || !reason.trim()}
                  loading={isLoading}
                >
                  {isLoading ? 'Activando...' : 'CONFIRMAR KILL SWITCH'}
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Resume Dialog - Same as compact mode */}
      <AnimatePresence>
        {showResumeDialog && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
            onClick={() => setShowResumeDialog(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="w-full max-w-md p-6 bg-slate-900 border border-emerald-500/30 rounded-xl shadow-xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-emerald-500/20 rounded-lg">
                  <PlayCircle className="h-6 w-6 text-emerald-500" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-emerald-400">Resumir Trading</h3>
                  <p className="text-sm text-slate-400">Reactivar el sistema de trading</p>
                </div>
              </div>

              {status?.reason && (
                <div className="p-3 mb-4 bg-slate-800 border border-slate-600 rounded-lg">
                  <p className="text-xs text-slate-400 mb-1">Razon del kill switch:</p>
                  <p className="text-sm text-slate-200">{status.reason}</p>
                  <p className="text-xs text-slate-500 mt-1">
                    Por: {status.activated_by} | {status.activated_at && new Date(status.activated_at).toLocaleString()}
                  </p>
                </div>
              )}

              <div className="mb-4">
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Para resumir, escriba <span className="font-mono text-emerald-400">CONFIRM_RESUME</span>:
                </label>
                <input
                  type="text"
                  value={confirmCode}
                  onChange={(e) => setConfirmCode(e.target.value.toUpperCase())}
                  placeholder="CONFIRM_RESUME"
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-slate-200 font-mono placeholder-slate-500 focus:outline-none focus:border-emerald-500"
                />
              </div>

              <div className="flex gap-3">
                <Button
                  variant="outline"
                  className="flex-1"
                  onClick={() => setShowResumeDialog(false)}
                  disabled={isLoading}
                >
                  Cancelar
                </Button>
                <Button
                  variant="gradient"
                  className="flex-1"
                  onClick={resumeTrading}
                  disabled={isLoading || confirmCode !== 'CONFIRM_RESUME'}
                  loading={isLoading}
                >
                  {isLoading ? 'Resumiendo...' : 'Resumir Trading'}
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default KillSwitch;

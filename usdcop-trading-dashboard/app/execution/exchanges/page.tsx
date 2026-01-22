'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Link2,
  Plus,
  RefreshCw,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Trash2,
  Eye,
  EyeOff,
  ShieldCheck,
  ShieldAlert,
  Wallet,
  X,
} from 'lucide-react';
import { exchangeService } from '@/lib/services/execution';
import {
  SUPPORTED_EXCHANGES,
  EXCHANGE_METADATA,
  type SupportedExchange,
  type ConnectedExchange,
  type ExchangeBalances,
} from '@/lib/contracts/execution/exchange.contract';

interface ConnectModalState {
  isOpen: boolean;
  exchange: SupportedExchange | null;
}

export default function ExchangesPage() {
  const [exchanges, setExchanges] = useState<ConnectedExchange[]>([]);
  const [balances, setBalances] = useState<Record<string, ExchangeBalances>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connectModal, setConnectModal] = useState<ConnectModalState>({ isOpen: false, exchange: null });

  const fetchData = useCallback(async () => {
    try {
      setError(null);
      const [exchangesList, allBalances] = await Promise.all([
        exchangeService.getExchanges(),
        exchangeService.getAllBalances(),
      ]);
      setExchanges(exchangesList);

      // Convert balances array to record
      const balanceRecord: Record<string, ExchangeBalances> = {};
      allBalances.forEach(b => {
        balanceRecord[b.exchange] = b;
      });
      setBalances(balanceRecord);
    } catch (err) {
      console.error('Failed to fetch exchanges:', err);
      setError(err instanceof Error ? err.message : 'Failed to load exchanges');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleDisconnect = async (exchange: SupportedExchange) => {
    const confirmed = window.confirm(`Are you sure you want to disconnect ${EXCHANGE_METADATA[exchange].name}?`);
    if (!confirmed) return;

    try {
      await exchangeService.disconnectExchange(exchange);
      await fetchData();
    } catch (err) {
      alert('Failed to disconnect exchange');
    }
  };

  const connectedExchangeIds = new Set(exchanges.map(e => e.exchange));

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <RefreshCw className="w-8 h-8 text-cyan-500 animate-spin" />
          <p className="text-gray-400">Loading exchanges...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Exchange Connections</h1>
          <p className="text-gray-400">Connect and manage your exchange API keys</p>
        </div>
        <button
          onClick={fetchData}
          className="p-2 text-gray-400 hover:text-white transition-colors"
        >
          <RefreshCw className="w-5 h-5" />
        </button>
      </div>

      {/* Security Notice */}
      <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 flex items-start gap-3">
        <ShieldCheck className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
        <div>
          <h3 className="font-medium text-amber-400">Security Notice</h3>
          <p className="text-sm text-amber-300/80 mt-1">
            Only connect API keys with <strong>Spot Trading</strong> permissions.
            Never use API keys with withdrawal permissions. All credentials are encrypted
            with AES-256-GCM before storage.
          </p>
        </div>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-red-400" />
          <span className="text-red-400">{error}</span>
        </div>
      )}

      {/* Connected Exchanges */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-white">Connected Exchanges</h2>

        {exchanges.length === 0 ? (
          <div className="bg-gray-900/50 border border-gray-800/50 rounded-xl p-8 text-center">
            <Link2 className="w-12 h-12 text-gray-600 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-300 mb-2">No Exchanges Connected</h3>
            <p className="text-gray-500 mb-4">Connect an exchange to start trading</p>
          </div>
        ) : (
          <div className="grid gap-4">
            {exchanges.map((exchange) => {
              const meta = EXCHANGE_METADATA[exchange.exchange];
              const balance = balances[exchange.exchange];
              return (
                <motion.div
                  key={exchange.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-gray-900/50 border border-gray-800/50 rounded-xl p-5"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 bg-gray-800 rounded-xl flex items-center justify-center text-2xl">
                        {meta?.logo || '🔗'}
                      </div>
                      <div>
                        <h3 className="font-bold text-white">{meta?.name || exchange.exchange}</h3>
                        <div className="flex items-center gap-2 mt-1">
                          {exchange.is_valid ? (
                            <span className="flex items-center gap-1 text-xs text-green-400">
                              <CheckCircle2 className="w-3 h-3" /> Connected
                            </span>
                          ) : (
                            <span className="flex items-center gap-1 text-xs text-red-400">
                              <XCircle className="w-3 h-3" /> Invalid
                            </span>
                          )}
                          <span className="text-xs text-gray-500">
                            Key: {exchange.key_fingerprint}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-4">
                      {/* Balance */}
                      {balance && (
                        <div className="text-right">
                          <p className="text-sm text-gray-400">Balance</p>
                          <p className="font-bold text-white">
                            ${balance.total_usd.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                          </p>
                        </div>
                      )}

                      {/* Actions */}
                      <button
                        onClick={() => handleDisconnect(exchange.exchange)}
                        className="p-2 text-gray-400 hover:text-red-400 transition-colors"
                        title="Disconnect"
                      >
                        <Trash2 className="w-5 h-5" />
                      </button>
                    </div>
                  </div>

                  {/* Balance Details */}
                  {balance && balance.balances.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-gray-800/50">
                      <div className="flex flex-wrap gap-3">
                        {balance.balances.slice(0, 5).map((b) => (
                          <div
                            key={b.asset}
                            className="px-3 py-1.5 bg-gray-800/50 rounded-lg text-sm"
                          >
                            <span className="text-gray-400">{b.asset}:</span>{' '}
                            <span className="text-white font-medium">
                              {b.total.toLocaleString(undefined, { maximumFractionDigits: 4 })}
                            </span>
                          </div>
                        ))}
                        {balance.balances.length > 5 && (
                          <span className="text-gray-500 text-sm self-center">
                            +{balance.balances.length - 5} more
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                </motion.div>
              );
            })}
          </div>
        )}
      </div>

      {/* Available Exchanges */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-white">Available Exchanges</h2>

        <div className="grid md:grid-cols-2 gap-4">
          {SUPPORTED_EXCHANGES.map((exchange) => {
            const meta = EXCHANGE_METADATA[exchange];
            const isConnected = connectedExchangeIds.has(exchange);
            return (
              <div
                key={exchange}
                className={`bg-gray-900/50 border rounded-xl p-5 transition-all ${
                  isConnected
                    ? 'border-green-500/30 bg-green-500/5'
                    : 'border-gray-800/50 hover:border-cyan-500/30'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-gray-800 rounded-xl flex items-center justify-center text-2xl">
                      {meta?.logo || '🔗'}
                    </div>
                    <div>
                      <h3 className="font-bold text-white">{meta?.name || exchange}</h3>
                      <p className="text-sm text-gray-400">{meta?.description || 'Exchange'}</p>
                    </div>
                  </div>

                  {isConnected ? (
                    <span className="px-3 py-1.5 bg-green-500/20 text-green-400 text-sm rounded-lg">
                      Connected
                    </span>
                  ) : (
                    <button
                      onClick={() => setConnectModal({ isOpen: true, exchange })}
                      className="flex items-center gap-2 px-4 py-2 bg-cyan-500/20 text-cyan-400 rounded-lg hover:bg-cyan-500/30 transition-colors"
                    >
                      <Plus className="w-4 h-4" />
                      Connect
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Connect Modal */}
      <AnimatePresence>
        {connectModal.isOpen && connectModal.exchange && (
          <ConnectExchangeModal
            exchange={connectModal.exchange}
            onClose={() => setConnectModal({ isOpen: false, exchange: null })}
            onSuccess={() => {
              setConnectModal({ isOpen: false, exchange: null });
              fetchData();
            }}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

// Connect Exchange Modal Component
function ConnectExchangeModal({
  exchange,
  onClose,
  onSuccess,
}: {
  exchange: SupportedExchange;
  onClose: () => void;
  onSuccess: () => void;
}) {
  const [apiKey, setApiKey] = useState('');
  const [apiSecret, setApiSecret] = useState('');
  const [passphrase, setPassphrase] = useState('');
  const [showSecret, setShowSecret] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [validationResult, setValidationResult] = useState<{
    success: boolean;
    message: string;
    hasWithdraw?: boolean;
  } | null>(null);

  const meta = EXCHANGE_METADATA[exchange];
  const requiresPassphrase = exchange === 'mexc'; // MEXC doesn't need passphrase but some exchanges do

  const handleValidate = async () => {
    if (!apiKey || !apiSecret) {
      setValidationResult({ success: false, message: 'API Key and Secret are required' });
      return;
    }

    setIsValidating(true);
    setValidationResult(null);

    try {
      const result = await exchangeService.connectExchange(exchange, {
        api_key: apiKey,
        api_secret: apiSecret,
        passphrase: passphrase || undefined,
      });

      if (result.is_valid) {
        if (result.has_withdraw_permission) {
          setValidationResult({
            success: false,
            message: 'Warning: This API key has WITHDRAW permission. Please create a new key without withdrawal access.',
            hasWithdraw: true,
          });
        } else {
          setValidationResult({
            success: true,
            message: `Connected successfully! Balance: $${Object.values(result.balance_check || {}).reduce((a: number, b: unknown) => a + (b as number), 0).toFixed(2)}`,
          });
          setTimeout(onSuccess, 1500);
        }
      } else {
        setValidationResult({
          success: false,
          message: result.error_message || 'Failed to validate API key',
        });
      }
    } catch (err) {
      setValidationResult({
        success: false,
        message: err instanceof Error ? err.message : 'Connection failed',
      });
    } finally {
      setIsValidating(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-md bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-gray-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gray-800 rounded-lg flex items-center justify-center text-xl">
              {meta?.logo || '🔗'}
            </div>
            <div>
              <h2 className="font-bold text-white">Connect {meta?.name}</h2>
              <p className="text-xs text-gray-400">Enter your API credentials</p>
            </div>
          </div>
          <button onClick={onClose} className="p-2 text-gray-400 hover:text-white">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Form */}
        <div className="p-5 space-y-4">
          {/* API Key */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">API Key</label>
            <input
              type="text"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter your API key"
              className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500"
            />
          </div>

          {/* API Secret */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">API Secret</label>
            <div className="relative">
              <input
                type={showSecret ? 'text' : 'password'}
                value={apiSecret}
                onChange={(e) => setApiSecret(e.target.value)}
                placeholder="Enter your API secret"
                className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500 pr-12"
              />
              <button
                type="button"
                onClick={() => setShowSecret(!showSecret)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white"
              >
                {showSecret ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
              </button>
            </div>
          </div>

          {/* Passphrase (if required) */}
          {requiresPassphrase && (
            <div>
              <label className="block text-sm text-gray-400 mb-2">Passphrase (optional)</label>
              <input
                type="password"
                value={passphrase}
                onChange={(e) => setPassphrase(e.target.value)}
                placeholder="Enter passphrase if required"
                className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500"
              />
            </div>
          )}

          {/* Validation Result */}
          {validationResult && (
            <div className={`p-4 rounded-lg flex items-start gap-3 ${
              validationResult.success
                ? 'bg-green-500/10 border border-green-500/30'
                : validationResult.hasWithdraw
                  ? 'bg-amber-500/10 border border-amber-500/30'
                  : 'bg-red-500/10 border border-red-500/30'
            }`}>
              {validationResult.success ? (
                <ShieldCheck className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
              ) : validationResult.hasWithdraw ? (
                <ShieldAlert className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
              ) : (
                <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              )}
              <span className={`text-sm ${
                validationResult.success ? 'text-green-400' : validationResult.hasWithdraw ? 'text-amber-400' : 'text-red-400'
              }`}>
                {validationResult.message}
              </span>
            </div>
          )}

          {/* Security Warning */}
          <div className="p-3 bg-gray-800/30 rounded-lg">
            <p className="text-xs text-gray-400">
              <strong className="text-gray-300">Required permissions:</strong> Spot Trading, Read Account
              <br />
              <strong className="text-red-400">Do NOT enable:</strong> Withdrawal, Futures, Margin
            </p>
          </div>
        </div>

        {/* Actions */}
        <div className="p-5 border-t border-gray-800 flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 py-3 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleValidate}
            disabled={isValidating || !apiKey || !apiSecret}
            className="flex-1 py-3 bg-cyan-500 text-white rounded-lg hover:bg-cyan-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {isValidating ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                Validating...
              </>
            ) : (
              <>
                <ShieldCheck className="w-4 h-4" />
                Connect
              </>
            )}
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}

'use client';

/**
 * /execution/exchanges — registro de conexiones, re-skin GM (prototipo Var B:
 * form con nota AES-256/TLS, chips de seguridad, permisos con "Retiros — bloqueado",
 * exchanges conectados con balances). Presentación + i18n ÚNICAMENTE: el flujo de
 * conexión/validación/desconexión (exchangeService) queda intacto.
 */
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
  Lock,
  Ban,
  X,
} from 'lucide-react';
import { GmPageHeader, GmPanel } from '@/components/gm';
import { GM, GMT, Z } from '@/lib/ui/gm-tokens';
import { useGmT } from '@/lib/i18n/gm-core';
import { exchangeService } from '@/lib/services/execution';
import {
  SUPPORTED_EXCHANGES,
  EXCHANGE_METADATA,
  type SupportedExchange,
  type ConnectedExchange,
  type ExchangeBalances,
  type ValidationResult,
} from '@/lib/contracts/execution/exchange.contract';
import { EXEC_DICT, gmFill } from './../i18n';

interface ConnectModalState {
  isOpen: boolean;
  exchange: SupportedExchange | null;
}

// Copy-trading asset universe (mirrors the /catalog gate: live assets are selectable,
// the rest render as "próximamente"). Prototype Var B line 916.
const COPY_ASSETS: Array<{ symbol: string; icon: string; available: boolean }> = [
  { symbol: 'USD/COP', icon: '💱', available: true },
  { symbol: 'BTC/USDT', icon: '₿', available: true },
  { symbol: 'XAU/USD', icon: '🥇', available: false },
  { symbol: 'S&P 500', icon: '📈', available: false },
];

export default function ExchangesPage() {
  const t = useGmT(EXEC_DICT);
  const [exchanges, setExchanges] = useState<ConnectedExchange[]>([]);
  const [balances, setBalances] = useState<Record<string, ExchangeBalances>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connectModal, setConnectModal] = useState<ConnectModalState>({ isOpen: false, exchange: null });
  // Real key permissions from the last successful validate (server-truth), used to
  // render the "Permisos de la llave" panel instead of a hardcoded list.
  const [lastValidation, setLastValidation] = useState<ValidationResult | null>(null);
  // Copy-trading config (front-end preference; applied in live mode). Assets flagged
  // `available:false` mirror the catalog "próximamente" gate and are not selectable.
  const [copyExchange, setCopyExchange] = useState<SupportedExchange>('binance');
  const [copyAssets, setCopyAssets] = useState<Set<string>>(new Set(['USD/COP', 'BTC/USDT']));
  const toggleCopyAsset = useCallback((sym: string) => {
    setCopyAssets((prev) => {
      const next = new Set(prev);
      next.has(sym) ? next.delete(sym) : next.add(sym);
      return next;
    });
  }, []);

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
      setError(err instanceof Error ? err.message : t('loadError'));
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleDisconnect = async (exchange: SupportedExchange) => {
    const confirmed = window.confirm(
      gmFill(t('disconnectConfirm'), { name: EXCHANGE_METADATA[exchange].name })
    );
    if (!confirmed) return;

    try {
      await exchangeService.disconnectExchange(exchange);
      await fetchData();
    } catch (err) {
      alert(t('disconnectError'));
    }
  };

  const connectedExchangeIds = new Set(exchanges.map(e => e.exchange));

  if (isLoading) {
    return (
      <div className="min-h-[50vh] flex items-center justify-center" aria-busy>
        <div className="flex flex-col items-center gap-4">
          <RefreshCw className={`w-8 h-8 ${GM.accent} motion-safe:animate-spin`} aria-hidden />
          <p className={`${GMT.body} ${GM.textMuted}`}>{t('loadingExchanges')}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <GmPageHeader
        kicker={t('kicker')}
        title={t('exchTitle')}
        subtitle={t('exchSub')}
        actions={
          <button
            onClick={fetchData}
            aria-label={t('refresh')}
            className={`${GM.ctaGhost} ${GM.focus} w-11 h-11 flex items-center justify-center`}
          >
            <RefreshCw className="w-4 h-4" aria-hidden />
          </button>
        }
      />

      {/* Nota de seguridad (prototipo: AES-256 en reposo / TLS 1.3 en tránsito) */}
      <GmPanel
        title={
          <span className="flex items-center gap-2">
            <ShieldCheck className={`w-4 h-4 ${GM.accent}`} aria-hidden />
            {t('securityTitle')}
          </span>
        }
      >
        <p className={`${GMT.meta} ${GM.textMuted} leading-relaxed mb-4`}>{t('securityBody')}</p>
        <div className="flex gap-2 flex-wrap">
          {[t('secFeat1'), t('secFeat2'), t('secFeat3'), t('secFeat4')].map((f) => (
            <span
              key={f}
              className={`${GM.posBadge} flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg ${GMT.micro} font-semibold`}
            >
              <Lock className="w-3 h-3" aria-hidden />
              {f}
            </span>
          ))}
        </div>
      </GmPanel>

      {error && (
        <div className={`${GM.negBadge} rounded-xl p-4 flex items-center gap-3`} role="alert">
          <AlertTriangle className={`w-5 h-5 ${GM.neg}`} aria-hidden />
          <span className={`${GMT.body} ${GM.neg}`}>{error}</span>
        </div>
      )}

      {/* Exchanges conectados */}
      <div className="space-y-4">
        <h2 className={`${GMT.h2} ${GM.headline}`}>{t('connectedTitle')}</h2>

        {exchanges.length === 0 ? (
          <div className={`${GM.panel} p-8 text-center`}>
            <Link2 className={`w-12 h-12 ${GM.textFaint} mx-auto mb-4`} aria-hidden />
            <h3 className={`text-lg font-medium ${GM.textStrong} mb-2`}>{t('noExchTitle')}</h3>
            <p className={`${GMT.body} ${GM.textMuted} mb-4`}>{t('noExchBody')}</p>
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
                  className={`${GM.panel} p-5`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex items-center gap-4">
                      <div className={`${GM.panelInner} w-12 h-12 flex items-center justify-center text-2xl`}>
                        {meta?.logo || '🔗'}
                      </div>
                      <div>
                        <h3 className={`font-bold ${GM.headline}`}>{meta?.name || exchange.exchange}</h3>
                        <div className="flex items-center gap-2 mt-1">
                          {exchange.is_valid ? (
                            <span className={`flex items-center gap-1 ${GMT.micro} ${GM.pos}`}>
                              <CheckCircle2 className="w-3 h-3" aria-hidden /> {t('statusConnected')}
                            </span>
                          ) : (
                            <span className={`flex items-center gap-1 ${GMT.micro} ${GM.neg}`}>
                              <XCircle className="w-3 h-3" aria-hidden /> {t('statusInvalid')}
                            </span>
                          )}
                          <span className={`${GMT.micro} ${GM.textMuted} ${GMT.mono}`}>
                            {t('keyLabel')}: {exchange.key_fingerprint}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-4">
                      {/* Balance */}
                      {balance && (
                        <div className="text-right">
                          <p className={`${GMT.label} ${GM.textMuted}`}>{t('balanceLabel')}</p>
                          <p className={`font-bold ${GM.headline} ${GMT.mono}`}>
                            ${balance.total_usd.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                          </p>
                        </div>
                      )}

                      {/* Acciones */}
                      <button
                        onClick={() => handleDisconnect(exchange.exchange)}
                        className={`${GM.ctaDanger} ${GM.focus} w-11 h-11 flex items-center justify-center`}
                        title={t('disconnectLabel')}
                        aria-label={`${t('disconnectLabel')} ${meta?.name || exchange.exchange}`}
                      >
                        <Trash2 className="w-5 h-5" aria-hidden />
                      </button>
                    </div>
                  </div>

                  {/* Detalle de balances */}
                  {balance && balance.balances.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-[var(--gm-border)]">
                      <div className="flex flex-wrap gap-3">
                        {balance.balances.slice(0, 5).map((b) => (
                          <div
                            key={b.asset}
                            className={`${GM.panelInner} px-3 py-1.5 ${GMT.meta}`}
                          >
                            <span className={GM.textMuted}>{b.asset}:</span>{' '}
                            <span className={`${GM.text} font-medium ${GMT.mono}`}>
                              {b.total.toLocaleString(undefined, { maximumFractionDigits: 4 })}
                            </span>
                          </div>
                        ))}
                        {balance.balances.length > 5 && (
                          <span className={`${GMT.meta} ${GM.textMuted} self-center`}>
                            +{balance.balances.length - 5} {t('moreSuffix')}
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

      {/* Permisos de la llave — reflejan la última validación real (server-truth).
          Hasta validar una llave en esta sesión se muestra la política del sistema. */}
      <GmPanel title={t('permsTitle')}>
        {!lastValidation && (
          <p className={`${GMT.meta} ${GM.textMuted} normal-case tracking-normal mb-3`}>
            {t('permsPolicyNote')}
          </p>
        )}
        <div className="grid sm:grid-cols-2 gap-2.5 mb-4">
          {(lastValidation
            ? [
                {
                  label: t('permRead'),
                  ok: lastValidation.permissions.some((p) => /read/i.test(p)) || lastValidation.is_valid,
                },
                { label: t('permTrade'), ok: lastValidation.can_trade_spot },
                { label: t('permWithdraw'), ok: lastValidation.has_withdraw_permission },
                { label: t('permIp'), ok: true },
              ]
            : [
                { label: t('permRead'), ok: true },
                { label: t('permTrade'), ok: true },
                { label: t('permWithdraw'), ok: false },
                { label: t('permIp'), ok: true },
              ]
          ).map((p) => (
            <div key={p.label} className="flex items-center gap-2">
              {p.ok ? (
                <CheckCircle2 className={`w-4 h-4 ${GM.pos} flex-shrink-0`} aria-hidden />
              ) : (
                <Ban className={`w-4 h-4 ${GM.neg} flex-shrink-0`} aria-hidden />
              )}
              <span className={`${GMT.meta} ${p.ok ? GM.textStrong : GM.textMuted}`}>{p.label}</span>
            </div>
          ))}
        </div>
        <div className={`${GM.accentBadge} flex items-start gap-2.5 rounded-[11px] px-3.5 py-3`}>
          <ShieldCheck className={`w-4 h-4 ${GM.accent} flex-shrink-0 mt-0.5`} aria-hidden />
          <span className={`${GMT.meta} ${GM.textStrong} leading-relaxed normal-case tracking-normal font-medium`}>
            {t('permsCanDo')}
          </span>
        </div>
      </GmPanel>

      {/* Copy trading — configuración (prototipo Var B, líneas 911-917) */}
      <GmPanel title={t('copyTitle')}>
        <p className={`${GMT.meta} ${GM.textMuted} normal-case tracking-normal mb-2`}>{t('copyNote')}</p>
        {/* Honest: there is no copy-trading preference store on the backend yet, so this
            selection is a local preview and is not persisted server-side. */}
        <div className={`${GM.accentBadge} flex items-start gap-2.5 rounded-[11px] px-3.5 py-3 mb-4`}>
          <ShieldAlert className={`w-4 h-4 ${GM.accent} flex-shrink-0 mt-0.5`} aria-hidden />
          <span className={`${GMT.meta} ${GM.textStrong} leading-relaxed normal-case tracking-normal font-medium`}>
            {t('copyLocalNote')}
          </span>
        </div>

        <div className={`${GMT.label} ${GM.textMuted} mb-2`}>{t('copyExchangeLabel')}</div>
        <div className="flex flex-wrap gap-2 mb-5">
          {SUPPORTED_EXCHANGES.map((ex) => {
            const meta = EXCHANGE_METADATA[ex];
            const active = copyExchange === ex;
            return (
              <button
                key={ex}
                onClick={() => setCopyExchange(ex)}
                aria-pressed={active}
                className={`${GM.focus} flex items-center gap-2 h-9 px-3.5 rounded-lg ${GMT.meta} font-semibold transition-colors duration-[var(--gm-dur-fast)] ${
                  active ? `${GM.accentBadge} ${GM.textStrong}` : `${GM.panelInner} ${GM.textMuted} hover:text-[var(--gm-text)]`
                }`}
              >
                <span aria-hidden>{meta?.logo}</span>
                {meta?.displayName || ex}
              </button>
            );
          })}
        </div>

        <div className={`${GMT.label} ${GM.textMuted} mb-2`}>{t('copyAssetsLabel')}</div>
        <div className="flex flex-wrap gap-2">
          {COPY_ASSETS.map((a) => {
            const selected = copyAssets.has(a.symbol);
            if (!a.available) {
              return (
                <span key={a.symbol} className={`${GM.panelInner} ${GM.textFaint} flex items-center gap-2 h-9 px-3.5 rounded-lg ${GMT.meta} opacity-60`}>
                  <span aria-hidden>{a.icon}</span>
                  {a.symbol}
                  <span className={`${GMT.micro} ${GM.neutralBadge} rounded px-1.5 py-0.5`}>{t('copySoon')}</span>
                </span>
              );
            }
            return (
              <button
                key={a.symbol}
                onClick={() => toggleCopyAsset(a.symbol)}
                aria-pressed={selected}
                className={`${GM.focus} flex items-center gap-2 h-9 px-3.5 rounded-lg ${GMT.meta} font-semibold transition-colors duration-[var(--gm-dur-fast)] ${
                  selected ? `${GM.posBadge} ${GM.textStrong}` : `${GM.panelInner} ${GM.textMuted} hover:text-[var(--gm-text)]`
                }`}
              >
                <span aria-hidden>{a.icon}</span>
                {a.symbol}
                {selected && <CheckCircle2 className={`w-3.5 h-3.5 ${GM.pos}`} aria-hidden />}
              </button>
            );
          })}
        </div>
      </GmPanel>

      {/* Mejores prácticas (prototipo Var B, líneas 918-921) */}
      <GmPanel title={t('bestTitle')}>
        <div className="flex flex-col gap-2.5">
          {[t('best1'), t('best2'), t('best3'), t('best4')].map((b, i) => (
            <div key={i} className="flex items-start gap-2.5">
              <CheckCircle2 className={`w-4 h-4 ${GM.pos} flex-shrink-0 mt-0.5`} aria-hidden />
              <span className={`${GMT.meta} ${GM.textStrong} normal-case tracking-normal`}>{b}</span>
            </div>
          ))}
        </div>
      </GmPanel>

      {/* Exchanges disponibles */}
      <div className="space-y-4">
        <h2 className={`${GMT.h2} ${GM.headline}`}>{t('availableTitle')}</h2>

        <div className="grid md:grid-cols-2 gap-4">
          {SUPPORTED_EXCHANGES.map((exchange) => {
            const meta = EXCHANGE_METADATA[exchange];
            const isConnected = connectedExchangeIds.has(exchange);
            return (
              <div
                key={exchange}
                className={`rounded-2xl p-5 transition-colors duration-[var(--gm-dur-fast)] ${
                  isConnected
                    ? 'bg-[rgba(52,211,153,.05)] border border-[rgba(52,211,153,.3)]'
                    : `${GM.panel} hover:border-[rgba(34,211,238,.3)]`
                }`}
              >
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-4">
                    <div className={`${GM.panelInner} w-12 h-12 flex items-center justify-center text-2xl`}>
                      {meta?.logo || '🔗'}
                    </div>
                    <div>
                      <h3 className={`font-bold ${GM.headline}`}>{meta?.name || exchange}</h3>
                      <p className={`${GMT.meta} ${GM.textMuted}`}>{meta?.description || 'Exchange'}</p>
                    </div>
                  </div>

                  {isConnected ? (
                    <span className={`${GM.posBadge} px-3 py-1.5 rounded-lg ${GMT.meta} font-semibold`}>
                      {t('statusConnected')}
                    </span>
                  ) : (
                    <button
                      onClick={() => setConnectModal({ isOpen: true, exchange })}
                      className={`${GM.ctaSoft} ${GM.focus} flex items-center gap-2 h-11 px-4 text-[13px]`}
                    >
                      <Plus className="w-4 h-4" aria-hidden />
                      {t('connectLabel')}
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Modal de conexión */}
      <AnimatePresence>
        {connectModal.isOpen && connectModal.exchange && (
          <ConnectExchangeModal
            exchange={connectModal.exchange}
            onClose={() => setConnectModal({ isOpen: false, exchange: null })}
            onSuccess={(result) => {
              if (result) setLastValidation(result);
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
  onSuccess: (result?: ValidationResult) => void;
}) {
  const t = useGmT(EXEC_DICT);
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
  // Passphrase field only for exchanges that actually use one (metadata-driven).
  // Binance/MEXC don't — the previous `exchange === 'mexc'` was inverted.
  const requiresPassphrase = meta?.requiresPassphrase ?? false;

  const handleValidate = async () => {
    if (!apiKey || !apiSecret) {
      setValidationResult({ success: false, message: t('requiredError') });
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
            message: t('withdrawWarning'),
            hasWithdraw: true,
          });
        } else {
          setValidationResult({
            success: true,
            message: `${t('connectedOk')} $${Object.values(result.balance_check || {}).reduce((a: number, b: unknown) => a + (b as number), 0).toFixed(2)}`,
          });
          setTimeout(() => onSuccess(result), 1500);
        }
      } else {
        setValidationResult({
          success: false,
          message: result.error_message || t('validateFailed'),
        });
      }
    } catch (err) {
      setValidationResult({
        success: false,
        message: err instanceof Error ? err.message : t('connFailed'),
      });
    } finally {
      setIsValidating(false);
    }
  };

  const inputClass = `${GM.input} ${GM.focus} w-full h-11 px-4 ${GMT.mono}`;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className={`fixed inset-0 ${Z.modal} flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm`}
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        onClick={(e) => e.stopPropagation()}
        className={`${GM.popover} w-full max-w-md overflow-hidden`}
        role="dialog"
        aria-modal="true"
        aria-label={`${t('modalConnect')} ${meta?.name}`}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-[var(--gm-border)]">
          <div className="flex items-center gap-3">
            <div className={`${GM.panelInner} w-10 h-10 flex items-center justify-center text-xl`}>
              {meta?.logo || '🔗'}
            </div>
            <div>
              <h2 className={`font-bold ${GM.headline}`}>{t('modalConnect')} {meta?.name}</h2>
              <p className={`${GMT.micro} ${GM.textMuted}`}>{t('modalSub')}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label={t('closeLabel')}
            className={`${GM.ctaGhost} ${GM.focus} w-11 h-11 flex items-center justify-center`}
          >
            <X className="w-5 h-5" aria-hidden />
          </button>
        </div>

        {/* Form */}
        <div className="p-5 space-y-4">
          {/* API Key */}
          <div>
            <label className={`block ${GMT.label} ${GM.textMuted} mb-2`}>{t('apiKeyLabel')}</label>
            <input
              type="text"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={t('apiKeyPh')}
              className={inputClass}
            />
          </div>

          {/* API Secret */}
          <div>
            <label className={`block ${GMT.label} ${GM.textMuted} mb-2`}>{t('apiSecretLabel')}</label>
            <div className="relative">
              <input
                type={showSecret ? 'text' : 'password'}
                value={apiSecret}
                onChange={(e) => setApiSecret(e.target.value)}
                placeholder={t('apiSecretPh')}
                className={`${inputClass} pr-12`}
              />
              <button
                type="button"
                onClick={() => setShowSecret(!showSecret)}
                aria-label={showSecret ? t('hideSecret') : t('showSecret')}
                className={`${GM.focus} absolute right-1 top-1/2 -translate-y-1/2 w-9 h-9 flex items-center justify-center rounded-[8px] text-[var(--gm-text-muted)] hover:text-[var(--gm-text)] transition-colors duration-[var(--gm-dur-fast)]`}
              >
                {showSecret ? <EyeOff className="w-5 h-5" aria-hidden /> : <Eye className="w-5 h-5" aria-hidden />}
              </button>
            </div>
          </div>

          {/* Passphrase (if required) */}
          {requiresPassphrase && (
            <div>
              <label className={`block ${GMT.label} ${GM.textMuted} mb-2`}>{t('passphraseLabel')}</label>
              <input
                type="password"
                value={passphrase}
                onChange={(e) => setPassphrase(e.target.value)}
                placeholder={t('passphrasePh')}
                className={inputClass}
              />
            </div>
          )}

          {/* Chips de seguridad (prototipo `exSecFeatures`) */}
          <div className="flex gap-2 flex-wrap">
            {[t('secFeat1'), t('secFeat2'), t('secFeat3'), t('secFeat4')].map((f) => (
              <span
                key={f}
                className={`${GM.posBadge} flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg ${GMT.micro} font-semibold`}
              >
                <Lock className="w-3 h-3" aria-hidden />
                {f}
              </span>
            ))}
          </div>

          {/* Validation Result */}
          {validationResult && (
            <div
              role="alert"
              className={`p-4 rounded-[11px] flex items-start gap-3 ${
                validationResult.success
                  ? GM.posBadge
                  : validationResult.hasWithdraw
                    ? GM.warnBadge
                    : GM.negBadge
              }`}
            >
              {validationResult.success ? (
                <ShieldCheck className={`w-5 h-5 ${GM.pos} flex-shrink-0 mt-0.5`} aria-hidden />
              ) : validationResult.hasWithdraw ? (
                <ShieldAlert className={`w-5 h-5 ${GM.warn} flex-shrink-0 mt-0.5`} aria-hidden />
              ) : (
                <AlertTriangle className={`w-5 h-5 ${GM.neg} flex-shrink-0 mt-0.5`} aria-hidden />
              )}
              <span className={`${GMT.meta} font-medium`}>
                {validationResult.message}
              </span>
            </div>
          )}

          {/* Aviso de permisos */}
          <div className={`${GM.panelInner} p-3`}>
            <p className={`${GMT.micro} ${GM.textMuted} leading-relaxed`}>
              <strong className={GM.textStrong}>{t('permsRequiredLabel')}</strong> {t('permsRequiredValue')}
              <br />
              <strong className={GM.neg}>{t('doNotEnableLabel')}</strong> {t('doNotEnableValue')}
            </p>
          </div>
        </div>

        {/* Actions */}
        <div className="p-5 border-t border-[var(--gm-border)] flex gap-3">
          <button
            onClick={onClose}
            className={`${GM.ctaGhost} ${GM.focus} flex-1 h-11 text-[13px] font-semibold`}
          >
            {t('cancelLabel')}
          </button>
          <button
            onClick={handleValidate}
            disabled={isValidating || !apiKey || !apiSecret}
            className={`${GM.ctaPrimary} ${GM.focus} flex-1 h-11 text-[13px] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2`}
          >
            {isValidating ? (
              <>
                <RefreshCw className="w-4 h-4 motion-safe:animate-spin" aria-hidden />
                {t('validatingLabel')}
              </>
            ) : (
              <>
                <ShieldCheck className="w-4 h-4" aria-hidden />
                {t('connectLabel')}
              </>
            )}
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}

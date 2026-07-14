'use client';

/**
 * /execution/settings — límites de riesgo (techo del sistema), re-skin GM
 * (prototipo Var B: "Límites de riesgo (techo del sistema)"). Presentación +
 * i18n ÚNICAMENTE: fetch/save de límites (signalBridgeService) queda intacto.
 */
import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  Shield,
  User,
  Bell,
  DollarSign,
  Percent,
  Clock,
  TrendingDown,
  Save,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  ChevronRight,
} from 'lucide-react';
import { GmPageHeader, GmPanel } from '@/components/gm';
import { GM, GMT, type GmTone, GM_TONE_TEXT } from '@/lib/ui/gm-tokens';
import { useGmT } from '@/lib/i18n/gm-core';
import { signalBridgeService, userProfileService } from '@/lib/services/execution';
import { type UserRiskLimits } from '@/lib/contracts/execution/signal-bridge.contract';
import { EXEC_DICT, gmFill } from './../i18n';

type SettingsTab = 'risk' | 'profile' | 'notifications';

/** Sentinel para traducir el error de carga en render (ver fetchRiskLimits). */
const RISK_LOAD_ERROR_SENTINEL = '__risk_load_error__';

export default function SettingsPage() {
  const t = useGmT(EXEC_DICT);
  const [activeTab, setActiveTab] = useState<SettingsTab>('risk');
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Risk limits state
  const [riskLimits, setRiskLimits] = useState<UserRiskLimits | null>(null);
  const [riskForm, setRiskForm] = useState({
    max_daily_loss_pct: 2.0,
    max_trades_per_day: 10,
    max_position_size_usd: 1000,
    cooldown_minutes: 15,
    enable_short: false,
  });

  // Profile state (name/email — the fields SignalBridge persists via PATCH /users/me).
  const [profileForm, setProfileForm] = useState({ name: '', email: '' });
  const [isProfileSaving, setIsProfileSaving] = useState(false);

  // The user id is resolved SERVER-SIDE by the BFF from the session — the client
  // no longer reads/guesses a user id (no more localStorage 'user-id'/'current').
  const fetchRiskLimits = useCallback(async () => {
    try {
      setError(null);
      const limits = await signalBridgeService.getUserLimits();
      setRiskLimits(limits);
      setRiskForm({
        max_daily_loss_pct: limits.max_daily_loss_pct,
        max_trades_per_day: limits.max_trades_per_day,
        max_position_size_usd: limits.max_position_size_usd,
        cooldown_minutes: limits.cooldown_minutes,
        enable_short: limits.enable_short,
      });
    } catch (err) {
      console.error('Failed to fetch risk limits:', err);
      // Sentinel traducido al render — así `t` no entra a las deps del callback.
      setError(RISK_LOAD_ERROR_SENTINEL);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const fetchProfile = useCallback(async () => {
    try {
      const profile = await userProfileService.getProfile();
      setProfileForm({ name: profile.name ?? '', email: profile.email ?? '' });
    } catch (err) {
      // Non-fatal: the risk tab still works; profile inputs stay empty/editable.
      console.error('Failed to fetch profile:', err);
    }
  }, []);

  useEffect(() => {
    fetchRiskLimits();
    fetchProfile();
  }, [fetchRiskLimits, fetchProfile]);

  const handleSaveRiskLimits = async () => {
    setIsSaving(true);
    setError(null);
    setSuccess(null);

    try {
      await signalBridgeService.updateUserLimits(riskForm);
      setSuccess(t('riskSaved'));
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : t('riskSaveError'));
    } finally {
      setIsSaving(false);
    }
  };

  const handleSaveProfile = async () => {
    setIsProfileSaving(true);
    setError(null);
    setSuccess(null);

    try {
      const updated = await userProfileService.updateProfile({
        name: profileForm.name,
        email: profileForm.email,
      });
      setProfileForm({ name: updated.name ?? '', email: updated.email ?? '' });
      setSuccess(t('profileSaved'));
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : t('profileSaveError'));
    } finally {
      setIsProfileSaving(false);
    }
  };

  const tabs = [
    { id: 'risk' as const, label: t('tabRisk'), icon: Shield },
    { id: 'profile' as const, label: t('tabProfile'), icon: User },
    { id: 'notifications' as const, label: t('tabNotifications'), icon: Bell },
  ];

  const inputClass = `${GM.input} ${GM.focus} w-full h-11 px-4 ${GMT.mono}`;
  const labelClass = `flex items-center gap-2 ${GMT.label} ${GM.textMuted} mb-2`;
  const helpClass = `${GMT.micro} ${GM.textMuted} mt-1.5`;

  const presets: Array<{
    name: string;
    description: string;
    values: { max_daily_loss_pct: number; max_trades_per_day: number; max_position_size_usd: number; cooldown_minutes: number };
    tone: GmTone;
  }> = [
    {
      name: t('presetCons'),
      description: t('presetConsDesc'),
      values: { max_daily_loss_pct: 1.0, max_trades_per_day: 5, max_position_size_usd: 500, cooldown_minutes: 30 },
      tone: 'pos',
    },
    {
      name: t('presetMod'),
      description: t('presetModDesc'),
      values: { max_daily_loss_pct: 2.0, max_trades_per_day: 10, max_position_size_usd: 1000, cooldown_minutes: 15 },
      tone: 'accent',
    },
    {
      name: t('presetAggr'),
      description: t('presetAggrDesc'),
      values: { max_daily_loss_pct: 5.0, max_trades_per_day: 20, max_position_size_usd: 2500, cooldown_minutes: 5 },
      tone: 'warn',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <GmPageHeader kicker={t('kicker')} title={t('setTitle')} subtitle={t('setSub')} />

      {/* Tabs */}
      <div className="flex flex-wrap gap-2 border-b border-[var(--gm-border)] pb-4" role="tablist">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              role="tab"
              aria-selected={activeTab === tab.id}
              className={`flex items-center gap-2 h-11 px-4 rounded-[10px] text-[13px] font-semibold ${GM.focus} ${
                activeTab === tab.id ? GM.navActive : GM.navIdle
              }`}
            >
              <Icon className="w-4 h-4" aria-hidden />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Feedback Messages */}
      {error && (
        <div className={`${GM.negBadge} rounded-[11px] p-4 flex items-center gap-3`} role="alert">
          <AlertCircle className={`w-5 h-5 ${GM.neg}`} aria-hidden />
          <span className={`${GMT.body} ${GM.neg}`}>
            {error === RISK_LOAD_ERROR_SENTINEL ? t('riskLoadError') : error}
          </span>
        </div>
      )}

      {success && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className={`${GM.posBadge} rounded-[11px] p-4 flex items-center gap-3`}
          role="status"
        >
          <CheckCircle2 className={`w-5 h-5 ${GM.pos}`} aria-hidden />
          <span className={`${GMT.body} ${GM.pos}`}>{success}</span>
        </motion.div>
      )}

      {/* Tab Content */}
      {activeTab === 'risk' && (
        <div className="space-y-6">
          {isLoading ? (
            <div className="flex items-center justify-center py-12" aria-busy>
              <RefreshCw className={`w-6 h-6 ${GM.accent} motion-safe:animate-spin`} aria-hidden />
            </div>
          ) : (
            <>
              {/* Límites de riesgo (techo del sistema) */}
              <GmPanel
                title={
                  <span className="flex items-center gap-2">
                    <Shield className={`w-4 h-4 ${GM.accent}`} aria-hidden />
                    {t('riskTitle')}
                  </span>
                }
                className="space-y-6"
              >
                <div className="grid md:grid-cols-2 gap-6">
                  {/* Pérdida diaria máx. */}
                  <div>
                    <label className={labelClass}>
                      <TrendingDown className="w-4 h-4" aria-hidden />
                      {t('maxDailyLoss')}
                    </label>
                    <input
                      type="number"
                      step="0.5"
                      min="0.5"
                      max="10"
                      value={riskForm.max_daily_loss_pct}
                      onChange={(e) => setRiskForm({ ...riskForm, max_daily_loss_pct: parseFloat(e.target.value) })}
                      className={inputClass}
                    />
                    <p className={helpClass}>{t('maxDailyLossHelp')}</p>
                  </div>

                  {/* Trades máx. por día */}
                  <div>
                    <label className={labelClass}>
                      <Percent className="w-4 h-4" aria-hidden />
                      {t('maxTrades')}
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="100"
                      value={riskForm.max_trades_per_day}
                      onChange={(e) => setRiskForm({ ...riskForm, max_trades_per_day: parseInt(e.target.value) })}
                      className={inputClass}
                    />
                    <p className={helpClass}>{t('maxTradesHelp')}</p>
                  </div>

                  {/* Tamaño máx. de posición */}
                  <div>
                    <label className={labelClass}>
                      <DollarSign className="w-4 h-4" aria-hidden />
                      {t('maxPosition')}
                    </label>
                    <input
                      type="number"
                      min="10"
                      max="100000"
                      value={riskForm.max_position_size_usd}
                      onChange={(e) => setRiskForm({ ...riskForm, max_position_size_usd: parseFloat(e.target.value) })}
                      className={inputClass}
                    />
                    <p className={helpClass}>{t('maxPositionHelp')}</p>
                  </div>

                  {/* Enfriamiento */}
                  <div>
                    <label className={labelClass}>
                      <Clock className="w-4 h-4" aria-hidden />
                      {t('cooldownLabel')}
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="60"
                      value={riskForm.cooldown_minutes}
                      onChange={(e) => setRiskForm({ ...riskForm, cooldown_minutes: parseInt(e.target.value) })}
                      className={inputClass}
                    />
                    <p className={helpClass}>{t('cooldownHelp')}</p>
                  </div>
                </div>

                {/* Habilitar cortos */}
                <div className="pt-4 border-t border-[var(--gm-border)]">
                  <label className={`${GM.panelInner} flex items-center justify-between gap-4 p-4 cursor-pointer`}>
                    <div>
                      <h3 className={`font-medium ${GM.text}`}>{t('enableShort')}</h3>
                      <p className={`${GMT.meta} ${GM.textMuted} mt-1`}>{t('enableShortHelp')}</p>
                    </div>
                    <div className="relative flex-shrink-0">
                      <input
                        type="checkbox"
                        checked={riskForm.enable_short}
                        onChange={(e) => setRiskForm({ ...riskForm, enable_short: e.target.checked })}
                        className={`sr-only peer ${GM.focus}`}
                      />
                      <div className="w-11 h-6 bg-[rgba(148,163,184,.25)] rounded-full peer peer-checked:bg-[var(--gm-accent)] peer-focus-visible:ring-2 peer-focus-visible:ring-[var(--gm-accent)] transition-colors duration-[var(--gm-dur-fast)]" />
                      <div className="absolute left-1 top-1 w-4 h-4 bg-[var(--gm-headline)] rounded-full peer-checked:translate-x-5 transition-transform duration-[var(--gm-dur-fast)]" />
                    </div>
                  </label>
                </div>

                {/* Guardar */}
                <div className="flex justify-end pt-4">
                  <button
                    onClick={handleSaveRiskLimits}
                    disabled={isSaving}
                    className={`${GM.ctaPrimary} ${GM.focus} flex items-center gap-2 h-11 px-6 text-[13px] disabled:opacity-50`}
                  >
                    {isSaving ? (
                      <>
                        <RefreshCw className="w-4 h-4 motion-safe:animate-spin" aria-hidden />
                        {t('savingLabel')}
                      </>
                    ) : (
                      <>
                        <Save className="w-4 h-4" aria-hidden />
                        {t('saveChanges')}
                      </>
                    )}
                  </button>
                </div>
              </GmPanel>

              {/* Presets rápidos */}
              <GmPanel title={t('presetsTitle')}>
                <div className="grid md:grid-cols-3 gap-4">
                  {presets.map((preset) => (
                    <button
                      key={preset.name}
                      onClick={() => setRiskForm({ ...riskForm, ...preset.values })}
                      className={`${GM.panelInner} ${GM.focus} p-4 text-left hover:border-[rgba(34,211,238,.3)] transition-colors duration-[var(--gm-dur-fast)] group`}
                    >
                      <h3 className={`font-bold ${GM_TONE_TEXT[preset.tone]} mb-1`}>{preset.name}</h3>
                      <p className={`${GMT.meta} ${GM.textMuted}`}>{preset.description}</p>
                      <div className={`mt-2 ${GMT.micro} ${GM.textMuted} ${GMT.mono}`}>
                        {gmFill(t('presetMeta'), { a: preset.values.max_daily_loss_pct, b: preset.values.max_trades_per_day })}
                      </div>
                      <ChevronRight
                        className={`w-4 h-4 ${GM.textMuted} group-hover:text-[var(--gm-accent)] ml-auto mt-2 transition-colors duration-[var(--gm-dur-fast)]`}
                        aria-hidden
                      />
                    </button>
                  ))}
                </div>
              </GmPanel>
            </>
          )}
        </div>
      )}

      {activeTab === 'profile' && (
        <GmPanel
          title={
            <span className="flex items-center gap-2">
              <User className={`w-4 h-4 ${GM.accent}`} aria-hidden />
              {t('profileTitle')}
            </span>
          }
        >
          <div className="space-y-4">
            <div>
              <label className={`block ${GMT.label} ${GM.textMuted} mb-2`}>{t('displayName')}</label>
              <input
                type="text"
                placeholder={t('displayNamePh')}
                value={profileForm.name}
                onChange={(e) => setProfileForm({ ...profileForm, name: e.target.value })}
                className={`${GM.input} ${GM.focus} w-full h-11 px-4`}
              />
            </div>

            <div>
              <label className={`block ${GMT.label} ${GM.textMuted} mb-2`}>{t('emailLabel')}</label>
              <input
                type="email"
                placeholder="tu@email.com"
                value={profileForm.email}
                onChange={(e) => setProfileForm({ ...profileForm, email: e.target.value })}
                className={`${GM.input} ${GM.focus} w-full h-11 px-4`}
              />
            </div>

            <div>
              <label className={`block ${GMT.label} ${GM.textMuted} mb-2`}>{t('tzLabel')}</label>
              <select className={`${GM.input} ${GM.focus} w-full h-11 px-4`} disabled>
                <option value="America/Bogota">America/Bogota (COT)</option>
                <option value="America/New_York">America/New_York (EST)</option>
                <option value="UTC">UTC</option>
              </select>
              <p className={`${helpClass}`}>{t('tzLocalNote')}</p>
            </div>

            <div className="pt-4">
              <button
                onClick={handleSaveProfile}
                disabled={isProfileSaving}
                className={`${GM.ctaPrimary} ${GM.focus} flex items-center gap-2 h-11 px-6 text-[13px] disabled:opacity-50`}
              >
                {isProfileSaving ? (
                  <>
                    <RefreshCw className="w-4 h-4 motion-safe:animate-spin" aria-hidden />
                    {t('savingLabel')}
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4" aria-hidden />
                    {t('saveProfile')}
                  </>
                )}
              </button>
            </div>
          </div>
        </GmPanel>
      )}

      {activeTab === 'notifications' && (
        <GmPanel
          title={
            <span className="flex items-center gap-2">
              <Bell className={`w-4 h-4 ${GM.accent}`} aria-hidden />
              {t('notifTitle')}
            </span>
          }
        >
          <div className="space-y-4">
            {/* Honest state: SignalBridge has no notification-preferences store yet,
                so these toggles are shown disabled rather than wired to a dead handler
                that silently discards the choice. */}
            <div className={`${GM.accentBadge} flex items-start gap-2.5 rounded-[11px] px-3.5 py-3`}>
              <AlertCircle className={`w-4 h-4 ${GM.accent} flex-shrink-0 mt-0.5`} aria-hidden />
              <span className={`${GMT.meta} ${GM.textStrong} leading-relaxed normal-case tracking-normal font-medium`}>
                {t('notifNoBackend')}
              </span>
            </div>

            {[
              { id: 'trade_executed', label: t('nTradeExecuted'), description: t('nTradeExecutedDesc') },
              { id: 'risk_alert', label: t('nRiskAlert'), description: t('nRiskAlertDesc') },
              { id: 'kill_switch', label: t('nKillSwitch'), description: t('nKillSwitchDesc') },
              { id: 'daily_summary', label: t('nDailySummary'), description: t('nDailySummaryDesc') },
            ].map((notification) => (
              <label
                key={notification.id}
                className={`${GM.panelInner} flex items-center justify-between gap-4 p-4 opacity-60`}
              >
                <div>
                  <h3 className={`font-medium ${GM.text}`}>{notification.label}</h3>
                  <p className={`${GMT.meta} ${GM.textMuted}`}>{notification.description}</p>
                </div>
                <input
                  type="checkbox"
                  defaultChecked
                  disabled
                  className={`w-5 h-5 accent-[var(--gm-accent)] bg-[var(--gm-panel-inner)] border-[var(--gm-border)] rounded flex-shrink-0`}
                />
              </label>
            ))}

            <div className="pt-4">
              <button
                disabled
                title={t('notifNoBackend')}
                className={`${GM.ctaPrimary} flex items-center gap-2 h-11 px-6 text-[13px] opacity-50 cursor-not-allowed`}
              >
                <Save className="w-4 h-4" aria-hidden />
                {t('savePrefs')}
              </button>
            </div>
          </div>
        </GmPanel>
      )}
    </div>
  );
}

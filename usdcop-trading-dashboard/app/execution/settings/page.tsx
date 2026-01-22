'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  Settings,
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
import { signalBridgeService } from '@/lib/services/execution';
import { type UserRiskLimits } from '@/lib/contracts/execution/signal-bridge.contract';

type SettingsTab = 'risk' | 'profile' | 'notifications';

export default function SettingsPage() {
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

  const userId = typeof window !== 'undefined' ? localStorage.getItem('user-id') || 'current' : 'current';

  const fetchRiskLimits = useCallback(async () => {
    try {
      setError(null);
      const limits = await signalBridgeService.getUserLimits(userId);
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
      setError('Failed to load risk settings');
    } finally {
      setIsLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    fetchRiskLimits();
  }, [fetchRiskLimits]);

  const handleSaveRiskLimits = async () => {
    setIsSaving(true);
    setError(null);
    setSuccess(null);

    try {
      await signalBridgeService.updateUserLimits(userId, riskForm);
      setSuccess('Risk settings saved successfully');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save settings');
    } finally {
      setIsSaving(false);
    }
  };

  const tabs = [
    { id: 'risk' as const, label: 'Risk Management', icon: Shield },
    { id: 'profile' as const, label: 'Profile', icon: User },
    { id: 'notifications' as const, label: 'Notifications', icon: Bell },
  ];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Settings</h1>
        <p className="text-gray-400">Configure your trading preferences and risk limits</p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-gray-800/50 pb-4">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                activeTab === tab.id
                  ? 'bg-cyan-500/20 text-cyan-400'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
              }`}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Feedback Messages */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-red-400" />
          <span className="text-red-400">{error}</span>
        </div>
      )}

      {success && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-green-500/10 border border-green-500/30 rounded-lg p-4 flex items-center gap-3"
        >
          <CheckCircle2 className="w-5 h-5 text-green-400" />
          <span className="text-green-400">{success}</span>
        </motion.div>
      )}

      {/* Tab Content */}
      {activeTab === 'risk' && (
        <div className="space-y-6">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="w-6 h-6 text-cyan-500 animate-spin" />
            </div>
          ) : (
            <>
              {/* Risk Limits Form */}
              <div className="bg-gray-900/50 border border-gray-800/50 rounded-xl p-6 space-y-6">
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <Shield className="w-5 h-5 text-cyan-400" />
                  Risk Limits
                </h2>

                <div className="grid md:grid-cols-2 gap-6">
                  {/* Max Daily Loss */}
                  <div>
                    <label className="flex items-center gap-2 text-sm text-gray-400 mb-2">
                      <TrendingDown className="w-4 h-4" />
                      Max Daily Loss (%)
                    </label>
                    <input
                      type="number"
                      step="0.5"
                      min="0.5"
                      max="10"
                      value={riskForm.max_daily_loss_pct}
                      onChange={(e) => setRiskForm({ ...riskForm, max_daily_loss_pct: parseFloat(e.target.value) })}
                      className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                    />
                    <p className="text-xs text-gray-500 mt-1">Trading stops if daily loss exceeds this limit</p>
                  </div>

                  {/* Max Trades Per Day */}
                  <div>
                    <label className="flex items-center gap-2 text-sm text-gray-400 mb-2">
                      <Percent className="w-4 h-4" />
                      Max Trades Per Day
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="100"
                      value={riskForm.max_trades_per_day}
                      onChange={(e) => setRiskForm({ ...riskForm, max_trades_per_day: parseInt(e.target.value) })}
                      className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                    />
                    <p className="text-xs text-gray-500 mt-1">Maximum number of trades allowed per day</p>
                  </div>

                  {/* Max Position Size */}
                  <div>
                    <label className="flex items-center gap-2 text-sm text-gray-400 mb-2">
                      <DollarSign className="w-4 h-4" />
                      Max Position Size (USD)
                    </label>
                    <input
                      type="number"
                      min="10"
                      max="100000"
                      value={riskForm.max_position_size_usd}
                      onChange={(e) => setRiskForm({ ...riskForm, max_position_size_usd: parseFloat(e.target.value) })}
                      className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                    />
                    <p className="text-xs text-gray-500 mt-1">Maximum size for a single position</p>
                  </div>

                  {/* Cooldown Minutes */}
                  <div>
                    <label className="flex items-center gap-2 text-sm text-gray-400 mb-2">
                      <Clock className="w-4 h-4" />
                      Cooldown Period (minutes)
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="60"
                      value={riskForm.cooldown_minutes}
                      onChange={(e) => setRiskForm({ ...riskForm, cooldown_minutes: parseInt(e.target.value) })}
                      className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                    />
                    <p className="text-xs text-gray-500 mt-1">Minimum time between trades</p>
                  </div>
                </div>

                {/* Enable Short Trading */}
                <div className="pt-4 border-t border-gray-800/50">
                  <label className="flex items-center justify-between p-4 bg-gray-800/30 rounded-lg cursor-pointer">
                    <div>
                      <h3 className="font-medium text-white">Enable Short Selling</h3>
                      <p className="text-sm text-gray-400 mt-1">
                        Allow the system to open short positions when signal indicates SELL
                      </p>
                    </div>
                    <div className="relative">
                      <input
                        type="checkbox"
                        checked={riskForm.enable_short}
                        onChange={(e) => setRiskForm({ ...riskForm, enable_short: e.target.checked })}
                        className="sr-only peer"
                      />
                      <div className="w-11 h-6 bg-gray-700 rounded-full peer peer-checked:bg-cyan-500 transition-colors"></div>
                      <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full peer-checked:translate-x-5 transition-transform"></div>
                    </div>
                  </label>
                </div>

                {/* Save Button */}
                <div className="flex justify-end pt-4">
                  <button
                    onClick={handleSaveRiskLimits}
                    disabled={isSaving}
                    className="flex items-center gap-2 px-6 py-3 bg-cyan-500 text-white rounded-lg hover:bg-cyan-600 disabled:opacity-50 transition-colors"
                  >
                    {isSaving ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        Saving...
                      </>
                    ) : (
                      <>
                        <Save className="w-4 h-4" />
                        Save Changes
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Risk Presets */}
              <div className="bg-gray-900/50 border border-gray-800/50 rounded-xl p-6">
                <h2 className="text-lg font-bold text-white mb-4">Quick Presets</h2>
                <div className="grid md:grid-cols-3 gap-4">
                  {[
                    {
                      name: 'Conservative',
                      description: 'Lower risk, smaller positions',
                      values: { max_daily_loss_pct: 1.0, max_trades_per_day: 5, max_position_size_usd: 500, cooldown_minutes: 30 },
                      color: 'green',
                    },
                    {
                      name: 'Moderate',
                      description: 'Balanced risk and reward',
                      values: { max_daily_loss_pct: 2.0, max_trades_per_day: 10, max_position_size_usd: 1000, cooldown_minutes: 15 },
                      color: 'cyan',
                    },
                    {
                      name: 'Aggressive',
                      description: 'Higher risk, larger positions',
                      values: { max_daily_loss_pct: 5.0, max_trades_per_day: 20, max_position_size_usd: 2500, cooldown_minutes: 5 },
                      color: 'amber',
                    },
                  ].map((preset) => (
                    <button
                      key={preset.name}
                      onClick={() => setRiskForm({ ...riskForm, ...preset.values })}
                      className="p-4 bg-gray-800/50 border border-gray-700 rounded-lg text-left hover:border-cyan-500/50 transition-colors group"
                    >
                      <h3 className={`font-bold text-${preset.color}-400 mb-1`}>{preset.name}</h3>
                      <p className="text-sm text-gray-400">{preset.description}</p>
                      <div className="mt-2 text-xs text-gray-500">
                        {preset.values.max_daily_loss_pct}% loss | {preset.values.max_trades_per_day} trades/day
                      </div>
                      <ChevronRight className="w-4 h-4 text-gray-500 group-hover:text-cyan-400 ml-auto mt-2 transition-colors" />
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {activeTab === 'profile' && (
        <div className="bg-gray-900/50 border border-gray-800/50 rounded-xl p-6">
          <h2 className="text-lg font-bold text-white flex items-center gap-2 mb-6">
            <User className="w-5 h-5 text-cyan-400" />
            Profile Settings
          </h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">Display Name</label>
              <input
                type="text"
                placeholder="Your name"
                className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-cyan-500"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-2">Email</label>
              <input
                type="email"
                placeholder="your@email.com"
                className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-cyan-500"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-2">Timezone</label>
              <select className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-cyan-500">
                <option value="America/Bogota">America/Bogota (COT)</option>
                <option value="America/New_York">America/New_York (EST)</option>
                <option value="UTC">UTC</option>
              </select>
            </div>

            <div className="pt-4">
              <button className="flex items-center gap-2 px-6 py-3 bg-cyan-500 text-white rounded-lg hover:bg-cyan-600 transition-colors">
                <Save className="w-4 h-4" />
                Save Profile
              </button>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'notifications' && (
        <div className="bg-gray-900/50 border border-gray-800/50 rounded-xl p-6">
          <h2 className="text-lg font-bold text-white flex items-center gap-2 mb-6">
            <Bell className="w-5 h-5 text-cyan-400" />
            Notification Preferences
          </h2>

          <div className="space-y-4">
            {[
              { id: 'trade_executed', label: 'Trade Executed', description: 'Get notified when a trade is executed' },
              { id: 'risk_alert', label: 'Risk Alerts', description: 'Alerts when approaching risk limits' },
              { id: 'kill_switch', label: 'Kill Switch Events', description: 'Notifications about kill switch status' },
              { id: 'daily_summary', label: 'Daily Summary', description: 'End of day performance summary' },
            ].map((notification) => (
              <label
                key={notification.id}
                className="flex items-center justify-between p-4 bg-gray-800/30 rounded-lg cursor-pointer"
              >
                <div>
                  <h3 className="font-medium text-white">{notification.label}</h3>
                  <p className="text-sm text-gray-400">{notification.description}</p>
                </div>
                <input
                  type="checkbox"
                  defaultChecked
                  className="w-5 h-5 text-cyan-500 bg-gray-700 border-gray-600 rounded focus:ring-cyan-500"
                />
              </label>
            ))}

            <div className="pt-4">
              <button className="flex items-center gap-2 px-6 py-3 bg-cyan-500 text-white rounded-lg hover:bg-cyan-600 transition-colors">
                <Save className="w-4 h-4" />
                Save Preferences
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

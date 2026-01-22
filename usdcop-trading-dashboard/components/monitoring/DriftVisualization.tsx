'use client';

/**
 * DriftVisualization.tsx
 * ======================
 * Sprint 3: COMP-88 - Dashboard Improvements
 *
 * Component for visualizing feature drift detection results:
 * - Overall drift status indicator
 * - Per-feature drift metrics
 * - Multivariate drift methods (MMD, Wasserstein, PCA)
 * - Historical drift trends
 *
 * BACKEND DEPENDENCIES FOR REAL DATA:
 * ====================================
 *
 * 1. API Endpoint: GET /api/v1/monitoring/drift
 *    Returns: DriftReport with univariate and multivariate results
 *
 * 2. Backend Requirements (services/inference_api):
 *    - app.state.drift_detector: FeatureDriftDetector instance
 *    - app.state.multivariate_drift_detector: MultivariateDriftDetector instance
 *
 * 3. Reference Data:
 *    - Univariate: config/norm_stats.json (auto-loaded on startup)
 *    - Multivariate: POST /api/v1/monitoring/drift/reference/multivariate
 *      with 500-1000 historical observations
 *
 * 4. Observation Feeding (for live drift detection):
 *    - POST /api/v1/monitoring/drift/observe after each inference
 *    - Or POST /api/v1/monitoring/drift/observe/batch for bulk updates
 *
 * 5. Status Check:
 *    - GET /api/v1/monitoring/drift/status
 *    - Verify: ready_for_detection === true
 *
 * If backend is not configured, component uses MOCK_DRIFT_REPORT for demo.
 */

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  TrendingDown,
  BarChart3,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Info,
  Layers,
  Zap,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';

// ============================================================================
// Types
// ============================================================================
interface FeatureDriftResult {
  feature_name: string;
  is_drifted: boolean;
  p_value: number;
  statistic: number;
  drift_severity: 'none' | 'low' | 'medium' | 'high';
  reference_mean?: number;
  current_mean?: number;
  percent_change?: number;
}

interface MultivariateDriftResult {
  method: string;
  score: number;
  threshold: number;
  is_drifted: boolean;
  details?: Record<string, number>;
}

interface DriftReport {
  timestamp: string;
  features_checked: number;
  features_drifted: number;
  overall_drift_score: number;
  alert_active: boolean;
  univariate_results: FeatureDriftResult[];
  multivariate_results: {
    mmd?: MultivariateDriftResult;
    wasserstein?: MultivariateDriftResult;
    pca_reconstruction?: MultivariateDriftResult;
  };
  aggregate: {
    any_drifted: boolean;
    max_severity: 'none' | 'low' | 'medium' | 'high';
    methods_triggered: string[];
  };
}

interface DriftVisualizationProps {
  className?: string;
  compact?: boolean;
  onAlertTrigger?: (severity: string) => void;
}

// ============================================================================
// Mock data for development/demo
// ============================================================================
const MOCK_DRIFT_REPORT: DriftReport = {
  timestamp: new Date().toISOString(),
  features_checked: 15,
  features_drifted: 2,
  overall_drift_score: 0.18,
  alert_active: false,
  univariate_results: [
    { feature_name: 'rsi_9', is_drifted: false, p_value: 0.45, statistic: 0.12, drift_severity: 'none' },
    { feature_name: 'atr_pct', is_drifted: true, p_value: 0.008, statistic: 0.34, drift_severity: 'medium', percent_change: 15.2 },
    { feature_name: 'macd_hist', is_drifted: false, p_value: 0.23, statistic: 0.18, drift_severity: 'none' },
    { feature_name: 'log_ret_5m', is_drifted: false, p_value: 0.67, statistic: 0.08, drift_severity: 'none' },
    { feature_name: 'volatility_20', is_drifted: true, p_value: 0.003, statistic: 0.42, drift_severity: 'high', percent_change: 28.5 },
    { feature_name: 'adx_14', is_drifted: false, p_value: 0.31, statistic: 0.15, drift_severity: 'none' },
    { feature_name: 'bb_width', is_drifted: false, p_value: 0.52, statistic: 0.11, drift_severity: 'low' },
    { feature_name: 'obv_norm', is_drifted: false, p_value: 0.78, statistic: 0.06, drift_severity: 'none' },
  ],
  multivariate_results: {
    mmd: { method: 'MMD', score: 0.15, threshold: 0.3, is_drifted: false },
    wasserstein: { method: 'Wasserstein', score: 0.22, threshold: 0.4, is_drifted: false },
    pca_reconstruction: { method: 'PCA Recon', score: 0.08, threshold: 0.15, is_drifted: false },
  },
  aggregate: {
    any_drifted: true,
    max_severity: 'high',
    methods_triggered: [],
  },
};

// ============================================================================
// Severity utilities
// ============================================================================
const SEVERITY_CONFIG = {
  none: { color: 'emerald', bgClass: 'bg-emerald-500/10', textClass: 'text-emerald-400', borderClass: 'border-emerald-500/30' },
  low: { color: 'yellow', bgClass: 'bg-yellow-500/10', textClass: 'text-yellow-400', borderClass: 'border-yellow-500/30' },
  medium: { color: 'orange', bgClass: 'bg-orange-500/10', textClass: 'text-orange-400', borderClass: 'border-orange-500/30' },
  high: { color: 'red', bgClass: 'bg-red-500/10', textClass: 'text-red-400', borderClass: 'border-red-500/30' },
};

const getSeverityConfig = (severity: string) => SEVERITY_CONFIG[severity as keyof typeof SEVERITY_CONFIG] || SEVERITY_CONFIG.none;

// ============================================================================
// Main Component
// ============================================================================
export function DriftVisualization({ className, compact = false, onAlertTrigger }: DriftVisualizationProps) {
  const [report, setReport] = useState<DriftReport | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isExpanded, setIsExpanded] = useState(!compact);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchDriftReport = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/v1/monitoring/drift');
      if (!response.ok) {
        // Use mock data if API unavailable
        setReport(MOCK_DRIFT_REPORT);
        setLastUpdated(new Date());
        return;
      }
      const data = await response.json();
      setReport(data);
      setLastUpdated(new Date());

      // Trigger alert callback if drift detected
      if (data.aggregate?.any_drifted && onAlertTrigger) {
        onAlertTrigger(data.aggregate.max_severity);
      }
    } catch (err) {
      // Use mock data for demo
      setReport(MOCK_DRIFT_REPORT);
      setLastUpdated(new Date());
    } finally {
      setIsLoading(false);
    }
  }, [onAlertTrigger]);

  useEffect(() => {
    fetchDriftReport();
    // Auto-refresh every 5 minutes
    const interval = setInterval(fetchDriftReport, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [fetchDriftReport]);

  // Prepare data for charts
  const barChartData = report?.univariate_results.map((r) => ({
    name: r.feature_name.replace(/_/g, ' '),
    score: r.statistic,
    severity: r.drift_severity,
    pValue: r.p_value,
  })) || [];

  const radarData = report?.multivariate_results ? [
    {
      method: 'MMD',
      score: (report.multivariate_results.mmd?.score || 0) * 100,
      threshold: (report.multivariate_results.mmd?.threshold || 0.3) * 100,
      fullMark: 100,
    },
    {
      method: 'Wasserstein',
      score: (report.multivariate_results.wasserstein?.score || 0) * 100,
      threshold: (report.multivariate_results.wasserstein?.threshold || 0.4) * 100,
      fullMark: 100,
    },
    {
      method: 'PCA',
      score: (report.multivariate_results.pca_reconstruction?.score || 0) * 100,
      threshold: (report.multivariate_results.pca_reconstruction?.threshold || 0.15) * 100,
      fullMark: 100,
    },
  ] : [];

  if (isLoading && !report) {
    return (
      <Card className={cn("bg-slate-900/50 border-slate-700/50", className)}>
        <CardContent className="p-6">
          <div className="flex items-center justify-center gap-3">
            <RefreshCw className="h-5 w-5 animate-spin text-cyan-400" />
            <span className="text-slate-400">Loading drift data...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error && !report) {
    return (
      <Card className={cn("bg-slate-900/50 border-red-500/30", className)}>
        <CardContent className="p-6">
          <div className="flex items-center gap-3 text-red-400">
            <AlertTriangle className="h-5 w-5" />
            <span>{error}</span>
            <Button variant="ghost" size="sm" onClick={fetchDriftReport}>
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  const severityConfig = getSeverityConfig(report?.aggregate.max_severity || 'none');
  const driftedCount = report?.features_drifted || 0;
  const totalFeatures = report?.features_checked || 0;

  // Compact view for sidebar or smaller spaces
  if (compact && !isExpanded) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className={cn("p-4 rounded-xl border bg-slate-900/50", severityConfig.borderClass, className)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={cn("p-2 rounded-lg", severityConfig.bgClass)}>
              <Activity className={cn("h-4 w-4", severityConfig.textClass)} />
            </div>
            <div>
              <div className="text-sm font-medium text-slate-200">Feature Drift</div>
              <div className="text-xs text-slate-400">
                {driftedCount}/{totalFeatures} features drifted
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge
              variant="outline"
              className={cn("text-xs", severityConfig.bgClass, severityConfig.textClass, severityConfig.borderClass)}
            >
              {report?.aggregate.max_severity?.toUpperCase() || 'OK'}
            </Badge>
            <Button variant="ghost" size="sm" onClick={() => setIsExpanded(true)}>
              <ChevronDown className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <Card className={cn("bg-slate-900/50 border-slate-700/50", className)}>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={cn("p-2 rounded-xl", severityConfig.bgClass, severityConfig.borderClass)}>
              <Layers className={cn("h-5 w-5", severityConfig.textClass)} />
            </div>
            <div>
              <CardTitle className="text-lg">Feature Drift Monitor</CardTitle>
              <p className="text-xs text-slate-400 mt-1">
                Multivariate drift detection with KS, MMD, Wasserstein
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={fetchDriftReport}
              disabled={isLoading}
              className="gap-2"
            >
              <RefreshCw className={cn("h-4 w-4", isLoading && "animate-spin")} />
              {!compact && "Refresh"}
            </Button>
            {compact && (
              <Button variant="ghost" size="sm" onClick={() => setIsExpanded(false)}>
                <ChevronUp className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Summary Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {/* Overall Status */}
          <div className={cn("p-4 rounded-xl border", severityConfig.bgClass, severityConfig.borderClass)}>
            <div className="flex items-center gap-2 mb-2">
              {report?.aggregate.any_drifted ? (
                <AlertTriangle className={cn("h-4 w-4", severityConfig.textClass)} />
              ) : (
                <CheckCircle className="h-4 w-4 text-emerald-400" />
              )}
              <span className="text-xs font-medium text-slate-400">Status</span>
            </div>
            <div className={cn("text-xl font-bold", severityConfig.textClass)}>
              {report?.aggregate.any_drifted ? 'DRIFT' : 'STABLE'}
            </div>
          </div>

          {/* Drift Score */}
          <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="h-4 w-4 text-cyan-400" />
              <span className="text-xs font-medium text-slate-400">Score</span>
            </div>
            <div className="text-xl font-bold text-slate-100">
              {((report?.overall_drift_score || 0) * 100).toFixed(1)}%
            </div>
          </div>

          {/* Drifted Features */}
          <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50">
            <div className="flex items-center gap-2 mb-2">
              <Zap className="h-4 w-4 text-amber-400" />
              <span className="text-xs font-medium text-slate-400">Drifted</span>
            </div>
            <div className="text-xl font-bold text-slate-100">
              {driftedCount}<span className="text-sm text-slate-400">/{totalFeatures}</span>
            </div>
          </div>

          {/* Last Updated */}
          <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50">
            <div className="flex items-center gap-2 mb-2">
              <RefreshCw className="h-4 w-4 text-slate-400" />
              <span className="text-xs font-medium text-slate-400">Updated</span>
            </div>
            <div className="text-sm font-medium text-slate-100">
              {lastUpdated?.toLocaleTimeString('es-CO', { hour: '2-digit', minute: '2-digit' }) || '--:--'}
            </div>
          </div>
        </div>

        {/* Feature Drift Bar Chart */}
        <div>
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="h-4 w-4 text-slate-400" />
            <span className="text-sm font-medium text-slate-300">Univariate Drift by Feature</span>
          </div>
          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barChartData} layout="vertical" margin={{ left: 80, right: 20 }}>
                <XAxis type="number" domain={[0, 0.5]} tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <YAxis
                  type="category"
                  dataKey="name"
                  tick={{ fill: '#94a3b8', fontSize: 10 }}
                  width={75}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #334155',
                    borderRadius: '8px',
                  }}
                  formatter={(value: number, name: string, props: any) => [
                    `${(value * 100).toFixed(1)}%`,
                    `KS Statistic (p=${props.payload.pValue.toFixed(3)})`
                  ]}
                />
                <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                  {barChartData.map((entry, index) => {
                    const config = getSeverityConfig(entry.severity);
                    const colors: Record<string, string> = {
                      none: '#10b981',
                      low: '#eab308',
                      medium: '#f97316',
                      high: '#ef4444',
                    };
                    return <Cell key={`cell-${index}`} fill={colors[entry.severity] || '#10b981'} />;
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Multivariate Methods Radar */}
        <div>
          <div className="flex items-center gap-2 mb-4">
            <Layers className="h-4 w-4 text-slate-400" />
            <span className="text-sm font-medium text-slate-300">Multivariate Drift Detection</span>
          </div>
          <div className="grid md:grid-cols-2 gap-4">
            {/* Radar Chart */}
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#334155" />
                  <PolarAngleAxis dataKey="method" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <PolarRadiusAxis tick={{ fill: '#64748b', fontSize: 9 }} domain={[0, 100]} />
                  <Radar
                    name="Threshold"
                    dataKey="threshold"
                    stroke="#64748b"
                    fill="#64748b"
                    fillOpacity={0.2}
                    strokeDasharray="3 3"
                  />
                  <Radar
                    name="Score"
                    dataKey="score"
                    stroke="#06b6d4"
                    fill="#06b6d4"
                    fillOpacity={0.4}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* Method Details */}
            <div className="space-y-3">
              {Object.entries(report?.multivariate_results || {}).map(([key, result]) => {
                if (!result) return null;
                const ratio = result.score / result.threshold;
                const isDrifted = result.is_drifted;

                return (
                  <div
                    key={key}
                    className={cn(
                      "p-3 rounded-lg border",
                      isDrifted
                        ? "bg-red-500/10 border-red-500/30"
                        : "bg-slate-800/50 border-slate-700/50"
                    )}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-slate-200">{result.method}</span>
                      <Badge
                        variant="outline"
                        className={cn(
                          "text-xs",
                          isDrifted
                            ? "bg-red-500/10 text-red-400 border-red-500/30"
                            : "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                        )}
                      >
                        {isDrifted ? 'DRIFT' : 'OK'}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${Math.min(ratio * 100, 100)}%` }}
                          className={cn(
                            "h-full rounded-full",
                            ratio > 1 ? "bg-red-500" : ratio > 0.7 ? "bg-amber-500" : "bg-cyan-500"
                          )}
                        />
                      </div>
                      <span className="text-xs text-slate-400 w-20 text-right">
                        {result.score.toFixed(3)} / {result.threshold.toFixed(2)}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Drifted Features List */}
        {driftedCount > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-3">
              <AlertTriangle className="h-4 w-4 text-amber-400" />
              <span className="text-sm font-medium text-slate-300">Drifted Features</span>
            </div>
            <div className="space-y-2">
              {report?.univariate_results
                .filter((r) => r.is_drifted)
                .map((r) => {
                  const config = getSeverityConfig(r.drift_severity);
                  return (
                    <motion.div
                      key={r.feature_name}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      className={cn(
                        "flex items-center justify-between p-3 rounded-lg",
                        config.bgClass,
                        "border",
                        config.borderClass
                      )}
                    >
                      <div className="flex items-center gap-3">
                        <span className={cn("font-mono text-sm", config.textClass)}>
                          {r.feature_name}
                        </span>
                        <Badge variant="outline" className={cn("text-xs", config.textClass, config.borderClass)}>
                          {r.drift_severity.toUpperCase()}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-4 text-xs">
                        <span className="text-slate-400">
                          KS: {r.statistic.toFixed(3)}
                        </span>
                        <span className="text-slate-400">
                          p: {r.p_value.toFixed(4)}
                        </span>
                        {r.percent_change && (
                          <span className={config.textClass}>
                            {r.percent_change > 0 ? '+' : ''}{r.percent_change.toFixed(1)}%
                          </span>
                        )}
                      </div>
                    </motion.div>
                  );
                })}
            </div>
          </div>
        )}

        {/* Info Footer */}
        <div className="flex items-start gap-2 p-3 rounded-lg bg-slate-800/30 border border-slate-700/30">
          <Info className="h-4 w-4 text-slate-500 mt-0.5" />
          <div className="text-xs text-slate-500">
            <p>
              <strong>Univariate:</strong> Kolmogorov-Smirnov test per feature (p &lt; 0.01 = drift).
            </p>
            <p className="mt-1">
              <strong>Multivariate:</strong> MMD (distribution similarity), Wasserstein (optimal transport),
              PCA (reconstruction error).
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default DriftVisualization;

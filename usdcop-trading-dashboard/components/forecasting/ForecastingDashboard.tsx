'use client';

/**
 * ForecastingDashboard - Professional Mobile-First Centered Design
 * ================================================================
 * Matches Landing Page patterns:
 * - All content centered
 * - Section headers with decorative lines
 * - Professional card styling
 * - Mobile-first responsive design
 */

import { useState, useEffect, useMemo } from 'react';
import {
  LayoutDashboard, Calendar, Filter, Clock, TrendingUp,
  BarChart3, Target, Activity, ChevronDown, Loader2, AlertCircle,
  LineChart, Percent, Settings2
} from 'lucide-react';
import Papa from 'papaparse';
import { ForecastingImageViewer } from './ForecastingImageViewer';
import { MetricsRankingPanel } from './MetricsRankingPanel';
import { ForecastRecord, ViewType, EnsembleVariant } from './types';
import { cn } from '@/lib/utils';

// ============================================================================
// Data Processing Utilities
// ============================================================================
const loadData = async (url: string): Promise<ForecastRecord[]> => {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  const text = await response.text();

  return new Promise((resolve, reject) => {
    Papa.parse(text, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        resolve(results.data as ForecastRecord[]);
      },
      error: (error) => {
        reject(error);
      }
    });
  });
};

const filterData = (data: ForecastRecord[], filters: { viewType: string; inferenceWeek: string }) => {
  if (!data) return [];
  return data.filter(row => {
    if (row.view_type !== filters.viewType) return false;
    if (filters.viewType === 'forward_forecast' && filters.inferenceWeek) {
      if (row.inference_week != filters.inferenceWeek) return false;
    }
    return true;
  });
};

const getUniqueValues = <T,>(data: T[], field: keyof T): string[] => {
  if (!data) return [];
  const values = new Set(data.map(row => String(row[field])).filter(v => v != null && v !== 'null' && v !== 'undefined'));
  return Array.from(values).sort();
};

// ============================================================================
// Section Header Component - Centered with decorative line
// ============================================================================
interface SectionHeaderProps {
  title: string;
  subtitle?: string;
  icon?: React.ReactNode;
}

function SectionHeader({ title, subtitle, icon }: SectionHeaderProps) {
  return (
    <div className="mb-8 sm:mb-10 lg:mb-12 text-center">
      <h2 className="text-xl sm:text-2xl lg:text-3xl font-bold text-white flex items-center justify-center gap-3">
        {icon && <span className="text-purple-400">{icon}</span>}
        {title}
      </h2>
      {subtitle && (
        <p className="mt-3 sm:mt-4 text-sm sm:text-base text-slate-400 max-w-2xl mx-auto leading-relaxed">
          {subtitle}
        </p>
      )}
      {/* Decorative line */}
      <div className="mt-4 sm:mt-6 flex items-center justify-center gap-1">
        <div className="h-0.5 w-6 rounded-full bg-gradient-to-r from-transparent to-purple-500/50" />
        <div className="h-0.5 w-16 rounded-full bg-gradient-to-r from-purple-500/50 to-pink-500/50" />
        <div className="h-0.5 w-6 rounded-full bg-gradient-to-r from-pink-500/50 to-transparent" />
      </div>
    </div>
  );
}

// ============================================================================
// KPI Card Component - Professional centered style
// ============================================================================
interface KPICardProps {
  title: string;
  value: string;
  icon: React.ReactNode;
  color?: string;
}

function KPICard({ title, value, icon, color = '#A855F7' }: KPICardProps) {
  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-5 sm:p-6 hover:border-slate-600 transition-all duration-300 hover:shadow-xl hover:shadow-purple-500/5">
      <div className="flex flex-col items-center text-center">
        <div
          className="p-3 rounded-xl mb-4"
          style={{ backgroundColor: `${color}15` }}
        >
          <div style={{ color }}>{icon}</div>
        </div>
        <span className="text-[10px] sm:text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">
          {title}
        </span>
        <div
          className="text-xl sm:text-2xl font-bold tracking-tight"
          style={{ color }}
        >
          {value}
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================
export function ForecastingDashboard() {
  const [data, setData] = useState<ForecastRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters state
  const [viewType, setViewType] = useState<ViewType>('forward_forecast');
  const [inferenceWeek, setInferenceWeek] = useState('');
  const [selectedModel, setSelectedModel] = useState('ALL');
  const [horizon, setHorizon] = useState('ALL');

  // Mobile filters state
  const [isFiltersOpen, setIsFiltersOpen] = useState(false);

  // Ensemble variants
  const ensembleVariants: EnsembleVariant[] = [
    { value: 'ENSEMBLE_BEST_OF_BREED', label: 'Best of Breed', imageKey: 'best_of_breed' },
    { value: 'ENSEMBLE_TOP_3', label: 'Top 3 Average', imageKey: 'top_3' },
    { value: 'ENSEMBLE_TOP_6_MEAN', label: 'Top 6 Average', imageKey: 'top_6_mean' }
  ];

  // Load data on mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        const csvUrl = '/forecasting/bi_dashboard_unified.csv';
        const loadedData = await loadData(csvUrl);
        setData(loadedData);

        const weeks = getUniqueValues(loadedData, 'inference_week');
        if (weeks.length > 0) setInferenceWeek(weeks[weeks.length - 1]);
      } catch (err) {
        console.error('Failed to load data:', err);
        setError('No se pudieron cargar los datos. Verifica que el archivo CSV este disponible.');
      }
      setLoading(false);
    };

    fetchData();
  }, []);

  // Filter logic
  const filteredData = useMemo(() => {
    let res = filterData(data, { viewType, inferenceWeek });

    if (selectedModel !== 'ALL') {
      res = res.filter(row => row.model_name === selectedModel);
    }

    if (horizon !== 'ALL') {
      res = res.filter(row => String(row.horizon_days) === String(horizon));
    }

    return res;
  }, [data, viewType, inferenceWeek, selectedModel, horizon]);

  const weeks = useMemo(() => getUniqueValues(data, 'inference_week'), [data]);
  const models = useMemo(() => getUniqueValues(data, 'model_name'), [data]);
  const availableHorizons = useMemo(() => {
    const filtered = data.filter(row => row.view_type === viewType);
    return getUniqueValues(filtered, 'horizon_days');
  }, [data, viewType]);

  // Helper functions
  const getModelDisplayLabel = (modelName: string) => {
    if (modelName === 'CONSENSUS') return 'Consensus (All Models Average)';
    if (modelName === 'ENSEMBLE') return 'Ensemble (Weighted)';
    const ensembleVar = ensembleVariants.find(v => v.value === modelName);
    if (ensembleVar) return ensembleVar.label;
    return modelName;
  };

  const isEnsembleVariant = (modelName: string) => {
    return ensembleVariants.some(v => v.value === modelName);
  };

  const getEnsembleImageKey = (modelName: string) => {
    const variant = ensembleVariants.find(v => v.value === modelName);
    return variant ? variant.imageKey : null;
  };

  const formatNum = (val: number | null | undefined, decimals = 2) => {
    if (val === null || val === undefined || isNaN(Number(val))) return 'N/A';
    return Number(val).toFixed(decimals);
  };

  // Render content based on selection
  const renderContent = () => {
    let imageSrc: string | null = null;
    let imageCaption = "";
    let metrics: { title: string; value: string; icon: React.ReactNode; color?: string }[] = [];

    // Case 1: Consolidated Forecast (All Models)
    if (viewType === 'forward_forecast' && selectedModel === 'ALL') {
      imageSrc = 'forward_forecast_consensus_all_models.png';
      imageCaption = "Consensus Forecast (All Models + Average)";

      if (filteredData.length > 0) {
        const row = filteredData[0];
        metrics = [
          { title: "Avg Direction Accuracy", value: `${formatNum(row.model_avg_direction_accuracy)}%`, icon: <Target className="w-5 h-5" />, color: '#10B981' },
          { title: "Avg RMSE", value: formatNum(row.model_avg_rmse, 4), icon: <Activity className="w-5 h-5" />, color: '#F59E0B' }
        ];
      }
    }
    // Case 2: Ensemble Variant Selected
    else if (isEnsembleVariant(selectedModel)) {
      const ensembleImageKey = getEnsembleImageKey(selectedModel);
      const displayLabel = getModelDisplayLabel(selectedModel);

      imageSrc = `forward_forecast_ensemble_${ensembleImageKey}.png`;
      imageCaption = `${displayLabel} Ensemble Forecast`;

      metrics = [
        { title: "Ensemble Method", value: displayLabel, icon: <BarChart3 className="w-5 h-5" />, color: '#A855F7' },
        { title: "Type", value: ensembleImageKey === 'best_of_breed' ? 'Best/horizon' :
                                ensembleImageKey === 'top_3' ? 'Top 3 Avg' :
                                'Top 6 Avg', icon: <LineChart className="w-5 h-5" />, color: '#EC4899' }
      ];
    }
    // Case 3: Specific Model Selected
    else {
      if (filteredData.length >= 1) {
        const row = filteredData[0];

        if (viewType === 'backtest') {
          imageSrc = row.image_backtest;
          imageCaption = `${row.model_name} Backtest (H=${row.horizon_days})`;
          metrics = [
            { title: "Direction Accuracy", value: `${formatNum(row.direction_accuracy)}%`, icon: <Target className="w-5 h-5" />, color: '#10B981' },
            { title: "RMSE", value: formatNum(row.rmse, 4), icon: <Activity className="w-5 h-5" />, color: '#F59E0B' },
            { title: "MAE", value: formatNum(row.mae, 4), icon: <BarChart3 className="w-5 h-5" />, color: '#3B82F6' },
            { title: "R2 Score", value: formatNum(row.r2, 4), icon: <Percent className="w-5 h-5" />, color: '#8B5CF6' }
          ];
        } else {
          imageSrc = row.image_forecast;
          imageCaption = `${row.model_name} Forecast (H=${row.horizon_days})`;
          metrics = [
            { title: "WF Direction Accuracy", value: `${formatNum(row.wf_direction_accuracy)}%`, icon: <Target className="w-5 h-5" />, color: '#10B981' },
            { title: "Sharpe Ratio", value: formatNum(row.sharpe, 2), icon: <TrendingUp className="w-5 h-5" />, color: '#3B82F6' },
            { title: "Profit Factor", value: formatNum(row.profit_factor, 2), icon: <BarChart3 className="w-5 h-5" />, color: '#F59E0B' },
            { title: "Max Drawdown", value: `${formatNum(row.max_drawdown ? row.max_drawdown * 100 : null, 1)}%`, icon: <Activity className="w-5 h-5" />, color: '#EF4444' }
          ];
        }
      } else {
        return (
          <div className="flex items-center justify-center h-64 text-gray-500">
            <div className="text-center">
              <AlertCircle className="w-12 h-12 mx-auto mb-3 text-gray-600" />
              <p className="text-sm">No hay datos para esta seleccion.</p>
            </div>
          </div>
        );
      }
    }

    return (
      <div className="space-y-12 sm:space-y-16 lg:space-y-20">
        {/* Rankings Panel - Only in Backtest view */}
        {viewType === 'backtest' && (
          <div className="text-center">
            <SectionHeader
              title="Ranking de Modelos"
              subtitle="Comparacion de rendimiento entre modelos"
              icon={<BarChart3 className="w-6 h-6 sm:w-7 sm:h-7" />}
            />
            <MetricsRankingPanel
              data={data}
              selectedHorizon={horizon}
              selectedModel={selectedModel}
            />
          </div>
        )}

        {/* Chart Section - Centered */}
        <div className="text-center">
          <SectionHeader
            title={viewType === 'backtest' ? 'Backtest Analysis' : 'Forecast Chart'}
            subtitle={imageCaption}
            icon={<LineChart className="w-6 h-6 sm:w-7 sm:h-7" />}
          />
          <div className="w-full rounded-2xl overflow-hidden border border-slate-700/50 shadow-2xl">
            <ForecastingImageViewer
              src={imageSrc}
              caption={imageCaption}
            />
          </div>
        </div>

        {/* Metrics Grid - Centered */}
        {metrics.length > 0 && (
          <div className="text-center">
            <SectionHeader
              title="Metricas Clave"
              subtitle="Indicadores de rendimiento del modelo"
              icon={<Target className="w-6 h-6 sm:w-7 sm:h-7" />}
            />
            <div className="w-full grid grid-cols-2 sm:grid-cols-4 gap-6 sm:gap-8">
              {metrics.map((m, idx) => (
                <KPICard
                  key={idx}
                  title={m.title}
                  value={m.value}
                  icon={m.icon}
                  color={m.color}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  // Loading state
  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex flex-col items-center gap-4 text-gray-400">
          <Loader2 className="w-10 h-10 animate-spin text-purple-500" />
          <span className="text-sm">Cargando datos del Forecasting...</span>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center text-red-400 max-w-md">
          <AlertCircle className="w-12 h-12 mx-auto mb-4" />
          <p className="text-sm">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-12 sm:space-y-16 lg:space-y-20">
      {/* Filters Section - Centered */}
      <div className="text-center">
        <SectionHeader
          title="Configuracion"
          subtitle="Selecciona el modo de vista, modelo y parametros"
          icon={<Settings2 className="w-6 h-6 sm:w-7 sm:h-7" />}
        />

        {/* Filters Grid - Centered */}
        <div className="w-full">
          {/* Mobile: Collapsible */}
          <div className="lg:hidden mb-6">
            <button
              onClick={() => setIsFiltersOpen(!isFiltersOpen)}
              className={cn(
                "w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl border transition-all",
                isFiltersOpen
                  ? "bg-purple-500/10 border-purple-500/30"
                  : "bg-slate-900/50 border-slate-800 hover:border-slate-700"
              )}
            >
              <Filter className="w-4 h-4 text-purple-400" />
              <span className="text-sm text-white font-medium">
                {isFiltersOpen ? 'Ocultar Filtros' : 'Mostrar Filtros'}
              </span>
              <ChevronDown className={cn(
                "w-4 h-4 text-gray-400 transition-transform duration-200",
                isFiltersOpen && "rotate-180"
              )} />
            </button>
          </div>

          {/* Filters Content - Centered grid */}
          <div className={cn(
            "flex flex-wrap justify-center gap-4 sm:gap-6",
            !isFiltersOpen && "hidden lg:flex"
          )}>
            {/* View Mode */}
            <div className="w-full sm:w-auto sm:min-w-[200px] bg-slate-900/50 rounded-xl p-4 border border-slate-800 text-center">
              <label className="flex items-center justify-center gap-2 text-xs font-medium text-gray-400 mb-3">
                <LayoutDashboard className="w-4 h-4 text-purple-400" />
                Modo de Vista
              </label>
              <select
                value={viewType}
                onChange={(e) => {
                  setViewType(e.target.value as ViewType);
                  setHorizon('ALL');
                }}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              >
                <option value="forward_forecast">Forward Forecast</option>
                <option value="backtest">Backtest Analysis</option>
              </select>
            </div>

            {/* Inference Week (only for forward forecast) */}
            {viewType === 'forward_forecast' && (
              <div className="w-full sm:w-auto sm:min-w-[200px] bg-slate-900/50 rounded-xl p-4 border border-slate-800 text-center">
                <label className="flex items-center justify-center gap-2 text-xs font-medium text-gray-400 mb-3">
                  <Calendar className="w-4 h-4 text-purple-400" />
                  Semana
                </label>
                <select
                  value={inferenceWeek}
                  onChange={(e) => setInferenceWeek(e.target.value)}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  {weeks.map(w => (
                    <option key={w} value={w}>Week {w}</option>
                  ))}
                </select>
              </div>
            )}

            {/* Model Filter */}
            <div className="w-full sm:w-auto sm:min-w-[220px] bg-slate-900/50 rounded-xl p-4 border border-slate-800 text-center">
              <label className="flex items-center justify-center gap-2 text-xs font-medium text-gray-400 mb-3">
                <Target className="w-4 h-4 text-purple-400" />
                Modelo
              </label>
              <select
                value={selectedModel}
                onChange={(e) => {
                  setSelectedModel(e.target.value);
                  if (e.target.value === 'ALL') setHorizon('ALL');
                }}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              >
                <option value="ALL">All Models (Consensus)</option>
                {models.filter(m => !m.includes('ENSEMBLE') && m !== 'CONSENSUS').map(m => (
                  <option key={m} value={m}>{getModelDisplayLabel(m)}</option>
                ))}
                <option disabled className="text-slate-600">--- Ensembles ---</option>
                {ensembleVariants.map(v => (
                  <option key={v.value} value={v.value}>{v.label}</option>
                ))}
              </select>
            </div>

            {/* Horizon Filter (only for backtest with model selected) */}
            {selectedModel !== 'ALL' && viewType === 'backtest' && (
              <div className="w-full sm:w-auto sm:min-w-[180px] bg-slate-900/50 rounded-xl p-4 border border-slate-800 text-center">
                <label className="flex items-center justify-center gap-2 text-xs font-medium text-gray-400 mb-3">
                  <Clock className="w-4 h-4 text-purple-400" />
                  Horizonte
                </label>
                <select
                  value={horizon}
                  onChange={(e) => setHorizon(e.target.value)}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  <option value="ALL">Overview (Avg)</option>
                  {availableHorizons.map(h => (
                    <option key={h} value={h}>H={h} Days</option>
                  ))}
                </select>
              </div>
            )}
          </div>

          {/* Quick Stats - Centered */}
          <div className="mt-8 flex flex-wrap items-center justify-center gap-6 sm:gap-10">
            <div className="text-center">
              <div className="text-2xl sm:text-3xl font-bold text-purple-400">{models.length}</div>
              <div className="text-xs text-gray-500 uppercase tracking-wider mt-1">Modelos</div>
            </div>
            <div className="text-center">
              <div className="text-2xl sm:text-3xl font-bold text-pink-400">{availableHorizons.length}</div>
              <div className="text-xs text-gray-500 uppercase tracking-wider mt-1">Horizontes</div>
            </div>
            <div className="text-center">
              <div className="text-2xl sm:text-3xl font-bold text-cyan-400">{filteredData.length}</div>
              <div className="text-xs text-gray-500 uppercase tracking-wider mt-1">Registros</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      {renderContent()}
    </div>
  );
}

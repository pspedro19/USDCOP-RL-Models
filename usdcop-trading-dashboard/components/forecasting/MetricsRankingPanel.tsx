'use client';

import { useMemo } from 'react';
import { Trophy, Star, AlertTriangle, Check, X } from 'lucide-react';
import { ForecastRecord, ModelMetrics } from './types';

interface MetricsRankingPanelProps {
  data: ForecastRecord[];
  selectedHorizon: string;
  selectedModel: string;
}

export function MetricsRankingPanel({ data, selectedHorizon, selectedModel }: MetricsRankingPanelProps) {
  const rankings = useMemo(() => {
    if (!data || data.length === 0) return [];

    // Filter only backtest rows with trading metrics
    const filtered = data.filter(row =>
      row.view_type === 'backtest' &&
      (selectedHorizon === 'ALL' || row.horizon_days === parseInt(selectedHorizon))
    );

    // Group by model and calculate averages
    const modelMetrics: Record<string, {
      model_id: string;
      da_values: number[];
      sharpe_values: number[];
      pf_values: number[];
      mdd_values: number[];
      return_values: number[];
    }> = {};

    filtered.forEach(row => {
      const modelId = row.model_id;
      if (!modelMetrics[modelId]) {
        modelMetrics[modelId] = {
          model_id: modelId,
          da_values: [],
          sharpe_values: [],
          pf_values: [],
          mdd_values: [],
          return_values: []
        };
      }
      const m = modelMetrics[modelId];

      const da = row.wf_direction_accuracy || row.direction_accuracy;
      if (da != null && !isNaN(da)) m.da_values.push(da);
      if (row.sharpe != null && !isNaN(row.sharpe)) m.sharpe_values.push(row.sharpe);
      if (row.profit_factor != null && !isNaN(row.profit_factor)) m.pf_values.push(row.profit_factor);
      if (row.max_drawdown != null && !isNaN(row.max_drawdown)) m.mdd_values.push(row.max_drawdown);
      if (row.total_return != null && !isNaN(row.total_return)) m.return_values.push(row.total_return);
    });

    const avg = (arr: number[]) => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : null;

    return Object.values(modelMetrics)
      .map(m => ({
        model_id: m.model_id,
        da: avg(m.da_values),
        sharpe: avg(m.sharpe_values),
        pf: avg(m.pf_values),
        mdd: avg(m.mdd_values),
        totalReturn: avg(m.return_values)
      }))
      .filter(m => m.da !== null)
      .sort((a, b) => (b.da || 0) - (a.da || 0));
  }, [data, selectedHorizon]);

  const formatMetric = (val: number | null, decimals = 2, suffix = '') => {
    if (val === null || val === undefined || isNaN(val)) return 'N/A';
    return val.toFixed(decimals) + suffix;
  };

  const formatDA = (da: number | null) => {
    if (da === null || da === undefined || isNaN(da)) return 'N/A';
    const pct = da < 1 ? da * 100 : da;
    return pct.toFixed(1) + '%';
  };

  const formatModelName = (modelId: string) => {
    return modelId
      .replace(/_/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase());
  };

  const getSharpeBadge = (sharpe: number | null) => {
    if (sharpe === null || isNaN(sharpe)) return { icon: null, color: 'text-gray-500' };
    if (sharpe >= 2) return { icon: <Star className="w-3 h-3" />, color: 'text-emerald-400' };
    if (sharpe >= 1) return { icon: <Check className="w-3 h-3" />, color: 'text-emerald-400' };
    if (sharpe >= 0) return { icon: null, color: 'text-amber-400' };
    return { icon: <X className="w-3 h-3" />, color: 'text-red-400' };
  };

  const getPFColor = (pf: number | null) => {
    if (pf === null || isNaN(pf)) return 'text-gray-500';
    if (pf >= 1.5) return 'text-emerald-400';
    if (pf >= 1.0) return 'text-emerald-400';
    return 'text-red-400';
  };

  const getMDDColor = (mdd: number | null) => {
    if (mdd === null || isNaN(mdd)) return 'text-gray-500';
    if (mdd < 0.1) return 'text-emerald-400';
    if (mdd < 0.2) return 'text-amber-400';
    return 'text-red-400';
  };

  const hasTradeMetrics = rankings.some(m => m.sharpe !== null || m.pf !== null);

  if (rankings.length === 0) {
    return (
      <div className="bg-gray-900/50 rounded-xl p-4 border border-gray-800">
        <div className="flex items-center gap-2 mb-3">
          <Trophy className="w-4 h-4 text-cyan-400" />
          <h3 className="text-sm font-semibold text-white">Model Rankings</h3>
        </div>
        <p className="text-xs text-gray-500">
          No hay datos de backtest disponibles para el horizonte seleccionado.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900/50 rounded-xl border border-gray-800 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gray-800/50 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <Trophy className="w-4 h-4 text-cyan-400" />
          <h3 className="text-sm font-semibold text-white">
            Model Rankings {selectedHorizon !== 'ALL' ? `(H=${selectedHorizon})` : ''}
          </h3>
        </div>
        {hasTradeMetrics && (
          <span className="text-xs text-gray-500 bg-gray-800 px-2 py-1 rounded">
            Walk-Forward Validated
          </span>
        )}
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="px-3 py-2 text-left text-gray-400 font-medium">#</th>
              <th className="px-3 py-2 text-left text-gray-400 font-medium">Model</th>
              <th className="px-3 py-2 text-right text-gray-400 font-medium">DA</th>
              {hasTradeMetrics && (
                <>
                  <th className="px-3 py-2 text-right text-gray-400 font-medium">Sharpe</th>
                  <th className="px-3 py-2 text-right text-gray-400 font-medium">PF</th>
                  <th className="px-3 py-2 text-right text-gray-400 font-medium">MaxDD</th>
                </>
              )}
            </tr>
          </thead>
          <tbody>
            {rankings.map((m, idx) => {
              const normalize = (s: string) => s.toLowerCase().replace(/[_ ]/g, '');
              const isSelected = selectedModel && normalize(m.model_id) === normalize(selectedModel);
              const isFirst = idx === 0;
              const sharpeBadge = getSharpeBadge(m.sharpe);

              return (
                <tr
                  key={m.model_id}
                  className={`
                    border-b border-gray-800/50 transition-colors
                    ${isSelected ? 'bg-cyan-500/10' : isFirst ? 'bg-emerald-500/5' : 'hover:bg-gray-800/30'}
                  `}
                >
                  <td className={`px-3 py-2 ${isFirst ? 'text-emerald-400 font-bold' : 'text-gray-500'}`}>
                    {idx + 1}
                  </td>
                  <td className={`px-3 py-2 ${isSelected ? 'text-cyan-400 font-semibold' : 'text-gray-300'}`}>
                    {formatModelName(m.model_id)}
                  </td>
                  <td className={`px-3 py-2 text-right font-semibold ${
                    m.da && (m.da > 0.55 || m.da > 55) ? 'text-emerald-400' : 'text-amber-400'
                  }`}>
                    {formatDA(m.da)}
                  </td>
                  {hasTradeMetrics && (
                    <>
                      <td className={`px-3 py-2 text-right ${sharpeBadge.color}`}>
                        <span className="flex items-center justify-end gap-1">
                          {sharpeBadge.icon}
                          {formatMetric(m.sharpe)}
                        </span>
                      </td>
                      <td className={`px-3 py-2 text-right ${getPFColor(m.pf)}`}>
                        {formatMetric(m.pf)}
                      </td>
                      <td className={`px-3 py-2 text-right ${getMDDColor(m.mdd)}`}>
                        {formatMetric(m.mdd ? m.mdd * 100 : null, 1, '%')}
                      </td>
                    </>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Legend */}
      {hasTradeMetrics && (
        <div className="px-4 py-2 bg-gray-800/30 border-t border-gray-700 text-xs text-gray-500">
          <div className="flex flex-wrap gap-4">
            <span><strong className="text-gray-400">Sharpe:</strong> {'>'}=2 Excelente | {'>'}=1 Bueno | {'<'}0 Malo</span>
            <span><strong className="text-gray-400">PF:</strong> {'>'}=1.5 Excelente | {'<'}1 Perdida</span>
            <span><strong className="text-gray-400">MaxDD:</strong> {'<'}10% OK | {'>'}20% Riesgo</span>
          </div>
        </div>
      )}
    </div>
  );
}

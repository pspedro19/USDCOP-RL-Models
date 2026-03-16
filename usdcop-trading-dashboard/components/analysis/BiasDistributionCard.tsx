'use client';

import { PoliticalBiasOutput } from '@/lib/contracts/weekly-analysis.contract';

interface BiasDistributionCardProps {
  biasData: PoliticalBiasOutput;
}

const BIAS_COLORS: Record<string, string> = {
  left: 'bg-blue-500',
  'center-left': 'bg-sky-400',
  center: 'bg-slate-400',
  'center-right': 'bg-orange-400',
  right: 'bg-red-500',
  unknown: 'bg-slate-600',
};

const BIAS_LABELS: Record<string, string> = {
  left: 'Izquierda',
  'center-left': 'Centro-Izq',
  center: 'Centro',
  'center-right': 'Centro-Der',
  right: 'Derecha',
  unknown: 'Sin clasificar',
};

const FACTUALITY_COLORS: Record<string, { bg: string; text: string }> = {
  high: { bg: 'bg-emerald-500/20', text: 'text-emerald-400' },
  mixed: { bg: 'bg-amber-500/20', text: 'text-amber-400' },
  low: { bg: 'bg-red-500/20', text: 'text-red-400' },
  unknown: { bg: 'bg-slate-500/20', text: 'text-slate-400' },
};

export function BiasDistributionCard({ biasData }: BiasDistributionCardProps) {
  const { source_bias_distribution, bias_diversity_score, factuality_distribution, flagged_articles, total_analyzed, bias_narrative, cluster_bias_assessments } = biasData;

  // Compute total for spectrum bar (exclude unknown)
  const spectrumOrder = ['left', 'center-left', 'center', 'center-right', 'right'] as const;
  const spectrumTotal = spectrumOrder.reduce((sum, key) => sum + (source_bias_distribution[key] || 0), 0);

  return (
    <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white">Sesgo Mediatico</h3>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">Diversidad:</span>
          <span className={`text-xs font-mono font-bold ${
            bias_diversity_score >= 0.7 ? 'text-emerald-400' :
            bias_diversity_score >= 0.4 ? 'text-amber-400' : 'text-red-400'
          }`}>
            {(bias_diversity_score * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Spectrum bar */}
      {spectrumTotal > 0 && (
        <div className="space-y-2">
          <div className="flex h-3 rounded-full overflow-hidden">
            {spectrumOrder.map((key) => {
              const count = source_bias_distribution[key] || 0;
              if (count === 0) return null;
              const pct = (count / spectrumTotal) * 100;
              return (
                <div
                  key={key}
                  className={`${BIAS_COLORS[key]} transition-all`}
                  style={{ width: `${pct}%` }}
                  title={`${BIAS_LABELS[key]}: ${count} (${pct.toFixed(0)}%)`}
                />
              );
            })}
          </div>
          <div className="flex justify-between text-[10px] text-slate-500">
            <span>Izquierda</span>
            <span>Centro</span>
            <span>Derecha</span>
          </div>
        </div>
      )}

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3">
        <div className="text-center">
          <div className="text-lg font-bold text-white">{total_analyzed}</div>
          <div className="text-[10px] text-slate-400 uppercase">Analizados</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-amber-400">{flagged_articles}</div>
          <div className="text-[10px] text-slate-400 uppercase">Con sesgo</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-slate-300">
            {source_bias_distribution['center'] || 0}
          </div>
          <div className="text-[10px] text-slate-400 uppercase">Centro</div>
        </div>
      </div>

      {/* Factuality badges */}
      <div className="flex gap-2 flex-wrap">
        {Object.entries(factuality_distribution)
          .filter(([k, v]) => k !== 'unknown' && v > 0)
          .map(([key, count]) => {
            const colors = FACTUALITY_COLORS[key] || FACTUALITY_COLORS.unknown;
            return (
              <span
                key={key}
                className={`px-2 py-0.5 rounded text-xs ${colors.bg} ${colors.text}`}
              >
                {key === 'high' ? 'Alta' : key === 'mixed' ? 'Media' : 'Baja'} fact.: {count}
              </span>
            );
          })}
      </div>

      {/* Cluster bias assessments */}
      {cluster_bias_assessments && cluster_bias_assessments.length > 0 && (
        <div className="space-y-1.5">
          <h4 className="text-xs font-medium text-slate-400 uppercase">Sesgo por Cluster</h4>
          {cluster_bias_assessments.map((assessment, i) => (
            <div key={i} className="flex items-center justify-between text-xs">
              <span className="text-slate-300 truncate max-w-[60%]">
                {assessment.cluster_label}
              </span>
              <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                assessment.bias_label === 'balanced' ? 'bg-slate-600 text-slate-300' :
                assessment.bias_label.includes('left') ? 'bg-blue-500/20 text-blue-400' :
                'bg-orange-500/20 text-orange-400'
              }`}>
                {assessment.bias_label.replace('_', ' ')}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Narrative */}
      {bias_narrative && (
        <p className="text-xs text-slate-400 leading-relaxed border-t border-slate-700/50 pt-3">
          {bias_narrative}
        </p>
      )}
    </div>
  );
}

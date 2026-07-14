'use client';

import { PoliticalBiasOutput } from '@/lib/contracts/weekly-analysis.contract';
import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT } from '@/lib/ui/gm-tokens';
import { GmBadge } from '@/components/gm';

import { ANALYSIS_DICT } from './gm-analysis';

interface BiasDistributionCardProps {
  biasData: PoliticalBiasOutput;
}

/** Political-spectrum segments — categorical mapping onto the GM semantic ramp. */
const BIAS_BAR_STYLE: Record<string, string> = {
  left: 'var(--gm-info)',
  'center-left': 'var(--gm-accent)',
  center: 'var(--gm-text-faint)',
  'center-right': 'var(--gm-warn)',
  right: 'var(--gm-neg)',
  unknown: 'var(--gm-border)',
};

export function BiasDistributionCard({ biasData }: BiasDistributionCardProps) {
  const t = useGmT(ANALYSIS_DICT);
  const { source_bias_distribution, bias_diversity_score, factuality_distribution, flagged_articles, total_analyzed, bias_narrative, cluster_bias_assessments } = biasData;

  const biasLabels: Record<string, string> = {
    left: t('left'),
    'center-left': t('centerLeft'),
    center: t('center'),
    'center-right': t('centerRight'),
    right: t('right'),
    unknown: t('unclassified'),
  };

  // Compute total for spectrum bar (exclude unknown)
  const spectrumOrder = ['left', 'center-left', 'center', 'center-right', 'right'] as const;
  const spectrumTotal = spectrumOrder.reduce((sum, key) => sum + (source_bias_distribution[key] || 0), 0);

  return (
    <div className={`${GM.panel} gm-contain p-5 space-y-4`}>
      <div className="flex items-center justify-between">
        <h3 className={`${GMT.panelTitle} ${GM.textStrong}`}>{t('biasTitle')}</h3>
        <div className="flex items-center gap-2">
          <span className={`${GMT.meta} ${GM.textSec}`}>{t('diversity')}:</span>
          <span className={`${GMT.meta} ${GMT.mono} font-bold ${
            bias_diversity_score >= 0.7 ? GM.pos :
            bias_diversity_score >= 0.4 ? GM.warn : GM.neg
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
                  className="transition-all duration-[var(--gm-dur-base)]"
                  style={{ width: `${pct}%`, background: BIAS_BAR_STYLE[key] }}
                  title={`${biasLabels[key]}: ${count} (${pct.toFixed(0)}%)`}
                />
              );
            })}
          </div>
          <div className={`flex justify-between ${GMT.micro} ${GM.textMuted}`}>
            <span>{t('left')}</span>
            <span>{t('center')}</span>
            <span>{t('right')}</span>
          </div>
        </div>
      )}

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3">
        <div className="text-center">
          <div className={`text-lg font-bold ${GMT.mono} ${GM.textStrong}`}>{total_analyzed}</div>
          <div className={`${GMT.label} ${GM.textMuted}`}>{t('analyzed')}</div>
        </div>
        <div className="text-center">
          <div className={`text-lg font-bold ${GMT.mono} ${GM.warn}`}>{flagged_articles}</div>
          <div className={`${GMT.label} ${GM.textMuted}`}>{t('flagged')}</div>
        </div>
        <div className="text-center">
          <div className={`text-lg font-bold ${GMT.mono} ${GM.text}`}>
            {source_bias_distribution['center'] || 0}
          </div>
          <div className={`${GMT.label} ${GM.textMuted}`}>{t('center')}</div>
        </div>
      </div>

      {/* Factuality badges */}
      <div className="flex gap-2 flex-wrap">
        {Object.entries(factuality_distribution)
          .filter(([k, v]) => k !== 'unknown' && v > 0)
          .map(([key, count]) => (
            <GmBadge
              key={key}
              tone={key === 'high' ? 'pos' : key === 'mixed' ? 'warn' : 'neg'}
              className="normal-case tracking-normal font-semibold"
            >
              {key === 'high' ? t('factHigh') : key === 'mixed' ? t('factMixed') : t('factLow')} {t('factSuffix')} {count}
            </GmBadge>
          ))}
      </div>

      {/* Cluster bias assessments */}
      {cluster_bias_assessments && cluster_bias_assessments.length > 0 && (
        <div className="space-y-1.5">
          <h4 className={`${GMT.label} ${GM.textMuted}`}>{t('biasByCluster')}</h4>
          {cluster_bias_assessments.map((assessment, i) => (
            <div key={i} className={`flex items-center justify-between ${GMT.meta}`}>
              <span className={`${GM.text} truncate max-w-[60%]`}>
                {assessment.cluster_label}
              </span>
              <GmBadge
                tone={
                  assessment.bias_label === 'balanced' ? 'neutral' :
                  assessment.bias_label.includes('left') ? 'info' : 'warn'
                }
                className="normal-case tracking-normal"
              >
                {assessment.bias_label.replace('_', ' ')}
              </GmBadge>
            </div>
          ))}
        </div>
      )}

      {/* Narrative */}
      {bias_narrative && (
        <p className={`${GMT.meta} ${GM.textSec} leading-relaxed border-t border-[var(--gm-border)] pt-3`}>
          {bias_narrative}
        </p>
      )}
    </div>
  );
}

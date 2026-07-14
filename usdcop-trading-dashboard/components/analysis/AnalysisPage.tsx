'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RefreshCw, AlertCircle } from 'lucide-react';
import { useAnalysisIndex, useWeeklyView, useUpcomingEvents, useAnalysisAssets, getCurrentISOWeek } from '@/hooks/useWeeklyAnalysis';
import { DEFAULT_ANALYSIS_ASSET } from '@/lib/contracts/analysis-assets';
import { normalizeNewsClusters } from '@/lib/analysis/normalize-news';
import { useAnalysisChatStore } from '@/stores/useAnalysisChatStore';
import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT } from '@/lib/ui/gm-tokens';

import { ANALYSIS_DICT } from './gm-analysis';

import { AssetSelector } from './AssetSelector';
import { WeekSelector } from './WeekSelector';
import { WeeklySummaryHeader } from './WeeklySummaryHeader';
import { SynthesisCard } from './SynthesisCard';
import { MtfAlignmentCard } from './MtfAlignmentCard';
import { FxContextCard } from './FxContextCard';
import { MacroSnapshotBar } from './MacroSnapshotBar';
import { MacroChartGrid } from './MacroChartGrid';
import { MacroDetailModal } from './MacroDetailModal';
import { SignalSummaryCards } from './SignalSummaryCards';
import { DailyTimeline } from './DailyTimeline';
import { UpcomingEventsPanel } from './UpcomingEventsPanel';
import { TechnicalAnalysisCard } from './TechnicalAnalysisCard';
import { TradingScenariosTable } from './TradingScenariosTable';
import { AssetTechnicalCard } from './AssetTechnicalCard';
import { isAssetTechnicalAnalysis, type TechnicalAnalysisOutput } from '@/lib/contracts/weekly-analysis.contract';
import { RegimeIndicator } from './RegimeIndicator';
import { NewsClusterCard } from './NewsClusterCard';
import { UnifiedMacroChart } from './UnifiedMacroChart';
import { BiasDistributionCard } from './BiasDistributionCard';
import { ReferencesSection } from './ReferencesSection';
import { MethodologySection } from './MethodologySection';

export function AnalysisPage() {
  const t = useGmT(ANALYSIS_DICT);
  const [selectedAsset, setSelectedAsset] = useState<string>(DEFAULT_ANALYSIS_ASSET);
  const [selectedYear, setSelectedYear] = useState<number | null>(null);
  const [selectedWeek, setSelectedWeek] = useState<number | null>(null);
  const [detailVariable, setDetailVariable] = useState<string | null>(null);

  // Data hooks (all asset-scoped)
  const { data: assetsData } = useAnalysisAssets();
  const { data: indexData, isLoading: indexLoading } = useAnalysisIndex(selectedAsset);
  const { data: weekData, isLoading: weekLoading, error: weekError } = useWeeklyView(selectedYear, selectedWeek, selectedAsset);
  const { data: eventsData } = useUpcomingEvents(selectedAsset);

  // Initialize (and re-initialize on asset switch) to the asset's latest week.
  // selectedYear is reset to null on asset change, which re-arms this effect.
  useEffect(() => {
    if (selectedYear !== null) return; // Already initialized for the current asset
    if (indexLoading) return; // Wait for index to finish loading

    if (indexData?.weeks?.length) {
      // weeks is sorted descending (W7, W6, ... W1), so [0] = most recent
      const latest = indexData.weeks[0];
      setSelectedYear(latest.year);
      setSelectedWeek(latest.week);
    } else {
      // No data for this asset — fallback to current week (renders the empty state)
      const current = getCurrentISOWeek();
      setSelectedYear(current.year);
      setSelectedWeek(current.week);
    }
  }, [indexData, indexLoading, selectedYear]);

  // Keep the floating chat's injected context in sync with what's on screen.
  const setChatContext = useAnalysisChatStore((s) => s.setContext);
  useEffect(() => {
    if (selectedYear !== null && selectedWeek !== null) {
      setChatContext(selectedAsset, selectedYear, selectedWeek);
    }
  }, [selectedAsset, selectedYear, selectedWeek, setChatContext]);

  const handleAssetSelect = useCallback((assetId: string) => {
    setSelectedAsset((prev) => {
      if (prev === assetId) return prev;
      // Reset week selection so the init effect picks the new asset's latest week.
      setSelectedYear(null);
      setSelectedWeek(null);
      return assetId;
    });
  }, []);

  const handleWeekSelect = useCallback((year: number, week: number) => {
    setSelectedYear(year);
    setSelectedWeek(week);
  }, []);

  const handleChartClick = useCallback((variableKey: string) => {
    setDetailVariable(variableKey);
  }, []);

  const isLoading = indexLoading || weekLoading;

  return (
    <div className="space-y-6">
      {/* Asset + Week Selectors */}
      <div className={`${GM.panel} gm-contain p-4 space-y-4`}>
        <div className="flex items-center justify-between gap-4 flex-wrap border-b border-[var(--gm-border)] pb-3">
          <AssetSelector
            assets={assetsData?.assets || []}
            selected={selectedAsset}
            onSelect={handleAssetSelect}
          />
        </div>
        <WeekSelector
          weeks={indexData?.weeks || []}
          currentYear={selectedYear || 2026}
          currentWeek={selectedWeek || 1}
          onSelect={handleWeekSelect}
        />
      </div>

      {/* Loading state */}
      {isLoading && (
        <div className="flex items-center justify-center py-12" aria-busy>
          <RefreshCw className={`w-6 h-6 ${GM.accent} motion-safe:animate-spin`} />
          <span className={`ml-3 ${GMT.body} ${GM.textSec}`}>{t('loading')}</span>
        </div>
      )}

      {/* Error state */}
      {weekError && !isLoading && (
        <div className={`${GM.panel} gm-contain border-[rgba(245,158,11,.25)] p-6 text-center`}>
          <AlertCircle className={`w-8 h-8 ${GM.warn} mx-auto mb-3`} />
          <p className={`${GMT.body} ${GM.textSec} mb-1`}>
            {t('noDataFor')} {assetsData?.assets?.find(a => a.asset_id === selectedAsset)?.display_name || selectedAsset} {t('inThisWeek')}
          </p>
          <p className={`${GMT.meta} ${GM.textMuted}`}>
            {t('runCmd')} <code className={`${GM.panelInner} ${GMT.mono} ${GM.accent} rounded px-1.5 py-0.5`}>
              python scripts/pipeline/generate_weekly_analysis.py --asset {selectedAsset} --week {selectedYear}-W{String(selectedWeek).padStart(2, '0')}
            </code>
          </p>
        </div>
      )}

      {/* Main content */}
      <AnimatePresence mode="wait">
        {weekData && !isLoading && (
          <motion.div
            key={`${selectedYear}-${selectedWeek}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="space-y-6"
          >
            {/* 1. Weekly Summary (with quality score). When a richer LangGraph
                 synthesis exists (USD/COP rich weeks), suppress the header's older
                 markdown body and render SynthesisCard below instead. */}
            {(() => {
              const hasSynthesis =
                typeof weekData.synthesis_markdown === 'string' && weekData.synthesis_markdown.trim().length > 0;
              return (
                <>
                  {weekData.weekly_summary && (
                    <WeeklySummaryHeader
                      summary={weekData.weekly_summary}
                      qualityScore={weekData.quality_score}
                      newsArticleCount={weekData.news_intelligence?.total_articles || weekData.news_context?.article_count}
                      clusterCount={weekData.news_intelligence?.clusters?.length}
                      sourceBreakdown={weekData.news_context?.source_breakdown}
                      hideMarkdown={hasSynthesis}
                    />
                  )}
                  {hasSynthesis && <SynthesisCard markdown={weekData.synthesis_markdown} />}
                </>
              );
            })()}

            {/* 2. Technical Analysis + Trading Scenarios.
                 USD/COP → rich LangGraph schema; Gold/BTC → lean asset schema. */}
            {weekData.technical_analysis && (
              isAssetTechnicalAnalysis(weekData.technical_analysis) ? (
                <AssetTechnicalCard
                  ta={weekData.technical_analysis}
                  symbol={weekData.chart_symbol ?? weekData.symbol}
                />
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <TechnicalAnalysisCard ta={weekData.technical_analysis as TechnicalAnalysisOutput} />
                  {(weekData.technical_analysis as TechnicalAnalysisOutput).scenarios?.length > 0 && (
                    <TradingScenariosTable
                      scenarios={(weekData.technical_analysis as TechnicalAnalysisOutput).scenarios}
                      noTradeZone={(weekData.technical_analysis as TechnicalAnalysisOutput).support_resistance?.no_trade_zone}
                    />
                  )}
                </div>
              )
            )}

            {/* 2b. Multi-timeframe alignment + FX context (rich USD/COP-week
                 LangGraph fields; Gold/BTC and stale weeks omit them → each card
                 self-guards and renders nothing). */}
            {weekData.mtf_analysis && <MtfAlignmentCard mtf={weekData.mtf_analysis} />}
            {weekData.fx_context && <FxContextCard fx={weekData.fx_context} />}

            {/* 3. Regime + Signal Cards */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {weekData.macro_regime && (
                <RegimeIndicator regime={weekData.macro_regime} />
              )}
              <div className={weekData.macro_regime ? 'lg:col-span-2' : 'lg:col-span-3'}>
                {weekData.signals && (
                  <SignalSummaryCards signals={weekData.signals} />
                )}
              </div>
            </div>

            {/* 4. Macro Snapshot Bar */}
            {weekData.macro_snapshots && Object.keys(weekData.macro_snapshots).length > 0 && (
              <div>
                <h2 className={`${GMT.h2} ${GM.textStrong} mb-3`}>{t('macroTitle')}</h2>
                <MacroSnapshotBar
                  snapshots={weekData.macro_snapshots}
                  charts={weekData.macro_charts}
                  onVariableClick={handleChartClick}
                />
              </div>
            )}

            {/* 5. Unified Macro Chart + per-variable grid */}
            {weekData.macro_charts && Object.keys(weekData.macro_charts).length > 0 && (
              <>
                <UnifiedMacroChart
                  charts={weekData.macro_charts}
                  onVariableClick={handleChartClick}
                />
                <MacroChartGrid
                  charts={weekData.macro_charts}
                  onChartClick={handleChartClick}
                />
              </>
            )}

            {/* 6a. Bias Distribution (Phase 3 — from LangGraph) */}
            {weekData.political_bias_analysis && weekData.political_bias_analysis.total_analyzed > 0 && (
              <BiasDistributionCard biasData={weekData.political_bias_analysis} />
            )}

            {/* 6b. News Intelligence Clusters (from LangGraph) */}
            {weekData.news_intelligence?.clusters && weekData.news_intelligence.clusters.length > 0 && (
              <NewsClusterCard clusters={normalizeNewsClusters(weekData.news_intelligence.clusters)} />
            )}

            {/* 7. Two-column layout: Timeline + Events */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Daily Timeline (2/3 width) */}
              <div className="lg:col-span-2">
                <DailyTimeline entries={weekData.daily_entries} />
              </div>

              {/* Upcoming Events (1/3 width) */}
              <div>
                <UpcomingEventsPanel events={eventsData?.events || weekData.upcoming_events || []} />
              </div>
            </div>

            {/* 8. Methodology & Explainability */}
            <MethodologySection />

            {/* 9. References & Data Sources */}
            <ReferencesSection weekData={weekData} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Detail Modal */}
      <MacroDetailModal
        isOpen={detailVariable !== null}
        onClose={() => setDetailVariable(null)}
        variableKey={detailVariable || ''}
        snapshot={detailVariable && weekData?.macro_snapshots ? weekData.macro_snapshots[detailVariable] || null : null}
        chartData={detailVariable && weekData?.macro_charts ? weekData.macro_charts[detailVariable] || null : null}
      />
    </div>
  );
}

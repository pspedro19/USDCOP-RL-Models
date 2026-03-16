'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RefreshCw, AlertCircle } from 'lucide-react';
import { useAnalysisIndex, useWeeklyView, useUpcomingEvents, getCurrentISOWeek } from '@/hooks/useWeeklyAnalysis';

import { WeekSelector } from './WeekSelector';
import { WeeklySummaryHeader } from './WeeklySummaryHeader';
import { MacroSnapshotBar } from './MacroSnapshotBar';
import { MacroChartGrid } from './MacroChartGrid';
import { MacroDetailModal } from './MacroDetailModal';
import { SignalSummaryCards } from './SignalSummaryCards';
import { DailyTimeline } from './DailyTimeline';
import { UpcomingEventsPanel } from './UpcomingEventsPanel';
import { TechnicalAnalysisCard } from './TechnicalAnalysisCard';
import { TradingScenariosTable } from './TradingScenariosTable';
import { RegimeIndicator } from './RegimeIndicator';
import { NewsClusterCard } from './NewsClusterCard';
import { UnifiedMacroChart } from './UnifiedMacroChart';
import { BiasDistributionCard } from './BiasDistributionCard';
import { ReferencesSection } from './ReferencesSection';
import { MethodologySection } from './MethodologySection';

export function AnalysisPage() {
  const [selectedYear, setSelectedYear] = useState<number | null>(null);
  const [selectedWeek, setSelectedWeek] = useState<number | null>(null);
  const [detailVariable, setDetailVariable] = useState<string | null>(null);

  // Data hooks
  const { data: indexData, isLoading: indexLoading } = useAnalysisIndex();
  const { data: weekData, isLoading: weekLoading, error: weekError } = useWeeklyView(selectedYear, selectedWeek);
  const { data: eventsData } = useUpcomingEvents();

  // Initialize to latest available week once index data loads
  useEffect(() => {
    if (selectedYear !== null) return; // Already initialized
    if (indexLoading) return; // Wait for index to finish loading

    if (indexData?.weeks?.length) {
      // weeks is sorted descending (W7, W6, ... W1), so [0] = most recent
      const latest = indexData.weeks[0];
      setSelectedYear(latest.year);
      setSelectedWeek(latest.week);
    } else {
      // No data available — fallback to current week
      const current = getCurrentISOWeek();
      setSelectedYear(current.year);
      setSelectedWeek(current.week);
    }
  }, [indexData, indexLoading, selectedYear]);

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
      {/* Week Selector */}
      <div className="bg-gray-900/60 backdrop-blur-sm rounded-xl border border-gray-800/50 p-4">
        <WeekSelector
          weeks={indexData?.weeks || []}
          currentYear={selectedYear || 2026}
          currentWeek={selectedWeek || 1}
          onSelect={handleWeekSelect}
        />
      </div>

      {/* Loading state */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <RefreshCw className="w-6 h-6 text-cyan-400 animate-spin" />
          <span className="ml-3 text-gray-400">Cargando analisis...</span>
        </div>
      )}

      {/* Error state */}
      {weekError && !isLoading && (
        <div className="bg-gray-900/40 rounded-xl border border-amber-500/20 p-6 text-center">
          <AlertCircle className="w-8 h-8 text-amber-400 mx-auto mb-3" />
          <p className="text-gray-400 text-sm mb-1">Sin datos de analisis para esta semana</p>
          <p className="text-gray-600 text-xs">
            Ejecuta: <code className="bg-gray-800 rounded px-1.5 py-0.5 text-cyan-400">
              python scripts/generate_weekly_analysis.py --week {selectedYear}-W{String(selectedWeek).padStart(2, '0')}
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
            {/* 1. Weekly Summary (with quality score) */}
            {weekData.weekly_summary && (
              <WeeklySummaryHeader
                summary={weekData.weekly_summary}
                qualityScore={weekData.quality_score}
                newsArticleCount={weekData.news_intelligence?.total_articles || weekData.news_context?.article_count}
                clusterCount={weekData.news_intelligence?.clusters?.length}
                sourceBreakdown={weekData.news_context?.source_breakdown}
              />
            )}

            {/* 2. Technical Analysis + Trading Scenarios (from LangGraph) */}
            {weekData.technical_analysis && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <TechnicalAnalysisCard ta={weekData.technical_analysis} />
                {weekData.technical_analysis.scenarios?.length > 0 && (
                  <TradingScenariosTable
                    scenarios={weekData.technical_analysis.scenarios}
                    noTradeZone={weekData.technical_analysis.support_resistance?.no_trade_zone}
                  />
                )}
              </div>
            )}

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
                <h2 className="text-base font-semibold text-white mb-3">Indicadores Macro</h2>
                <MacroSnapshotBar
                  snapshots={weekData.macro_snapshots}
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
              <NewsClusterCard clusters={weekData.news_intelligence.clusters} />
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

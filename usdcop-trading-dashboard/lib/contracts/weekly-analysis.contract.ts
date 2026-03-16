/**
 * Weekly Analysis Contract (SDD-08)
 * ====================================
 * TypeScript types for the /analysis page data model.
 *
 * Contract: CTR-ANALYSIS-FRONTEND-001
 */

// ---------------------------------------------------------------------------
// Macro snapshot (from MacroAnalyzer)
// ---------------------------------------------------------------------------

export interface MacroChartPoint {
  date: string;
  value: number | null;
  sma20: number | null;
  bb_upper: number | null;
  bb_lower: number | null;
  rsi: number | null;
}

export interface MacroVariableSnapshot {
  snapshot_date: string;
  variable_key: string;
  variable_name: string;
  value: number;
  sma_5: number | null;
  sma_10: number | null;
  sma_20: number | null;
  sma_50: number | null;
  bollinger_upper_20: number | null;
  bollinger_lower_20: number | null;
  bollinger_width_20: number | null;
  rsi_14: number | null;
  macd_line: number | null;
  macd_signal: number | null;
  macd_histogram: number | null;
  roc_5: number | null;
  roc_20: number | null;
  z_score_20: number | null;
  trend: string | null;
  signal: string | null;
  chart_data?: MacroChartPoint[];
  png_url?: string;
}

// ---------------------------------------------------------------------------
// Daily analysis entry
// ---------------------------------------------------------------------------

export interface DailyAnalysisEntry {
  analysis_date: string;
  iso_year: number;
  iso_week: number;
  day_of_week: number;  // 0=Mon, 4=Fri
  headline: string | null;
  summary_markdown: string | null;
  sentiment: 'bullish' | 'bearish' | 'neutral' | 'mixed' | null;
  usdcop_close: number | null;
  usdcop_change_pct: number | null;
  usdcop_high: number | null;
  usdcop_low: number | null;
  h1_signal: Record<string, unknown>;
  h5_status: Record<string, unknown>;
  macro_publications: MacroPublication[];
  economic_events: EconomicEvent[];
  news_highlights: NewsHighlight[];
}

export interface MacroPublication {
  variable: string;
  value: number;
  previous?: number;
  change_pct?: number;
}

export interface EconomicEvent {
  event: string;
  date: string;
  impact_level: 'high' | 'medium' | 'low';
  actual?: string;
  forecast?: string;
}

export interface NewsHighlight {
  title: string;
  source: string;
  sentiment: string | null;
  url?: string;
}

// ---------------------------------------------------------------------------
// Multi-Agent Analysis outputs (Phase 4)
// ---------------------------------------------------------------------------

export interface TradingScenario {
  direction: 'long' | 'short';
  entry_condition: string;
  entry_price: number | null;
  stop_loss: number | null;
  targets: number[];
  risk_reward: number | null;
  confidence: 'high' | 'medium' | 'low';
  profile: 'scalp' | 'intraday' | 'swing';
}

export interface TechnicalAnalysisOutput {
  current_price: number;
  atr: number | null;
  atr_pct: number | null;
  volatility_regime: 'low' | 'normal' | 'high';
  dominant_bias: 'bullish' | 'bearish' | 'neutral';
  bias_confidence: number;
  bullish_signals: string[];
  bearish_signals: string[];
  scenarios: TradingScenario[];
  watch_list: string[];
  ichimoku?: Record<string, unknown>;
  supertrend?: Record<string, unknown>;
  macd?: Record<string, unknown>;
  rsi?: number | null;
  bollinger_width?: number | null;
  bollinger_position?: string;
  fibonacci?: Record<string, unknown>;
  support_resistance?: {
    key_supports: number[];
    key_resistances: number[];
    no_trade_zone: [number, number];
  };
}

export interface MultiTimeframeAnalysis {
  reports: Record<string, TechnicalAnalysisOutput>;
  alignment_score: number;
  alignment_label: string;
  confluent_supports: number[];
  confluent_resistances: number[];
  dominant_timeframe: string;
}

export interface NewsArticleRef {
  title: string;
  url?: string;
  source: string;
  date: string;
  tone: number;
  category?: string;
  bias_label?: string;
  factuality?: string;
}

export interface NewsClusterOutput {
  cluster_id: number;
  label: string;
  article_count: number;
  avg_sentiment: number;
  dominant_category: string;
  bias_distribution: Record<string, number>;
  representative_titles: string[];
  narrative_summary: string;
  articles: NewsArticleRef[];
}

export interface NewsIntelligenceOutput {
  total_articles: number;
  relevant_articles: number;
  avg_sentiment: number;
  sentiment_distribution: { positive: number; negative: number; neutral: number };
  clusters: NewsClusterOutput[];
  top_stories: NewsArticleRef[];
  source_diversity: Record<string, number>;
}

export interface RegimeState {
  label: 'risk_on' | 'transition' | 'risk_off';
  since: string | null;
  confidence: number;
  transition_probabilities: Record<string, number>;
}

export interface GrangerLeader {
  variable: string;
  optimal_lag: number;
  f_statistic: number;
  p_value: number;
  direction: string;
}

export interface ZScoreAlert {
  variable: string;
  variable_name: string;
  z_score: number;
  direction: 'extreme_high' | 'extreme_low';
  interpretation: string;
}

export interface MacroRegimeOutput {
  regime: RegimeState;
  correlations: Record<string, { current: number | null; avg_60d: number | null; expected_direction: string }>;
  granger_leaders: GrangerLeader[];
  zscore_alerts: ZScoreAlert[];
  changepoints: Array<{ date: string; variable: string; direction: string; magnitude: number }>;
  insights: string[];
}

export interface RiskFactor {
  factor: string;
  severity: 'low' | 'medium' | 'high';
  direction: 'cop_weakening' | 'cop_strengthening';
  description: string;
}

export interface FXContextOutput {
  carry_trade: {
    ibr_rate: number | null;
    fed_funds_rate: number | null;
    differential_pct: number | null;
    differential_trend: string;
    carry_attractiveness: string;
    interpretation: string;
  };
  oil_impact: {
    wti_current: number | null;
    wti_weekly_change_pct: number | null;
    brent_current: number | null;
    estimated_cop_impact_pct: number | null;
    interpretation: string;
  };
  banrep: {
    tpm_current: number | null;
    ibr_current: number | null;
    next_meeting: string | null;
    rate_expectation: 'cut' | 'hold' | 'hike';
    interpretation: string;
  };
  risk_factors: RiskFactor[];
  fx_narrative: string;
  cop_weekly_change_pct: number | null;
  cop_level: number | null;
  sensitivity_impacts: Record<string, {
    current: number;
    weekly_change_pct: number;
    sensitivity: number;
    estimated_cop_impact_pct: number;
  }>;
}

// ---------------------------------------------------------------------------
// Political Bias Detection (Phase 3)
// ---------------------------------------------------------------------------

export interface ClusterBiasAssessment {
  cluster_label: string;
  bias_label: 'balanced' | 'slightly_left' | 'slightly_right' | 'left_leaning' | 'right_leaning';
  confidence: number;
  article_count: number;
}

export interface PoliticalBiasOutput {
  source_bias_distribution: Record<string, number>;   // {left: N, center-left: N, center: N, ...}
  bias_diversity_score: number;                        // 0-1 (1 = balanced across spectrum)
  factuality_distribution: Record<string, number>;     // {high: N, mixed: N, low: N}
  cluster_bias_assessments: ClusterBiasAssessment[];
  flagged_articles: number;
  bias_narrative: string;
  total_analyzed: number;
}

// ---------------------------------------------------------------------------
// Weekly summary
// ---------------------------------------------------------------------------

export interface WeeklySummary {
  headline: string;
  markdown: string;
  sentiment: 'bullish' | 'bearish' | 'neutral' | 'mixed';
  themes: (WeeklyTheme | string)[];
  ohlcv: OHLCVSummary;
}

export interface WeeklyTheme {
  theme: string;
  description: string;
  impact: 'positive' | 'negative' | 'neutral';
}

export interface OHLCVSummary {
  open: number;
  high: number;
  low: number;
  close: number;
  change_pct: number;
  range_pct?: number;
}

// ---------------------------------------------------------------------------
// Signal summaries
// ---------------------------------------------------------------------------

export interface SignalSummaries {
  h5: H5SignalSummary;
  h1: H1SignalSummary;
}

export interface H5SignalSummary {
  direction?: string;
  confidence?: string;
  predicted_return?: number;
  leverage?: number;
}

export interface H1SignalSummary {
  direction?: string;
  magnitude?: number;
  signals?: Array<{ date: string; direction: string; signal_strength: number }>;
}

// ---------------------------------------------------------------------------
// Macro chart data (for Recharts)
// ---------------------------------------------------------------------------

export interface MacroChartData {
  png_url: string | null;
  data: MacroChartPoint[];
}

// ---------------------------------------------------------------------------
// News context
// ---------------------------------------------------------------------------

export interface NewsContext {
  article_count: number;
  top_categories: Record<string, number>;
  avg_sentiment: number;
  cross_refs?: number;
  source_breakdown?: Record<string, number>;
  highlights?: Array<{
    title: string;
    source: string;
    date?: string;
    url?: string;
    news_source?: string;
  }>;
}

// ---------------------------------------------------------------------------
// Full weekly view export (matches Python WeeklyViewExport)
// ---------------------------------------------------------------------------

export interface WeeklyViewData {
  weekly_summary: WeeklySummary;
  daily_entries: DailyAnalysisEntry[];
  macro_snapshots: Record<string, MacroVariableSnapshot>;
  signals: SignalSummaries;
  upcoming_events: EconomicEvent[];
  macro_charts: Record<string, MacroChartData>;
  news_context: NewsContext;

  // Multi-Agent Analysis outputs (Phase 4 — optional, added by LangGraph pipeline)
  technical_analysis?: TechnicalAnalysisOutput;
  mtf_analysis?: MultiTimeframeAnalysis;
  news_intelligence?: NewsIntelligenceOutput;
  macro_regime?: MacroRegimeOutput;
  fx_context?: FXContextOutput;
  political_bias_analysis?: PoliticalBiasOutput;  // Phase 3: Bias detection
  quality_score?: number;
  synthesis_markdown?: string;
}

// ---------------------------------------------------------------------------
// Analysis index
// ---------------------------------------------------------------------------

export interface AnalysisWeekEntry {
  year: number;
  week: number;
  start: string;
  end: string;
  sentiment: string | null;
  headline: string | null;
  has_weekly: boolean;
  daily_count: number;
}

export interface AnalysisIndex {
  weeks: AnalysisWeekEntry[];
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  tokens_used?: number;
}

export interface ChatSession {
  session_id: string;
  messages: ChatMessage[];
  context_year: number;
  context_week: number;
}

// ---------------------------------------------------------------------------
// Sentiment colors (reusable)
// ---------------------------------------------------------------------------

export const SENTIMENT_COLORS = {
  bullish: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30' },
  bearish: { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/30' },
  neutral: { bg: 'bg-slate-500/20', text: 'text-slate-400', border: 'border-slate-500/30' },
  mixed:   { bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/30' },
} as const;

export function getSentimentColor(sentiment: string | null) {
  return SENTIMENT_COLORS[sentiment as keyof typeof SENTIMENT_COLORS] || SENTIMENT_COLORS.neutral;
}

// ---------------------------------------------------------------------------
// Day names (Spanish)
// ---------------------------------------------------------------------------

export const DAY_NAMES_ES = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes'] as const;

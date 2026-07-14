'use client';

/**
 * GM re-skin shared bits for the /analysis area (CTR-GM-UI-001 · prototype Var B
 * "ANALYSIS" view, layout 720-855 + view-model 2557-2634).
 *
 * - ANALYSIS_DICT: local ES/EN dictionary (UI chrome strings; ES copied from the
 *   as-built page, EN from the prototype view-model). Long-form editorial prose
 *   (methodology bodies, source descriptions, disclaimer) is CONTENT — like the
 *   AI markdown from the API it stays ES-only, outside the dict.
 * - sentimentTone: maps API sentiment/bias labels → GmTone (replaces the legacy
 *   contract getSentimentColor() tailwind classes for presentation).
 * - CHART: literal colors for chart-library props (recharts / lightweight-charts
 *   can't consume Tailwind classes). Same precedent + values as
 *   components/gm/views/{ProductionView,ForecastingView}.tsx.
 * - GmSpark: local sparkline SVG (prototype 80×30) until components/gm/Spark exists.
 */

import { defineGmDict } from '@/lib/i18n/gm-core';
import { GM_CHART, type GmTone } from '@/lib/ui/gm-tokens';

// ─────────────────────────────────────────────────────────────── tones

/** bullish/positive → pos · bearish/negative → neg · mixed → warn · else neutral. */
export function sentimentTone(sentiment: string | null | undefined): GmTone {
  switch ((sentiment || '').toLowerCase()) {
    case 'bullish':
    case 'positive':
      return 'pos';
    case 'bearish':
    case 'negative':
      return 'neg';
    case 'mixed':
      return 'warn';
    default:
      return 'neutral';
  }
}

/** Economic-event impact level → GmTone (high neg · medium warn · low neutral). */
export function impactTone(level: string | null | undefined): GmTone {
  return level === 'high' ? 'neg' : level === 'medium' ? 'warn' : 'neutral';
}

// ─────────────────────────────────────────────────────────────── chart-lib colors

/**
 * Chart-library prop palette. Hex values come from `GM_CHART` (single hex home,
 * CTR-GM-UI-001); only rgba grid/border strokes are local. Do NOT use these in
 * Tailwind classes; there, use GM/GMT tokens.
 */
export const CHART = {
  ...GM_CHART,
  grid: 'rgba(148,163,184,.10)',
  border: 'rgba(148,163,184,.16)',
} as const;

// ─────────────────────────────────────────────────────────────── sparkline

/** Local sparkline (prototype `spark()` 80×30) — SVG polyline, tone via var(--gm-*). */
export function GmSpark({ values, tone = 'accent', width = 80, height = 30 }: {
  values: Array<number | null | undefined>;
  tone?: GmTone;
  width?: number;
  height?: number;
}) {
  const clean = values.filter((v): v is number => v != null && !Number.isNaN(v));
  if (clean.length < 2) return null;

  const min = Math.min(...clean);
  const max = Math.max(...clean);
  const span = max - min || 1;
  const pad = 2;
  const points = clean
    .map((v, i) => {
      const x = pad + (i / (clean.length - 1)) * (width - pad * 2);
      const y = height - pad - ((v - min) / span) * (height - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');

  const stroke = tone === 'neutral' ? 'var(--gm-text-faint)' : `var(--gm-${tone})`;

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden className="shrink-0">
      <polyline
        points={points}
        fill="none"
        style={{ stroke }}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────── i18n dictionary

export const ANALYSIS_DICT = defineGmDict({
  es: {
    // Selectors / page shell
    pair: 'Par',
    week: 'Semana',
    weekWord: 'SEMANA',
    days: 'días',
    loading: 'Cargando análisis…',
    noDataFor: 'Sin datos de análisis para',
    inThisWeek: 'en esta semana',
    runCmd: 'Ejecuta:',
    prevWeek: 'Semana anterior',
    nextWeek: 'Semana siguiente',
    jumpToWeek: 'Ir a la semana',

    // Weekly summary / recap OHLC
    open: 'Apertura',
    high: 'Máximo',
    low: 'Mínimo',
    close: 'Cierre',
    change: 'Cambio',
    articles: 'artículos',
    clustersWord: 'clusters',
    quality: 'Calidad',
    showLess: 'Ver menos',
    showFullReport: 'Ver informe completo',

    // Sentiment / bias labels
    bullish: 'Alcista',
    bearish: 'Bajista',
    neutral: 'Neutral',
    mixed: 'Mixto',

    // Technical analysis
    techTitle: 'Análisis técnico',
    confidence: 'Confianza',
    volatility: 'Volatilidad',
    volLow: 'Baja',
    volNormal: 'Normal',
    volHigh: 'Alta',
    bullishGroup: 'Alcistas',
    bearishGroup: 'Bajistas',
    watch: 'Vigilar',

    // Trading scenarios
    scenTitle: 'Escenarios de trading',
    noTradeZone: 'Zona de no-operar',
    thDir: 'Dir',
    thEntry: 'Entrada',
    thStop: 'Stop',
    thTargets: 'Objetivo(s)',
    thRR: 'R:R',
    thConf: 'Confianza',
    thProfile: 'Perfil',
    profileScalp: 'Scalp',
    profileIntraday: 'Intradía',
    profileSwing: 'Swing',

    // Asset technical (lean Gold/BTC schema)
    signalWord: 'Señal',
    atrLabel: 'ATR (14)',
    supportLabel: 'Soporte',
    resistanceLabel: 'Resistencia',
    thScenario: 'Escenario',
    thTrigger: 'Disparo',
    thTarget: 'Objetivo',
    thInvalid: 'Invalidación',
    sigOverbought: 'Sobrecompra',
    sigOversold: 'Sobreventa',
    sigNeutral: 'Neutral',

    // Macro regime
    regimeTitle: 'Régimen macro',
    riskOn: 'Risk-On',
    riskOff: 'Risk-Off',
    transition: 'Transición',
    activeSince: 'Activo desde',
    zAlerts: 'Alertas Z-Score',
    leaders: 'Variables líderes',

    // Signals
    h5Title: 'H5 Semanal',
    h1Title: 'H1 Diario',
    predictedReturn: 'Retorno predicho',
    leverage: 'Apalancamiento',
    magnitude: 'Magnitud',
    lastSignals: 'Últimas señales',
    noTradeWeek: 'Sin operación esta semana',
    noH5: 'Sin señal H5 activa',
    noH1: 'Sin señal H1 activa',
    dataUnavailable: 'Datos no disponibles',

    // Macro indicators
    macroTitle: 'Indicadores macro',
    source: 'Fuente',
    normalized: 'Normalizado',
    absolute: 'Absoluto',
    noData: 'Sin datos',
    noChartData: 'Sin datos de gráfico',

    // Detail modal
    currentValue: 'Valor Actual',
    trend: 'Tendencia',
    signalWord: 'Señal',
    closeModal: 'Cerrar',
    trendGoldenCross: 'Golden Cross',
    trendDeathCross: 'Death Cross',
    trendAbove: 'Por encima SMA20',
    trendBelow: 'Por debajo SMA20',
    trendNeutral: 'Neutral',

    // Timeline
    timelineTitle: 'Timeline diario',
    noDaily: 'Sin análisis diarios para esta semana',
    range: 'Rango',
    showFullAnalysis: 'Ver análisis completo',

    // Upcoming events
    eventsTitle: 'Próximos eventos',
    noEvents: 'Sin eventos próximos',
    forecastAbbr: 'Est:',
    impact: 'Impacto',

    // News clusters
    newsClusters: 'Clusters de noticias',
    articlesIn: 'artículos en',
    showMorePrefix: 'Ver',
    morePlural: 'más',
    hide: 'Ocultar',
    view: 'Ver',
    sourcesWord: 'fuentes',
    positive: 'Positivo',
    negative: 'Negativo',

    // Media bias
    biasTitle: 'Sesgo mediático',
    diversity: 'Diversidad',
    left: 'Izquierda',
    centerLeft: 'Centro-Izq',
    center: 'Centro',
    centerRight: 'Centro-Der',
    right: 'Derecha',
    unclassified: 'Sin clasificar',
    analyzed: 'Analizados',
    flagged: 'Con sesgo',
    biasByCluster: 'Sesgo por cluster',
    factHigh: 'Alta',
    factMixed: 'Media',
    factLow: 'Baja',
    factSuffix: 'fact.:',

    // Synthesis / MTF alignment / FX context (rich USD/COP-week LangGraph fields)
    synthTitle: 'Síntesis ejecutiva',
    mtfTitle: 'Alineación multi-timeframe',
    mtfAlignment: 'Alineación',
    mtfDominantTf: 'TF dominante',
    mtfConfSupports: 'Soportes confluentes',
    mtfConfResistances: 'Resistencias confluentes',
    thTimeframe: 'Timeframe',
    thBias: 'Sesgo',
    thRsi: 'RSI',
    fxTitle: 'Contexto FX',
    fxCarry: 'Carry trade',
    fxOil: 'Impacto del petróleo',
    fxBanrep: 'BanRep',
    fxNarrative: 'Narrativa FX',
    fxSensitivity: 'Sensibilidad del COP',
    fxDifferential: 'Diferencial',
    fxCopImpact: 'Impacto COP est.',
    fxWeeklyChange: 'Cambio sem.',
    fxSensitivityCoef: 'Sensibilidad',
    fxRateExpectation: 'Expectativa de tasa',
    fxNextMeeting: 'Próxima reunión',
    fxCopLevel: 'Nivel COP',
    fxIbr: 'IBR',
    fxFedFunds: 'Fed Funds',
    fxWti: 'WTI',
    fxBrent: 'Brent',
    fxVariable: 'Variable',
    carryAttractive: 'Atractivo',
    carryNeutral: 'Neutral',
    carryUnattractive: 'Poco atractivo',
    rateHold: 'Mantener',
    rateCut: 'Recortar',
    rateHike: 'Subir',

    // Chat
    chatTitle: 'Asistente de análisis',
    chatEmpty: 'Pregunta sobre el análisis semanal',
    chatPlaceholder: 'Pregunta sobre el análisis…',
    newSession: 'Nueva sesión',
    thinking: 'Pensando…',
    connError: 'Error de conexión. Intenta de nuevo.',
    tokensWord: 'tokens',
    openChat: 'Abrir asistente de análisis',
    qaSummary: 'Resumen',
    qaMacro: 'Macro',
    qaSignals: 'Señales',
    qaOutlook: 'Perspectiva',
    qaSummaryPrompt: 'Dame un resumen rápido de la semana',
    qaMacroPrompt: '¿Cómo están los indicadores macro clave?',
    qaSignalsPrompt: '¿Qué dicen las señales de los modelos H1 y H5?',
    qaOutlookPrompt: '¿Cuál es la perspectiva para la próxima semana?',

    // References / methodology (section chrome)
    refsTitle: 'Referencias y fuentes de datos',
    refsMacro: 'Fuentes de datos macroeconómicos',
    refsNews: 'Fuentes de noticias',
    refsCited: 'Artículos citados esta semana',
    refsInfra: 'Infraestructura de datos',
    methodTitle: 'Metodología e interpretabilidad del análisis',
    methodSub: 'Cómo funciona este reporte, qué impulsa el USD/COP, y cómo interpretar cada sección.',
  },
  en: {
    pair: 'Pair',
    week: 'Week',
    weekWord: 'WEEK',
    days: 'days',
    loading: 'Loading analysis…',
    noDataFor: 'No analysis data for',
    inThisWeek: 'for this week',
    runCmd: 'Run:',
    prevWeek: 'Previous week',
    nextWeek: 'Next week',
    jumpToWeek: 'Jump to week',

    open: 'Open',
    high: 'High',
    low: 'Low',
    close: 'Close',
    change: 'Change',
    articles: 'articles',
    clustersWord: 'clusters',
    quality: 'Quality',
    showLess: 'Show less',
    showFullReport: 'View full report',

    bullish: 'Bullish',
    bearish: 'Bearish',
    neutral: 'Neutral',
    mixed: 'Mixed',

    techTitle: 'Technical analysis',
    confidence: 'Confidence',
    volatility: 'Volatility',
    volLow: 'Low',
    volNormal: 'Normal',
    volHigh: 'High',
    bullishGroup: 'Bullish',
    bearishGroup: 'Bearish',
    watch: 'Watch',

    scenTitle: 'Trading scenarios',
    noTradeZone: 'No-trade zone',
    thDir: 'Dir',
    thEntry: 'Entry',
    thStop: 'Stop',
    thTargets: 'Target(s)',
    thRR: 'R:R',
    thConf: 'Conf.',
    thProfile: 'Profile',
    profileScalp: 'Scalp',
    profileIntraday: 'Intraday',
    profileSwing: 'Swing',

    signalWord: 'Signal',
    atrLabel: 'ATR (14)',
    supportLabel: 'Support',
    resistanceLabel: 'Resistance',
    thScenario: 'Scenario',
    thTrigger: 'Trigger',
    thTarget: 'Target',
    thInvalid: 'Invalidation',
    sigOverbought: 'Overbought',
    sigOversold: 'Oversold',
    sigNeutral: 'Neutral',

    regimeTitle: 'Macro regime',
    riskOn: 'Risk-On',
    riskOff: 'Risk-Off',
    transition: 'Transition',
    activeSince: 'Active since',
    zAlerts: 'Z-Score alerts',
    leaders: 'Leading variables',

    h5Title: 'H5 Weekly',
    h1Title: 'H1 Daily',
    predictedReturn: 'Predicted return',
    leverage: 'Leverage',
    magnitude: 'Magnitude',
    lastSignals: 'Latest signals',
    noTradeWeek: 'No trade this week',
    noH5: 'No active H5 signal',
    noH1: 'No active H1 signal',
    dataUnavailable: 'Data unavailable',

    macroTitle: 'Macro indicators',
    source: 'Source',
    normalized: 'Normalized',
    absolute: 'Absolute',
    noData: 'No data',
    noChartData: 'No chart data',

    currentValue: 'Current Value',
    trend: 'Trend',
    signalWord: 'Signal',
    closeModal: 'Close',
    trendGoldenCross: 'Golden Cross',
    trendDeathCross: 'Death Cross',
    trendAbove: 'Above SMA20',
    trendBelow: 'Below SMA20',
    trendNeutral: 'Neutral',

    timelineTitle: 'Daily timeline',
    noDaily: 'No daily analyses for this week',
    range: 'Range',
    showFullAnalysis: 'View full analysis',

    eventsTitle: 'Upcoming events',
    noEvents: 'No upcoming events',
    forecastAbbr: 'Est:',
    impact: 'Impact',

    newsClusters: 'News clusters',
    articlesIn: 'articles in',
    showMorePrefix: 'View',
    morePlural: 'more',
    hide: 'Hide',
    view: 'View',
    sourcesWord: 'sources',
    positive: 'Positive',
    negative: 'Negative',

    biasTitle: 'Media bias',
    diversity: 'Diversity',
    left: 'Left',
    centerLeft: 'Center-Left',
    center: 'Center',
    centerRight: 'Center-Right',
    right: 'Right',
    unclassified: 'Unclassified',
    analyzed: 'Analyzed',
    flagged: 'Flagged',
    biasByCluster: 'Bias by cluster',
    factHigh: 'High',
    factMixed: 'Mixed',
    factLow: 'Low',
    factSuffix: 'fact.:',

    synthTitle: 'Executive synthesis',
    mtfTitle: 'Multi-timeframe alignment',
    mtfAlignment: 'Alignment',
    mtfDominantTf: 'Dominant TF',
    mtfConfSupports: 'Confluent supports',
    mtfConfResistances: 'Confluent resistances',
    thTimeframe: 'Timeframe',
    thBias: 'Bias',
    thRsi: 'RSI',
    fxTitle: 'FX context',
    fxCarry: 'Carry trade',
    fxOil: 'Oil impact',
    fxBanrep: 'BanRep',
    fxNarrative: 'FX narrative',
    fxSensitivity: 'COP sensitivity',
    fxDifferential: 'Differential',
    fxCopImpact: 'Est. COP impact',
    fxWeeklyChange: 'Weekly chg.',
    fxSensitivityCoef: 'Sensitivity',
    fxRateExpectation: 'Rate expectation',
    fxNextMeeting: 'Next meeting',
    fxCopLevel: 'COP level',
    fxIbr: 'IBR',
    fxFedFunds: 'Fed Funds',
    fxWti: 'WTI',
    fxBrent: 'Brent',
    fxVariable: 'Variable',
    carryAttractive: 'Attractive',
    carryNeutral: 'Neutral',
    carryUnattractive: 'Unattractive',
    rateHold: 'Hold',
    rateCut: 'Cut',
    rateHike: 'Hike',

    chatTitle: 'Analysis assistant',
    chatEmpty: 'Ask about the weekly analysis',
    chatPlaceholder: 'Ask about the analysis…',
    newSession: 'New session',
    thinking: 'Thinking…',
    connError: 'Connection error. Try again.',
    tokensWord: 'tokens',
    openChat: 'Open analysis assistant',
    qaSummary: 'Summary',
    qaMacro: 'Macro',
    qaSignals: 'Signals',
    qaOutlook: 'Outlook',
    qaSummaryPrompt: 'Give me a quick summary of the week',
    qaMacroPrompt: 'How are the key macro indicators?',
    qaSignalsPrompt: 'What do the H1 and H5 model signals say?',
    qaOutlookPrompt: 'What is the outlook for next week?',

    refsTitle: 'References & data sources',
    refsMacro: 'Macroeconomic data sources',
    refsNews: 'News sources',
    refsCited: 'Articles cited this week',
    refsInfra: 'Data infrastructure',
    methodTitle: 'Methodology & interpretability of the analysis',
    methodSub: 'How this report works, what drives USD/COP, and how to read each section.',
  },
});

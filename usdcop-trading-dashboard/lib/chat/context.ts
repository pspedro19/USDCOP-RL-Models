/**
 * Chat context builder (CTR-CHAT-001) — SRP.
 *
 * Turns a weekly-analysis JSON into a compact, grounded prompt context: OHLC,
 * model signals, technical, macro regime, news-cluster themes, source-bias
 * landscape and upcoming events. Pure + defensive (tolerates missing fields).
 */

import type { AnalysisAsset } from '@/lib/contracts/analysis-assets';

export function buildWeekContext(weekData: any, asset: AnalysisAsset): string {
  if (!weekData) return 'No hay contexto semanal disponible.';
  const parts: string[] = [];

  if (weekData.weekly_summary?.headline) parts.push(`Titular: ${weekData.weekly_summary.headline}`);
  if (weekData.weekly_summary?.sentiment) parts.push(`Sentimiento: ${weekData.weekly_summary.sentiment}`);
  const o = weekData.weekly_summary?.ohlcv;
  if (o) {
    parts.push(`${asset.chart_symbol}: Apertura ${o.open}, Máx ${o.high}, Mín ${o.low}, Cierre ${o.close}, Cambio ${o.change_pct}%`);
  }
  if (weekData.signals?.h5?.direction) {
    parts.push(`Señal H5: ${weekData.signals.h5.direction} (conf: ${weekData.signals.h5.confidence})`);
  }
  if (weekData.signals?.h1?.direction) parts.push(`Señal H1: ${weekData.signals.h1.direction}`);

  const ta = weekData.technical_analysis;
  if (ta) {
    const rsi = ta.indicators?.rsi_14;
    const sup = ta.support_resistance?.support;
    const res = ta.support_resistance?.resistance;
    parts.push(
      `Técnico: tendencia ${ta.trend ?? 'n/d'}${rsi != null ? `, RSI(14) ${rsi}` : ''}` +
      `${sup != null && res != null ? `, soporte ${sup} / resistencia ${res}` : ''}`,
    );
  }
  if (weekData.macro_regime?.regime?.label) {
    const r = weekData.macro_regime.regime;
    parts.push(`Régimen macro: ${r.label}${r.confidence != null ? ` (conf ${r.confidence})` : ''}`);
  }
  if (weekData.news_context) {
    parts.push(`Noticias: ${weekData.news_context.article_count} artículos, sentimiento medio ${weekData.news_context.avg_sentiment}`);
  }
  const clusters = weekData.news_intelligence?.clusters;
  if (Array.isArray(clusters) && clusters.length > 0) {
    const themes = clusters
      .slice(0, 5)
      .map((c: any) => `${c.label ?? c.theme ?? c.dominant_category ?? 'tema'} (${c.article_count ?? c.count ?? 0})`)
      .join(', ');
    parts.push(`Temas de noticias: ${themes}`);
  }
  const bias = weekData.political_bias_analysis;
  if (bias?.total_analyzed) {
    parts.push(`Sesgo de fuentes: ${bias.total_analyzed} analizadas, diversidad ${bias.bias_diversity_score ?? 'n/d'}, ${bias.flagged_articles ?? 0} marcadas`);
  }
  if (Array.isArray(weekData.upcoming_events) && weekData.upcoming_events.length > 0) {
    const ev = weekData.upcoming_events
      .slice(0, 4)
      .map((e: any) => e.title ?? e.event ?? e.name)
      .filter(Boolean)
      .join('; ');
    if (ev) parts.push(`Próximos eventos: ${ev}`);
  }

  return parts.join('\n') || 'No hay contexto semanal disponible.';
}

export function buildSystemPrompt(asset: AnalysisAsset, context: string): string {
  return `Eres un asistente financiero del sistema de trading para ${asset.display_name} (${asset.symbol}). Tienes acceso al contexto de la semana actual incluyendo señales de modelos, indicadores macro, y noticias relevantes.

Reglas:
- Responde en español
- Sé conciso (máximo 300 palabras por respuesta)
- Basa tus respuestas en los datos proporcionados en el contexto
- Si no tienes información suficiente, dilo
- No des consejos de inversión directos — presenta datos y análisis

Contexto semanal:
${context}`;
}

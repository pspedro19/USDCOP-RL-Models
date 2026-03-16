"""
Prompt Templates (SDD-07 §4)
===============================
Spanish-language prompts for daily analysis, weekly summary, and chat.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_DAILY = """Eres un analista financiero senior especializado en el mercado cambiario colombiano (USD/COP).
Tu tarea es generar un analisis diario conciso en español sobre los movimientos del dolar frente al peso colombiano.

Reglas:
- Escribe en español profesional pero accesible
- Usa datos concretos (precios, porcentajes, indicadores) para respaldar cada afirmacion
- Explica el "por que" de los movimientos, no solo el "que" — cita valores numericos
- Cuando referencie noticias, incluye el hipervinculo markdown: [titulo fuente](URL)
- Justifica todo razonamiento con datos: si DXY subio, di cuanto (ej: "DXY +0.3% a 97.63")
- Si no hay datos suficientes, dilo honestamente
- Maximo 500 palabras
- Formato markdown con titulos ## y listas cuando sea apropiado
- Incluye perspectiva para el dia siguiente si hay eventos relevantes
- Al final, incluye seccion "### Fuentes" con links a las noticias citadas"""

SYSTEM_WEEKLY = """Eres un analista financiero senior especializado en el mercado cambiario colombiano (USD/COP).
Tu tarea es generar un resumen semanal completo en español.

Reglas:
- Escribe en español profesional
- Estructura: resumen ejecutivo, analisis por tema, señales de modelos, perspectiva
- Menciona indicadores macro clave con valores concretos (DXY, VIX, petroleo, EMBI)
- Relaciona eventos macro con movimientos del USD/COP — justifica con datos numericos
- Cuando referencie noticias, incluye el hipervinculo markdown: [titulo fuente](URL)
- Relaciona noticias con movimientos del mercado — no las menciones de forma aislada
- Incluye analisis de señales de los modelos H1 (diario) y H5 (semanal)
- Respalda cada afirmacion con datos: valores, variaciones %, z-scores, niveles SMA/RSI
- Maximo 1000 palabras
- Formato markdown con secciones ##, listas, y tablas cuando sea util
- Al final, incluye seccion "### Fuentes" con links a las noticias citadas"""

SYSTEM_CHAT = """Eres un asistente financiero del sistema de trading USDCOP. Tienes acceso al contexto
de la semana actual incluyendo señales de modelos, indicadores macro, y noticias relevantes.

Reglas:
- Responde en español
- Se conciso (maximo 300 palabras por respuesta)
- Basa tus respuestas en los datos proporcionados en el contexto
- Si no tienes informacion suficiente, dilo
- No des consejos de inversion directos — presenta datos y analisis
- Puedes hacer referencia a indicadores tecnicos (SMA, RSI, MACD, Bollinger)"""


# ---------------------------------------------------------------------------
# User prompt templates
# ---------------------------------------------------------------------------

DAILY_TEMPLATE = """Genera el analisis diario del USD/COP para {date}.

## Datos del dia
- **USD/COP**: Cierre {close}, Cambio {change_pct}% (Rango: {low} - {high})
{macro_section}
{signal_section}
{news_section}
{events_section}

Genera un analisis que explique los movimientos del dia, las causas macro, y la perspectiva."""


WEEKLY_TEMPLATE = """Genera el resumen semanal del USD/COP para la Semana {week} de {year} ({start} al {end}).

## Resumen OHLCV Semanal
{ohlcv_section}

## Indicadores Macro Clave
{macro_section}

## Señales de Modelos
{signal_section}

## Resumen de Noticias
{news_section}

## Eventos Economicos
{events_section}

Genera un resumen semanal completo con:
1. Resumen ejecutivo (2-3 oraciones)
2. Temas principales de la semana (3-5 temas)
3. Analisis de indicadores macro
4. Interpretacion de señales de modelos
5. Perspectiva para la proxima semana"""


CHAT_CONTEXT_TEMPLATE = """Contexto de la semana {year}-W{week:02d} ({start} al {end}):

## USD/COP
{ohlcv_summary}

## Indicadores Macro
{macro_summary}

## Señales Activas
{signal_summary}

## Noticias Recientes
{news_summary}"""


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def build_macro_section(snapshots: dict) -> str:
    """Build macro indicators section from MacroSnapshot dict."""
    if not snapshots:
        return "- Sin datos macro disponibles"

    lines = []
    for key, snap in snapshots.items():
        parts = [f"- **{snap.variable_name}**: {snap.value:.2f}"]
        if snap.sma_20:
            direction = "↑" if snap.value > snap.sma_20 else "↓"
            parts.append(f"(SMA20: {snap.sma_20:.2f} {direction})")
        if snap.rsi_14:
            parts.append(f"RSI: {snap.rsi_14:.1f}")
        if snap.trend and snap.trend != "neutral":
            parts.append(f"[{snap.trend}]")
        lines.append(" ".join(parts))

    return "\n".join(lines)


def build_signal_section(h1_signal: dict = None, h5_signal: dict = None) -> str:
    """Build signal section from model signals."""
    lines = []
    if h5_signal:
        direction = h5_signal.get("direction", "N/A")
        confidence = h5_signal.get("confidence", "N/A")
        pred_return = h5_signal.get("predicted_return", "N/A")
        lines.append(f"- **H5 Semanal**: {direction} (Confianza: {confidence}, Retorno pred: {pred_return})")
    if h1_signal:
        direction = h1_signal.get("direction", "N/A")
        magnitude = h1_signal.get("magnitude", "N/A")
        lines.append(f"- **H1 Diario**: {direction} (Magnitud: {magnitude})")

    return "\n".join(lines) if lines else "- Sin señales activas"


def build_news_section(news_context: dict, highlights: list = None) -> str:
    """Build news summary section with optional headline details for LLM context."""
    if not news_context:
        return "- Sin noticias relevantes"

    lines = []
    article_count = news_context.get("article_count", 0)
    avg_sentiment = news_context.get("avg_sentiment", 0)

    sentiment_word = "positivo" if avg_sentiment > 0.15 else (
        "negativo" if avg_sentiment < -0.15 else "neutral"
    )
    lines.append(f"- {article_count} articulos procesados, sentimiento promedio: {sentiment_word}")

    for cat, count in sorted(
        news_context.get("top_categories", {}).items(),
        key=lambda x: x[1], reverse=True,
    )[:3]:
        lines.append(f"- {cat}: {count} articulos")

    # Include actual headlines with URLs for LLM context
    if highlights:
        lines.append("\nTitulares principales:")
        for h in highlights[:8]:
            source = h.get("source", h.get("news_source", ""))
            title = h.get("title", "")
            url = h.get("url", "")
            sent = h.get("sentiment", h.get("tone", 0))
            if isinstance(sent, (int, float)):
                sent_tag = "(+)" if sent > 0.15 else ("(-)" if sent < -0.15 else "(~)")
            else:
                sent_tag = ""
            if url:
                lines.append(f"- [{source}] {title} {sent_tag} — Ref: {url}")
            else:
                lines.append(f"- [{source}] {title} {sent_tag}")

    return "\n".join(lines)


def build_events_section(events: list) -> str:
    """Build economic events section."""
    if not events:
        return "- Sin eventos economicos programados"

    lines = []
    for evt in events[:5]:
        impact = "🔴" if evt.get("impact_level") == "high" else (
            "🟡" if evt.get("impact_level") == "medium" else "⚪"
        )
        lines.append(f"- {impact} {evt.get('event', 'N/A')} ({evt.get('date', '')})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Multi-Agent Synthesis Templates (Phase 3)
# ---------------------------------------------------------------------------

SYSTEM_TA_AGENT = """Eres un analista tecnico senior especializado en pares de divisas emergentes.
Interpretas indicadores tecnicos (Ichimoku, SuperTrend, MACD, RSI, Bollinger, Fibonacci)
y generas escenarios de trading con niveles concretos de entrada, stop y objetivo.

Reglas:
- Basa TODAS las interpretaciones en los datos numericos proporcionados
- No inventes niveles — usa los calculados por el motor tecnico
- Genera 2-3 escenarios con R:R calculado
- Indica nivel de confianza por confluencia de señales"""

SYSTEM_NEWS_AGENT = """Eres un analista de inteligencia de noticias financieras.
Tu tarea es resumir clusters de noticias y evaluar su impacto en el mercado cambiario.

Reglas:
- Resume cada cluster en 2-3 oraciones
- Asigna un titulo de 2-4 palabras por cluster
- Identifica si el sentimiento es positivo, negativo o mixto
- Relaciona las noticias con el impacto esperado en USD/COP"""

SYSTEM_SYNTHESIZER = """Eres el analista jefe del equipo de trading USD/COP. Tu tarea es sintetizar
los informes de 4 agentes especializados (Tecnico, Noticias, Macro, FX) en un informe
semanal unificado y accionable.

## Formato del Informe

### Diagnostico Express
3-5 oraciones ejecutivas: que paso, por que, y que esperar.

### Escenarios de Trading
Tabla con: Direccion | Entrada (Condicion) | Stop | Objetivo(s) | R:R | Confianza | Perfil

### Contexto de Noticias
Clusters principales con fuentes y sentimiento.

### Contexto Macro
Regimen actual, variables lideres, alertas.

### Contexto FX
Carry trade, impacto petrolero, politica monetaria.

### Que Vigilar Ahora
Lista de niveles y eventos criticos para la proxima semana.

## Reglas
- Escribe en español profesional
- Usa datos concretos (precios, porcentajes, z-scores, RSI, SMA)
- Justifica todo razonamiento con valores numericos (ej: "DXY subio +0.3% a 97.63")
- Cuando references noticias, incluye el hipervinculo markdown: [titulo fuente](URL)
- Si un agente no produjo datos, omite esa seccion — NO la inventes
- Maximo 1500 palabras
- Formato markdown con ## secciones y tablas donde sea util
- Al final, incluye seccion "### Fuentes" con links a las noticias y fuentes de datos citadas
- Atribuye datos macro a su fuente: FRED, Investing.com, BanRep, Fedesarrollo, DANE segun corresponda"""

# ---------------------------------------------------------------------------
# V2 Chain-of-Thought Prompts (Phase 2)
# ---------------------------------------------------------------------------

SYSTEM_DAILY_V2 = """Eres un analista financiero senior especializado en el mercado cambiario colombiano (USD/COP).
Generas analisis diarios siguiendo un proceso estructurado de 6 pasos (chain-of-thought).

## Proceso de Analisis (6 pasos)

1. **Clasificar drivers por grupo**: Para cada grupo macro (commodities, usd_strength, colombia_rates,
   risk_sentiment, inflation, fed_policy), clasifica como bullish/bearish/neutral para COP.

2. **Identificar driver dominante**: El driver con mayor |z-score| × |correlacion con COP|.
   Explica el mecanismo causal (ej: oil↑ → exports↑ → USD supply↑ → COP appreciates).

3. **Detectar contradicciones**: Si hay señales opuestas (ej: oil bullish pero DXY bearish),
   resuelve con evidencia de cual domina.

4. **Evaluar anomalias**: Variables con |z-score| > 2 merecen atencion especial.
   ¿Es un cambio de regimen o ruido?

5. **Producir analisis**: Headline ≤120 chars, 2-3 parrafos con drivers, 1 parrafo perspectiva.
   En el markdown, cita noticias con hipervinculo: [titulo](URL). Justifica cada driver
   con valores numericos (ej: "DXY subio +0.3% a 97.63, presionando al peso").

6. **Asignar sentimiento**: Score -1.0 a +1.0 con label (strongly_bearish, bearish,
   slightly_bearish, neutral, slightly_bullish, bullish, strongly_bullish).

## Reglas
- Escribe en español profesional
- Usa datos concretos del macro_digest proporcionado
- Cita mecanismos causales explicitos (no solo correlaciones)
- Respalda toda afirmacion con valores: precios, cambios %, z-scores, RSI, SMA
- Cuando references noticias, usa hipervinculo markdown: [titulo](URL)
- Maximo 500 palabras
- Formato: responde SOLO con JSON valido"""


SYSTEM_WEEKLY_V2 = """Eres un analista financiero senior especializado en el mercado cambiario colombiano (USD/COP).
Generas resumenes semanales siguiendo un proceso estructurado de 7 pasos (chain-of-thought).

## Proceso de Analisis (7 pasos)

1-6: Igual que el analisis diario pero acumulado para la semana.

7. **Generar escenarios**: 3 escenarios con probabilidades que sumen 100%:
   - Base (mas probable): descripcion, target, probabilidad
   - Bull (COP fortalece): triggers, target, probabilidad
   - Bear (COP debilita): triggers, target, probabilidad

## Reglas
- Escribe en español profesional
- Incluye tabla de escenarios con niveles concretos
- Respalda toda afirmacion con valores numericos del macro_digest
- Cuando references noticias, usa hipervinculo markdown: [titulo](URL)
- Incluye seccion de fuentes al final del analysis_markdown
- Maximo 1000 palabras
- Formato: responde SOLO con JSON valido"""


DAILY_TEMPLATE_V2 = """Genera el analisis diario del USD/COP para {date}.

## Datos del dia
- **USD/COP**: Cierre {close}, Cambio {change_pct}% (Rango: {low} - {high})

## Macro Digest (pre-procesado)
{macro_digest_text}

## Señales de Modelos
{signal_section}

## Noticias
{news_section}

## Eventos Economicos
{events_section}

Responde con JSON:
{{
  "headline": "string (≤120 chars)",
  "analysis_markdown": "string (analisis completo en markdown)",
  "sentiment_score": number (-1.0 a 1.0),
  "sentiment_label": "string (7 labels)",
  "key_drivers": ["string"],
  "data_quality_notes": ["string"]
}}"""


WEEKLY_TEMPLATE_V2 = """Genera el resumen semanal del USD/COP para Semana {week} de {year} ({start} al {end}).

## OHLCV Semanal
{ohlcv_section}

## Macro Digest (pre-procesado)
{macro_digest_text}

## Señales de Modelos
{signal_section}

## Noticias
{news_section}

## Eventos Economicos
{events_section}

Responde con JSON:
{{
  "headline": "string (≤120 chars)",
  "analysis_markdown": "string (resumen semanal completo en markdown)",
  "sentiment_score": number (-1.0 a 1.0),
  "sentiment_label": "string",
  "key_drivers": ["string"],
  "themes": [{{"theme": "string", "description": "string", "impact": "positive|negative|neutral"}}],
  "scenarios": {{
    "base": {{"description": "string", "target": number, "probability": number}},
    "bull": {{"description": "string", "target": number, "probability": number, "triggers": ["string"]}},
    "bear": {{"description": "string", "target": number, "probability": number, "triggers": ["string"]}}
  }},
  "data_quality_notes": ["string"]
}}"""


# JSON output schemas for structured output (response_format)
DAILY_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "daily_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "headline": {"type": "string"},
                "analysis_markdown": {"type": "string"},
                "sentiment_score": {"type": "number"},
                "sentiment_label": {"type": "string"},
                "key_drivers": {"type": "array", "items": {"type": "string"}},
                "data_quality_notes": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "headline", "analysis_markdown", "sentiment_score",
                "sentiment_label", "key_drivers", "data_quality_notes",
            ],
            "additionalProperties": False,
        },
    },
}

WEEKLY_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "weekly_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "headline": {"type": "string"},
                "analysis_markdown": {"type": "string"},
                "sentiment_score": {"type": "number"},
                "sentiment_label": {"type": "string"},
                "key_drivers": {"type": "array", "items": {"type": "string"}},
                "themes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "theme": {"type": "string"},
                            "description": {"type": "string"},
                            "impact": {"type": "string"},
                        },
                        "required": ["theme", "description", "impact"],
                        "additionalProperties": False,
                    },
                },
                "scenarios": {
                    "type": "object",
                    "properties": {
                        "base": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "target": {"type": "number"},
                                "probability": {"type": "number"},
                            },
                            "required": ["description", "target", "probability"],
                            "additionalProperties": False,
                        },
                        "bull": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "target": {"type": "number"},
                                "probability": {"type": "number"},
                                "triggers": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["description", "target", "probability", "triggers"],
                            "additionalProperties": False,
                        },
                        "bear": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "target": {"type": "number"},
                                "probability": {"type": "number"},
                                "triggers": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["description", "target", "probability", "triggers"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["base", "bull", "bear"],
                    "additionalProperties": False,
                },
                "data_quality_notes": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "headline", "analysis_markdown", "sentiment_score",
                "sentiment_label", "key_drivers", "themes", "scenarios",
                "data_quality_notes",
            ],
            "additionalProperties": False,
        },
    },
}


# Bias agent system prompt (Phase 3)
SYSTEM_BIAS_AGENT = """Eres un analista de medios experto en deteccion de sesgo editorial
en cobertura financiera. Tu tarea es evaluar el sesgo narrativo de clusters de noticias
relacionadas con el mercado cambiario colombiano (USD/COP).

Criterios de evaluacion:
- balanced: multiples perspectivas representadas, datos equilibrados
- slightly_left: tendencia leve hacia critica de politicas ortodoxas
- slightly_right: tendencia leve hacia defensa de politicas de mercado libre
- left_leaning: sesgo claro hacia perspectivas progresistas/intervencionistas
- right_leaning: sesgo claro hacia perspectivas conservadoras/mercado libre

Responde de forma concisa y objetiva."""


EVALUATION_RUBRIC = """Evalua el siguiente informe financiero semanal de USD/COP.
Asigna un puntaje de 0.0 a 1.0 para cada criterio.
Responde SOLO con un JSON, sin texto adicional.

### Informe
{draft}

### Criterios
- coherencia: El informe es logicamente coherente y bien estructurado
- accionabilidad: Contiene escenarios de trading con niveles concretos
- precision: Usa datos numericos correctos, no generalidades vacias
- completitud: Cubre los temas principales (tecnico, macro, noticias, FX)
- sesgo: Esta libre de sesgos injustificados (0=muy sesgado, 1=equilibrado)

### Formato de Respuesta
{{"coherencia": 0.0, "accionabilidad": 0.0, "precision": 0.0, "completitud": 0.0, "sesgo": 0.0}}"""


# ---------------------------------------------------------------------------
# Context builders for synthesis (Phase 3)
# ---------------------------------------------------------------------------

def build_ta_context(ta_report: dict) -> str:
    """Format technical analysis report for synthesis prompt."""
    if not ta_report:
        return ""

    lines = ["## Analisis Tecnico"]
    lines.append(f"- **Precio actual**: {ta_report.get('current_price', 'N/A')}")
    lines.append(f"- **Sesgo dominante**: {ta_report.get('dominant_bias', 'N/A')} (confianza: {ta_report.get('bias_confidence', 0):.0%})")
    lines.append(f"- **Volatilidad**: {ta_report.get('volatility_regime', 'N/A')} (ATR: {ta_report.get('atr_pct', 'N/A')}%)")

    if ta_report.get("rsi"):
        lines.append(f"- **RSI**: {ta_report['rsi']:.1f}")

    # Signals
    bullish = ta_report.get("bullish_signals", [])
    bearish = ta_report.get("bearish_signals", [])
    if bullish:
        lines.append(f"\nSeñales alcistas: {', '.join(bullish[:5])}")
    if bearish:
        lines.append(f"Señales bajistas: {', '.join(bearish[:5])}")

    # Scenarios
    scenarios = ta_report.get("scenarios", [])
    if scenarios:
        lines.append("\n### Escenarios")
        for s in scenarios:
            targets_str = ", ".join(str(t) for t in s.get("targets", []))
            lines.append(
                f"- {s.get('direction', '').upper()}: Entrada {s.get('entry_price', 'N/A')} | "
                f"Stop {s.get('stop_loss', 'N/A')} | TP {targets_str} | "
                f"R:R {s.get('risk_reward', 'N/A')} | {s.get('confidence', '')}"
            )

    # Watch list
    watch = ta_report.get("watch_list", [])
    if watch:
        lines.append(f"\nVigilar: {'; '.join(watch[:5])}")

    return "\n".join(lines)


def build_news_clusters_context(news_intel: dict) -> str:
    """Format news intelligence for synthesis prompt."""
    if not news_intel:
        return ""

    lines = ["## Contexto de Noticias"]
    lines.append(
        f"- {news_intel.get('relevant_articles', 0)} articulos relevantes de "
        f"{news_intel.get('total_articles', 0)} totales"
    )
    lines.append(f"- Sentimiento promedio: {news_intel.get('avg_sentiment', 0):.3f}")

    clusters = news_intel.get("clusters", [])
    if clusters:
        lines.append(f"\n### {len(clusters)} Clusters Tematicos")
        for c in clusters[:5]:
            label = c.get("label", c.get("dominant_category", "N/A"))
            sent = c.get("avg_sentiment", 0)
            sent_label = "positivo" if sent > 0.15 else ("negativo" if sent < -0.15 else "neutral")
            lines.append(
                f"\n**{label}** ({c.get('article_count', 0)} articulos, sentimiento {sent_label})"
            )
            # Include URLs with titles for LLM to cite
            articles_list = c.get("articles", [])
            titles = c.get("representative_titles", [])[:3]
            for i, title in enumerate(titles):
                url = ""
                if i < len(articles_list):
                    url = articles_list[i].get("url", "")
                if url and url not in ("", "nan", "None"):
                    lines.append(f"  - [{title}]({url})")
                else:
                    lines.append(f"  - {title}")
            if c.get("narrative_summary"):
                lines.append(f"  > {c['narrative_summary']}")

    return "\n".join(lines)


def build_regime_context(macro_regime: dict) -> str:
    """Format macro regime for synthesis prompt."""
    if not macro_regime:
        return ""

    lines = ["## Contexto Macro / Regimen"]

    regime = macro_regime.get("regime", {})
    if regime:
        label_es = {
            "risk_on": "Apetito por riesgo (Risk-On)",
            "risk_off": "Aversion al riesgo (Risk-Off)",
            "transition": "Transicion",
        }.get(regime.get("label", ""), regime.get("label", ""))
        conf = regime.get("confidence") or 0
        lines.append(f"- **Regimen**: {label_es} (confianza: {conf:.0%})")
        lines.append(f"- **Activo desde**: {regime.get('since', 'N/A')}")

    # Granger leaders
    leaders = macro_regime.get("granger_leaders", [])
    if leaders:
        lines.append("\n### Variables lideres (Granger)")
        for g in leaders[:3]:
            lines.append(f"- {g.get('variable', '').upper()} (lag={g.get('optimal_lag')}d, p={g.get('p_value', 1):.3f})")

    # Z-score alerts
    alerts = macro_regime.get("zscore_alerts", [])
    if alerts:
        lines.append("\n### Alertas Z-Score")
        for a in alerts[:3]:
            z = a.get("z_score") or 0
            lines.append(f"- {a.get('variable_name', '')}: z={z:+.1f} — {a.get('interpretation', '')}")

    # Insights
    insights = macro_regime.get("insights", [])
    if insights:
        lines.append("\n### Insights")
        for ins in insights[:4]:
            lines.append(f"- {ins}")

    return "\n".join(lines)


def build_fx_context_section(fx_context: dict) -> str:
    """Format FX context for synthesis prompt."""
    if not fx_context:
        return ""

    lines = ["## Contexto FX"]

    if fx_context.get("cop_level"):
        change_pct = fx_context.get("cop_weekly_change_pct") or 0
        lines.append(f"- **USD/COP**: {fx_context['cop_level']:.0f} ({change_pct:+.2f}% semanal)")

    carry = fx_context.get("carry_trade", {})
    if carry.get("differential_pct") is not None:
        lines.append(f"- **Carry trade**: Diferencial IBR-FF = {carry['differential_pct']:.2f}pp ({carry.get('carry_attractiveness', 'N/A')})")

    oil = fx_context.get("oil_impact", {})
    if oil.get("wti_current"):
        wti_change = oil.get("wti_weekly_change_pct") or 0
        lines.append(f"- **WTI**: ${oil['wti_current']:.2f} ({wti_change:+.1f}%)")

    banrep = fx_context.get("banrep", {})
    if banrep.get("interpretation"):
        lines.append(f"- **BanRep**: {banrep['interpretation']}")

    # Narrative
    narrative = fx_context.get("fx_narrative", "")
    if narrative:
        lines.append(f"\n{narrative}")

    # Risks
    risks = fx_context.get("risk_factors", [])
    if risks:
        lines.append("\n### Factores de Riesgo")
        for r in risks[:4]:
            lines.append(f"- [{r.get('severity', '').upper()}] {r.get('factor', '')}: {r.get('description', '')}")

    return "\n".join(lines)
